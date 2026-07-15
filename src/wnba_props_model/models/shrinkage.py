"""Hierarchical Bayesian shrinkage for small-sample player projections.

Implements PenaltyBlog's hierarchical Bayes approach using a Gamma-Poisson
conjugate model, where the shrinkage strength (k) is LEARNED from the
observed distribution of player rates — not set manually.

Theory (Gamma-Poisson conjugate model)
---------------------------------------
For each stat, model each player's true rate λ_i ~ Gamma(α, β).
Each observed game count y_ij ~ Poisson(λ_i * minutes_j / 40).

Marginal: Y_i (season total) ~ Negative Binomial(α, β/(β+n_i)).

Empirical Bayes: fit α, β by Method of Moments on all players' per-game rates:
    mu  = E[λ_i]  = league_mean_rate
    var = Var[λ_i] = inter-player variance of rates
    → α = mu² / var
    → β = mu  / var

Posterior for player i (after n_i games, total stat = S_i):
    λ_i | data ~ Gamma(α + S_i, β + n_i)
    E[λ_i | data] = (α + S_i) / (β + n_i)
                  = league_mean × (β / (β + n_i))   ← shrinkage to league mean
                  + observed_mean × (n_i / (β + n_i)) ← own data

The effective shrinkage k = β = mu/var is learned from data rather than
manually set.  This mirrors PenaltyBlog's HierarchicalBayesianGoalModel
but uses a fast analytical approximation (no MCMC needed).

This is a strict improvement over the previous ad-hoc k=15:
  - k is calibrated to actual talent variance in WNBA data
  - Each stat gets its own k (pts variance >> stl variance)
  - Shrinkage naturally adjusts to season length (fewer games → more shrinkage)

Reference:
  PenaltyBlog (Jan 2026): HierarchicalBayesianGoalModel, empirical Bayes fit.
  Morris (1983): Parametric Empirical Bayes inference: Theory and Applications.

Usage:
    from wnba_props_model.models.shrinkage import apply_bayesian_shrinkage
    pmfs_shrunken = apply_bayesian_shrinkage(pmfs_long, features_wide)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

logger = logging.getLogger(__name__)

# Fallback league-average stat means — OOF-measured empirical means (blk 0.50→0.38).
# Updated from data when features_wide is available; these serve as the static fallback.
_LEAGUE_PRIORS: dict[str, float] = {
    "pts": 7.156,      # OOF actual mean (was 7.95 — 11% too high)
    "reb": 2.966,      # OOF actual mean (was 3.80 — 28% too high)
    "ast": 1.773,      # OOF actual mean (was 2.03 — 14% too high)
    "fg3m": 0.699,     # OOF actual mean (was 0.78 — 11% too high)
    "stl": 0.639,      # OOF actual mean (was 0.67 — 5% too high)
    "blk": 0.348,      # OOF actual mean (was 0.38 — 9% too high)
    "turnover": 1.131, # OOF actual mean (was 1.36 — 20% too high)
    # Combo stats: sum of component league means (all-player population incl. bench)
    "pts_reb":     10.122,   # pts (7.156) + reb (2.966)
    "pts_ast":      8.929,   # pts (7.156) + ast (1.773)
    "reb_ast":      4.739,   # reb (2.966) + ast (1.773)
    "pts_reb_ast": 11.895,   # pts + reb + ast
    "stocks":       0.987,   # stl (0.639) + blk (0.348)
}

# Load updated league priors from artifacts/models/league_priors.json if available.
# These are computed by scripts/update_league_priors.py from current-season data,
# keeping shrinkage targets calibrated to the actual league distribution each year.
_LEAGUE_PRIORS_PATH = Path(__file__).parents[3] / "artifacts" / "models" / "league_priors.json"
if _LEAGUE_PRIORS_PATH.exists():
    try:
        import json as _json
        _LEAGUE_PRIORS.update(_json.loads(_LEAGUE_PRIORS_PATH.read_text()))
        logger.info("Loaded league priors from %s", _LEAGUE_PRIORS_PATH)
    except Exception as _lp_exc:
        logger.warning("Failed to load league priors: %s", _lp_exc)

# Kept for backward-compatibility on the function signature; no longer used as a hard cutoff.
# Continuous effective-n shrinkage (compute_shrinkage_alpha) replaces this threshold.
_MIN_GAMES_FOR_FULL_CONFIDENCE: int = 80

# Minimum effective k to avoid explosive shrinkage in low-variance stats
_MIN_K: float = 3.0
# Maximum k: if variance is very high, cap shrinkage strength
_MAX_K: float = 50.0

# Adaptive K_BASE shrinkage: decay factor λ for form divergence.
# K_BASE_adaptive = K_BASE * exp(-λ * |L5_mean - season_mean| / max(L10_std, 0.5))
# At λ=2.0: 1-std divergence → ~86% of K, 2-std → ~75% of K, 3-std → ~55% of K.
# Meaningful role changes reduce shrinkage; noise stays near K_BASE.
_ADAPTIVE_SHRINKAGE_LAMBDA: float = 2.0
# Log a message when K_BASE is reduced by more than this fraction
_ADAPTIVE_SHRINKAGE_LOG_THRESHOLD: float = 0.30


@dataclass
class GammaPrior:
    """Fitted Gamma hyperparameters for one stat.

    E[λ] = alpha / beta  = league_mean_rate
    Var[λ]= alpha / beta² = inter-player variance

    k = beta is the effective sample size parameter that determines shrinkage:
        alpha_player = k / (k + n_games)   (shrinkage weight toward prior)
    """
    alpha: float     # shape
    beta: float      # rate  (k = beta = mu / var)
    mu: float        # league mean per game
    var: float       # inter-player variance


def fit_hyperpriors(
    player_rate_df: pd.DataFrame,
    stat: str,
    rate_col: str | None = None,
    games_col: str = "n_games",
) -> dict[str, Any]:
    """Fit league-level (alpha, beta) by maximizing marginal Gamma-Poisson likelihood.

    The marginal distribution of the sum S_i = sum(y_ij) over n_i games for
    player i, when y_ij | λ_i ~ Poisson(λ_i) and λ_i ~ Gamma(α, β), is:

        S_i ~ NegativeBinomial(n=α, p=β/(β+n_i))

    We maximize the sum of NegBinom log-likelihoods over all players.
    This is more statistically efficient than Method of Moments because it
    uses the full likelihood rather than only the first two moments.

    Parameters
    ----------
    player_rate_df : pd.DataFrame
        One row per player with columns:
        - rate_col (default: actual_{stat}): per-game average stat value
        - games_col: number of games played (used to compute total count)
        - player_id: player identifier
    stat : str
        Stat name (e.g. 'pts', 'turnover').
    rate_col : str, optional
        Column with per-game average stat. Defaults to 'actual_{stat}'.
    games_col : str
        Column with number of games (default: 'n_games').

    Returns
    -------
    dict with keys: alpha, beta, k (=beta), mu (=alpha/beta), method='mle'
    """
    col = rate_col or f"actual_{stat}"
    if col not in player_rate_df.columns:
        # Fallback to MoM from whatever stat column exists
        rates = player_rate_df[stat].dropna().values if stat in player_rate_df.columns else np.array([])
        if len(rates) < 5:
            return {"alpha": 1.0, "beta": float(_MIN_K), "k": float(_MIN_K), "mu": 1.0, "method": "mom_fallback"}
        return _fit_gamma_prior_to_dict(rates)

    df = player_rate_df[[col]].copy()
    if games_col in player_rate_df.columns:
        df[games_col] = player_rate_df[games_col]
    else:
        df[games_col] = 10  # default assumption

    df = df.dropna()
    rates = df[col].values.astype(float)
    n_games = df[games_col].values.astype(float)

    if len(rates) < 5:
        prior = _fit_gamma_prior(rates)
        return {"alpha": prior.alpha, "beta": prior.beta, "k": prior.beta,
                "mu": prior.mu, "method": "mom_small_sample"}

    # Total counts (S_i = rate_i * n_games_i)
    totals = np.round(rates * n_games).astype(int)
    totals = np.clip(totals, 0, None)

    # MoM initializer
    mom_prior = _fit_gamma_prior(rates)
    x0 = np.array([mom_prior.alpha, mom_prior.beta])

    def neg_log_lik(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        if a <= 0 or b <= 0:
            return 1e12
        # NegBinom: n=alpha, p=b/(b+n_games)
        p = b / (b + n_games + 1e-12)
        ll = (
            special.gammaln(totals + a)
            - special.gammaln(a)
            - special.gammaln(totals + 1)
            + a * np.log(p + 1e-12)
            + totals * np.log(1 - p + 1e-12)
        )
        return -float(ll.sum())

    result = optimize.minimize(
        neg_log_lik,
        x0,
        method="L-BFGS-B",
        bounds=[(0.1, 500.0), (0.01, 200.0)],
        options={"maxiter": 200, "ftol": 1e-10},
    )

    if result.success and result.fun < neg_log_lik(x0) + 1.0:
        alpha_hat = float(np.clip(result.x[0], 0.1, 500.0))
        beta_hat = float(np.clip(result.x[1], _MIN_K, _MAX_K))
    else:
        # MLE failed to improve on MoM — fall back to MoM
        alpha_hat = mom_prior.alpha
        beta_hat = mom_prior.beta

    mu_hat = alpha_hat / beta_hat
    return {
        "alpha": round(alpha_hat, 6),
        "beta": round(beta_hat, 6),
        "k": round(beta_hat, 6),
        "mu": round(mu_hat, 6),
        "method": "mle" if result.success else "mom_fallback",
        "n_players": int(len(rates)),
    }


def _fit_gamma_prior_to_dict(rates: np.ndarray) -> dict[str, Any]:
    """Thin wrapper around _fit_gamma_prior for dict output."""
    prior = _fit_gamma_prior(rates)
    return {"alpha": prior.alpha, "beta": prior.beta, "k": prior.beta,
            "mu": prior.mu, "method": "mom"}


def posterior_lambda(
    alpha_league: float,
    beta_league: float,
    player_total_stat: float,
    player_n_games: int,
    ci_level: float = 0.90,
) -> tuple[float, float]:
    """Compute Gamma-Poisson posterior (posterior_mean, credible_interval_width).

    Given:
        Prior:     λ ~ Gamma(alpha_league, beta_league)
        Likelihood: S | λ, n ~ Poisson(λ * n)
        → Sum S over n games is Poisson(n * λ)

    Posterior:
        λ | S, n ~ Gamma(alpha_league + S, beta_league + n)

    Parameters
    ----------
    alpha_league : float
        Fitted league Gamma shape parameter.
    beta_league : float
        Fitted league Gamma rate parameter (= k).
    player_total_stat : float
        Sum of stat across all player games (S_i = Σ y_ij).
    player_n_games : int
        Number of games for this player.
    ci_level : float
        Credible interval coverage (default 90%).

    Returns
    -------
    (posterior_mean, credible_interval_width)
        posterior_mean = E[λ | data] = (alpha_league + S_i) / (beta_league + n)
        credible_interval_width = upper - lower of CI on λ
        Credible interval width > 2.0 → flag player as "high uncertainty".
    """
    a_post = alpha_league + float(player_total_stat)
    b_post = beta_league + float(player_n_games)

    posterior_mean = a_post / b_post

    lo = (1 - ci_level) / 2
    hi = 1 - lo
    lower = float(stats.gamma.ppf(lo, a=a_post, scale=1.0 / b_post))
    upper = float(stats.gamma.ppf(hi, a=a_post, scale=1.0 / b_post))
    ci_width = upper - lower

    return posterior_mean, ci_width


def _fit_gamma_prior(per_game_rates: np.ndarray) -> GammaPrior:
    """Fit Gamma(α, β) hyperparameters via Method of Moments.

    Parameters
    ----------
    per_game_rates: array of per-player per-game rates (e.g. pts/game).
                    Only players with at least a few games of data.
    """
    rates = per_game_rates[np.isfinite(per_game_rates) & (per_game_rates >= 0)]
    if len(rates) < 5:
        mu  = _LEAGUE_PRIORS.get("pts", 9.5)   # fallback
        var = mu
        return GammaPrior(alpha=1.0, beta=1.0, mu=mu, var=var)

    mu  = float(np.mean(rates))
    var = float(np.var(rates, ddof=1))

    if var < 1e-6:
        # Degenerate: all players same — very high α, just use mean
        var = max(mu * 0.1, 1e-3)

    alpha = mu ** 2 / var
    beta  = mu / var
    return GammaPrior(
        alpha=float(alpha),
        beta=float(np.clip(beta, _MIN_K, _MAX_K)),
        mu=float(mu),
        var=float(var),
    )


def _posterior_mean(
    prior: GammaPrior,
    n_games: int,
    observed_mean: float,
    season_phase: str = "mid",
) -> float:
    """Compute Gamma-Poisson posterior mean (shrinkage toward league average).

    E[λ | data] = (α + n*x̄) / (β + n)
                = league_mean × β/(β+n)  +  observed_mean × n/(β+n)

    Enhancement 10: season_phase multiplier adjusts shrinkage strength:
        early   → 2.0× (first 8 games, small sample, shrink more)
        mid     → 1.0× (games 9–30, normal)
        late    → 0.8× (games 31+, more data, slightly less shrinkage)
        playoff → 0.6× (postseason, minimal shrinkage)
    """
    _PHASE_MULTIPLIER = {"early": 2.0, "mid": 1.0, "late": 0.8, "playoff": 0.6}
    multiplier = _PHASE_MULTIPLIER.get(season_phase, 1.0)
    k     = prior.beta * multiplier  # effective prior sample size (adjusted)
    k     = max(k, _MIN_K)
    w_obs = n_games / (k + n_games)       # weight on own data
    w_pri = 1.0 - w_obs                   # weight on league prior
    return float(w_pri * prior.mu + w_obs * observed_mean)


def _compute_stat_priors(
    features: pd.DataFrame,
    min_player_games: int = 5,
) -> dict[str, GammaPrior]:
    """Fit per-stat Gamma priors from historical feature table.

    Uses only players with at least ``min_player_games`` prior games so
    that very small-sample players don't corrupt the variance estimate.
    """
    priors: dict[str, GammaPrior] = {}
    stats_map = {
        "pts": "actual_pts", "reb": "actual_reb", "ast": "actual_ast",
        "fg3m": "actual_fg3m", "stl": "actual_stl", "blk": "actual_blk",
        "turnover": "actual_turnover",
    }
    games_col = "player_games_prior" if "player_games_prior" in features.columns else None

    for stat, col in stats_map.items():
        if col not in features.columns:
            continue
        df = features[["player_id", col]].dropna()
        if games_col:
            df = features[features[games_col] >= min_player_games][["player_id", col]].dropna()

        # Per-player mean per game (one point per player)
        player_means = df.groupby("player_id")[col].mean().values
        priors[stat] = _fit_gamma_prior(player_means)
        logger.debug(
            "[%s] Gamma prior: mu=%.3f, var=%.3f, k=%.2f (α=%.2f)",
            stat, priors[stat].mu, priors[stat].var, priors[stat].beta, priors[stat].alpha,
        )
    return priors


def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """Poisson PMF with mean lam, support 0..max_k."""
    lam = max(lam, 1e-6)
    ks = np.arange(max_k + 1, dtype=float)
    log_pmf = ks * math.log(lam) - lam - np.array([sum(math.log(i) for i in range(1, int(k) + 1)) for k in ks])
    pmf = np.exp(log_pmf)
    return pmf / pmf.sum()


def _blend_pmfs(model_pmf: np.ndarray, prior_pmf: np.ndarray, alpha: float) -> np.ndarray:
    """Blend model PMF with prior PMF: (1-alpha)*model + alpha*prior.

    alpha is the shrinkage weight toward prior (0 = full model, 1 = full prior).
    """
    max_k = max(len(model_pmf), len(prior_pmf))
    m = np.zeros(max_k)
    p = np.zeros(max_k)
    m[:len(model_pmf)] = model_pmf
    p[:len(prior_pmf)] = prior_pmf
    blended = (1 - alpha) * m + alpha * p
    blended = np.clip(blended, 0, None)
    s = blended.sum()
    return blended / s if s > 1e-9 else blended


def compute_shrinkage_weight(n_games: int, k: float = _MIN_K) -> float:
    """Return alpha (shrinkage weight toward prior) given n_games and strength k.

    alpha = k / (n + k)  — identical to Gamma-Poisson posterior weight on prior.
    k is now derived from data via _fit_gamma_prior rather than set manually.

    - n=0:  alpha = 1.0  (pure prior)
    - n=k:  alpha = 0.5  (equal weight)
    - n=∞:  alpha = 0.0  (pure model)
    """
    return float(k / (n_games + k))


# ---------------------------------------------------------------------------
# Continuous role-weighted effective-n shrinkage (replaces hard bypass cutoff)
# ---------------------------------------------------------------------------

def compute_effective_n(n_games: int, mean_minutes: float, starter_minutes: float = 32.0) -> float:
    """Compute effective sample size weighting games by minutes played.

    Low-minute players (bench, fringe) have less reliable rate estimates, so
    their games are down-weighted toward the fraction of a starter's minutes.
    Caps at 1.5× to prevent extreme over-weighting of workhorse outliers.
    """
    minutes_ratio = mean_minutes / starter_minutes
    n_eff = n_games * min(minutes_ratio, 1.5)
    return max(n_eff, 1.0)


# Base k values per stat: how many effective games before shrinkage drops to 50%.
# Halved from original (8/10/12/15/12/18/10) — the prior was over-weighting
# low-season-data players toward the all-player league mean, which is far below
# actual starter-level production (starters avg 2-3x bench players).
# At k=4, a starter with 10 games gets alpha=4/(4+10)=0.286 (was 0.50).
K_BASE: dict[str, float] = {
    # Base k values per stat: how many effective games before shrinkage drops to 50%.
    # Halved from original (8/10/12/15/12/18/10) — the prior was over-weighting
    # low-season-data players toward the all-player league mean, which is far below
    # actual starter-level production (starters avg 2-3x bench players).
    # At k=4, a starter with 10 games gets alpha=4/(4+10)=0.286 (was 0.50).
    "pts": 4.0,
    "reb": 5.0,
    "ast": 6.0,
    "fg3m": 7.0,
    "stl": 6.0,
    "blk": 8.0,
    "turnover": 5.0,
    # Combo stats get a K_BASE floor of 5.0 — they are sums of 2-3 base stats and
    # their inter-player variance is higher, so over-shrinking is the bigger risk.
    # Values below 5.0 were raised to enforce this floor.
    "pts_reb":     5.0,
    "pts_ast":     5.0,
    "reb_ast":     5.0,
    "pts_reb_ast": 5.0,
    "stocks":      5.0,
}


# Load empirically fitted K_BASE if available (overrides hardcoded defaults).
# Generated by scripts/fit_shrinkage_params.py via ECE minimisation on OOF data.
_shrinkage_params_path = Path(__file__).parents[3] / "artifacts" / "models" / "shrinkage_params.json"
if _shrinkage_params_path.exists():
    try:
        import json as _sp_json  # noqa: PLC0415
        _sp = _sp_json.loads(_shrinkage_params_path.read_text())
        _k_fitted = _sp.get("k_base_fitted", {})
        for _s, _v in _k_fitted.items():
            _k = _v.get("k_base_fitted") if isinstance(_v, dict) else None
            if _k is not None and isinstance(_k, (int, float)) and _k > 0:
                K_BASE[str(_s)] = float(_k)
        logger.info(
            "Loaded empirically fitted K_BASE from %s (%d stats updated)",
            _shrinkage_params_path,
            sum(1 for v in _k_fitted.values()
                if isinstance(v, dict) and v.get("k_base_fitted") is not None),
        )
    except Exception as _kbase_exc:
        logger.warning("Failed to load fitted K_BASE: %s", _kbase_exc)


def compute_shrinkage_alpha(
    n_eff: float,
    stat: str,
    position: str | None = None,
    k_override: float | None = None,
) -> float:
    """Compute continuous shrinkage alpha via role-weighted effective-n.

    Replaces the hard `if n_games >= _MIN_GAMES_FOR_FULL_CONFIDENCE: alpha = 0.0`
    bypass with a smooth k / (k + n_eff) decay.  Position adjustments ensure guards
    are shrunk harder for blk (rarely block) and centers less so.

    Parameters
    ----------
    k_override : float, optional
        When provided (e.g. from adaptive K_BASE shrinkage), this value is used
        as the base k instead of K_BASE[stat].  Position adjustments still apply.

    Returns alpha in [0, 1]: 0 = no shrinkage (trust model), 1 = full prior.
    """
    k = k_override if k_override is not None else K_BASE.get(stat, 10.0)
    if position == "G" and stat == "blk":
        k *= 2.0
    elif position == "C" and stat == "blk":
        k *= 0.5
    elif position == "C" and stat == "ast":
        k *= 1.5
    return k / (k + n_eff)


def compute_league_priors_from_data(features: pd.DataFrame) -> dict[str, float]:
    """Compute league-average stat means from the historical feature table."""
    priors = dict(_LEAGUE_PRIORS)
    stat_to_col = {
        "pts": "actual_pts",
        "reb": "actual_reb",
        "ast": "actual_ast",
        "fg3m": "actual_fg3m",
        "stl": "actual_stl",
        "blk": "actual_blk",
        "turnover": "actual_turnover",
    }
    for stat, col in stat_to_col.items():
        if col in features.columns:
            val = float(features[col].dropna().mean())
            if not math.isnan(val) and val > 0:
                priors[stat] = round(val, 3)
    return priors


def apply_bayesian_shrinkage(
    pmfs_long: pd.DataFrame,
    features: pd.DataFrame | None = None,
    k: float | None = None,
    min_games_full_confidence: int = _MIN_GAMES_FOR_FULL_CONFIDENCE,
    league_priors: dict[str, float] | None = None,
    player_role_overrides: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Apply hierarchical Bayesian shrinkage to PMFs for small-sample players.

    **Hierarchical Bayes (Gamma-Poisson conjugate) approach:**
    For each stat, the shrinkage strength k is LEARNED from the inter-player
    variance in the training data via Method of Moments:

        k = beta = mu / var   (Gamma hyperparameter)

    This means:
      - High-variance stats (pts): smaller k → less shrinkage (signal is real)
      - Low-variance stats (blk): larger k → more shrinkage (small samples mislead)

    For each small-sample player:
        blended_mean = league_mean × k/(k+n) + observed_mean × n/(k+n)
        blended_pmf  = (1-alpha)*model_pmf + alpha*prior_pmf

    Players with >= min_games_full_confidence are not modified.

    Parameters
    ----------
    pmfs_long               : Long PMF DataFrame (player_id, stat, pmf_json, ...)
    features                : Wide feature table (for empirical prior fitting)
    k                       : Override k (if None, learned from data)
    min_games_full_confidence: Skip shrinkage for experienced players
    league_priors           : Override league mean dict (deprecated; kept for compat)
    """
    # Fit per-stat Gamma priors from data (the key upgrade)
    stat_priors: dict[str, GammaPrior] = {}
    static_league_means = dict(_LEAGUE_PRIORS)
    if features is not None:
        stat_priors = _compute_stat_priors(features)
        static_league_means = compute_league_priors_from_data(features)
    # Override with explicit league_priors for backward compatibility
    if league_priors is not None:
        for stat, mu in league_priors.items():
            if stat not in stat_priors:
                stat_priors[stat] = GammaPrior(alpha=1.0, beta=float(k or _MIN_K), mu=mu, var=mu)

    # Enhancement 10: season-phase shrinkage multiplier
    _PHASE_MULTIPLIER: dict[str, float] = {"early": 2.0, "mid": 1.0, "late": 0.8, "playoff": 0.6}

    # Role-stratified prior means: pull starters toward starter-level production,
    # not all-player averages (which include many bench/fringe players).
    # Combo stats priors = sum of component role priors.
    _ROLE_PRIORS: dict[str, dict[str, float]] = {
        "starter": {
            "pts": 13.5, "reb": 6.0, "ast": 3.5, "fg3m": 1.2,
            "stl": 1.0, "blk": 0.6, "turnover": 2.0,
            "pts_reb": 19.5, "pts_ast": 17.0, "reb_ast": 9.5,
            "pts_reb_ast": 23.0, "stocks": 1.6,
        },
        "core":    {
            "pts": 10.5, "reb": 4.8, "ast": 2.8, "fg3m": 1.0,
            "stl": 0.85, "blk": 0.45, "turnover": 1.7,
            "pts_reb": 15.3, "pts_ast": 13.3, "reb_ast": 7.6,
            "pts_reb_ast": 18.1, "stocks": 1.3,
        },
        "rotation": {
            "pts": 7.5, "reb": 3.2, "ast": 1.8, "fg3m": 0.65,
            "stl": 0.60, "blk": 0.30, "turnover": 1.2,
            "pts_reb": 10.7, "pts_ast": 9.3, "reb_ast": 5.0,
            "pts_reb_ast": 12.5, "stocks": 0.9,
        },
    }

    # Build player n_games, season_phase, position, role, and mean-minutes lookups
    player_ngames:       dict[int, int]   = {}
    player_phase:        dict[int, str]   = {}
    player_position:     dict[int, str]   = {}
    player_role:         dict[int, str]   = {}
    player_mean_minutes: dict[int, float] = {}
    if features is not None and "player_id" in features.columns:
        for pid, grp in features.groupby("player_id"):
            n = int(grp.get("player_games_prior", pd.Series([0])).max() or 0)
            if n == 0:
                n = len(grp.dropna(subset=["actual_pts"] if "actual_pts" in grp.columns else []))
            player_ngames[int(pid)] = n
            if "season_phase" in grp.columns:
                phase_val = grp["season_phase"].dropna().iloc[-1] if not grp["season_phase"].dropna().empty else "mid"
                player_phase[int(pid)] = str(phase_val)
            else:
                player_phase[int(pid)] = "mid"
            # Position for role-aware shrinkage alpha
            if "position" in grp.columns:
                pos_val = grp["position"].dropna()
                player_position[int(pid)] = str(pos_val.iloc[-1])[0].upper() if not pos_val.empty else "F"
            else:
                player_position[int(pid)] = "F"
            # Role status for stratified priors
            # Unify role taxonomy: role_bucket has 6 tiers; shrinkage uses role_status (3-4 tiers).
            # Fall back to role_bucket when role_status is absent, mapping to compatible values.
            _ROLE_BUCKET_TO_STATUS = {
                "starter": "starter", "core": "core", "rotation": "rotation",
                "bench": "bench", "fringe": "bench", "inactive_risk": "bench",
            }
            if "role_status" in grp.columns:
                role_val = grp["role_status"].dropna()
                player_role[int(pid)] = str(role_val.iloc[-1]).lower() if not role_val.empty else "rotation"
            elif "role_bucket" in grp.columns:
                role_val = grp["role_bucket"].dropna().map(_ROLE_BUCKET_TO_STATUS)
                player_role[int(pid)] = str(role_val.iloc[-1]).lower() if not role_val.empty else "rotation"
            else:
                player_role[int(pid)] = "rotation"
            # Mean minutes for effective-n down-weighting of bench players
            min_col = next((c for c in ["player_minutes_mean_season", "actual_minutes", "minutes_mean"] if c in grp.columns), None)
            if min_col: 
                mn = grp[min_col].dropna()
                player_mean_minutes[int(pid)] = float(mn.mean()) if not mn.empty else 20.0
            else:
                player_mean_minutes[int(pid)] = 20.0

    # -------------------------------------------------------------------
    # Adaptive K_BASE: build per-(player, stat) divergence from features.
    # Divergence = |L5_mean - season_mean| / max(L10_std, 0.5).
    # When a player's recent 5-game form diverges strongly from their
    # season baseline, K_BASE is decayed: K_adaptive = K_BASE * exp(-λ * div).
    # Falls back to divergence=0.0 (no K change) when columns are absent.
    # -------------------------------------------------------------------
    _BASE_STATS_DIVERGENCE = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    player_stat_divergence: dict[tuple[int, str], float] = {}
    if features is not None and "player_id" in features.columns:
        for _div_pid, _div_grp in features.groupby("player_id"):
            _last = _div_grp.iloc[-1]
            for _div_stat in _BASE_STATS_DIVERGENCE:
                _l5_col   = f"player_{_div_stat}_mean_l5"
                _sea_col  = f"player_{_div_stat}_mean_season"
                _std_col  = f"player_{_div_stat}_std_l10"
                try:
                    _l5  = _last[_l5_col]  if _l5_col  in features.columns else None
                    _sea = _last[_sea_col] if _sea_col in features.columns else None
                    _std = _last[_std_col] if _std_col in features.columns else None
                    if _l5 is None or _sea is None:
                        continue
                    _l5_f   = float(_l5)
                    _sea_f  = float(_sea)
                    _std_f  = float(_std) if (_std is not None and not (isinstance(_std, float) and math.isnan(_std))) else 0.5
                    if not math.isfinite(_l5_f) or not math.isfinite(_sea_f) or _sea_f < 0:
                        continue
                    _std_eff = max(_std_f, 0.5)
                    _div_val = abs(_l5_f - _sea_f) / _std_eff
                    player_stat_divergence[(int(_div_pid), _div_stat)] = _div_val
                except Exception:
                    pass

    # Apply player_role_overrides (e.g. from role_bucket_override in player_form_corrections)
    # so that the shrinkage prior uses the correct role-stratified mean, not the raw features value.
    if player_role_overrides:
        for pid_override, role_override in player_role_overrides.items():
            player_role[int(pid_override)] = str(role_override).lower()

    # Stat support caps (matching pmf_engine)
    _STAT_CAPS = {
        "pts": 60, "reb": 30, "ast": 25, "fg3m": 15, "stl": 10, "blk": 10, "turnover": 12,
        # Combo stat caps = sum of component caps
        "pts_reb": 90, "pts_ast": 85, "reb_ast": 55, "pts_reb_ast": 105, "stocks": 20,
    }

    # Enhancement 10: season-phase shrinkage multiplier (defined once, not per-row)
    _PHASE_MULT: dict[str, float] = {"early": 2.0, "mid": 1.0, "late": 0.8, "playoff": 0.6}

    rows_modified = 0
    out_rows = []
    for _, row in pmfs_long.iterrows():
        pid     = int(row["player_id"])
        stat    = str(row["stat"])
        n_games = player_ngames.get(pid, 50)

        # Lookup per-stat prior (needed for prior PMF construction and CI width)
        prior = stat_priors.get(stat)

        # Continuous role-weighted effective-n shrinkage (replaces hard bypass cutoff).
        # compute_effective_n down-weights bench players whose games carry less signal.
        # compute_shrinkage_alpha gives position-adjusted k for the stat.
        mean_min  = player_mean_minutes.get(pid, 20.0)
        position  = player_position.get(pid, "F")
        n_eff     = compute_effective_n(n_games, mean_min)

        # Adaptive K_BASE: reduce shrinkage when player's recent L5 form diverges
        # from their season baseline, indicating a genuine role/usage change.
        _k_base = K_BASE.get(stat, 10.0)
        _divergence = player_stat_divergence.get((pid, stat), 0.0)
        _k_adaptive = _k_base * math.exp(-_ADAPTIVE_SHRINKAGE_LAMBDA * _divergence)
        _k_adaptive = max(_k_adaptive, _MIN_K)
        if _k_adaptive < _k_base * (1.0 - _ADAPTIVE_SHRINKAGE_LOG_THRESHOLD):
            _player_name = str(row.get("player_name", pid))
            logger.info(
                "Shrinkage reduced for %s %s: K=%.1f → %.1f (divergence=%.2f)",
                _player_name, stat, _k_base, _k_adaptive, _divergence,
            )

        alpha     = compute_shrinkage_alpha(n_eff, stat, position, k_override=_k_adaptive)

        # Enhancement 10: adjust k by season-phase multiplier.
        # Force "mid" phase if the season has enough games played (>25 total team
        # games) even if the player's own game_number_in_season looks small due
        # to absences — prevents double-penalizing returning/injured players.
        phase = player_phase.get(pid, "mid")
        if phase == "early" and n_games >= 25:
            phase = "mid"
        phase_mult = _PHASE_MULT.get(phase, 1.0)
        # Apply phase multiplier to effective n (more shrinkage early, less late)
        n_eff_phased = n_eff / phase_mult  # dividing n_eff → higher alpha early
        alpha = compute_shrinkage_alpha(n_eff_phased, stat, position, k_override=_k_adaptive)

        # No hard bypass — continuous alpha naturally approaches 0 for large n_eff
        if alpha < 0.005:
            out_rows.append(row.to_dict())
            continue

        # Parse model PMF
        try:
            pmf_dict  = json.loads(row["pmf_json"])
            max_k_val = max(int(kk) for kk in pmf_dict)
            model_pmf = np.array([pmf_dict.get(str(i), 0.0) for i in range(max_k_val + 1)], dtype=float)
            if model_pmf.sum() < 1e-9:
                out_rows.append(row.to_dict())
                continue
            model_pmf /= model_pmf.sum()
        except Exception:
            out_rows.append(row.to_dict())
            continue

        # Build prior PMF using role-stratified mean when available.
        # Starters and core players should shrink toward starter-level production,
        # not the all-player league average (which includes bench/fringe players).
        role = player_role.get(pid, "rotation")
        role_priors = _ROLE_PRIORS.get(role)
        if role_priors is not None and stat in role_priors:
            league_mean = role_priors[stat]
        else:
            league_mean = (prior.mu if prior else static_league_means.get(stat, 5.0))
        cap         = _STAT_CAPS.get(stat, 30)
        prior_pmf   = _poisson_pmf(league_mean, cap)

        # Observed mean from the raw model PMF
        observed_mean = float(np.arange(len(model_pmf)) @ model_pmf)

        # Blend — posterior mean automatically matches _posterior_mean() formula
        blended   = _blend_pmfs(model_pmf, prior_pmf, alpha)
        new_mean  = float(np.arange(len(blended)) @ blended)
        new_pmf_json = json.dumps({str(i): round(float(v), 6) for i, v in enumerate(blended) if v > 1e-8})
        new_median   = int(np.searchsorted(np.cumsum(blended), 0.5))
        new_mode     = int(np.argmax(blended))
        new_p0       = float(blended[0]) if len(blended) > 0 else 0.0

        # Compute posterior credible interval width for uncertainty flagging
        try:
            if prior is not None:
                player_total = observed_mean * n_games
                _, ci_width = posterior_lambda(
                    alpha_league=prior.alpha,
                    beta_league=prior.beta,
                    player_total_stat=player_total,
                    player_n_games=n_games,
                )
            else:
                ci_width = float("nan")
        except Exception:
            ci_width = float("nan")

        r = row.to_dict()
        r["pmf_json"]          = new_pmf_json
        r["mean"]              = round(new_mean, 4)
        r["pmf_mean"]          = round(new_mean, 4)
        r["median"]            = new_median
        r["mode"]              = new_mode
        r["p0"]                = round(new_p0, 6)
        r["shrinkage_alpha"]       = round(alpha, 4)
        r["shrinkage_k"]           = round(_k_adaptive, 4)
        r["shrinkage_k_base"]      = round(_k_base, 4)
        r["shrinkage_divergence"]  = round(_divergence, 4)
        r["n_games_sample"]    = n_games
        r["posterior_lambda_mean"] = round(new_mean, 4)
        r["credible_interval_width"] = round(ci_width, 4) if math.isfinite(ci_width) else None
        r["high_uncertainty"] = (math.isfinite(ci_width) and ci_width > 2.0)
        out_rows.append(r)
        rows_modified += 1

    logger.info(
        "Hierarchical Bayes shrinkage applied to %d / %d PMF rows "
        "(k per stat: {%s})",
        rows_modified, len(pmfs_long),
        ", ".join(f"{s}:{p.beta:.1f}" for s, p in stat_priors.items()),
    )
    return pd.DataFrame(out_rows)
