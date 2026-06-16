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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fallback league-average stat means (updated from data when features_wide available)
_LEAGUE_PRIORS: dict[str, float] = {
    "pts": 9.5,
    "reb": 4.2,
    "ast": 2.1,
    "fg3m": 1.1,
    "stl": 0.9,
    "blk": 0.5,
    "turnover": 1.7,
}

# Minimum games before shrinkage is bypassed entirely
_MIN_GAMES_FOR_FULL_CONFIDENCE: int = 40

# Minimum effective k to avoid explosive shrinkage in low-variance stats
_MIN_K: float = 3.0
# Maximum k: if variance is very high, cap shrinkage strength
_MAX_K: float = 50.0


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
) -> float:
    """Compute Gamma-Poisson posterior mean (shrinkage toward league average).

    E[λ | data] = (α + n*x̄) / (β + n)
                = league_mean × β/(β+n)  +  observed_mean × n/(β+n)
    """
    k     = prior.beta  # effective prior sample size
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

    # Build player n_games lookup
    player_ngames: dict[int, int] = {}
    if features is not None and "player_id" in features.columns:
        for pid, grp in features.groupby("player_id"):
            n = int(grp.get("player_games_prior", pd.Series([0])).max() or 0)
            if n == 0:
                n = len(grp.dropna(subset=["actual_pts"] if "actual_pts" in grp.columns else []))
            player_ngames[int(pid)] = n

    # Stat support caps (matching pmf_engine)
    _STAT_CAPS = {"pts": 60, "reb": 30, "ast": 25, "fg3m": 15, "stl": 10, "blk": 10, "turnover": 12}

    rows_modified = 0
    out_rows = []
    for _, row in pmfs_long.iterrows():
        pid   = int(row["player_id"])
        stat  = str(row["stat"])
        n_games = player_ngames.get(pid, 50)

        # Lookup per-stat prior (Gamma parameters)
        prior = stat_priors.get(stat)
        k_stat = float(k) if k is not None else (prior.beta if prior else _MIN_K)
        alpha  = compute_shrinkage_weight(n_games, k_stat)

        if alpha < 0.05 or n_games >= min_games_full_confidence:
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

        # Build prior PMF
        league_mean = (prior.mu if prior else static_league_means.get(stat, 5.0))
        cap         = _STAT_CAPS.get(stat, 30)
        prior_pmf   = _poisson_pmf(league_mean, cap)

        # Blend — posterior mean automatically matches _posterior_mean() formula
        blended   = _blend_pmfs(model_pmf, prior_pmf, alpha)
        new_mean  = float(np.arange(len(blended)) @ blended)
        new_pmf_json = json.dumps({str(i): round(float(v), 6) for i, v in enumerate(blended) if v > 1e-8})
        new_median   = int(np.searchsorted(np.cumsum(blended), 0.5))
        new_mode     = int(np.argmax(blended))
        new_p0       = float(blended[0]) if len(blended) > 0 else 0.0

        r = row.to_dict()
        r["pmf_json"]          = new_pmf_json
        r["mean"]              = round(new_mean, 4)
        r["pmf_mean"]          = round(new_mean, 4)
        r["median"]            = new_median
        r["mode"]              = new_mode
        r["p0"]                = round(new_p0, 6)
        r["shrinkage_alpha"]   = round(alpha, 4)
        r["shrinkage_k"]       = round(k_stat, 4)
        r["n_games_sample"]    = n_games
        out_rows.append(r)
        rows_modified += 1

    logger.info(
        "Hierarchical Bayes shrinkage applied to %d / %d PMF rows "
        "(k per stat: {%s})",
        rows_modified, len(pmfs_long),
        ", ".join(f"{s}:{p.beta:.1f}" for s, p in stat_priors.items()),
    )
    return pd.DataFrame(out_rows)
