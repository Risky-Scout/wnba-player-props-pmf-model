"""Bayesian shrinkage for small-sample player projections.

Implements empirical Bayes shrinkage following PenaltyBlog's recommendation
for handling players with limited history. The principle: for a player with
n games, blend their model prediction toward the league prior with weight
proportional to (k / (n + k)), where k controls prior strength.

This prevents the model from generating overconfident projections for:
  - Rookies (first season)
  - Players returning from injury
  - Players with very few games in the current season

Reference: PenaltyBlog — Bayesian Approaches to Player Projection,
  "Use a hierarchical prior: as observation count → 0, projection → league mean"

Usage:
    from wnba_props_model.models.shrinkage import apply_bayesian_shrinkage
    pmfs_shrunken = apply_bayesian_shrinkage(pmfs_long, features_wide)
"""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# League-average stat means (2024-2025 WNBA, used as empirical prior)
# These are updated weekly from the historical feature table
_LEAGUE_PRIORS: dict[str, float] = {
    "pts": 9.5,
    "reb": 4.2,
    "ast": 2.1,
    "fg3m": 1.1,
    "stl": 0.9,
    "blk": 0.5,
    "turnover": 1.7,
}

# League-average projected minutes (used to build Poisson prior PMFs)
_LEAGUE_MINUTES_PRIOR: float = 22.0

# PenaltyBlog shrinkage strength: k = effective prior sample size
# Interpretation: with k=15, a player with 15 games is 50% shrunken toward the prior
_DEFAULT_SHRINKAGE_K: float = 15.0

# Minimum games before shrinkage is applied (very experienced players are not shrunken)
_MIN_GAMES_FOR_FULL_CONFIDENCE: int = 40


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


def compute_shrinkage_weight(n_games: int, k: float = _DEFAULT_SHRINKAGE_K) -> float:
    """Return alpha (shrinkage weight toward prior) given n_games and strength k.

    alpha = k / (n + k)
    - n=0:  alpha = 1.0 (pure prior)
    - n=k:  alpha = 0.5 (equal weight)
    - n=40: alpha ≈ 0.27 (mostly model)
    - n=∞:  alpha = 0.0 (pure model)
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
    k: float = _DEFAULT_SHRINKAGE_K,
    min_games_full_confidence: int = _MIN_GAMES_FOR_FULL_CONFIDENCE,
    league_priors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply PenaltyBlog-style Bayesian shrinkage to PMFs for small-sample players.

    For players with few games (< min_games_full_confidence):
      blended_pmf = (1 - alpha) * model_pmf + alpha * prior_pmf
      where alpha = k / (n_games + k)

    Players with >= min_games_full_confidence are not modified.

    Parameters
    ----------
    pmfs_long: Long PMF DataFrame (player_id, stat, pmf_json, mean, ...)
    features:  Wide feature DataFrame (for n_games and league prior computation)
    k:         Shrinkage strength (prior effective sample size)
    """
    # Build league priors
    if league_priors is None:
        if features is not None:
            league_priors = compute_league_priors_from_data(features)
        else:
            league_priors = dict(_LEAGUE_PRIORS)

    # Build player n_games lookup
    player_ngames: dict[int, int] = {}
    if features is not None and "player_id" in features.columns:
        for pid, grp in features.groupby("player_id"):
            n = int(grp.get("player_games_prior", pd.Series([0])).max() or 0)
            if n == 0:
                # Fall back to counting actual rows
                n = len(grp.dropna(subset=["actual_pts"] if "actual_pts" in grp.columns else []))
            player_ngames[int(pid)] = n

    # Stat support caps (matching pmf_engine)
    _STAT_CAPS = {"pts": 60, "reb": 30, "ast": 25, "fg3m": 15, "stl": 10, "blk": 10, "turnover": 12}

    rows_modified = 0
    out_rows = []
    for _, row in pmfs_long.iterrows():
        pid = int(row["player_id"])
        stat = str(row["stat"])
        n_games = player_ngames.get(pid, 50)  # assume experienced if unknown

        alpha = compute_shrinkage_weight(n_games, k)
        if alpha < 0.05 or n_games >= min_games_full_confidence:
            # Experienced player — no shrinkage needed
            out_rows.append(row.to_dict())
            continue

        # Parse model PMF
        try:
            pmf_dict = json.loads(row["pmf_json"])
            max_k = max(int(k_) for k_ in pmf_dict)
            model_pmf = np.array([pmf_dict.get(str(i), 0.0) for i in range(max_k + 1)], dtype=float)
            if model_pmf.sum() < 1e-9:
                out_rows.append(row.to_dict())
                continue
            model_pmf /= model_pmf.sum()
        except Exception:
            out_rows.append(row.to_dict())
            continue

        # Build prior PMF (Poisson with league mean)
        league_mean = league_priors.get(stat, 5.0)
        cap = _STAT_CAPS.get(stat, 30)
        prior_pmf = _poisson_pmf(league_mean, cap)

        # Blend
        blended = _blend_pmfs(model_pmf, prior_pmf, alpha)
        new_mean = float(np.arange(len(blended)) @ blended)

        # Encode back to JSON
        new_pmf_json = json.dumps({str(i): round(float(v), 6) for i, v in enumerate(blended) if v > 1e-8})
        new_median = int(np.searchsorted(np.cumsum(blended), 0.5))
        new_mode = int(np.argmax(blended))
        new_p0 = float(blended[0]) if len(blended) > 0 else 0.0

        r = row.to_dict()
        r["pmf_json"] = new_pmf_json
        r["mean"] = round(new_mean, 4)
        r["median"] = new_median
        r["mode"] = new_mode
        r["p0"] = round(new_p0, 6)
        r["shrinkage_alpha"] = round(alpha, 4)
        r["n_games_sample"] = n_games
        out_rows.append(r)
        rows_modified += 1

    logger.info(
        "Bayesian shrinkage applied to %d / %d PMF rows (k=%.1f)",
        rows_modified, len(pmfs_long), k,
    )
    return pd.DataFrame(out_rows)
