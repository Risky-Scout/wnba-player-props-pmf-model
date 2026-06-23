"""Game-Total-Conditioned Player Props (Enhancement 20).

Anchors player projections to the game-total market — the MOST efficiently
priced WNBA market (more liquidity, more books, more sharp action).

When the sum of independent player point projections diverges from the game-
total market implied mean, this module applies a proportional scaling factor
so that the projections are COHERENT with the game-total:

    scale_home = (game_total × home_share) / sum(projected_pts_home)
    conditioned_pts_home = projected_pts_home × scale_home
    conditioned_ast_home = projected_ast_home × (1 + 0.3 × (scale - 1))
    conditioned_reb_home = projected_reb_home  # no scaling (not pts-linked)

Reference
---------
Terner & Franks (2020). Modeling Player and Team Performance in Basketball.
Annual Review of Statistics and Its Application.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Standard deviation of the game-total distribution (calibrated for WNBA)
# WNBA mean game total ≈ 155; σ ≈ 11 (empirical from 2021-2024 seasons)
WNBA_GAME_TOTAL_SIGMA = 11.0

# Stats whose projections scale with pts (partial scaling for assists)
FULL_SCALE_STATS    = ["pts", "fg3m", "turnover"]
PARTIAL_SCALE_STATS = ["ast"]       # scale factor = 1 + 0.3 * (s - 1)
NO_SCALE_STATS      = ["reb", "stl", "blk"]  # not directly pts-linked


def sample_game_total(
    market_line:      float,
    market_over_odds: float | None = None,
    sigma:            float = WNBA_GAME_TOTAL_SIGMA,
    n_samples:        int = 1,
) -> float | np.ndarray:
    """Sample one or more game totals from the market-implied Normal distribution.

    Parameters
    ----------
    market_line      : the posted over/under line.
    market_over_odds : American odds on the over side (e.g. -110).
                       Used to shift the implied mean.  If None, treats the
                       line as the distributional mean (symmetric market).
    sigma            : standard deviation of the game-total distribution.
    n_samples        : number of samples to draw.

    Returns
    -------
    float (single) or np.ndarray (n_samples > 1)
    """
    # Adjust mean if market is not exactly 50/50
    if market_over_odds is not None and market_over_odds != 0:
        try:
            # Convert American odds to implied probability
            if market_over_odds < 0:
                p_over = (-market_over_odds) / (-market_over_odds + 100)
            else:
                p_over = 100 / (market_over_odds + 100)
            p_over = np.clip(p_over, 0.10, 0.90)
            # Shift mean so that P(X > market_line) ≈ p_over
            z = sp_stats.norm.ppf(1.0 - p_over)
            mu = market_line - z * sigma
        except Exception:
            mu = market_line
    else:
        mu = market_line

    samples = np.random.normal(mu, sigma, n_samples)
    samples = np.clip(samples, 80.0, 260.0)   # WNBA historical range

    return samples  # always return ndarray for consistency


def condition_player_props_on_total(
    player_projections:   list[dict[str, Any]],
    game_total:           float,
    home_team_share:      float = 0.50,
) -> list[dict[str, Any]]:
    """Adjust player projections to be coherent with a sampled game total.

    Parameters
    ----------
    player_projections : list of dicts; each must have:
        "team" ("home" or "away"), "{stat}_projection" for each stat.
    game_total         : sampled total game score.
    home_team_share    : fraction of total scored by home team (default 0.50).

    Returns
    -------
    Same list with added "{stat}_projection_conditioned" keys.
    """
    home_budget = game_total * home_team_share
    away_budget = game_total * (1.0 - home_team_share)

    for team, budget in [("home", home_budget), ("away", away_budget)]:
        team_players = [p for p in player_projections if p.get("team", "home") == team]
        if not team_players:
            continue

        total_projected_pts = sum(
            p.get("pts_projection", 0.0) for p in team_players
        )
        if total_projected_pts < 1.0:
            continue

        scale = budget / total_projected_pts

        for p in team_players:
            # Points and correlated counting stats
            for stat in FULL_SCALE_STATS:
                k = f"{stat}_projection"
                if k in p:
                    p[f"{k}_conditioned"] = p[k] * scale

            # Assists: partial scaling (0.3 of full adjustment)
            for stat in PARTIAL_SCALE_STATS:
                k = f"{stat}_projection"
                if k in p:
                    partial_scale = 1.0 + 0.3 * (scale - 1.0)
                    p[f"{k}_conditioned"] = p[k] * partial_scale

            # Rebounds, steals, blocks: no scaling
            for stat in NO_SCALE_STATS:
                k = f"{stat}_projection"
                if k in p:
                    p[f"{k}_conditioned"] = p[k]

            p["game_total_conditioning_scale"] = round(float(scale), 4)

    return player_projections


def apply_game_total_conditioning(
    player_projections:   list[dict[str, Any]],
    market_line:          float,
    market_over_odds:     float | None = None,
    home_team_share:      float = 0.50,
    sigma:                float = WNBA_GAME_TOTAL_SIGMA,
    n_mc_samples:         int = 1,
) -> list[dict[str, Any]]:
    """Sample a game total and apply conditioning to player projections.

    If n_mc_samples > 1, returns the AVERAGE conditioned projection across
    multiple game-total samples (Monte Carlo integration).
    """
    if n_mc_samples == 1:
        game_total = float(sample_game_total(market_line, market_over_odds, sigma, n_samples=1)[0])
        return condition_player_props_on_total(player_projections, game_total, home_team_share)

    # Monte Carlo: average conditioned projections across samples
    totals = sample_game_total(market_line, market_over_odds, sigma, n_samples=n_mc_samples)
    all_scales: dict[int, list[float]] = {}

    import copy
    base = copy.deepcopy(player_projections)
    for i, proj in enumerate(base):
        all_scales[i] = []

    for gt in totals:
        conditioned = condition_player_props_on_total(
            copy.deepcopy(player_projections), float(gt), home_team_share
        )
        for i, p in enumerate(conditioned):
            all_scales[i].append(p.get("game_total_conditioning_scale", 1.0))

    # Apply average scale
    for i, p in enumerate(base):
        avg_scale = float(np.mean(all_scales[i]))
        for stat in FULL_SCALE_STATS:
            k = f"{stat}_projection"
            if k in p:
                p[f"{k}_conditioned"] = p[k] * avg_scale
        for stat in PARTIAL_SCALE_STATS:
            k = f"{stat}_projection"
            if k in p:
                p[f"{k}_conditioned"] = p[k] * (1.0 + 0.3 * (avg_scale - 1.0))
        for stat in NO_SCALE_STATS:
            k = f"{stat}_projection"
            if k in p:
                p[f"{k}_conditioned"] = p[k]
        p["game_total_conditioning_scale"] = round(avg_scale, 4)

    logger.info(
        "E20: game-total conditioning applied (line=%.1f, mean_scale=%.3f)",
        market_line,
        float(np.mean([p.get("game_total_conditioning_scale", 1.0) for p in base])),
    )
    return base


def mc_condition_player_props(
    player_projections: list[dict[str, Any]],
    game_total:         float,
    home_team_share:    float = 0.50,
) -> list[dict[str, Any]]:
    """Alias for condition_player_props_on_total used by deliver.py.

    Returns conditioned projections; each dict gains a
    "coherence_scale_factor" key.
    """
    result = condition_player_props_on_total(player_projections, game_total, home_team_share)
    for p in result:
        p["coherence_scale_factor"] = p.get("game_total_conditioning_scale", 1.0)
    return result
