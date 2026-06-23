"""Game-Total Conditioned Player Props (Enhancement 20).

The game total market is the MOST EFFICIENTLY PRICED WNBA market
(more liquidity, more books, more sharp action).  By conditioning player
props on a sampled game total, the model anchors individual projections
to the market's best estimate of team scoring — preventing the most
common coherence error: 8 starters summing to 90 pts when the game total
implies only 75 pts per team.

Strategy:
    1. Sample a game total from the market-implied Normal distribution.
    2. Compute the "budget" available for each team.
    3. Scale player stats proportionally so sum(player_pts) ≈ team_budget.
    4. Apply partial scaling to correlated stats (ast, fg3m) and
       no scaling to orthogonal stats (reb, stl, blk).

The scaling factor is:
    scale = team_budget / sum(projected_pts)

If scale ≈ 1.0, the model's projections are already coherent and no
adjustment is made.

Reference:
    Terner & Franks (2020). Modeling Player and Team Performance in Basketball.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Stats that scale WITH team scoring (partially or fully)
FULL_SCALE_STATS  = {"pts"}
PARTIAL_SCALE_STATS = {"ast": 0.60, "fg3m": 0.40, "turnover": 0.20}
NO_SCALE_STATS    = {"reb", "stl", "blk"}

# WNBA historical sigma for game totals (pts per game, both teams combined)
# Based on 2022-2024 WNBA season data: mean ≈ 160, SD ≈ 11
WNBA_GAME_TOTAL_SIGMA: float = 11.0


def sample_game_total(
    market_line: float,
    market_over_odds: float = -110,
    sigma: float = WNBA_GAME_TOTAL_SIGMA,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample game totals from the market-implied distribution.

    The market line + vig-free implied probability define a Normal
    distribution over the game total.

    Parameters
    ----------
    market_line      : the posted over/under line (e.g. 162.5)
    market_over_odds : American odds on the over (e.g. -110 → 110 vig)
    sigma            : std dev of game total distribution (default: WNBA-calibrated)
    n_samples        : number of Monte Carlo samples

    Returns
    -------
    1D array of sampled game totals (floored at 100).
    """
    rng = rng or np.random.default_rng()

    # Vig-free implied probability of the over
    if market_over_odds < 0:
        imp_prob = (-market_over_odds) / (-market_over_odds + 100)
    else:
        imp_prob = 100 / (market_over_odds + 100)
    imp_prob = float(np.clip(imp_prob, 0.01, 0.99))

    # Infer mean from vig-free probability assuming symmetric market
    # If imp_prob ≈ 0.5, market_line ≈ true mean
    mu = market_line  # simple assumption; can be refined with vig removal

    samples = rng.normal(mu, sigma, n_samples)
    return np.maximum(samples, 100.0)


def condition_player_props_on_total(
    player_projections: list[dict[str, Any]],
    game_total: float,
    home_team_id: Any | None = None,
    home_scoring_share: float = 0.50,
) -> list[dict[str, Any]]:
    """Adjust player projections to be coherent with a sampled game total.

    Parameters
    ----------
    player_projections  : list of player projection dicts, each with:
        - team            : team identifier
        - pts_projection  : base pts projection
        - reb_projection  : base reb projection
        - ast_projection  : etc.
    game_total          : sampled game total (both teams combined)
    home_team_id        : optional team id to split home vs away
    home_scoring_share  : fraction of game total attributed to home team (default 0.50)

    Returns
    -------
    player_projections enriched with *_conditioned columns.
    """
    if game_total <= 0 or not player_projections:
        return player_projections

    # Determine team budgets
    teams = list({p.get("team") for p in player_projections})
    if len(teams) == 2:
        t1, t2 = teams
        budgets = {t1: game_total * home_scoring_share,
                   t2: game_total * (1 - home_scoring_share)}
    else:
        budget_per_team = game_total / max(len(teams), 1)
        budgets = {t: budget_per_team for t in teams}

    result = []
    for team, budget in budgets.items():
        team_players = [p for p in player_projections if p.get("team") == team]
        total_proj   = sum(
            float(p.get("pts_projection", p.get("pts_proj", 0.0)))
            for p in team_players
        )
        scale = budget / total_proj if total_proj > 0 else 1.0

        # Clamp scale: don't adjust by more than ±50%
        scale = float(np.clip(scale, 0.50, 1.50))

        for p in team_players:
            conditioned = dict(p)
            for stat in list(FULL_SCALE_STATS) + list(PARTIAL_SCALE_STATS.keys()) + list(NO_SCALE_STATS):
                proj_key = f"{stat}_projection"
                if proj_key not in p:
                    proj_key = f"{stat}_proj"
                if proj_key not in p:
                    continue

                base = float(p[proj_key])
                if math.isnan(base):
                    conditioned[f"{stat}_projection_conditioned"] = base
                    continue

                if stat in FULL_SCALE_STATS:
                    conditioned[f"{stat}_projection_conditioned"] = base * scale
                elif stat in PARTIAL_SCALE_STATS:
                    partial = PARTIAL_SCALE_STATS[stat]
                    conditioned[f"{stat}_projection_conditioned"] = (
                        base * (1 + partial * (scale - 1))
                    )
                else:  # NO_SCALE_STATS
                    conditioned[f"{stat}_projection_conditioned"] = base

            conditioned["coherence_scale_factor"] = round(scale, 4)
            conditioned["implied_team_total"]      = round(budget, 2)
            result.append(conditioned)

    # Append any players without a team match (shouldn't happen)
    matched_ids = {id(p) for grp_team in budgets for p in
                   player_projections if p.get("team") == grp_team}
    for p in player_projections:
        if id(p) not in matched_ids:
            result.append(p)

    return result


def mc_condition_player_props(
    player_projections: list[dict[str, Any]],
    market_line: float,
    market_over_odds: float = -110,
    n_samples: int = 1_000,
    home_scoring_share: float = 0.50,
) -> list[dict[str, Any]]:
    """Monte Carlo version: sample n game totals and average conditioned projections.

    This produces expectation-corrected projections that account for
    uncertainty in the game total, not just a single point estimate.

    Parameters
    ----------
    player_projections : same format as condition_player_props_on_total
    market_line        : posted game total line
    market_over_odds   : American odds on the over
    n_samples          : MC samples over game total distribution

    Returns
    -------
    player_projections with averaged *_conditioned columns.
    """
    rng = np.random.default_rng(42)
    totals = sample_game_total(market_line, market_over_odds, n_samples=n_samples, rng=rng)

    # Accumulate conditioned projections across MC samples
    stat_accumulators: dict[int, dict[str, list[float]]] = {
        i: {} for i in range(len(player_projections))
    }

    for total in totals:
        conditioned = condition_player_props_on_total(
            player_projections, float(total), home_scoring_share=home_scoring_share
        )
        for i, p in enumerate(conditioned):
            for k, v in p.items():
                if k.endswith("_conditioned") and isinstance(v, float):
                    stat_accumulators[i].setdefault(k, []).append(v)

    # Average over samples
    result = [dict(p) for p in player_projections]
    for i, acc in stat_accumulators.items():
        for k, vals in acc.items():
            result[i][k] = round(float(np.mean(vals)), 4) if vals else math.nan

    return result
