"""Pi Ratings for player form and opponent defensive strength.

PenaltyBlog's Pi Ratings track form as a running exponential moving average
of the "surprise" (actual - predicted).  In soccer they track home and away
ratings separately because home advantage is large.  WNBA has a modest home
advantage, but travel patterns differ enough that separate home/away form
ratings capture real signal.

Implementation
--------------
For each player × stat:
    form_home[t+1] = (1 - α) * form_home[t] + α * (actual[t] - predicted[t])
    form_away[t+1] = (1 - α) * form_away[t] + α * (actual[t] - predicted[t])

where `predicted[t]` is the rolling mean (l5) as a simple baseline.

For each team × stat (opponent defensive rating):
    def_pi[t+1] = (1 - β) * def_pi[t] + β * (opp_actual - opp_expected)

All ratings are *shift-1* (the rating entering game t uses information from
games 0..t-1 only) so they are leak-free features.

The output is a DataFrame with one row per (player_id, game_id) and columns:
    player_{stat}_pi_home_form   — home-game residual momentum
    player_{stat}_pi_away_form   — away-game residual momentum
    team_{stat}_def_pi           — team defensive suppression rating (per stat)

These are written to the feature table by ``build_pi_rating_features`` and
should be treated as ordinary numeric features (already shift-1 safe).

Usage
-----
    from wnba_props_model.models.pi_ratings import build_pi_rating_features

    wide = pd.read_parquet("data/processed/wnba_player_game_features_wide.parquet")
    pi_df = build_pi_rating_features(wide)
    # pi_df has same index / row count as wide; merge on (player_id, game_id)
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Learning rates (tuned for ~40-game WNBA season)
_PLAYER_FORM_ALPHA  = 0.25   # player residual form update speed
_TEAM_DEF_BETA      = 0.20   # team defense Pi update speed

_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover")


def _pi_update_group(
    actual: np.ndarray,
    predicted: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return shift-1 Pi rating for a sorted chronological series.

    result[0] = 0.0  (no prior information)
    result[i] = (1-α)*result[i-1] + α*(actual[i-1] - predicted[i-1])

    The shift-1 guarantee: result[i] depends only on games 0..i-1.
    """
    n = len(actual)
    ratings = np.zeros(n, dtype=float)
    for i in range(1, n):
        prev_residual = float(actual[i - 1]) - float(predicted[i - 1])
        if not np.isfinite(prev_residual):
            prev_residual = 0.0
        ratings[i] = (1.0 - alpha) * ratings[i - 1] + alpha * prev_residual
    return ratings


def build_pi_rating_features(
    wide_df: pd.DataFrame,
    stats: Iterable[str] = _STATS,
    player_alpha: float = _PLAYER_FORM_ALPHA,
    team_beta: float = _TEAM_DEF_BETA,
) -> pd.DataFrame:
    """Compute Pi rating features for each (player_id, game_id) row.

    Assumes ``wide_df`` is sorted by (player_id, game_date) or that
    (player_id, game_date, game_id) sorts are already applied.

    Returns a DataFrame indexed the same as wide_df with Pi columns added.
    If a required actual-stat column is missing, the corresponding Pi column
    is filled with NaN (graceful degradation).

    Parameters
    ----------
    wide_df     : wide feature table (one row per player × game)
    stats       : stat names to compute Pi ratings for
    player_alpha: learning rate for player form EMA
    team_beta   : learning rate for team defense EMA
    """
    df = wide_df.copy()
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    pi_cols: list[str] = []
    stats_list = list(stats)

    # ------------------------------------------------------------------
    # 1. Player form Pi ratings (home & away separate)
    # ------------------------------------------------------------------
    for stat in stats_list:
        actual_col   = f"actual_{stat}"
        baseline_col = f"player_{stat}_mean_l5"
        home_pi_col  = f"player_{stat}_pi_home_form"
        away_pi_col  = f"player_{stat}_pi_away_form"

        for pi_col in (home_pi_col, away_pi_col):
            df[pi_col] = np.nan
        pi_cols += [home_pi_col, away_pi_col]

        if actual_col not in df.columns or baseline_col not in df.columns:
            continue
        if "is_home" not in df.columns:
            continue

        player_groups = df.groupby("player_id", sort=False)

        for player_id, grp in player_groups:
            idx = grp.index.values  # original indices in df

            actual    = grp[actual_col].fillna(0.0).values
            predicted = grp[baseline_col].fillna(grp[baseline_col].mean()).fillna(0.0).values
            is_home   = grp["is_home"].fillna(0).astype(int).values

            n = len(actual)
            home_ratings = np.zeros(n, dtype=float)
            away_ratings = np.zeros(n, dtype=float)

            # Each game updates only one split (home or away)
            # Running state variables for home & away tracks
            home_rating = 0.0
            away_rating = 0.0

            for i in range(1, n):
                prev_home = home_rating
                prev_away = away_rating
                residual = actual[i - 1] - predicted[i - 1]
                if not np.isfinite(residual):
                    residual = 0.0

                if is_home[i - 1] == 1:
                    home_rating = (1.0 - player_alpha) * prev_home + player_alpha * residual
                else:
                    away_rating = (1.0 - player_alpha) * prev_away + player_alpha * residual

                home_ratings[i] = home_rating
                away_ratings[i] = away_rating

            df.loc[idx, home_pi_col] = home_ratings
            df.loc[idx, away_pi_col] = away_ratings

    # ------------------------------------------------------------------
    # 2. Team defensive Pi ratings (per stat)
    # For each team, track whether they are suppressing or yielding more
    # than expected for each stat.  "Expected" = rolling opponent baseline.
    # ------------------------------------------------------------------
    for stat in stats_list:
        actual_col  = f"actual_{stat}"
        opp_col     = f"opp_{stat}_allowed_mean_l5"
        def_pi_col  = f"team_{stat}_def_pi"

        df[def_pi_col] = np.nan
        pi_cols.append(def_pi_col)

        if actual_col not in df.columns:
            continue

        if "team_id" not in df.columns:
            continue

        # team_def_pi: for each team × game, how much did their opponents score vs expectation?
        # We need to aggregate to team level first.
        team_df = (
            df.sort_values(["team_id", "game_date", "game_id"])
            .groupby(["team_id", "game_id", "game_date"], as_index=False)
            .agg(
                team_actual_stat_allowed=(actual_col, "mean"),
                team_opp_baseline=(opp_col, "mean") if opp_col in df.columns else (actual_col, "mean"),
            )
        )
        team_df = team_df.sort_values(["team_id", "game_date", "game_id"])

        team_pi_rows = []
        for team_id, t_grp in team_df.groupby("team_id", sort=False):
            t_idx   = t_grp.index.values
            t_actual = t_grp["team_actual_stat_allowed"].fillna(0.0).values
            t_base   = t_grp["team_opp_baseline"].fillna(t_actual.mean()).fillna(0.0).values
            n        = len(t_actual)
            rating   = 0.0
            ratings  = np.zeros(n, dtype=float)
            for i in range(1, n):
                residual = t_actual[i - 1] - t_base[i - 1]
                if not np.isfinite(residual):
                    residual = 0.0
                rating = (1.0 - team_beta) * rating + team_beta * residual
                ratings[i] = rating
            for j, idx in enumerate(t_idx):
                team_pi_rows.append({
                    "team_id": team_id,
                    "game_id": t_grp.iloc[j]["game_id"],
                    def_pi_col: ratings[j],
                })

        if team_pi_rows:
            team_pi_df = pd.DataFrame(team_pi_rows)
            df = df.merge(
                team_pi_df.rename(columns={"team_id": "_def_pi_team"}),
                left_on=["opponent_team_id", "game_id"],
                right_on=["_def_pi_team", "game_id"],
                how="left",
                suffixes=("", "_new"),
            )
            # Prefer freshly computed (right side)
            if f"{def_pi_col}_new" in df.columns:
                df[def_pi_col] = df[f"{def_pi_col}_new"].combine_first(df[def_pi_col])
                df = df.drop(columns=[f"{def_pi_col}_new", "_def_pi_team"], errors="ignore")
            else:
                df = df.drop(columns=["_def_pi_team"], errors="ignore")

    # ------------------------------------------------------------------
    # Return only the Pi columns + identity columns for easy merge
    # ------------------------------------------------------------------
    identity_cols = [c for c in ("player_id", "game_id") if c in df.columns]
    return df[identity_cols + [c for c in pi_cols if c in df.columns]].copy()


def attach_pi_ratings(
    wide_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Compute and merge Pi rating columns into wide_df in-place.

    Returns a new DataFrame (does not mutate input).  Suitable for calling
    inside ``build_wide_table`` or as a post-processing step.
    """
    pi_df = build_pi_rating_features(wide_df, **kwargs)
    merge_cols = [c for c in ("player_id", "game_id") if c in pi_df.columns]
    pi_cols = [c for c in pi_df.columns if c not in merge_cols]
    if not pi_cols:
        logger.warning("Pi rating computation produced no feature columns.")
        return wide_df
    result = wide_df.merge(pi_df, on=merge_cols, how="left", suffixes=("", "_pi_dup"))
    # Drop any duplicate columns from merge
    dup_cols = [c for c in result.columns if c.endswith("_pi_dup")]
    result = result.drop(columns=dup_cols, errors="ignore")
    logger.info("Attached %d Pi rating columns to wide table.", len(pi_cols))
    return result
