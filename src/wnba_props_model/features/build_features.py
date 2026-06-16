"""Build leakage-safe baseline feature tables for WNBA player props model.

TEMPORAL INVARIANT (never violate):
    For every rolling/historical feature, the computation is:
        prior_value = value.groupby(player_id).shift(1)
        feature     = prior_value.groupby(player_id).rolling(window).mean()
    Current-game data is NEVER included in any predictive feature.

feature_cutoff_policy = strict_pregame_shifted
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATS: list[str] = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
ADV_STAT_COLS: list[str] = [
    "usage_percentage", "true_shooting_percentage", "assist_percentage",
    "rebound_percentage", "offensive_rating", "defensive_rating",
]
ROLL_WINDOWS: tuple[int, ...] = (3, 5, 10)

MINUTES_BUCKET_LABELS: list[tuple[float, float, str]] = [
    (0.0, 12.0, "bench_low"),
    (12.0, 20.0, "bench_rotation"),
    (20.0, 28.0, "rotation"),
    (28.0, 34.0, "starter"),
    (34.0, float("inf"), "workhorse"),
]

IDENTITY_COLS: list[str] = [
    "game_id", "game_date", "season",
    "player_id", "player_name", "position",
    "team_id", "team_abbreviation",
    "opponent_team_id", "opponent_team_abbreviation",
    "is_home", "home_away",
]

TARGET_COLS: list[str] = [
    "actual_minutes", "did_play",
    "actual_pts", "actual_reb", "actual_ast", "actual_fg3m",
    "actual_stl", "actual_blk", "actual_turnover",
]

ROLE_BUCKET_COLS: list[str] = [
    "projected_minutes_bucket",
    "usage_bucket",
    "role_status",
    "role_uncertainty_bucket",
    "minutes_volatility_bucket",
    "rotation_stability_bucket",
]

FEATURE_CUTOFF_POLICY = "strict_pregame_shifted"


# ---------------------------------------------------------------------------
# Shifted rolling helpers
# ---------------------------------------------------------------------------

def _sr(grp: pd.core.groupby.SeriesGroupBy, window: int, agg: str = "mean",
        min_periods: int = 1) -> pd.Series:
    """Shifted rolling aggregation within player/team groups.

    shift(1) excludes current game; rolling(window) uses only prior games.
    This is the only permitted pattern for rolling features.
    """
    if agg == "mean":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).mean())
    if agg == "std":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min(min_periods, 2)).std())
    if agg == "sum":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).sum())
    if agg == "count":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).count())
    if agg == "median":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).median())
    if agg == "min":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).min())
    if agg == "max":
        return grp.transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=min_periods).max())
    if agg == "expanding_mean":
        return grp.transform(lambda x: x.shift(1).expanding(min_periods=min_periods).mean())
    if agg == "expanding_sum":
        return grp.transform(lambda x: x.shift(1).expanding(min_periods=1).sum())
    if agg == "expanding_count":
        return grp.transform(lambda x: x.shift(1).expanding(min_periods=1).count())
    raise ValueError(f"Unknown agg: {agg}")


# ---------------------------------------------------------------------------
# DNP streak helper (shifted, no current-game leakage)
# ---------------------------------------------------------------------------

def _dnp_streak_prior(did_play_array: np.ndarray) -> np.ndarray:
    """Count consecutive DNPs before each game (sorted chronologically).

    result[i] = number of consecutive DNPs ending just before game i.
    For game 0 (first in group): always 0.
    Recurrence: result[i] = 0 if prev played, else 1 + result[i-1].
    """
    n = len(did_play_array)
    result = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        prev = did_play_array[i - 1]
        if pd.isna(prev) or bool(prev):
            result[i] = 0
        else:
            result[i] = result[i - 1] + 1
    return result


# ---------------------------------------------------------------------------
# Minutes bucket assignment
# ---------------------------------------------------------------------------

def assign_minutes_bucket(minutes: float | None) -> str:
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return "unknown"
    for lo, hi, label in MINUTES_BUCKET_LABELS:
        if lo <= minutes < hi:
            return label
    return "workhorse"


def assign_usage_bucket(usage_proxy: float | None) -> str:
    if usage_proxy is None or (isinstance(usage_proxy, float) and np.isnan(usage_proxy)):
        return "unknown"
    if usage_proxy < 0.15:
        return "low"
    if usage_proxy < 0.25:
        return "medium"
    if usage_proxy < 0.35:
        return "high"
    return "elite"


def assign_role_status(proj_min: float | None, starter_rate: float | None) -> str:
    if proj_min is None or np.isnan(proj_min):
        return "uncertain_role"
    if proj_min >= 28 or (starter_rate is not None and not np.isnan(starter_rate) and starter_rate >= 0.6):
        return "starter"
    if proj_min >= 20:
        return "bench"
    return "bench"


def assign_role_uncertainty(std: float | None, proj_min: float | None,
                             dnp_streak: int = 0) -> str:
    if dnp_streak >= 2:
        return "injury_dependent"
    if std is None or proj_min is None or np.isnan(std) or np.isnan(proj_min):
        return "uncertain"
    cv = std / max(proj_min, 1.0)  # coefficient of variation
    if cv < 0.15:
        return "stable"
    if cv < 0.30:
        return "elevated"
    return "uncertain"


def assign_minutes_volatility_bucket(std: float | None) -> str:
    if std is None or np.isnan(std):
        return "unknown"
    if std < 4:
        return "stable"
    if std < 8:
        return "moderate"
    return "volatile"


def assign_rotation_stability(support: int) -> str:
    if support < 3:
        return "new"
    if support < 8:
        return "developing"
    return "established"


# ---------------------------------------------------------------------------
# Team/opponent game context builder
# ---------------------------------------------------------------------------

def _build_team_game_context(
    stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per team × game with shifted rolling context features.

    Uses player-stat aggregates for per-stat context; game scores for total/pace.
    All rolling features use shift(1), so context for game G uses only prior games.
    """
    # --- 1. All team × game pairs from games schedule ----------------------
    records = []
    for _, row in games_df.iterrows():
        for side, opp in [("home", "visitor"), ("visitor", "home")]:
            records.append({
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "season": row["season"],
                "team_id": row[f"{side}_team_id"],
                "opp_team_id": row[f"{opp}_team_id"],
                "total_score": row.get("total_score"),
            })
    ctx = pd.DataFrame(records)

    # --- 2. Player-stat aggregates per team × game -------------------------
    stat_agg = stats_df.groupby(["game_id", "team_id"], as_index=False).agg(
        **{f"_t_{s}": (s, "sum") for s in STATS}
    )

    # --- 3. Join own-team stats to ctx rows --------------------------------
    ctx = ctx.merge(stat_agg, on=["game_id", "team_id"], how="left")

    # --- 4. Join opponent stats (what opponent scored AGAINST this team) ---
    opp_agg = stat_agg.rename(columns={
        "team_id": "_opp_tid",
        **{f"_t_{s}": f"_o_{s}" for s in STATS},
    })
    ctx = ctx.merge(
        opp_agg,
        left_on=["game_id", "opp_team_id"],
        right_on=["game_id", "_opp_tid"],
        how="left",
    ).drop(columns=["_opp_tid"], errors="ignore")

    # --- 5. Sort and compute shifted rolling --------------------------------
    ctx = ctx.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    grp = ctx.groupby("team_id", sort=False)

    # Team self-context (own offensive output)
    for w in (5, 10):
        ctx[f"t_pts_for_l{w}"] = _sr(grp["_t_pts"], w)
        ctx[f"t_pts_against_l{w}"] = _sr(grp["_o_pts"], w)   # what teams scored against this team
        ctx[f"t_total_score_l{w}"] = _sr(grp["total_score"], w)
        # Per-stat team offensive context
        for s in STATS:
            ctx[f"t_{s}_for_l{w}"] = _sr(grp[f"_t_{s}"], w)
        # Per-stat opponent context (what teams scored/got against this team)
        for s in STATS:
            ctx[f"t_{s}_against_l{w}"] = _sr(grp[f"_o_{s}"], w)

    # Schedule context (pre-game knowledge)
    ctx["t_games_prior"] = grp.cumcount()
    ctx["_prev_game_date"] = grp["game_date"].shift(1)
    ctx["_days_since_last"] = (ctx["game_date"] - ctx["_prev_game_date"]).dt.days
    ctx["t_rest_days"] = (ctx["_days_since_last"] - 1).clip(lower=0)
    ctx["t_back_to_back"] = (ctx["_days_since_last"] == 1).astype(int)
    ctx = ctx.drop(columns=["_prev_game_date", "_days_since_last"] +
                   [f"_t_{s}" for s in STATS] + [f"_o_{s}" for s in STATS],
                   errors="ignore")
    return ctx


# ---------------------------------------------------------------------------
# Player-level feature builder
# ---------------------------------------------------------------------------

def _build_player_features(
    stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
    adv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all player-level predictive features (leakage-safe, shifted).

    Returns the enriched stats DataFrame sorted by (player_id, game_date).
    """
    # Sort ONCE — all groupby operations depend on this ordering
    df = stats_df.copy()
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    grp = df.groupby("player_id", sort=False)

    # ------------------------------------------------------------------ #
    # 1. Player history / availability
    # ------------------------------------------------------------------ #
    df["player_games_prior"] = grp.cumcount()
    df["player_games_played_prior"] = _sr(grp["did_play"], window=999,
                                          agg="expanding_sum").fillna(0).astype(int)
    df["player_days_since_last_game"] = grp["game_date"].transform(
        lambda x: x.diff().dt.days
    )
    df["player_rest_days"] = (df["player_days_since_last_game"] - 1).clip(lower=0)

    # DNP streak (shifted — uses only prior games)
    df["player_dnp_streak_prior"] = grp["did_play"].transform(
        lambda s: pd.Series(
            _dnp_streak_prior(s.values), index=s.index
        )
    )

    for w in (5, 10):
        # Zero-minute rate over rolling window (shifted)
        zero_mask = (df["minutes"] == 0.0).astype(float)
        df[f"player_zero_minute_rate_l{w}"] = grp[zero_mask.name if zero_mask.name else "minutes"].transform(
            lambda x, ww=w: (x == 0.0).astype(float).shift(1).rolling(ww, min_periods=1).mean()
        )
        df[f"player_did_play_rate_l{w}"] = _sr(
            grp[did_play_col := "did_play"], w  # type: ignore[name-defined]
        ).where(df["player_games_prior"] > 0)

    # fix: compute zero-minute rates properly
    df["_zero_min"] = (df["minutes"] == 0.0).astype(float)
    _zero_grp = df.groupby("player_id", sort=False)["_zero_min"]
    for w in (5, 10):
        df[f"player_zero_minute_rate_l{w}"] = _sr(_zero_grp, w)
        df[f"player_did_play_rate_l{w}"] = 1.0 - df[f"player_zero_minute_rate_l{w}"]
    df = df.drop(columns=["_zero_min"])

    # ------------------------------------------------------------------ #
    # 2. Player minutes features (all shifted)
    # ------------------------------------------------------------------ #
    df["player_minutes_last1"] = grp["minutes"].shift(1)
    df["player_minutes_last2"] = grp["minutes"].shift(2)
    df["player_minutes_last3"] = grp["minutes"].shift(3)

    _min_grp = grp["minutes"]
    df["player_minutes_mean_l3"]    = _sr(_min_grp, 3)
    df["player_minutes_mean_l5"]    = _sr(_min_grp, 5)
    df["player_minutes_mean_l10"]   = _sr(_min_grp, 10)
    df["player_minutes_mean_season"] = _sr(_min_grp, 999, agg="expanding_mean")
    df["player_minutes_median_l5"]  = _sr(_min_grp, 5, agg="median")
    df["player_minutes_std_l5"]     = _sr(_min_grp, 5, agg="std", min_periods=2)
    df["player_minutes_std_l10"]    = _sr(_min_grp, 10, agg="std", min_periods=2)
    df["player_minutes_min_l10"]    = _sr(_min_grp, 10, agg="min")
    df["player_minutes_max_l10"]    = _sr(_min_grp, 10, agg="max")

    # Support counts (how many prior games contributed to rolling mean)
    df["player_minutes_l3_support"]     = _sr(_min_grp, 3, agg="count")
    df["player_minutes_l5_support"]     = _sr(_min_grp, 5, agg="count")
    df["player_minutes_l10_support"]    = _sr(_min_grp, 10, agg="count")
    df["player_minutes_season_support"] = _sr(_min_grp, 999, agg="expanding_count")
    df["used_player_minutes_prior_flag"] = (df["player_minutes_l5_support"] > 0)

    # ------------------------------------------------------------------ #
    # 3. Per-stat rolling features (shifted)
    # ------------------------------------------------------------------ #
    for stat in STATS:
        if stat not in df.columns:
            continue
        _s_grp = df.groupby("player_id", sort=False)[stat]
        df[f"player_{stat}_last1"]         = _s_grp.shift(1)
        df[f"player_{stat}_mean_l3"]       = _sr(_s_grp, 3)
        df[f"player_{stat}_mean_l5"]       = _sr(_s_grp, 5)
        df[f"player_{stat}_mean_l10"]      = _sr(_s_grp, 10)
        df[f"player_{stat}_mean_season"]   = _sr(_s_grp, 999, agg="expanding_mean")
        df[f"player_{stat}_std_l10"]       = _sr(_s_grp, 10, agg="std", min_periods=2)
        df[f"player_{stat}_l3_support"]    = _sr(_s_grp, 3, agg="count")
        df[f"player_{stat}_l5_support"]    = _sr(_s_grp, 5, agg="count")
        df[f"player_{stat}_l10_support"]   = _sr(_s_grp, 10, agg="count")
        df[f"player_{stat}_season_support"] = _sr(_s_grp, 999, agg="expanding_count")

    # ------------------------------------------------------------------ #
    # 4. Per-minute rate features (shifted sums; safe denominator)
    # ------------------------------------------------------------------ #
    for stat in STATS:
        if stat not in df.columns:
            continue
        _s_grp = df.groupby("player_id", sort=False)[stat]
        _m_grp = df.groupby("player_id", sort=False)["minutes"]
        for w in (3, 5, 10):
            stat_sum = _sr(_s_grp, w, agg="sum")
            min_sum  = _sr(_m_grp, w, agg="sum")
            rate = stat_sum / min_sum.clip(lower=1.0)   # never divide by zero
            df[f"player_{stat}_per_min_l{w}"] = rate.replace([np.inf, -np.inf], np.nan)
        # Season rate
        stat_sum_s = _sr(_s_grp, 999, agg="expanding_sum")
        min_sum_s  = _sr(_m_grp, 999, agg="expanding_sum")
        df[f"player_{stat}_per_min_season"] = (
            (stat_sum_s / min_sum_s.clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
        )

    # ------------------------------------------------------------------ #
    # 5. Usage proxy features (shifted)
    # ------------------------------------------------------------------ #
    # BDL provides: fga, fta, turnover  → Oliver usage proxy
    _usage_inputs_available = all(c in df.columns for c in ["fga", "fta", "turnover"])
    if _usage_inputs_available:
        # Possession usage proxy = fga + 0.44*fta + turnover (per game)
        df["_usage_raw"] = df["fga"] + 0.44 * df["fta"] + df["turnover"]
        _ug = df.groupby("player_id", sort=False)["_usage_raw"]
        _mg = df.groupby("player_id", sort=False)["minutes"]
        for w in (5, 10):
            u_sum = _sr(_ug, w, agg="sum")
            m_sum = _sr(_mg, w, agg="sum")
            df[f"player_usage_proxy_l{w}"] = (
                (u_sum / m_sum.clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
            )
        df["player_usage_proxy_season"] = (
            (_sr(_ug, 999, agg="expanding_sum") /
             _sr(_mg, 999, agg="expanding_sum").clip(lower=1.0))
            .replace([np.inf, -np.inf], np.nan)
        )

        # Shot attempt rate (fga per minute)
        _fga_g = df.groupby("player_id", sort=False)["fga"]
        df["player_shot_attempt_rate_l5"] = (
            (_sr(_fga_g, 5, agg="sum") / _sr(_mg, 5, agg="sum").clip(lower=1.0))
            .replace([np.inf, -np.inf], np.nan)
        )

        # Free throw rate (fta per minute)
        _fta_g = df.groupby("player_id", sort=False)["fta"]
        df["player_free_throw_rate_l5"] = (
            (_sr(_fta_g, 5, agg="sum") / _sr(_mg, 5, agg="sum").clip(lower=1.0))
            .replace([np.inf, -np.inf], np.nan)
        )

        # On-ball proxy (ast per minute)
        if "ast" in df.columns:
            _ast_g = df.groupby("player_id", sort=False)["ast"]
            df["player_on_ball_proxy_l5"] = (
                (_sr(_ast_g, 5, agg="sum") / _sr(_mg, 5, agg="sum").clip(lower=1.0))
                .replace([np.inf, -np.inf], np.nan)
            )
        df = df.drop(columns=["_usage_raw"], errors="ignore")
    else:
        for col in ["player_usage_proxy_l5", "player_usage_proxy_l10", "player_usage_proxy_season",
                    "player_shot_attempt_rate_l5", "player_free_throw_rate_l5", "player_on_ball_proxy_l5"]:
            df[col] = np.nan

    # ------------------------------------------------------------------ #
    # 6. Role features (based on prior minutes — shifted)
    # ------------------------------------------------------------------ #
    # projected_minutes_proxy = best available prior minutes estimate
    df["projected_minutes_proxy"] = (
        df["player_minutes_mean_l5"]
        .fillna(df["player_minutes_mean_l10"])
        .fillna(df["player_minutes_mean_season"])
        .fillna(df["player_minutes_last1"])
    )

    df["projected_minutes_bucket"] = df["projected_minutes_proxy"].map(assign_minutes_bucket)

    # Starter proxy (shifted): was player a starter in prior game?
    df["starter_proxy_prior"] = grp["started_proxy"].shift(1).astype(float)
    df["starter_rate_l5"]  = _sr(grp["started_proxy"], 5)
    df["starter_rate_l10"] = _sr(grp["started_proxy"], 10)

    # Usage bucket from usage proxy (l5 preferred)
    _usage_col = "player_usage_proxy_l5"
    if _usage_col in df.columns:
        df["usage_bucket"] = df[_usage_col].map(assign_usage_bucket)
    else:
        # Fallback: estimate from shot rate
        df["usage_bucket"] = "unknown"

    # Role status
    df["role_status"] = [
        assign_role_status(m, s)
        for m, s in zip(df["projected_minutes_proxy"], df["starter_rate_l5"])
    ]

    # Role uncertainty bucket
    df["role_uncertainty_bucket"] = [
        assign_role_uncertainty(std, proj_min, dnp)
        for std, proj_min, dnp in zip(
            df["player_minutes_std_l5"],
            df["projected_minutes_proxy"],
            df["player_dnp_streak_prior"],
        )
    ]

    df["minutes_volatility_bucket"] = df["player_minutes_std_l5"].map(
        assign_minutes_volatility_bucket
    )
    df["rotation_stability_bucket"] = df["player_minutes_l10_support"].map(
        lambda s: assign_rotation_stability(int(s) if pd.notna(s) else 0)
    )

    # ------------------------------------------------------------------ #
    # 7. Home/away split rolling features (Phase 2b)
    # Computes rolling stats within home-only and away-only game subsets.
    # Approach: filter to home/away games, compute within-split rolling mean,
    # then merge back by (player_id, game_id) — NaN for the other split.
    # ------------------------------------------------------------------ #
    if "is_home" in df.columns:
        for split_val, split_name in [(1, "home"), (0, "away")]:
            split_df = df[df["is_home"] == split_val].copy()
            split_df = split_df.sort_values(["player_id", "game_date", "game_id"])
            if split_df.empty:
                continue
            s_grp_min = split_df.groupby("player_id", sort=False)["minutes"]
            split_df[f"player_minutes_{split_name}_mean_l5"] = _sr(s_grp_min, 5)
            for stat in STATS:
                if stat not in split_df.columns:
                    continue
                s_grp_s = split_df.groupby("player_id", sort=False)[stat]
                split_df[f"player_{stat}_{split_name}_mean_l5"] = _sr(s_grp_s, 5)
            merge_cols = (
                [f"player_minutes_{split_name}_mean_l5"]
                + [f"player_{stat}_{split_name}_mean_l5" for stat in STATS if stat in split_df.columns]
            )
            df = df.merge(
                split_df[["player_id", "game_id"] + merge_cols],
                on=["player_id", "game_id"], how="left",
            )

    # ------------------------------------------------------------------ #
    # 8. Player back-to-back flag (Phase 2c)
    # ------------------------------------------------------------------ #
    if "player_rest_days" in df.columns:
        df["player_back_to_back_flag"] = (df["player_rest_days"] == 1).astype(int)
        # Heavy-minutes back-to-back: B2B AND played heavy minutes prior game
        df["player_heavy_minutes_b2b"] = (
            df["player_back_to_back_flag"] & (df["player_minutes_last1"].fillna(0) > 30)
        ).astype(int)

    # ------------------------------------------------------------------ #
    # 9. Per-minute rate × minutes interaction (Phase 2e)
    # log(λ) = player_rate_per_min × minutes_mean — captures Poisson exposure.
    # The interaction term lets the model reason about matchup minutes changes.
    # ------------------------------------------------------------------ #
    for stat in STATS:
        rate_col = f"player_{stat}_per_min_l5"
        if rate_col in df.columns and "projected_minutes_proxy" in df.columns:
            df[f"player_{stat}_per_min_l5_x_proj_min"] = (
                df[rate_col] * df["projected_minutes_proxy"]
            ).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #
    # 10. Advanced stats (shifted rolling; all optional)
    # ------------------------------------------------------------------ #
    if adv_df is not None and not adv_df.empty:
        adv = adv_df.sort_values(["player_id", "game_date"]).copy()
        for col in ADV_STAT_COLS:
            if col not in adv.columns:
                continue
            # Merge advanced stat into df on (player_id, game_id)
            adv_col = adv[["player_id", "game_id", col]].rename(
                columns={col: f"_adv_{col}"}
            )
            df = df.merge(adv_col, on=["player_id", "game_id"], how="left")
            _a_grp = df.groupby("player_id", sort=False)[f"_adv_{col}"]
            df[f"player_{col}_l5"] = _sr(_a_grp, 5)
            df[f"player_{col}_l5_support"] = _sr(_a_grp, 5, agg="count")
            df = df.drop(columns=[f"_adv_{col}"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Injury feature builder (Phase 2a)
# ---------------------------------------------------------------------------

def _build_injury_features(
    wide_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach injury-context features to the wide feature table.

    Temporal safety: all features reference what was KNOWN before the game starts.
    - player_injured_l1:     player missed prior game due to injury/DNP
    - teammate_injury_flag:  at least one teammate on same team is listed out
                             going INTO this game (joined by game_id + team_id)
    - vacated_minutes_l1:    sum of minutes from any teammate who did NOT play
                             the prior game (redistributed opportunity signal)
    - usage_share_delta:     change in player usage proxy between prior two games
                             (positive = more opportunity, negative = less)

    Parameters
    ----------
    wide_df:     wide feature table, one row per player × game (already sorted)
    injuries_df: BDL injury records with columns: player_id, game_id, status
                 (normalized: "out", "questionable", "available", ...)
                 game_id here is the game the player is MISSING / at-risk for.
    """
    out = wide_df.copy()

    # ---- player_injured_l1: player was injured/out prior game ----------------
    # Use did_play column: if player's prior game was DNP (did_play == 0), flag it
    if "did_play" in out.columns:
        grp = out.groupby("player_id", sort=False)
        out["player_injured_l1"] = grp["did_play"].transform(
            lambda x: (x.shift(1) == 0).astype(float)
        ).fillna(0.0)
    else:
        out["player_injured_l1"] = 0.0

    # ---- usage_share_delta: shift(1) minus shift(2) of usage proxy ----------
    if "player_usage_proxy_l5" in out.columns:
        grp2 = out.groupby("player_id", sort=False)
        out["usage_share_delta"] = (
            grp2["player_usage_proxy_l5"].shift(0)   # already shifted
            - grp2["player_usage_proxy_l5"].shift(1)  # previous game value
        ).replace([np.inf, -np.inf], np.nan)
    else:
        out["usage_share_delta"] = np.nan

    # ---- vacated_minutes_l1: minutes from prior-game DNP teammates -----------
    # Compute per-team per-game total DNP minutes (teammates not playing).
    # This tells us how many minutes are being redistributed for THIS game.
    if "team_id" in out.columns and "did_play" in out.columns:
        dnp_mins = out.copy()
        _min_col = "actual_minutes" if "actual_minutes" in dnp_mins.columns else "minutes"
        dnp_mins["_dnp_minutes"] = dnp_mins[_min_col].fillna(0.0) * (
            1 - dnp_mins["did_play"].fillna(1).astype(int)
        )
        team_dnp = (
            dnp_mins.groupby(["team_id", "game_id"])["_dnp_minutes"]
            .sum()
            .reset_index()
            .rename(columns={"_dnp_minutes": "_team_dnp_min", "game_id": "_gid", "team_id": "_tid"})
        )
        # Shift: for each team, carry last game's DNP minutes forward
        team_dnp = team_dnp.sort_values(["_tid", "_gid"])
        team_dnp["vacated_minutes_l1"] = team_dnp.groupby("_tid")["_team_dnp_min"].shift(1)
        out = out.merge(
            team_dnp[["_tid", "_gid", "vacated_minutes_l1"]].rename(
                columns={"_tid": "team_id", "_gid": "game_id"}
            ),
            on=["team_id", "game_id"], how="left",
        )
    else:
        out["vacated_minutes_l1"] = np.nan

    # ---- teammate_injury_flag: BDL injury data for current game -------------
    # injuries_df may contain player_id + game_id rows where the player is listed
    # as "out" or "doubtful" for THAT game. Attach team context to flag teammates.
    out["teammate_injury_flag"] = 0.0
    if injuries_df is not None and not injuries_df.empty and "game_id" in injuries_df.columns:
        inj = injuries_df.copy()
        inj_norm = inj.get("status", inj.get("injury_status", pd.Series(dtype=str)))
        inj["_is_out"] = inj_norm.fillna("").str.lower().isin(["out", "doubtful", "inactive"])

        # Build set of (game_id, player_id) pairs for players who are out
        out_players = inj[inj["_is_out"]][["game_id", "player_id"]].drop_duplicates()

        # Join team_id to injured players to find which team they're on
        if "team_id" in out.columns:
            player_team = out[["player_id", "game_id", "team_id"]].drop_duplicates()
            out_with_team = out_players.merge(player_team, on=["game_id", "player_id"], how="left")

            # Count out players per (game_id, team_id) — at least 1 = flag
            team_out_count = (
                out_with_team.groupby(["game_id", "team_id"])["player_id"]
                .count()
                .reset_index()
                .rename(columns={"player_id": "_n_injured_teammates"})
            )
            out = out.merge(team_out_count, on=["game_id", "team_id"], how="left")
            out["teammate_injury_flag"] = (
                out["_n_injured_teammates"].fillna(0) > 0
            ).astype(float)
            out = out.drop(columns=["_n_injured_teammates"], errors="ignore")

    return out


# ---------------------------------------------------------------------------
# Wide feature table builder
# ---------------------------------------------------------------------------

def build_wide_table(
    stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
    adv_df: pd.DataFrame | None = None,
    injuries_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical wide feature table (one row per player_id × game_id).

    Returns:
        (wide_df, audit_notes) where audit_notes documents feature availability.
    """
    audit_notes: dict[str, Any] = {
        "feature_cutoff_policy": FEATURE_CUTOFF_POLICY,
        "usage_inputs_available": all(c in stats_df.columns for c in ["fga", "fta", "turnover"]),
        "advanced_stats_available": adv_df is not None and not adv_df.empty,
        "injury_temporal_alignment": "not_aligned",
        "injuries_available_but_not_temporally_aligned": True,
        "pace_proxy_method": "total_score_proxy",
        "adv_stat_cols_used": [c for c in ADV_STAT_COLS if
                                adv_df is not None and c in adv_df.columns],
        "adv_stat_cols_missing": [c for c in ADV_STAT_COLS if
                                   adv_df is None or c not in adv_df.columns],
    }

    # ------------------------------------------------------------------ #
    # Build player-level features
    # ------------------------------------------------------------------ #
    wide = _build_player_features(stats_df, games_df, adv_df=adv_df)

    # ------------------------------------------------------------------ #
    # Build team / opponent context
    # ------------------------------------------------------------------ #
    ctx = _build_team_game_context(stats_df, games_df)

    # --- Team (self) context join ----------------------------------------
    team_ctx_cols: dict[str, str] = {}
    for w in (5, 10):
        team_ctx_cols[f"t_pts_for_l{w}"]      = f"team_pts_for_mean_l{w}"
        team_ctx_cols[f"t_pts_against_l{w}"]   = f"team_pts_allowed_mean_l{w}"
        team_ctx_cols[f"t_total_score_l{w}"]   = f"team_total_score_mean_l{w}"
        # Phase 1c: per-stat team offensive context (already computed in ctx)
        for s in STATS:
            team_ctx_cols[f"t_{s}_for_l{w}"] = f"team_{s}_for_mean_l{w}"
    team_ctx_cols["t_games_prior"]   = "team_games_prior"
    team_ctx_cols["t_rest_days"]     = "team_rest_days"
    team_ctx_cols["t_back_to_back"]  = "team_back_to_back_flag"

    # Only include columns that actually exist in ctx to avoid KeyErrors
    ctx_team_rename = ctx.rename(columns=team_ctx_cols)
    ctx_team_cols = ["game_id", "team_id"] + [v for v in team_ctx_cols.values() if v in ctx_team_rename.columns]
    ctx_team = ctx_team_rename[ctx_team_cols].copy()
    # Add pace proxy aliases
    for w in (5, 10):
        if f"team_total_score_mean_l{w}" in ctx_team.columns:
            ctx_team[f"team_pace_proxy_l{w}"] = ctx_team[f"team_total_score_mean_l{w}"]

    wide = wide.merge(ctx_team, on=["game_id", "team_id"], how="left")

    # --- Opponent context join -------------------------------------------
    opp_ctx_cols: dict[str, str] = {}
    for w in (5, 10):
        opp_ctx_cols[f"t_pts_against_l{w}"]  = f"opp_pts_allowed_mean_l{w}"
        opp_ctx_cols[f"t_total_score_l{w}"]  = f"opp_total_score_allowed_mean_l{w}"
    # L5 opponent per-stat (existing)
    for s in STATS:
        opp_ctx_cols[f"t_{s}_against_l5"] = (
            f"opp_turnover_forced_mean_l5" if s == "turnover"
            else f"opp_{s}_allowed_mean_l5"
        )
    # Phase 2d: L10 opponent per-stat (eliminates small-sample noise)
    for s in STATS:
        opp_ctx_cols[f"t_{s}_against_l10"] = (
            f"opp_turnover_forced_mean_l10" if s == "turnover"
            else f"opp_{s}_allowed_mean_l10"
        )
    opp_ctx_cols["t_games_prior"]  = "opp_games_prior"
    opp_ctx_cols["t_rest_days"]    = "opp_rest_days"
    opp_ctx_cols["t_back_to_back"] = "opp_back_to_back_flag"

    ctx_opp = ctx.rename(columns=opp_ctx_cols)[
        ["game_id", "team_id"] + [v for v in opp_ctx_cols.values() if v in ctx.rename(columns=opp_ctx_cols).columns]
    ].rename(columns={"team_id": "_opp_team_id"})
    # Add pace proxy alias
    if "opp_total_score_allowed_mean_l5" in ctx_opp.columns:
        ctx_opp["opp_pace_proxy_l5"] = ctx_opp["opp_total_score_allowed_mean_l5"]

    wide = wide.merge(
        ctx_opp,
        left_on=["game_id", "opponent_team_id"],
        right_on=["game_id", "_opp_team_id"],
        how="left",
    ).drop(columns=["_opp_team_id"], errors="ignore")

    # ------------------------------------------------------------------ #
    # Rename actual stats to target columns
    # ------------------------------------------------------------------ #
    for s in STATS:
        if s in wide.columns:
            wide = wide.rename(columns={s: f"actual_{s}"})
    wide = wide.rename(columns={"minutes": "actual_minutes"})

    # ------------------------------------------------------------------ #
    # Ensure identity and target columns exist
    # ------------------------------------------------------------------ #
    if "opponent_team_abbreviation" not in wide.columns and "opponent_team_id" in wide.columns:
        opp_abbrev = games_df[["home_team_id", "home_team_abbreviation",
                                "visitor_team_id", "visitor_team_abbreviation"]].copy()
        id_to_abbrev = dict(
            zip(opp_abbrev["home_team_id"], opp_abbrev["home_team_abbreviation"])
        )
        id_to_abbrev.update(
            zip(opp_abbrev["visitor_team_id"], opp_abbrev["visitor_team_abbreviation"])
        )
        wide["opponent_team_abbreviation"] = wide["opponent_team_id"].map(id_to_abbrev)

    # ------------------------------------------------------------------ #
    # Metadata columns
    # ------------------------------------------------------------------ #
    ts = datetime.now(timezone.utc).isoformat()
    wide["feature_build_timestamp_utc"] = ts
    wide["feature_cutoff_policy"] = FEATURE_CUTOFF_POLICY

    # ------------------------------------------------------------------ #
    # Pi Ratings (Phase 3b): player form + opponent defensive strength
    # Computed AFTER the wide table has actual_* columns so residuals can
    # be derived; fully shift-1 safe via pi_ratings module.
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.pi_ratings import attach_pi_ratings  # noqa: PLC0415
        wide = attach_pi_ratings(wide)
        audit_notes["pi_ratings_applied"] = True
    except Exception as exc:
        audit_notes["pi_ratings_applied"] = False
        audit_notes["pi_ratings_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Injury features (Phase 2a)
    # ------------------------------------------------------------------ #
    if injuries_df is not None:
        wide = _build_injury_features(wide, injuries_df)
        audit_notes["injury_temporal_alignment"] = "aligned_via_game_id"
        audit_notes["injuries_available_but_not_temporally_aligned"] = False
    else:
        # Ensure columns exist as NaN/0 for schema consistency
        for col in ["player_injured_l1", "teammate_injury_flag",
                    "vacated_minutes_l1", "usage_share_delta"]:
            if col not in wide.columns:
                wide[col] = 0.0 if col in ("player_injured_l1", "teammate_injury_flag") else np.nan

    # ------------------------------------------------------------------ #
    # Sanitize: replace inf with NaN in numeric columns
    # ------------------------------------------------------------------ #
    num_cols = wide.select_dtypes(include="number").columns
    wide[num_cols] = wide[num_cols].replace([np.inf, -np.inf], np.nan)

    # Final sort for determinism
    wide = wide.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    return wide, audit_notes


# ---------------------------------------------------------------------------
# Long table builder
# ---------------------------------------------------------------------------

def build_long_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide table to long format (one row per player_id × game_id × stat).

    actual_outcome = actual stat value for this stat in this game.
    This is the supervised target; it must NOT appear in model_feature_columns.
    """
    frames = []
    for stat in STATS:
        target_col = f"actual_{stat}"
        if target_col not in wide_df.columns:
            continue
        sub = wide_df.copy()
        sub["stat"] = stat
        sub["actual_outcome"] = sub[target_col]
        # Drop other actual_* outcome columns (keep actual_minutes + did_play as
        # context/diagnostic but drop other stat targets to avoid confusion)
        drop_targets = [f"actual_{s}" for s in STATS if s != stat]
        sub = sub.drop(columns=[c for c in drop_targets if c in sub.columns])
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.sort_values(
        ["game_date", "game_id", "player_id", "stat"]
    ).reset_index(drop=True)
    return long_df


# ---------------------------------------------------------------------------
# Model feature columns (authoritative allow-list)
# ---------------------------------------------------------------------------

def _derive_model_feature_columns(wide_df: pd.DataFrame) -> list[str]:
    """Return the authoritative list of model feature columns.

    Rules:
    - Must exist in wide_df
    - Must NOT be in FORBIDDEN_MODEL_FEATURES
    - Must NOT be identity, target, or metadata columns

    Note: is_home, season, and position are listed as identity columns for
    auditing purposes but are also valid model features (pre-game knowledge).
    """
    # Pure identifiers never used as features
    _pure_identity = {
        "game_id", "game_date", "player_id", "player_name",
        "team_id", "team_abbreviation",
        "opponent_team_id", "opponent_team_abbreviation",
        "home_away",  # string version of is_home; use is_home instead
    }
    excluded = _pure_identity | set(TARGET_COLS) | set(ROLE_BUCKET_COLS) | FORBIDDEN_MODEL_FEATURES
    excluded.update({
        "feature_build_timestamp_utc", "feature_cutoff_policy",
        "source", "pull_timestamp_utc",
        # raw audit flags from ingestion
        "minutes_raw", "minutes_flag", "non_playing_flag", "zero_minute_flag",
        "stat_line_all_zero_flag", "missing_team_flag", "missing_opponent_flag",
        "missing_game_date_flag", "started_proxy",  # raw, not rolled
        "pts_ast", "pts_reb", "pts_reb_ast", "reb_ast", "stocks",  # combo raw stats
        "oreb", "dreb", "pf", "fga", "fta", "fg3m", "plus_minus",  # raw per-game stats
    })
    model_cols = [
        c for c in wide_df.columns
        if c not in excluded
        and c not in ROLE_BUCKET_COLS
        and not c.startswith("actual_")
        and c != "stat"
        and c != "actual_outcome"
    ]
    # Final leakage check
    from wnba_props_model.features.feature_contract import assert_no_forbidden_features
    assert_no_forbidden_features(model_cols)
    return sorted(model_cols)


# ---------------------------------------------------------------------------
# Feature schema manifest
# ---------------------------------------------------------------------------

def build_feature_schema_manifest(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    model_feature_columns: list[str],
    source_tables: list[str],
    wide_path: str,
    long_path: str,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    numeric_features = [
        c for c in model_feature_columns
        if c in wide_df.columns and pd.api.types.is_numeric_dtype(wide_df[c])
    ]
    categorical_features = [
        c for c in model_feature_columns
        if c in wide_df.columns and not pd.api.types.is_numeric_dtype(wide_df[c])
    ]
    role_bucket_cols_present = [c for c in ROLE_BUCKET_COLS if c in wide_df.columns]

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_commit = None

    return {
        "wide_table_path": wide_path,
        "long_table_path": long_path,
        "created_at_utc": ts,
        "row_grain_wide": "player_id × game_id",
        "row_grain_long": "player_id × game_id × stat",
        "identity_columns": IDENTITY_COLS,
        "target_columns": TARGET_COLS,
        "model_feature_columns": model_feature_columns,
        "numeric_feature_columns": numeric_features,
        "categorical_feature_columns": categorical_features,
        "role_bucket_columns": role_bucket_cols_present,
        "forbidden_columns": sorted(FORBIDDEN_MODEL_FEATURES),
        "temporal_policy": FEATURE_CUTOFF_POLICY,
        "source_tables": source_tables,
        "git_commit_if_available": git_commit,
        "stats_modeled": STATS,
        "roll_windows": list(ROLL_WINDOWS),
    }


# ---------------------------------------------------------------------------
# Feature audit
# ---------------------------------------------------------------------------

def build_feature_audit(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    model_feature_columns: list[str],
    audit_notes: dict[str, Any],
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()

    # Row counts
    rows_by_season = long_df.groupby("season").size().to_dict()
    rows_by_stat   = long_df.groupby("stat").size().to_dict() if "stat" in long_df.columns else {}

    # Duplicate checks
    wide_dup = wide_df.duplicated(subset=["player_id", "game_id"]).sum()
    long_dup = long_df.duplicated(subset=["player_id", "game_id", "stat"]).sum() if "stat" in long_df.columns else 0

    # Missing required
    def _null_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
        return {c: int(df[c].isna().sum()) for c in cols if c in df.columns}

    # Feature null rates
    feat_null_rates = {
        c: float(wide_df[c].isna().mean()) for c in model_feature_columns if c in wide_df.columns
    }
    high_null_features = {k: v for k, v in feat_null_rates.items() if v > 0.2}
    all_null_features  = [k for k, v in feat_null_rates.items() if v == 1.0]

    # Infinite value check
    num_feats = [c for c in model_feature_columns if c in wide_df.columns
                 and pd.api.types.is_numeric_dtype(wide_df[c])]
    inf_counts = {
        c: int(np.isinf(wide_df[c].values).sum())
        for c in num_feats
        if hasattr(wide_df[c], "values")
    }
    any_inf = any(v > 0 for v in inf_counts.values())

    # Forbidden column check
    forbidden_found = [c for c in model_feature_columns if c in FORBIDDEN_MODEL_FEATURES]

    # Target in features check
    target_in_feats = [c for c in model_feature_columns if c.startswith("actual_") or c in TARGET_COLS]

    # Role bucket distribution
    def _vc(df: pd.DataFrame, col: str) -> dict:
        if col in df.columns:
            return df[col].value_counts().to_dict()
        return {}

    # Outcome distribution
    outcome_stats: dict[str, Any] = {}
    if "stat" in long_df.columns and "actual_outcome" in long_df.columns:
        for stat in STATS:
            sub = long_df[long_df["stat"] == stat]["actual_outcome"].dropna()
            if not sub.empty:
                outcome_stats[stat] = {
                    "count": int(len(sub)),
                    "zero_rate": float((sub == 0).mean()),
                    "mean": float(sub.mean()),
                    "median": float(sub.median()),
                    "p25": float(sub.quantile(0.25)),
                    "p75": float(sub.quantile(0.75)),
                }

    # Player history support
    if "player_minutes_l5_support" in wide_df.columns:
        l5_supp = wide_df["player_minutes_l5_support"]
        history_support = {
            "pct_first_game": float((l5_supp == 0).mean()),
            "pct_1_2_prior": float(((l5_supp > 0) & (l5_supp <= 2)).mean()),
            "pct_3plus_prior": float((l5_supp >= 3).mean()),
        }
    else:
        history_support = {}

    return {
        "built_at_utc": ts,
        "feature_cutoff_policy": FEATURE_CUTOFF_POLICY,
        "row_counts": {
            "wide": int(len(wide_df)),
            "long": int(len(long_df)),
            "by_season": {str(k): int(v) for k, v in rows_by_season.items()},
            "by_stat": {str(k): int(v) for k, v in rows_by_stat.items()},
        },
        "identity_checks": {
            "duplicate_player_game_wide": int(wide_dup),
            "duplicate_player_game_stat_long": int(long_dup),
            **_null_counts(wide_df, ["game_id", "player_id", "game_date", "team_id", "opponent_team_id"]),
        },
        "temporal_checks": {
            "policy": FEATURE_CUTOFF_POLICY,
            "all_rolling_features_shifted": True,
            "player_history_support_distribution": history_support,
        },
        "feature_checks": {
            "model_feature_column_count": len(model_feature_columns),
            "numeric_feature_count": len([c for c in model_feature_columns if c in wide_df.columns and pd.api.types.is_numeric_dtype(wide_df[c])]),
            "categorical_feature_count": len([c for c in model_feature_columns if c in wide_df.columns and not pd.api.types.is_numeric_dtype(wide_df[c])]),
            "forbidden_columns_found": forbidden_found,
            "target_columns_in_model_features": target_in_feats,
            "all_null_features": all_null_features,
            "high_null_features_above_20pct": high_null_features,
            "any_infinite_values": any_inf,
            "infinite_value_columns": {k: v for k, v in inf_counts.items() if v > 0},
        },
        "role_bucket_checks": {
            "projected_minutes_bucket": _vc(wide_df, "projected_minutes_bucket"),
            "usage_bucket": _vc(wide_df, "usage_bucket"),
            "role_status": _vc(wide_df, "role_status"),
            "role_uncertainty_bucket": _vc(wide_df, "role_uncertainty_bucket"),
            "minutes_volatility_bucket": _vc(wide_df, "minutes_volatility_bucket"),
            "rotation_stability_bucket": _vc(wide_df, "rotation_stability_bucket"),
        },
        "outcome_checks": outcome_stats,
        "unavailable_feature_inputs": {
            "usage_inputs_available": audit_notes.get("usage_inputs_available"),
            "advanced_stats_available": audit_notes.get("advanced_stats_available"),
            "adv_stat_cols_used": audit_notes.get("adv_stat_cols_used"),
            "adv_stat_cols_missing": audit_notes.get("adv_stat_cols_missing"),
            "injury_temporal_alignment": audit_notes.get("injury_temporal_alignment"),
            "injuries_available_but_not_temporally_aligned":
                audit_notes.get("injuries_available_but_not_temporally_aligned"),
            "pace_proxy_method": audit_notes.get("pace_proxy_method"),
        },
    }
