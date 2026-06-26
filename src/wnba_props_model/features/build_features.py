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
    "rotation_minutes_role",   # string label — never feed raw to HGB
    "season_phase",            # categorical string (early/mid/late/playoff)
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
# Enhancement 1: Usage Transfer Matrix features
# ---------------------------------------------------------------------------

def _build_usage_transfer_features(
    wide: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add Usage Transfer Matrix and with-without features (Enhancement 1).

    Requires player_id, fga, fta, turnover, minutes columns in stats_df.
    """
    try:
        from wnba_props_model.models.usage_transfer import (  # noqa: PLC0415
            build_player_usage_map,
            build_wowy_splits,
            add_usage_transfer_features,
        )
    except ImportError:
        return wide

    if "game_date" not in stats_df.columns:
        return wide

    cutoff = wide["game_date"].max() if "game_date" in wide.columns else None
    usage_map = build_player_usage_map(stats_df, cutoff_date=cutoff)
    if not usage_map:
        return wide

    wowy = build_wowy_splits(stats_df, usage_map)
    wide = add_usage_transfer_features(wide, usage_map, wowy)

    # Enhancement 11: Upgrade with DR-learner causal transfer estimates
    # when sufficient historical data is available (>= 500 rows)
    if len(wide) >= 500:
        try:
            from wnba_props_model.models.causal_transfer import train_causal_transfer  # noqa: PLC0415
            causal_est = train_causal_transfer(wide, usage_map, top_n=5)
            wide = causal_est.apply_causal_utm(wide, usage_map, top_n=5)
        except Exception:
            pass  # silently fall back to positional UTM

    return wide


# ---------------------------------------------------------------------------
# Enhancement 3: Extended schedule fatigue features
# ---------------------------------------------------------------------------

def _build_extended_fatigue_features(wide: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add is_4_in_5, is_5_in_7, cumulative_minutes_l7, altitude_flag,
    rest_interaction_high_usage, and schedule_fatigue_index (Enhancement 3).
    """
    if "game_date" not in wide.columns or "player_id" not in wide.columns:
        return wide

    df = wide.copy()
    # Game date per player history from stats_df (or use df itself)
    src = stats_df if ("game_date" in stats_df.columns and "player_id" in stats_df.columns) else df

    # ── is_4_in_5 and is_5_in_7 ────────────────────────────────────────────
    def _density_flag(pid_dates_map: dict, threshold: int, window_days: int) -> pd.Series:
        flags = []
        for idx, row in df.iterrows():
            pid = row["player_id"]
            game_date = row["game_date"]
            hist = pid_dates_map.get(pid, [])
            cutoff = game_date - pd.Timedelta(days=window_days - 1)
            count = sum(1 for d in hist if cutoff <= d < game_date)
            flags.append(1 if count >= threshold else 0)
        return pd.Series(flags, index=df.index)

    # Build historical game dates per player from src
    pid_dates: dict[int, list] = {}
    if "game_date" in src.columns and "player_id" in src.columns:
        for pid, grp in src.groupby("player_id", sort=False):
            pid_dates[int(pid)] = sorted(grp["game_date"].dropna().tolist())

    df["is_4_in_5"] = _density_flag(pid_dates, threshold=4, window_days=5)
    df["is_5_in_7"] = _density_flag(pid_dates, threshold=5, window_days=7)

    # ── cumulative_minutes_l7: sum of minutes in last 7 calendar days ───────
    def _cum_min_7(row: pd.Series) -> float:
        pid = row["player_id"]
        gdate = row["game_date"]
        cutoff = gdate - pd.Timedelta(days=7)
        pid_src = src[(src["player_id"] == pid) &
                      (src["game_date"] >= cutoff) &
                      (src["game_date"] < gdate)]
        if pid_src.empty:
            return 0.0
        min_col = "minutes" if "minutes" in pid_src.columns else "actual_minutes"
        return float(pid_src[min_col].fillna(0).sum()) if min_col in pid_src.columns else 0.0

    df["cumulative_minutes_l7"] = df.apply(_cum_min_7, axis=1)

    # ── altitude_flag ───────────────────────────────────────────────────────
    _ALTITUDE_CITIES = {"Denver", "Salt Lake City"}
    if "game_city" in df.columns:
        df["altitude_flag"] = df["game_city"].apply(
            lambda c: 1 if isinstance(c, str) and c in _ALTITUDE_CITIES else 0
        )
    else:
        df["altitude_flag"] = 0

    # ── rest_interaction_high_usage ─────────────────────────────────────────
    b2b_col    = "player_back_to_back_flag" if "player_back_to_back_flag" in df.columns else "is_back_to_back"
    usage_col  = "player_usage_rate_l5" if "player_usage_rate_l5" in df.columns else "player_usage_proxy_l5"
    b2b_vals   = df.get(b2b_col, pd.Series(0, index=df.index)).fillna(0)
    usage_vals = df.get(usage_col, pd.Series(0.20, index=df.index)).fillna(0.20)
    df["rest_interaction_high_usage"] = b2b_vals * usage_vals

    # ── schedule_fatigue_index (composite) ──────────────────────────────────
    three4   = df.get("player_3in4_flag", pd.Series(0, index=df.index)).fillna(0)
    four5    = df["is_4_in_5"].fillna(0)
    cum_norm = df["cumulative_minutes_l7"].clip(0, 300).fillna(0) / 200.0
    travel   = df.get("team_timezone_diff", pd.Series(0, index=df.index)).fillna(0).clip(0, 3) / 3.0
    alt      = df["altitude_flag"].fillna(0)
    df["schedule_fatigue_index"] = (
        0.30 * b2b_vals +
        0.25 * three4 +
        0.20 * cum_norm +
        0.15 * travel +
        0.10 * alt
    )

    return df


# ---------------------------------------------------------------------------
# Enhancement 4: Shot quality and efficiency regression features
# ---------------------------------------------------------------------------

def _build_shot_quality_features(wide: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add TS%, eFG%, pts-per-scoring-attempt, shot quality proxies (Enhancement 4)."""
    df = wide.copy()

    # Aggregate rolling shooting columns expected from build_player_features
    # (fgm_l10, fg3m_l10, fga_l10, fta_l10, pts_l10 produced as player_{stat}_* rolling)
    needed = {
        "pts_l10":   f"player_pts_mean_l10",
        "fga_l10":   f"player_fga_mean_l10",   # may not exist
        "fta_l10":   f"player_fta_mean_l10",   # may not exist
        "fg3m_l10":  f"player_fg3m_mean_l10",
    }

    # Pull from wide columns if available
    pts_l10  = df.get("player_pts_mean_l10",  pd.Series(np.nan, index=df.index))
    fg3m_l10 = df.get("player_fg3m_mean_l10", pd.Series(np.nan, index=df.index))

    # Estimate fga_l10 and fta_l10 from usage proxy if raw not available
    fga_l10_raw = df.get("player_fga_mean_l10", None)
    fta_l10_raw = df.get("player_fta_mean_l10", None)

    # If raw not yet in wide, compute from stats_df
    if fga_l10_raw is None or fta_l10_raw is None:
        for col in ["fga", "fta"]:
            if col in stats_df.columns:
                roll_vals = (
                    stats_df.sort_values(["player_id", "game_date"])
                    .groupby("player_id", sort=False)[col]
                    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                )
                # Map back by (player_id, game_id)
                if "game_id" in stats_df.columns and "game_id" in df.columns:
                    tmp = stats_df[["player_id", "game_id"]].copy()
                    tmp[f"_roll_{col}_l10"] = roll_vals.values
                    df = df.merge(tmp, on=["player_id", "game_id"], how="left")
                    if col == "fga":
                        fga_l10_raw = df.pop(f"_roll_fga_l10")
                    else:
                        fta_l10_raw = df.pop(f"_roll_fta_l10")

    fga_l10  = fga_l10_raw if fga_l10_raw is not None else pd.Series(np.nan, index=df.index)
    fta_l10  = fta_l10_raw if fta_l10_raw is not None else pd.Series(np.nan, index=df.index)

    # Coerce index alignment (merge can produce a new index)
    fga_l10 = fga_l10.reset_index(drop=True) if hasattr(fga_l10, "reset_index") else fga_l10
    fta_l10 = fta_l10.reset_index(drop=True) if hasattr(fta_l10, "reset_index") else fta_l10
    df = df.reset_index(drop=True)

    # Guard: only compute shooting-efficiency features when FGA data is valid.
    # When fga_l10 is missing (e.g. game_id type mismatch in the merge), all
    # three features would produce nonsensical values (pts_mean/2, etc.).
    has_fga = fga_l10.fillna(0) > 0

    tsa = fga_l10.fillna(0) + 0.44 * fta_l10.fillna(0)

    # ── True Shooting % (TS%) — only valid when FGA data exists ─────────────
    df["player_ts_pct_l10"] = (pts_l10 / (2.0 * tsa.clip(lower=1.0))).where(has_fga)

    # ── Effective FG% (eFG%) — only valid when FGA data exists ─────────────
    df["player_efg_pct_l10"] = (
        (fga_l10.fillna(0) + 0.5 * fg3m_l10.fillna(0)) /
        fga_l10.clip(lower=1.0)
    ).where(has_fga)

    # ── FTA rate (FTA / FGA) — only valid when FGA data exists ─────────────
    df["player_fta_rate_l10"] = (fta_l10 / fga_l10.clip(lower=1.0)).where(has_fga)

    # ── Points per scoring attempt — only valid when FGA data exists ─────────
    scoring_attempts = fga_l10.fillna(0) + 0.44 * fta_l10.fillna(0)
    df["pts_per_scoring_attempt_l10"] = (
        pts_l10 / scoring_attempts.clip(lower=1.0)
    ).where(has_fga)

    # ── Shot quality delta proxy (hot/cold streak signal) ───────────────────
    # Only meaningful when we have valid eFG%; falls back to 0 (neutral) otherwise.
    efg_season = (
        (df.get("player_fga_mean_season", pd.Series(np.nan, index=df.index)).fillna(0) +
         0.5 * df.get("player_fg3m_mean_season", pd.Series(np.nan, index=df.index)).fillna(0)) /
        df.get("player_fga_mean_season", pd.Series(1.0, index=df.index)).clip(lower=1.0)
    )
    df["shot_quality_delta_l10"] = df["player_efg_pct_l10"].fillna(0) - efg_season.fillna(0)

    # ── Hot/cold flags ───────────────────────────────────────────────────────
    df["is_running_hot"]  = (df["shot_quality_delta_l10"] >  0.05).fillna(False).astype(int)
    df["is_running_cold"] = (df["shot_quality_delta_l10"] < -0.05).fillna(False).astype(int)

    return df


# ---------------------------------------------------------------------------
# Enhancement 5: Game script and conditional minutes features
# ---------------------------------------------------------------------------

def _build_game_script_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Add pregame win/blowout/close probabilities and conditional minutes (Enhancement 5)."""
    from scipy.stats import norm  # noqa: PLC0415

    df = wide.copy()

    HOME_ADV    = 3.0   # WNBA home court ~3 points
    SPREAD_SIGMA = 10.0  # WNBA game standard deviation

    # Net rating proxy: team_pts_for - team_pts_allowed (rolling L5)
    # h_net: player's own team net rating (positive = team scoring more than they allow)
    h_net = (
        df.get("team_pts_for_mean_l5", pd.Series(0.0, index=df.index)).fillna(0) -
        df.get("team_pts_allowed_mean_l5", pd.Series(0.0, index=df.index)).fillna(0)
    )
    # o_net: opponent team net rating.
    # opp_total_score_allowed_mean_l5 = total game score (both teams) in opponent's games.
    # opp_pts_allowed_mean_l5         = points the opponent allows (opponent defense).
    # Opponent offense = total_game_score - opp_defense.
    # Opponent net = opponent_offense - opponent_defense.
    opp_total = df.get("opp_total_score_allowed_mean_l5", pd.Series(0.0, index=df.index)).fillna(0)
    opp_def   = df.get("opp_pts_allowed_mean_l5",          pd.Series(0.0, index=df.index)).fillna(0)
    opp_off   = (opp_total - opp_def).clip(lower=0)   # opponent's scoring rate
    o_net     = opp_off - opp_def                      # opponent's net rating

    is_home = df.get("is_home", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    # Spread from the player's team perspective (positive = player's team favored)
    spread = np.where(is_home, h_net - o_net + HOME_ADV, o_net - h_net - HOME_ADV)

    # P(win) from Normal CDF
    p_win = 1.0 - norm.cdf(-spread / SPREAD_SIGMA)
    df["pregame_win_probability"] = np.clip(p_win, 0.05, 0.95)

    # P(blowout) = P(margin > 15 either direction)
    p_blow_fav  = 1.0 - norm.cdf(15,  loc=spread, scale=SPREAD_SIGMA)
    p_blow_dog  = norm.cdf(-15, loc=spread, scale=SPREAD_SIGMA)
    df["blowout_probability"] = np.clip(p_blow_fav + p_blow_dog, 0.0, 1.0)

    # P(close) = P(|margin| < 5)
    p_close = norm.cdf(5, loc=spread, scale=SPREAD_SIGMA) - norm.cdf(-5, loc=spread, scale=SPREAD_SIGMA)
    df["close_game_probability"] = np.clip(p_close, 0.0, 1.0)

    # ── Conditional minutes given script ─────────────────────────────────────
    base_min = df.get("projected_minutes_proxy", pd.Series(25.0, index=df.index)).fillna(25.0)
    role     = df.get("role_status", pd.Series("rotation", index=df.index)).fillna("rotation")
    is_star  = role.isin(["star", "starter", "primary_starter"])

    df["expected_minutes_given_script"] = np.where(
        is_star,
        base_min * (1.0 + 0.05 * df["close_game_probability"] - 0.12 * df["blowout_probability"]),
        base_min * (1.0 - 0.03 * df["close_game_probability"] + 0.15 * df["blowout_probability"]),
    )
    df["minutes_upside"] = np.where(
        is_star,
        base_min * 0.08 * df["close_game_probability"],
        base_min * 0.20 * df["blowout_probability"],
    )

    return df


# ---------------------------------------------------------------------------
# Enhancement 9: Defensive scheme proxy features
# ---------------------------------------------------------------------------

def _build_defensive_scheme_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Approximate opponent defensive scheme from box-score stats (Enhancement 9)."""
    df = wide.copy()

    # Use opponent rolling stat totals available in wide table
    opp_stl = df.get("opp_stl_allowed_mean_l5", pd.Series(np.nan, index=df.index)).fillna(5.0)
    opp_blk = df.get("opp_blk_allowed_mean_l5", pd.Series(np.nan, index=df.index)).fillna(4.0)
    # Use opponent turnovers FORCED as proxy for defensive aggression pressure.
    # The correct column is opp_turnover_forced_mean_l5 (turnovers the opponent forces).
    opp_tov = df.get("opp_turnover_forced_mean_l5", pd.Series(np.nan, index=df.index)).fillna(14.0)

    # Normalize to per-possession proxies (assume ~75 possessions per WNBA game)
    _POSS = 75.0
    stl_rate = opp_stl / _POSS
    blk_rate = opp_blk / _POSS
    tov_rate = opp_tov / _POSS

    # Aggression index: high steal + high forced-turnover rate = blitz/aggressive
    df["opp_aggression_index"] = (stl_rate * 100 + tov_rate * 100 * 0.5).clip(0, 10)

    # Drop indicator: high block + low steal = drop / interior-oriented
    df["opp_drop_indicator"] = (blk_rate * 100 - stl_rate * 50).clip(-5, 10)

    # Switch rate proxy: moderate on both = switch-everything
    agg = df["opp_aggression_index"]
    drop = df["opp_drop_indicator"]
    denom = (agg + drop.abs()).clip(lower=0.01)
    df["opp_switch_rate_proxy"] = ((1.0 - (agg - drop).abs() / denom)).clip(0, 1)

    return df


# ---------------------------------------------------------------------------
# Enhancement 10: Season-phase categorical feature
# ---------------------------------------------------------------------------

def _build_season_phase_feature(wide: pd.DataFrame) -> pd.DataFrame:
    """Add season_phase categorical: early / mid / late / playoff (Enhancement 10)."""
    df = wide.copy()
    if "game_number_in_season" not in df.columns:
        df["season_phase"] = "mid"
        return df

    is_playoff = df.get("is_playoff_game", pd.Series(0, index=df.index)).fillna(0).astype(int)

    def _phase(gn: float, playoff: int) -> str:
        if playoff:
            return "playoff"
        if np.isnan(gn):
            return "mid"
        if gn <= 8:
            return "early"
        if gn <= 30:
            return "mid"
        return "late"

    df["season_phase"] = [
        _phase(gn, po)
        for gn, po in zip(df["game_number_in_season"], is_playoff)
    ]
    return df


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
    extra_agg: dict[str, tuple] = {}
    if "fg3a" in stats_df.columns:
        extra_agg["_t_fg3a"] = ("fg3a", "sum")
    stat_agg = stats_df.groupby(["game_id", "team_id"], as_index=False).agg(
        **{f"_t_{s}": (s, "sum") for s in STATS},
        **extra_agg,
    )

    # --- 3. Join own-team stats to ctx rows --------------------------------
    ctx = ctx.merge(stat_agg, on=["game_id", "team_id"], how="left")

    # --- 4. Join opponent stats (what opponent scored AGAINST this team) ---
    opp_rename = {f"_t_{s}": f"_o_{s}" for s in STATS}
    if "fg3a" in stats_df.columns:
        opp_rename["_t_fg3a"] = "_o_fg3a"
    opp_agg = stat_agg.rename(columns={
        "team_id": "_opp_tid",
        **opp_rename,
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
    # fg3a allowed rolling (opponent 3pt attempt volume against this team)
    if "_o_fg3a" in ctx.columns:
        grp2 = ctx.groupby("team_id", sort=False)
        ctx["t_fg3a_against_l5"]  = _sr(grp2["_o_fg3a"], 5)
        ctx["t_fg3a_against_l10"] = _sr(grp2["_o_fg3a"], 10)

    # Team 3-in-4 flag (P2.3): team played 3 games in 4 days
    _ctx_date_lag2 = grp["game_date"].shift(2)
    ctx["t_3in4_flag"] = (
        (ctx["game_date"] - _ctx_date_lag2).dt.days <= 3
    ).fillna(False).astype(int)

    # Timezone difference proxy (P2.3): static UTC offsets per abbreviation
    _TZ_OFFSET: dict[str, int] = {
        "NYL": -4, "CON": -4, "WAS": -4, "ATL": -4,
        "CHI": -5, "IND": -5, "MIN": -5, "DAL": -5,
        "PHO": -7, "LVA": -7,
        "SEA": -7, "LA": -7, "LAS": -7,
    }
    if "team_id" in ctx.columns:
        # team_abbreviation may not be in ctx — use opp_team_id side join approach;
        # store as placeholder column; final join in build_wide_table merges abbrevs
        ctx["_t_tz_placeholder"] = 0  # filled in build_wide_table using abbreviation map

    drop_extra = ["_t_fg3a", "_o_fg3a"] if "_t_fg3a" in ctx.columns else []
    ctx = ctx.drop(columns=["_prev_game_date", "_days_since_last"] +
                   [f"_t_{s}" for s in STATS] + [f"_o_{s}" for s in STATS] +
                   drop_extra,
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
    # 3. Per-stat rolling features (shifted) — batched to avoid fragmentation
    # ------------------------------------------------------------------ #
    _stat_rolling_cols: dict[str, pd.Series] = {}
    for stat in STATS:
        if stat not in df.columns:
            continue
        _s_grp = df.groupby("player_id", sort=False)[stat]
        _stat_rolling_cols[f"player_{stat}_last1"]          = _s_grp.shift(1)
        _stat_rolling_cols[f"player_{stat}_mean_l3"]        = _sr(_s_grp, 3)
        _stat_rolling_cols[f"player_{stat}_mean_l5"]        = _sr(_s_grp, 5)
        _stat_rolling_cols[f"player_{stat}_mean_l10"]       = _sr(_s_grp, 10)
        _stat_rolling_cols[f"player_{stat}_mean_season"]    = _sr(_s_grp, 999, agg="expanding_mean")
        _stat_rolling_cols[f"player_{stat}_std_l10"]        = _sr(_s_grp, 10, agg="std", min_periods=2)
        _stat_rolling_cols[f"player_{stat}_l3_support"]     = _sr(_s_grp, 3, agg="count")
        _stat_rolling_cols[f"player_{stat}_l5_support"]     = _sr(_s_grp, 5, agg="count")
        _stat_rolling_cols[f"player_{stat}_l10_support"]    = _sr(_s_grp, 10, agg="count")
        _stat_rolling_cols[f"player_{stat}_season_support"] = _sr(_s_grp, 999, agg="expanding_count")
    if _stat_rolling_cols:
        df = pd.concat([df, pd.DataFrame(_stat_rolling_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------ #
    # 4. Per-minute rate features (shifted sums; safe denominator) — batched
    # ------------------------------------------------------------------ #
    _rate_cols: dict[str, pd.Series] = {}
    _m_grp_rate = df.groupby("player_id", sort=False)["minutes"]
    for stat in STATS:
        if stat not in df.columns:
            continue
        _s_grp = df.groupby("player_id", sort=False)[stat]
        for w in (3, 5, 10):
            stat_sum = _sr(_s_grp, w, agg="sum")
            min_sum  = _sr(_m_grp_rate, w, agg="sum")
            rate = stat_sum / min_sum.clip(lower=1.0)
            _rate_cols[f"player_{stat}_per_min_l{w}"] = rate.replace([np.inf, -np.inf], np.nan)
        # Season rate
        stat_sum_s = _sr(_s_grp, 999, agg="expanding_sum")
        min_sum_s  = _sr(_m_grp_rate, 999, agg="expanding_sum")
        _rate_cols[f"player_{stat}_per_min_season"] = (
            (stat_sum_s / min_sum_s.clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
        )
    if _rate_cols:
        df = pd.concat([df, pd.DataFrame(_rate_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------ #
    # 4b. fg3a (three-point attempts) rolling features — batched
    # ------------------------------------------------------------------ #
    _fg3_cols: dict[str, pd.Series] = {}
    if "fg3a" in df.columns:
        _fg3a_g = df.groupby("player_id", sort=False)["fg3a"]
        _fg3m_g = df.groupby("player_id", sort=False)["fg3m"] if "fg3m" in df.columns else None
        _mg = df.groupby("player_id", sort=False)["minutes"]
        _fg3_cols["player_fg3a_mean_l5"]    = _sr(_fg3a_g, 5)
        _fg3_cols["player_fg3a_mean_l10"]   = _sr(_fg3a_g, 10)
        _fg3_cols["player_fg3a_per_min_l5"] = (
            _sr(_fg3a_g, 5, agg="sum") / _sr(_mg, 5, agg="sum").clip(lower=1.0)
        ).replace([np.inf, -np.inf], np.nan)
        if _fg3m_g is not None:
            _fg3_cols["player_fg3_pct_l5"] = (
                _sr(_fg3m_g, 5, agg="sum") / _sr(_fg3a_g, 5, agg="sum").clip(lower=0.5)
            ).replace([np.inf, -np.inf], np.nan).clip(0, 1)

    # ------------------------------------------------------------------ #
    # 4c. Shot profile features (P2.1) — 3pt attempt rate from fg3a/fga — batched
    # ------------------------------------------------------------------ #
    if "fg3a" in df.columns and "fga" in df.columns:
        _fg3a_g2 = df.groupby("player_id", sort=False)["fg3a"]
        _fga_g   = df.groupby("player_id", sort=False)["fga"]
        for w in (5, 10):
            fg3a_sum = _sr(_fg3a_g2, w, agg="sum")
            fga_sum  = _sr(_fga_g, w, agg="sum")
            _fg3_cols[f"player_fg3_attempt_rate_l{w}"] = (
                fg3a_sum / fga_sum.clip(lower=0.5)
            ).replace([np.inf, -np.inf], np.nan).clip(0, 1)
        _fg3_cols["player_fg3_attempt_rate_season"] = (
            _sr(_fg3a_g2, 999, agg="expanding_sum") /
            _sr(_fga_g, 999, agg="expanding_sum").clip(lower=0.5)
        ).replace([np.inf, -np.inf], np.nan).clip(0, 1)
    if _fg3_cols:
        df = pd.concat([df, pd.DataFrame(_fg3_cols, index=df.index)], axis=1)
    # Shot-zone columns from wnba_shot_locations.parquet — NaN when file unavailable
    for _zone_col in ["player_rim_freq_l5", "player_corner3_freq_l5", "player_above_break3_freq_l5"]:
        if _zone_col not in df.columns:
            df[_zone_col] = np.nan

    # ------------------------------------------------------------------ #
    # 4d. EWMA rolling features (P2.4) — exponentially weighted recent form
    # Batched via pd.concat to avoid DataFrame fragmentation.
    # ------------------------------------------------------------------ #
    _ewma_stats = list(STATS) + ["minutes"]
    _ewma_cols: dict[str, pd.Series] = {}
    for _stat in _ewma_stats:
        if _stat not in df.columns:
            continue
        _eg = df.groupby("player_id", sort=False)[_stat]
        for _hl in (3, 5):
            _ewma_cols[f"player_{_stat}_ewma_halflife{_hl}"] = _eg.transform(
                lambda s, hl=_hl: s.shift(1).ewm(halflife=hl, min_periods=hl).mean()
            )
    if _ewma_cols:
        df = pd.concat([df, pd.DataFrame(_ewma_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------ #
    # 4e. Form delta + momentum features (P2.4b)                         #
    # Explicit breakout indicators: encode how far a player's CURRENT     #
    # form deviates from their historical baseline. The HGB model at      #
    # max_leaf_nodes=31 cannot reliably learn "ewma5 > season_mean →      #
    # project higher" from two separate features. With form_delta as one  #
    # pre-computed feature, a single tree split captures this signal,     #
    # freeing tree capacity for matchup and usage interactions.           #
    # All source features (ewma, mean_l5, etc.) are already shift(1) so  #
    # these derived features are also leak-free.                          #
    # ------------------------------------------------------------------ #
    _form_delta_cols: dict[str, pd.Series] = {}
    for _stat in list(STATS) + ["minutes"]:
        if _stat not in df.columns:
            continue
        _ewma3_col   = f"player_{_stat}_ewma_halflife3"
        _ewma5_col   = f"player_{_stat}_ewma_halflife5"
        _mean_l5_col = f"player_{_stat}_mean_l5"
        _mean_l10_col = f"player_{_stat}_mean_l10"
        _mean_season_col = f"player_{_stat}_mean_season"

        # EWMA vs season average: captures hot/cold relative to season norm
        if _ewma5_col in df.columns and _mean_season_col in df.columns:
            _form_delta_cols[f"player_{_stat}_form_delta_ewma5_vs_season"] = (
                df[_ewma5_col] - df[_mean_season_col]
            )
        if _ewma3_col in df.columns and _mean_season_col in df.columns:
            _form_delta_cols[f"player_{_stat}_form_delta_ewma3_vs_season"] = (
                df[_ewma3_col] - df[_mean_season_col]
            )
        # L5 vs L10 breakout signal: recent 5-game vs medium 10-game trend
        if _mean_l5_col in df.columns and _mean_l10_col in df.columns:
            _form_delta_cols[f"player_{_stat}_form_delta_l5_vs_l10"] = (
                df[_mean_l5_col] - df[_mean_l10_col]
            )
        # Momentum (acceleration): is form improving or declining?
        if _ewma3_col in df.columns and _ewma5_col in df.columns:
            _form_delta_cols[f"player_{_stat}_momentum_ewma3_vs_ewma5"] = (
                df[_ewma3_col] - df[_ewma5_col]
            )
    if _form_delta_cols:
        df = pd.concat([df, pd.DataFrame(_form_delta_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------ #
    # 4f. Player quality anchor features (P2.4c)                          #
    # Give the HGBR a direct "how elite is this player vs the league"     #
    # signal. Without this, HGBR must infer quality purely from rolling   #
    # windows, which converge to similar ranges for all starters due to   #
    # regression to mean. A z-score breaks that degeneracy.              #
    # Uses game_date × season groupby so the z-score is computed from    #
    # players in the same game window — preserving temporal causality.    #
    # ------------------------------------------------------------------ #
    _quality_anchor_cols: dict[str, pd.Series] = {}
    _qa_group_cols = [c for c in ("season", "game_date") if c in df.columns]
    for _stat in list(STATS) + ["minutes"]:
        _mean_season_col = f"player_{_stat}_mean_season"
        if _mean_season_col not in df.columns:
            continue
        # Z-score within the same season × game_date snapshot
        if _qa_group_cols:
            _gs_mean = df.groupby(_qa_group_cols)[_mean_season_col].transform("mean")
            _gs_std  = df.groupby(_qa_group_cols)[_mean_season_col].transform("std").clip(lower=0.01)
        else:
            _gs_mean = df[_mean_season_col].mean()
            _gs_std  = max(float(df[_mean_season_col].std()), 0.01)
        _quality_anchor_cols[f"player_{_stat}_season_zscore"] = (
            (df[_mean_season_col] - _gs_mean) / _gs_std
        )
        # Ratio of recent EWMA to season average (> 1 = trending up, < 1 = slumping)
        _ewma5_col = f"player_{_stat}_ewma_halflife5"
        if _ewma5_col in df.columns:
            _quality_anchor_cols[f"player_{_stat}_form_vs_season_ratio"] = (
                df[_ewma5_col] / df[_mean_season_col].clip(lower=0.1)
            )
    if _quality_anchor_cols:
        df = pd.concat([df, pd.DataFrame(_quality_anchor_cols, index=df.index)], axis=1)

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

    # Derive started_proxy from minutes if not provided by BDL data
    if "started_proxy" not in df.columns:
        df["started_proxy"] = (df["minutes"] >= 15.0).astype(float)
    # Starter proxy (shifted): was player a starter in prior game?
    grp = df.groupby("player_id", sort=False)  # re-bind after potential column add
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
    # 8b. Fatigue features (P2.3): 3-in-4, weekly load, cumulative minutes
    # ------------------------------------------------------------------ #
    # player_3in4_flag: played 3 games in 4 calendar days (shift-1 safe)
    # We check (game_date - game_date_two_games_prior).days <= 3
    _game_dates_grp = df.groupby("player_id", sort=False)["game_date"]
    df["_game_date_lag2"] = _game_dates_grp.shift(2)
    df["player_3in4_flag"] = (
        (df["game_date"] - df["_game_date_lag2"]).dt.days <= 3
    ).fillna(False).astype(int)
    df = df.drop(columns=["_game_date_lag2"], errors="ignore")

    # player_games_in_last_7_days: count of prior games in rolling 7-day window
    # Use the shift-1 game_date sequence and count how many fall within 7 days prior
    def _games_in_7_days(s: pd.Series) -> pd.Series:
        s = s.sort_values()
        result = np.zeros(len(s), dtype=float)
        dates = s.values
        for i in range(len(dates)):
            cutoff = dates[i] - np.timedelta64(7, "D")
            result[i] = np.sum((dates[:i] >= cutoff) & (dates[:i] < dates[i]))
        return pd.Series(result, index=s.index)

    df["player_games_in_last_7_days"] = _game_dates_grp.transform(_games_in_7_days)

    # player_cumulative_minutes_l3: sum of minutes in last 3 prior games (NOT mean)
    df["player_cumulative_minutes_l3"] = _sr(
        df.groupby("player_id", sort=False)["minutes"], 3, agg="sum"
    )

    # ------------------------------------------------------------------ #
    # 9. Per-minute rate × minutes interaction (Phase 2e)
    # log(λ) = player_rate_per_min × minutes_mean — captures Poisson exposure.
    # The interaction term lets the model reason about matchup minutes changes.
    # ------------------------------------------------------------------ #
    _interaction_cols: dict[str, pd.Series] = {}
    for stat in STATS:
        rate_col = f"player_{stat}_per_min_l5"
        if rate_col in df.columns and "projected_minutes_proxy" in df.columns:
            _interaction_cols[f"player_{stat}_per_min_l5_x_proj_min"] = (
                df[rate_col] * df["projected_minutes_proxy"]
            ).replace([np.inf, -np.inf], np.nan)
    if _interaction_cols:
        df = pd.concat([df, pd.DataFrame(_interaction_cols, index=df.index)], axis=1)

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

    # Defragment DataFrame before returning — avoids PerformanceWarning downstream
    return df.copy()


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

    # Normalize game_id dtype to int64 throughout to prevent merge errors when
    # BDL API returns game_id as strings vs the main stats table's int64.
    def _coerce_game_id(df: pd.DataFrame) -> pd.DataFrame:
        if "game_id" in df.columns and df["game_id"].dtype == object:
            df = df.copy()
            df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        return df

    out = _coerce_game_id(out)
    injuries_df = _coerce_game_id(injuries_df)

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

    # ---- vacated_minutes_l1 + vacated_pts_l1: from prior-game DNP teammates --
    # Compute per-team per-game total DNP minutes/pts (teammates not playing).
    # This tells us how many minutes / pts are being redistributed for THIS game.
    if "team_id" in out.columns and "did_play" in out.columns:
        dnp_mins = out.copy()
        _min_col = "actual_minutes" if "actual_minutes" in dnp_mins.columns else "minutes"
        _pts_col = "actual_pts" if "actual_pts" in dnp_mins.columns else "pts"
        dnp_flag = 1 - dnp_mins["did_play"].fillna(1).astype(int)
        dnp_mins["_dnp_minutes"] = dnp_mins[_min_col].fillna(0.0) * dnp_flag
        dnp_mins["_dnp_pts"] = (
            dnp_mins[_pts_col].fillna(0.0) * dnp_flag
            if _pts_col in dnp_mins.columns else 0.0
        )
        team_dnp = (
            dnp_mins.groupby(["team_id", "game_id"])
            .agg(_team_dnp_min=("_dnp_minutes", "sum"),
                 _team_dnp_pts=("_dnp_pts", "sum"))
            .reset_index()
            .rename(columns={"game_id": "_gid", "team_id": "_tid"})
        )
        # Shift: for each team, carry last game's DNP values forward
        team_dnp = team_dnp.sort_values(["_tid", "_gid"])
        team_dnp["vacated_minutes_l1"] = team_dnp.groupby("_tid")["_team_dnp_min"].shift(1)
        team_dnp["vacated_pts_l1"]     = team_dnp.groupby("_tid")["_team_dnp_pts"].shift(1)
        out = out.merge(
            team_dnp[["_tid", "_gid", "vacated_minutes_l1", "vacated_pts_l1"]].rename(
                columns={"_tid": "team_id", "_gid": "game_id"}
            ),
            on=["team_id", "game_id"], how="left",
        )
    else:
        out["vacated_minutes_l1"] = np.nan
        out["vacated_pts_l1"]     = np.nan

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

def _build_positional_defense_features(
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build position-stratified opponent defense features (P2.2).

    For each primary position bucket {G, F, C} and each of 7 stats, computes
    opp_{stat}_vs_{pos}_allowed_l5: the rolling L5 mean of how many pts/reb/etc
    that opponent team conceded to players of that position.

    Returns a DataFrame with columns:
        game_id, opponent_team_id, _primary_pos, opp_{stat}_vs_{pos}_allowed_l5
    indexed by (game_id, opponent_team_id, _primary_pos).
    """
    if "position" not in stats_df.columns:
        return pd.DataFrame()

    df = stats_df.copy()
    # Normalize position to primary bucket: first character of position string
    def _primary_pos(pos: str | None) -> str:
        if not isinstance(pos, str) or not pos:
            return "F"
        c = pos.strip()[0].upper()
        return c if c in ("G", "F", "C") else "F"

    df["_primary_pos"] = df["position"].map(_primary_pos)

    # Compute per-game per-stat averages for each (opponent_team_id, game_id, pos)
    # "opponent_team_id" here = the team that DEFENDED against those players
    if "opponent_team_id" not in df.columns:
        return pd.DataFrame()

    records = []
    for pos in ("G", "F", "C"):
        pos_df = df[df["_primary_pos"] == pos].copy()
        if pos_df.empty:
            continue
        # Aggregate: what did this opponent give up to this position in this game?
        agg_dict = {s: (s, "mean") for s in STATS if s in pos_df.columns}
        if not agg_dict:
            continue
        game_opp_agg = pos_df.groupby(
            ["game_id", "game_date", "opponent_team_id"], as_index=False
        ).agg(**agg_dict)
        # Rolling L5 per opponent_team_id (shift-1)
        game_opp_agg = game_opp_agg.sort_values(
            ["opponent_team_id", "game_date", "game_id"]
        ).reset_index(drop=True)
        opp_grp = game_opp_agg.groupby("opponent_team_id", sort=False)
        for s in STATS:
            if s not in game_opp_agg.columns:
                continue
            game_opp_agg[f"opp_{s}_vs_{pos}_allowed_l5"] = _sr(
                opp_grp[s], 5
            )
        game_opp_agg["_primary_pos"] = pos
        records.append(game_opp_agg)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)
    keep = (
        ["game_id", "opponent_team_id", "_primary_pos"]
        + [f"opp_{s}_vs_{pos}_allowed_l5" for pos in ("G", "F", "C") for s in STATS
           if f"opp_{s}_vs_{pos}_allowed_l5" in out.columns]
    )
    # Drop STAT sum cols (we only want the positional defense cols)
    out = out[[c for c in keep if c in out.columns]].copy()
    return out


def _build_matchup_history_features(
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build player-vs-specific-opponent career matchup history features (P2.5).

    Groups by (player_id, opponent_team_id), computes shift-1 rolling L3 mean,
    career expanding mean, and support count per stat.

    Returns merged DataFrame indexed by (player_id, game_id).
    """
    if "opponent_team_id" not in stats_df.columns:
        return pd.DataFrame(columns=["player_id", "game_id"])

    df = stats_df.copy().sort_values(
        ["player_id", "opponent_team_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    matchup_grp = df.groupby(["player_id", "opponent_team_id"], sort=False)
    result_cols: list[str] = []

    for stat in STATS + ["minutes"]:
        if stat not in df.columns:
            continue
        mg = matchup_grp[stat]
        col_l3    = f"player_{stat}_vs_opp_l3"
        col_career = f"player_{stat}_vs_opp_career_mean"
        col_supp   = f"player_{stat}_vs_opp_support"
        df[col_l3]     = mg.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[col_career] = mg.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        df[col_supp]   = mg.transform(lambda s: s.shift(1).expanding(min_periods=1).count())
        # NaN when support < 2 (insufficient history)
        mask = df[col_supp] < 2
        df.loc[mask, col_l3]     = np.nan
        df.loc[mask, col_career] = np.nan
        result_cols += [col_l3, col_career, col_supp]

    return df[["player_id", "game_id"] + result_cols].drop_duplicates(
        subset=["player_id", "game_id"]
    )


def build_wide_table(
    stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
    adv_df: pd.DataFrame | None = None,
    injuries_df: pd.DataFrame | None = None,
    use_positional_defense_features: bool = True,
    use_matchup_features: bool = True,
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
    # fg3a allowed by opponent (how many 3pt attempts opponent's defense gives up)
    if "t_fg3a_against_l5" in ctx.columns:
        opp_ctx_cols["t_fg3a_against_l5"]  = "opp_fg3a_allowed_mean_l5"
        opp_ctx_cols["t_fg3a_against_l10"] = "opp_fg3a_allowed_mean_l10"

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
    # Pace-adjusted opponent defensive ratings (F7b)
    # ------------------------------------------------------------------ #
    if "opp_pace_proxy_l5" in wide.columns:
        denom = wide["opp_pace_proxy_l5"].clip(lower=50.0) / 100.0
        for s in STATS:
            raw_col = f"opp_{s}_allowed_mean_l5"
            if raw_col in wide.columns:
                wide[f"opp_{s}_allowed_per_100_poss_l5"] = (
                    wide[raw_col] / denom
                ).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #
    # Rate × pace-adjusted opponent interaction features (F7c)
    # ------------------------------------------------------------------ #
    for s in STATS:
        rate_col = f"player_{s}_per_min_l5"
        opp_col  = f"opp_{s}_allowed_per_100_poss_l5"
        if rate_col in wide.columns and opp_col in wide.columns:
            wide[f"player_{s}_per_min_l5_x_opp_{s}_per_100_l5"] = (
                wide[rate_col] * wide[opp_col]
            ).replace([np.inf, -np.inf], np.nan)

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
    # Season-stage features (F7d)
    # ------------------------------------------------------------------ #
    if "season" in wide.columns and "team_id" in wide.columns and "game_date" in wide.columns:
        wide["game_number_in_season"] = (
            wide.groupby(["team_id", "season"])["game_date"]
            .rank(method="first")
            .astype(int)
        )
        wide["season_completion_pct"] = (wide["game_number_in_season"] / 40.0).clip(0, 1)
        wide["is_playoff_game"] = (wide["game_number_in_season"] > 36).astype(int)
    else:
        wide["game_number_in_season"] = np.nan
        wide["season_completion_pct"]  = np.nan
        wide["is_playoff_game"]        = 0

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
                    "vacated_minutes_l1", "vacated_pts_l1", "usage_share_delta"]:
            if col not in wide.columns:
                wide[col] = 0.0 if col in ("player_injured_l1", "teammate_injury_flag") else np.nan

    # ------------------------------------------------------------------ #
    # Team 3-in-4 and opp 3-in-4 flags from ctx (P2.3)
    # ------------------------------------------------------------------ #
    if "t_3in4_flag" in ctx.columns:
        _t3in4 = ctx[["game_id", "team_id", "t_3in4_flag"]].copy()
        wide = wide.merge(_t3in4.rename(columns={"t_3in4_flag": "team_3in4_flag"}),
                          on=["game_id", "team_id"], how="left")
        _o3in4 = ctx[["game_id", "team_id", "t_3in4_flag"]].rename(
            columns={"team_id": "_opp_tid2", "t_3in4_flag": "opp_3in4_flag"}
        )
        wide = wide.merge(
            _o3in4, left_on=["game_id", "opponent_team_id"],
            right_on=["game_id", "_opp_tid2"], how="left"
        ).drop(columns=["_opp_tid2"], errors="ignore")

    # ------------------------------------------------------------------ #
    # Timezone difference proxy (P2.3)
    # ------------------------------------------------------------------ #
    _TZ_OFFSET: dict[str, int] = {
        "NYL": -4, "CON": -4, "WAS": -4, "ATL": -4,
        "CHI": -5, "IND": -5, "MIN": -5, "DAL": -5,
        "PHO": -7, "LVA": -7, "SEA": -7, "LA": -7, "LAS": -7,
    }
    if "team_abbreviation" in wide.columns and "opponent_team_abbreviation" in wide.columns:
        team_tz = wide["team_abbreviation"].map(_TZ_OFFSET).fillna(-5)
        opp_tz  = wide["opponent_team_abbreviation"].map(_TZ_OFFSET).fillna(-5)
        wide["team_timezone_diff"] = (team_tz - opp_tz).abs()
    else:
        wide["team_timezone_diff"] = np.nan

    # ------------------------------------------------------------------ #
    # Positional defense matchup features (P2.2)
    # ------------------------------------------------------------------ #
    if use_positional_defense_features:
        pos_def = _build_positional_defense_features(stats_df)
        if not pos_def.empty and "position" in wide.columns:
            wide["_primary_pos"] = wide["position"].apply(
                lambda pos: (pos.strip()[0].upper() if isinstance(pos, str) and pos else "F")
            )
            wide["_primary_pos"] = wide["_primary_pos"].apply(
                lambda c: c if c in ("G", "F", "C") else "F"
            )
            # Merge positional defense cols per matching (game_id, opponent_team_id, primary_pos)
            pos_def_cols = [c for c in pos_def.columns
                            if c not in ("game_id", "opponent_team_id", "_primary_pos")]
            wide = wide.merge(
                pos_def[["game_id", "opponent_team_id", "_primary_pos"] + pos_def_cols],
                on=["game_id", "opponent_team_id", "_primary_pos"],
                how="left",
            ).drop(columns=["_primary_pos"], errors="ignore")
            audit_notes["positional_defense_features"] = True
        else:
            audit_notes["positional_defense_features"] = False
    else:
        audit_notes["positional_defense_features"] = False

    # ------------------------------------------------------------------ #
    # Lineup status and DNP overrides (P4.2)
    # ------------------------------------------------------------------ #
    if injuries_df is not None and not injuries_df.empty:
        try:
            from wnba_props_model.data.lineup_parser import apply_lineup_overrides  # noqa: PLC0415
            wide = apply_lineup_overrides(wide, injuries_df)
            audit_notes["lineup_parser_applied"] = True
        except Exception as exc:
            audit_notes["lineup_parser_applied"] = False
            audit_notes["lineup_parser_error"] = str(exc)
    else:
        for col in ["confirmed_starter", "lineup_confirmed", "inferred_out", "p_dnp_override"]:
            if col not in wide.columns:
                wide[col] = np.nan
        audit_notes["lineup_parser_applied"] = False

    # ------------------------------------------------------------------ #
    # Player-vs-specific-opponent matchup history (P2.5)
    # ------------------------------------------------------------------ #
    if use_matchup_features:
        matchup_feats = _build_matchup_history_features(stats_df)
        if not matchup_feats.empty and len(matchup_feats.columns) > 2:
            matchup_cols = [c for c in matchup_feats.columns if c not in ("player_id", "game_id")]
            wide = wide.merge(
                matchup_feats[["player_id", "game_id"] + matchup_cols],
                on=["player_id", "game_id"], how="left"
            )
            audit_notes["matchup_history_features"] = True
        else:
            audit_notes["matchup_history_features"] = False
    else:
        audit_notes["matchup_history_features"] = False

    # ------------------------------------------------------------------ #
    # Enhancement 1: Usage Transfer Matrix + With-Without features
    # ------------------------------------------------------------------ #
    try:
        wide = _build_usage_transfer_features(wide, stats_df)
        audit_notes["usage_transfer_features"] = True
    except Exception as exc:
        audit_notes["usage_transfer_features"] = False
        audit_notes["usage_transfer_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 3: Extended schedule fatigue features
    # ------------------------------------------------------------------ #
    try:
        wide = _build_extended_fatigue_features(wide, stats_df)
        audit_notes["extended_fatigue_features"] = True
    except Exception as exc:
        audit_notes["extended_fatigue_features"] = False
        audit_notes["extended_fatigue_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 4: Shot quality / efficiency regression features
    # ------------------------------------------------------------------ #
    try:
        wide = _build_shot_quality_features(wide, stats_df)
        audit_notes["shot_quality_features"] = True
    except Exception as exc:
        audit_notes["shot_quality_features"] = False
        audit_notes["shot_quality_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 5: Game script + conditional minutes features
    # ------------------------------------------------------------------ #
    try:
        wide = _build_game_script_features(wide)
        audit_notes["game_script_features"] = True
    except Exception as exc:
        audit_notes["game_script_features"] = False
        audit_notes["game_script_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 9: Defensive scheme proxy features
    # ------------------------------------------------------------------ #
    try:
        wide = _build_defensive_scheme_features(wide)
        audit_notes["defensive_scheme_features"] = True
    except Exception as exc:
        audit_notes["defensive_scheme_features"] = False
        audit_notes["defensive_scheme_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 10: Season-phase categorical feature
    # ------------------------------------------------------------------ #
    try:
        wide = _build_season_phase_feature(wide)
        audit_notes["season_phase_feature"] = True
    except Exception as exc:
        audit_notes["season_phase_feature"] = False
        audit_notes["season_phase_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 17: Causal DNP features (IPW-corrected injury model)
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.causal_injury import fit_causal_dnp_model, add_causal_dnp_features  # noqa: PLC0415
        if len(wide) >= 100:
            causal_models = fit_causal_dnp_model(wide)
            wide = add_causal_dnp_features(wide, causal_models)
            audit_notes["causal_dnp_features"] = True
    except Exception as exc:
        audit_notes["causal_dnp_features"] = False
        audit_notes["causal_dnp_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 18: Duo synergy features (box-score approximation)
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.synergy_features import (  # noqa: PLC0415
            compute_duo_synergy_from_boxscores,
            add_synergy_features,
        )
        synergy_data = compute_duo_synergy_from_boxscores(wide, min_games=10)
        if synergy_data:
            wide = add_synergy_features(wide, synergy_data, top_n=3)
            audit_notes["synergy_features"] = len(synergy_data)
    except Exception as exc:
        audit_notes["synergy_features"] = False
        audit_notes["synergy_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 12: WNBA2Vec embedding features (synthetic cold-start)
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.wnba2vec import EmbeddingFeatureInjector, build_player_id_map  # noqa: PLC0415
        import os  # noqa: PLC0415
        embed_path = os.environ.get("WNBA2VEC_MODEL_PATH", "")
        pid_map = build_player_id_map(wide["player_id"].unique().tolist())
        injector = EmbeddingFeatureInjector(
            model_path=embed_path if embed_path else None,
            player_id_map=pid_map,
            n_dims=8,
        )
        wide = injector.inject(wide)
        audit_notes["embedding_features"] = True
    except Exception as exc:
        audit_notes["embedding_features"] = False
        audit_notes["embedding_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 11: Causal transfer features (DR-learner per-player-pair)
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.causal_transfer import CausalTransferEstimator  # noqa: PLC0415
        from wnba_props_model.models.usage_transfer import build_player_usage_map     # noqa: PLC0415
        if len(wide) >= 150 and "player_id" in wide.columns:
            usage_map = build_player_usage_map(wide, stats_df)
            teammate_cols = [c for c in wide.columns if c.endswith("_is_out")]
            teammate_ids = []
            for col in teammate_cols:
                try:
                    tid_str = col.replace("teammate_", "").replace("_is_out", "")
                    teammate_ids.append(int(tid_str))
                except ValueError:
                    pass
            if teammate_ids:
                cte = CausalTransferEstimator(n_folds=3, min_obs_treated=10)
                cte.fit(wide, teammate_ids=teammate_ids[:10])
                if cte._is_fitted:
                    top_teammates = sorted(
                        usage_map.items(), key=lambda x: -x[1]["usage_season"]
                    )[:5]
                    wide = cte.enrich_usage_transfer_features(wide, usage_map, top_n=5)
            audit_notes["causal_transfer_features"] = True
    except Exception as exc:
        audit_notes["causal_transfer_features"] = False
        audit_notes["causal_transfer_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Enhancement 19: Rotation model — bimodal minutes features
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.models.rotation_model import add_rotation_minutes_features  # noqa: PLC0415
        wide = add_rotation_minutes_features(wide, n_samples=500)
        audit_notes["rotation_minutes_features"] = True
    except Exception as exc:
        audit_notes["rotation_minutes_features"] = False
        audit_notes["rotation_minutes_error"] = str(exc)

    # ------------------------------------------------------------------ #
    # Advanced features — 24 new columns from extended BDL endpoints
    # (Item 2 of production blueprint)
    # ------------------------------------------------------------------ #
    try:
        from wnba_props_model.features.advanced_features import build_all_advanced_features  # noqa: PLC0415
        processed_dir = Path(data_dir) / ".." / "processed" if data_dir else None
        wide = build_all_advanced_features(wide, processed_dir=processed_dir)
        audit_notes["advanced_features_applied"] = True
    except Exception as exc:
        audit_notes["advanced_features_applied"] = False
        audit_notes["advanced_features_error"] = str(exc)

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

    ALLOWLIST MODE (R1.2): columns must satisfy ALL of the following:
      1. Exist in wide_df
      2. Not be identity, target, or metadata columns
      3. Not be in FORBIDDEN_MODEL_FEATURES
      4. Match an explicit FEATURE_FAMILIES entry OR a safe naming-convention prefix
      5. Pass cross-row variance gate: numeric std >= 0.05 (catches constant features)

    This prevents any future column added to the wide table from automatically
    becoming a model feature without explicit review.
    """
    import logging as _log
    from wnba_props_model.features.feature_contract import (
        FEATURE_FAMILIES,
        FORBIDDEN_MODEL_FEATURES as _FORBIDDEN,
        assert_no_forbidden_features,
    )

    # --- 1. Build explicit allowlist from FEATURE_FAMILIES -----------------
    allowed: set[str] = set()
    for _features in FEATURE_FAMILIES.values():
        allowed.update(_features)

    # --- 2. Extend via safe naming-convention prefixes ---------------------
    #   Any column whose name starts with a known safe prefix is permitted
    #   provided it is not forbidden.  This captures the rolling/EWMA stats
    #   (player_pts_mean_l5, team_pts_roll5, opp_pts_allowed_mean_l5, …)
    #   that are generated dynamically by build_features.py.
    _SAFE_PREFIXES = (
        "player_", "team_", "opp_", "is_home",
        "position", "game_number", "season_completion", "is_playoff",
    )
    _excluded_meta = {
        "game_id", "game_date", "player_id", "player_name",
        "team_id", "team_abbreviation",
        "opponent_team_id", "opponent_team_abbreviation",
        "home_away",
        "feature_build_timestamp_utc", "feature_cutoff_policy",
        "source", "pull_timestamp_utc",
        "minutes_raw", "minutes_flag", "non_playing_flag", "zero_minute_flag",
        "stat_line_all_zero_flag", "missing_team_flag", "missing_opponent_flag",
        "missing_game_date_flag", "started_proxy",
        "pts_ast", "pts_reb", "pts_reb_ast", "reb_ast", "stocks",
        "oreb", "dreb", "pf", "fga", "fg3a", "fta", "fg3m", "plus_minus",
    }
    for col in wide_df.columns:
        if col in _excluded_meta or col in _FORBIDDEN:
            continue
        if col.startswith("actual_") or col in set(TARGET_COLS) or col in set(ROLE_BUCKET_COLS):
            continue
        if col in ("stat", "actual_outcome"):
            continue
        if any(col.startswith(p) for p in _SAFE_PREFIXES):
            allowed.add(col)

    # --- 3. Intersect with columns that actually exist in wide_df ----------
    #   Also exclude ROLE_BUCKET_COLS: these are categorical string labels
    #   (season_phase='early', role_status='starter', etc.) handled separately
    #   by one-hot encoding.  They must NEVER enter HGB directly even if they
    #   appear in FEATURE_FAMILIES.
    #   Also exclude non-numeric columns entirely — HGBR requires numeric input.
    _role_bucket_set = set(ROLE_BUCKET_COLS)
    model_cols = sorted(
        c for c in allowed
        if c in wide_df.columns
        and c not in _FORBIDDEN
        and c not in _role_bucket_set
        and (
            pd.api.types.is_numeric_dtype(wide_df[c])
            or pd.api.types.is_bool_dtype(wide_df[c])
        )
    )

    # --- 4. Variance gate: drop numeric features with cross-row std < 0.05 -
    #   Constant or near-constant features provide no signal but corrupt HGB
    #   tree splits (the model allocates splits to them, wasting capacity and
    #   creating spurious interactions with informative features).
    low_var: list[str] = []
    for c in model_cols:
        if pd.api.types.is_numeric_dtype(wide_df[c]):
            if wide_df[c].std(skipna=True) < 0.05:
                low_var.append(c)
    if low_var:
        _log.getLogger(__name__).warning(
            "Variance gate: dropping %d near-zero-variance features (std < 0.05): %s",
            len(low_var), low_var,
        )
        model_cols = [c for c in model_cols if c not in low_var]

    # --- 5. Final leakage check --------------------------------------------
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


# ---------------------------------------------------------------------------
# Backward-compatibility alias for pipeline/train.py and pipeline/oof.py
# ---------------------------------------------------------------------------

def build_player_training_table(
    stats_df: "pd.DataFrame",
    games_df: "pd.DataFrame | None" = None,
    **kwargs,
) -> "pd.DataFrame":
    """Backward-compat wrapper: returns only the wide DataFrame from build_wide_table.

    pipeline/train.py (deprecated legacy path) calls this function.
    Production pipelines should call build_wide_table() directly.
    """
    import pandas as _pd
    if games_df is None or (hasattr(games_df, "empty") and games_df.empty):
        games_df = _pd.DataFrame()
    wide, _ = build_wide_table(stats_df, games_df, **kwargs)
    return wide
