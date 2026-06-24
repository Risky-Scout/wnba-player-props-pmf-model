"""Usage Transfer Matrix (UTM) for WNBA player props.

Enhancement 1: Usage Transfer / With-Without Features
Priority: CRITICAL — #1 source of CLV edge in WNBA

When a high-usage teammate is out, production redistributes to remaining
players in a position-weighted manner.  This module computes:
  - player_usage_rate_l5 / season
  - usage_shift = l5 - season  (detects permanent role changes)
  - teammate_{pid}_is_out / _usage_rate  (teammate-specific flags)
  - without_{tid}_{stat}_delta  (with-without splits from historical lineup data)
  - projected_usage_given_absences  (UTM-projected usage after absences)
  - usage_transfer_delta  (delta from baseline)

All features are shift-1 safe — they use only information known BEFORE the
target game.

Reference:
  Deshpande & Jensen (2016). Estimating an NBA player's impact on his
  team's chances of winning.  Journal of Quantitative Analysis in Sports.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Position-aware transfer weights (row = absent player pos, col = beneficiary pos)
# guard/wing/big taxonomy — values are fraction of absent usage that transfers.
POS_TRANSFER_WEIGHTS: dict[tuple[str, str], float] = {
    ("guard", "guard"): 0.45, ("guard", "wing"):  0.30, ("guard", "big"):  0.10,
    ("wing",  "guard"): 0.25, ("wing",  "wing"):  0.40, ("wing",  "big"):  0.20,
    ("big",   "guard"): 0.10, ("big",   "wing"):  0.25, ("big",   "big"):  0.45,
}

# Stats whose volumes are most directly linked to usage
USAGE_SENSITIVE_STATS = ["pts", "fgm", "ftm", "turnover", "ast"]

# Top-N teammates to track individually (balance signal vs column explosion)
TOP_TEAMMATES_N = 5
TOP_WOWY_N = 3  # with-without split teammates


def _normalize_position(pos: str | None) -> str:
    """Coerce raw position string to guard / wing / big."""
    if not isinstance(pos, str) or not pos:
        return "wing"
    p = pos.strip().upper()[:2]
    if p in ("G", "PG", "SG"):
        return "guard"
    if p in ("C",):
        return "big"
    return "wing"   # SF, PF, F all treated as wing


def build_player_usage_map(
    stats_df: pd.DataFrame,
    cutoff_date: Any = None,
) -> dict[int, dict[str, Any]]:
    """Compute per-player usage rates from historical stats.

    Usage proxy = (fga + 0.44*fta + turnover) per minute played.

    Parameters
    ----------
    stats_df : one-row-per-player-game DataFrame with fga, fta, turnover, minutes
    cutoff_date : date before which rows are included (for leakage prevention)

    Returns
    -------
    {player_id: {"usage_l5": float, "usage_season": float,
                 "position_group": str, "team_id": int}}
    """
    df = stats_df.copy()
    if cutoff_date is not None:
        df = df[df["game_date"] < cutoff_date]

    required = {"player_id", "game_date", "fga", "fta", "turnover", "minutes"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("build_player_usage_map: missing columns %s — returning empty", missing)
        return {}

    df = df.sort_values(["player_id", "game_date"])
    df["_usage_raw"] = df["fga"] + 0.44 * df["fta"] + df["turnover"]
    df["_min"] = df["minutes"].clip(lower=0.5)

    result: dict[int, dict[str, Any]] = {}
    for pid, grp in df.groupby("player_id", sort=False):
        grp = grp.tail(40)  # cap to season
        if grp["_min"].sum() < 1:
            continue
        usage_per_min = grp["_usage_raw"] / grp["_min"]
        l5 = usage_per_min.tail(5).mean()
        season_mean = usage_per_min.mean()

        pos = "wing"
        if "position" in grp.columns:
            pos = _normalize_position(grp["position"].dropna().iloc[-1] if not grp["position"].dropna().empty else None)

        team_id = int(grp["team_id"].iloc[-1]) if "team_id" in grp.columns else 0

        result[int(pid)] = {
            "usage_l5":    float(l5) if np.isfinite(l5) else 0.20,
            "usage_season": float(season_mean) if np.isfinite(season_mean) else 0.20,
            "position_group": pos,
            "team_id": team_id,
        }
    return result


def build_wowy_splits(
    stats_df: pd.DataFrame,
    player_usage_map: dict[int, dict[str, Any]],
    top_n: int = TOP_WOWY_N,
) -> dict[int, dict[int, dict[str, dict[str, float]]]]:
    """Approximate with-or-without-you (WOWY) splits from box scores.

    Without true PBP lineup data we approximate WOWY via:
      "without" game = a game where the teammate had 0 minutes / DNP.
      "with" game    = a game where both players played.

    Returns
    -------
    {player_id: {teammate_id: {"with": {stat: rate}, "without": {stat: rate}}}}
    """
    if "did_play" not in stats_df.columns and "minutes" not in stats_df.columns:
        return {}

    # Build top-N teammates per team (by usage)
    team_top_map: dict[int, list[int]] = {}
    sorted_by_usage = sorted(player_usage_map.items(),
                             key=lambda x: -x[1]["usage_season"])
    for pid, info in sorted_by_usage:
        tid = info["team_id"]
        if tid not in team_top_map:
            team_top_map[tid] = []
        if len(team_top_map[tid]) < top_n:
            team_top_map[tid].append(pid)

    df = stats_df.copy()
    if "did_play" not in df.columns:
        df["did_play"] = (df["minutes"].fillna(0) > 0).astype(int)

    result: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
    wowy_stats = ["pts", "reb", "ast", "fg3m", "turnover"]

    for pid, info in player_usage_map.items():
        tid = info["team_id"]
        top_teammates = [t for t in team_top_map.get(tid, []) if t != pid]
        if not top_teammates:
            continue

        player_rows = df[df["player_id"] == pid].copy()
        if player_rows.empty:
            continue

        result[pid] = {}
        for tid2 in top_teammates[:top_n]:
            tm_rows = df[df["player_id"] == tid2][["game_id", "did_play", "minutes"]].rename(
                columns={"did_play": "_tm_played", "minutes": "_tm_min"}
            )
            merged = player_rows.merge(tm_rows, on="game_id", how="inner")
            if merged.empty:
                continue

            tm_played = merged["_tm_played"].fillna(0).astype(int)
            with_mask    = tm_played == 1
            without_mask = tm_played == 0

            splits: dict[str, dict[str, float]] = {"with": {}, "without": {}}
            for stat in wowy_stats:
                if stat not in merged.columns:
                    continue
                min_col = "minutes" if "minutes" in merged.columns else "actual_minutes"
                if min_col not in merged.columns:
                    continue
                mins = merged[min_col].fillna(0).clip(lower=0.5)

                if with_mask.sum() >= 3:
                    rate_with = (merged.loc[with_mask, stat].fillna(0) /
                                 mins[with_mask]).mean()
                    splits["with"][stat] = float(rate_with) if np.isfinite(rate_with) else 0.0
                if without_mask.sum() >= 2:
                    rate_wo = (merged.loc[without_mask, stat].fillna(0) /
                               mins[without_mask]).mean()
                    splits["without"][stat] = float(rate_wo) if np.isfinite(rate_wo) else 0.0

            result[pid][tid2] = splits

    return result


def add_usage_transfer_features(
    df: pd.DataFrame,
    player_usage_map: dict[int, dict[str, Any]],
    wowy_splits: dict[int, dict[int, dict[str, dict[str, float]]]] | None = None,
    top_n: int = TOP_TEAMMATES_N,
) -> pd.DataFrame:
    """Add Usage Transfer Matrix features to the wide feature DataFrame.

    Parameters
    ----------
    df : wide feature DataFrame (one row per player-game)
    player_usage_map : {player_id: {"usage_l5", "usage_season", "position_group", "team_id"}}
    wowy_splits : {player_id: {teammate_id: {"with": {stat: rate}, "without": {stat: rate}}}}
    top_n : number of teammates to generate individual flags for

    Returns
    -------
    df with new UTM feature columns added in-place (returns copy).
    """
    df = df.copy()
    if not player_usage_map:
        logger.warning("add_usage_transfer_features: empty player_usage_map — skipping")
        return df

    if "player_id" not in df.columns:
        return df

    # ── 1. Usage rate features ──────────────────────────────────────────────
    pid_series = df["player_id"].astype(int)
    df["player_usage_rate_l5"] = pid_series.map(
        {pid: v["usage_l5"] for pid, v in player_usage_map.items()}
    ).fillna(0.20)
    df["player_usage_rate_season"] = pid_series.map(
        {pid: v["usage_season"] for pid, v in player_usage_map.items()}
    ).fillna(0.20)
    df["usage_shift"] = df["player_usage_rate_l5"] - df["player_usage_rate_season"]
    df["usage_shift_abs"] = df["usage_shift"].abs()

    # ── 2. Teammate-specific absence features ───────────────────────────────
    # Sort teammates by usage (desc) and keep top_n
    sorted_teammates = sorted(
        player_usage_map.items(), key=lambda x: -x[1]["usage_season"]
    )[:top_n]

    # We need "teammate_injury_ids" which lists out teammate IDs for each row.
    # Fall back to building it from injuries_df columns that may already be merged.
    # If not present, we create sparse flags as zeros (no confirmed injury data).
    has_injury_ids = "teammate_injury_ids" in df.columns

    utm_cols: dict[str, pd.Series] = {}
    for tid_val, t_info in sorted_teammates:
        flag_col  = f"teammate_{tid_val}_is_out"
        usage_col = f"teammate_{tid_val}_usage_rate"
        if has_injury_ids:
            utm_cols[flag_col] = (
                df["teammate_injury_ids"]
                .fillna("")
                .apply(lambda x, _t=str(tid_val): 1 if _t in str(x) else 0)
                .astype(int)
            )
        else:
            utm_cols[flag_col] = pd.Series(0, index=df.index)
        utm_cols[usage_col] = utm_cols[flag_col] * t_info["usage_season"]

    if utm_cols:
        df = pd.concat([df, pd.DataFrame(utm_cols, index=df.index)], axis=1)

    # ── 3. With-or-Without splits ───────────────────────────────────────────
    if wowy_splits:
        wowy_cols: dict[str, list[float]] = {}
        wowy_stats = ["pts", "reb", "ast", "fg3m", "turnover"]
        top3_tms = [t for t, _ in sorted_teammates[:TOP_WOWY_N]]

        for stat in wowy_stats:
            for tid_val in top3_tms:
                col = f"without_{tid_val}_{stat}_delta"
                vals = []
                for pid_val in df["player_id"].astype(int):
                    splits = wowy_splits.get(pid_val, {}).get(tid_val, {})
                    wo  = splits.get("without", {}).get(stat, None)
                    wi  = splits.get("with",    {}).get(stat, None)
                    if wo is not None and wi is not None:
                        vals.append(wo - wi)
                    else:
                        vals.append(0.0)
                wowy_cols[col] = vals

        if wowy_cols:
            df = pd.concat([df, pd.DataFrame(wowy_cols, index=df.index)], axis=1)

    # ── 4. Projected usage given absences (full UTM) ─────────────────────
    pos_map = {pid: v["position_group"] for pid, v in player_usage_map.items()}
    base_usage = df["player_usage_rate_season"].copy()
    projected_usage = base_usage.copy()

    for idx in df.index:
        pid_val = int(df.at[idx, "player_id"])
        player_pos = pos_map.get(pid_val, "wing")
        for tid_val, t_info in sorted_teammates:
            flag_col = f"teammate_{tid_val}_is_out"
            if flag_col not in df.columns:
                continue
            is_out = df.at[idx, flag_col]
            if not is_out:
                continue
            absent_pos = t_info.get("position_group", "wing")
            tk = (player_pos, absent_pos)
            w = POS_TRANSFER_WEIGHTS.get(tk, 0.15)
            transfer = t_info["usage_season"] * w
            projected_usage.at[idx] = projected_usage.at[idx] + transfer

    df["projected_usage_given_absences"] = projected_usage
    df["usage_transfer_delta"] = projected_usage - base_usage

    logger.info(
        "add_usage_transfer_features: added UTM cols; "
        "usage_transfer_delta mean=%.4f",
        df["usage_transfer_delta"].mean(),
    )
    return df


def compute_team_usage_maps(
    stats_df: pd.DataFrame,
    cutoff_date: Any = None,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Compute per-team usage maps (team_id → player_id → usage info).

    Convenience wrapper around build_player_usage_map that organises by team.
    """
    usage_map = build_player_usage_map(stats_df, cutoff_date)
    team_maps: dict[int, dict[int, dict[str, Any]]] = {}
    for pid, info in usage_map.items():
        tid = info["team_id"]
        if tid not in team_maps:
            team_maps[tid] = {}
        team_maps[tid][pid] = info
    return team_maps


# ===========================================================================
# Item 9: Automated Usage Transfer Matrix (blueprint specification)
# Replaces manual overrides in apply_injury_news.py with data-driven
# USG%-based redistribution from BDL player_season_advanced_stats.
# ===========================================================================

class UsageTransferMatrix:
    """Automatic usage redistribution using BDL USG% data.

    When a player is ruled out (or leaves mid-game), their USG% and
    minutes must be redistributed to remaining teammates proportionally
    to each available player's baseline USG%.

    Redistribution rule:
      1. Sum the out player's USG% (from BDL season advanced stats)
      2. Redistribute proportionally to remaining players' USG%
      3. Scale minutes: out player's minutes distributed by each
         teammate's share of remaining total USG%

    Example:
      Player A (USG% = 25%) is OUT
      Remaining: B(20%), C(18%), D(12%), E(8%), F(5%) → total = 63%
      B gets: 25% * (20/63) = 7.94% additional USG%
      B's minutes boost = A_minutes * (20/63) = ~9.5 extra min
    """

    def __init__(self, season_usage_df: pd.DataFrame) -> None:
        """
        Args:
            season_usage_df: DataFrame from player_season_adv_usage.parquet
                Required columns: player_id, usg_pct_season (or usage_USG_PCT)
        """
        usg_col = next(
            (c for c in ["usg_pct_season", "usage_USG_PCT", "usage_pct", "usage_percentage"]
             if c in season_usage_df.columns),
            None,
        )
        if usg_col:
            self.usage_lookup: dict[int, float] = dict(
                zip(season_usage_df["player_id"],
                    season_usage_df[usg_col].fillna(0.20))
            )
        else:
            logger.warning("UsageTransferMatrix: no USG% column found — using uniform 0.20")
            self.usage_lookup = {}

    def get_usage(self, player_id: int) -> float:
        return float(self.usage_lookup.get(player_id, 0.20))

    def redistribute(
        self,
        roster: list[dict],
        out_player_ids: list[int],
        out_minutes_dict: dict[int, float] | None = None,
    ) -> tuple[list[dict], dict]:
        """Redistribute usage and minutes for out players.

        Args:
            roster: list of dicts with keys: player_id, projected_minutes
            out_player_ids: list of player_ids who are OUT
            out_minutes_dict: {player_id: projected_minutes} for out players
                (if None, uses roster's projected_minutes for out players)

        Returns:
            (updated_roster, transfer_report)
        """
        out_ids_set = set(out_player_ids)
        available = [p for p in roster if p["player_id"] not in out_ids_set]
        out_players = [p for p in roster if p["player_id"] in out_ids_set]

        if not available or not out_players:
            return roster, {"transferred": False, "reason": "no_available_or_no_out"}

        # Sum USG% of available players
        available_usg = {p["player_id"]: self.get_usage(p["player_id"]) for p in available}
        total_available_usg = max(sum(available_usg.values()), 1e-6)

        # Sum USG% and minutes of out players
        total_out_usg = sum(self.get_usage(p["player_id"]) for p in out_players)
        total_out_minutes = sum(
            (out_minutes_dict or {}).get(p["player_id"], p.get("projected_minutes", 0.0))
            for p in out_players
        )

        transfers = []
        for p in roster:
            if p["player_id"] in out_ids_set:
                p["projected_minutes"] = 0.0
                p["status"] = "out"
                continue
            share = available_usg.get(p["player_id"], 0.10) / total_available_usg
            extra_min = total_out_minutes * share
            extra_usg = total_out_usg * share
            p["projected_minutes_original"] = p.get("projected_minutes", 0.0)
            p["projected_minutes"] = p.get("projected_minutes", 0.0) + extra_min
            p["usage_pct_adjusted"] = available_usg.get(p["player_id"], 0.10) + extra_usg
            transfers.append({
                "player_id": p["player_id"],
                "extra_minutes": round(extra_min, 1),
                "extra_usage_pct": round(extra_usg * 100, 2),
            })

        return roster, {
            "transferred": True,
            "n_out": len(out_players),
            "n_available": len(available),
            "total_out_usage_pct": round(total_out_usg * 100, 2),
            "total_out_minutes": round(total_out_minutes, 1),
            "transfers": transfers,
        }
