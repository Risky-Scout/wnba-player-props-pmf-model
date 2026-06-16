"""Player minutes override and DNP redistribution pipeline.

Supports three NoVig operational requirements:
  1. Mark a player DNP — zero their minutes, redistribute to active teammates
  2. Manually set projected minutes for a player
  3. Remove a player and see projected impact on teammates

All overrides operate on the feature DataFrame BEFORE inference,
so the full PMF generation pipeline naturally propagates changes.

Usage:
    from wnba_props_model.pipeline.overrides import apply_overrides

    slate = pd.read_parquet("deliveries/next_game/slate_2026-06-16.parquet")
    overridden = apply_overrides(
        slate,
        dnp_player_ids=[341],
        minutes_overrides={419: 32.0},
    )
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns used for the minutes model — these are updated when overrides are applied
_MINUTES_PROXY_COL = "projected_minutes_proxy"
_LAST1_COL = "player_minutes_last1"
_MEAN_L3_COL = "player_minutes_mean_l3"
_MEAN_L5_COL = "player_minutes_mean_l5"
_ZERO_FLAG_COL = "zero_minute_flag"


def _team_minutes_baseline(df: pd.DataFrame, team_id: int, exclude_player_ids: list[int]) -> dict[int, float]:
    """Return baseline projected minutes per active player on team, excluding DNPs."""
    team = df[(df["team_id"] == team_id) & (~df["player_id"].isin(exclude_player_ids))].copy()
    result: dict[int, float] = {}
    for _, row in team.iterrows():
        mins = float(row.get(_MINUTES_PROXY_COL) or row.get(_MEAN_L5_COL) or 0.0)
        if mins > 0:
            result[int(row["player_id"])] = mins
    return result


def _redistribute_minutes(
    df: pd.DataFrame,
    team_id: int,
    freed_minutes: float,
    exclude_player_ids: list[int],
) -> pd.DataFrame:
    """Add freed_minutes to active teammates proportionally to their baseline share.

    When a player goes DNP, their expected minutes are distributed to their active
    teammates in proportion to each teammate's current projected_minutes_proxy.
    This models the empirical finding that missing player minutes diffuse across the roster.
    """
    out = df.copy()
    baseline = _team_minutes_baseline(out, team_id, exclude_player_ids)
    total_baseline = sum(baseline.values())
    if total_baseline <= 0 or freed_minutes <= 0:
        return out

    for pid, base_mins in baseline.items():
        share = base_mins / total_baseline
        extra = freed_minutes * share
        mask = out["player_id"] == pid
        for col in (_MINUTES_PROXY_COL, _MEAN_L5_COL, _MEAN_L3_COL):
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col].fillna(base_mins) + extra

    return out


def apply_dnp_overrides(
    df: pd.DataFrame,
    dnp_player_ids: list[int],
) -> pd.DataFrame:
    """Mark players as DNP and redistribute their minutes to teammates.

    For each DNP player:
      - Zero their projected_minutes_proxy and related features
      - Set zero_minute_flag = True
      - Add their freed minutes to active teammates (proportional to baseline)
      - Set override_applied = True, override_source = 'manual_dnp'
    """
    out = df.copy()
    if "override_applied" not in out.columns:
        out["override_applied"] = False
        out["override_source"] = None

    already_dnp: list[int] = []

    for pid in dnp_player_ids:
        mask = out["player_id"] == pid
        if not mask.any():
            logger.warning("DNP override: player_id %s not found in slate", pid)
            continue

        # Get how many minutes this player was expected to play
        freed = float(out.loc[mask, _MINUTES_PROXY_COL].fillna(
            out.loc[mask, _MEAN_L5_COL].fillna(0)
        ).iloc[0])

        # Zero out their minutes features
        for col in (_MINUTES_PROXY_COL, _LAST1_COL, _MEAN_L3_COL, _MEAN_L5_COL):
            if col in out.columns:
                out.loc[mask, col] = 0.0
        if _ZERO_FLAG_COL in out.columns:
            out.loc[mask, _ZERO_FLAG_COL] = True

        out.loc[mask, "override_applied"] = True
        out.loc[mask, "override_source"] = "manual_dnp"

        player_name = out.loc[mask, "player_name"].iloc[0] if "player_name" in out.columns else str(pid)
        logger.info("DNP: %s (player_id=%s, freed %.1f minutes)", player_name, pid, freed)

        # Redistribute freed minutes to active teammates
        team_id = int(out.loc[mask, "team_id"].iloc[0])
        already_dnp.append(pid)
        out = _redistribute_minutes(out, team_id, freed, already_dnp)

    return out


def apply_minutes_overrides(
    df: pd.DataFrame,
    minutes_overrides: Mapping[int, float],
) -> pd.DataFrame:
    """Manually set projected minutes for specific players.

    Does NOT redistribute minutes — use this when you have confirmed lineup
    information (e.g. a starter confirmed for 36 min per coach).

    minutes_overrides: {player_id: minutes} mapping
    """
    out = df.copy()
    if "override_applied" not in out.columns:
        out["override_applied"] = False
        out["override_source"] = None

    for pid, mins in minutes_overrides.items():
        mins = float(np.clip(mins, 0.0, 45.0))
        mask = out["player_id"] == pid
        if not mask.any():
            logger.warning("Minutes override: player_id %s not found in slate", pid)
            continue

        for col in (_MINUTES_PROXY_COL, _MEAN_L5_COL, _MEAN_L3_COL):
            if col in out.columns:
                out.loc[mask, col] = mins
        if _ZERO_FLAG_COL in out.columns:
            out.loc[mask, _ZERO_FLAG_COL] = mins < 1.0

        out.loc[mask, "override_applied"] = True
        out.loc[mask, "override_source"] = "manual_minutes"

        player_name = out.loc[mask, "player_name"].iloc[0] if "player_name" in out.columns else str(pid)
        logger.info("Minutes override: %s (player_id=%s) → %.1f min", player_name, pid, mins)

    return out


def apply_overrides(
    df: pd.DataFrame,
    dnp_player_ids: list[int] | None = None,
    minutes_overrides: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Apply all overrides in the correct order: DNP first, then minutes."""
    out = df.copy()
    if dnp_player_ids:
        out = apply_dnp_overrides(out, dnp_player_ids)
    if minutes_overrides:
        out = apply_minutes_overrides(out, minutes_overrides)
    return out


def override_summary(original: pd.DataFrame, overridden: pd.DataFrame) -> dict:
    """Return a summary of what changed between original and overridden slate."""
    changes = []
    for pid in overridden["player_id"].unique():
        orig = original[original["player_id"] == pid]
        new = overridden[overridden["player_id"] == pid]
        if orig.empty or new.empty:
            continue
        orig_min = float(orig[_MINUTES_PROXY_COL].iloc[0]) if _MINUTES_PROXY_COL in orig.columns else None
        new_min = float(new[_MINUTES_PROXY_COL].iloc[0]) if _MINUTES_PROXY_COL in new.columns else None
        if orig_min is not None and new_min is not None and abs(orig_min - new_min) > 0.05:
            changes.append({
                "player_id": int(pid),
                "player_name": str(new["player_name"].iloc[0]) if "player_name" in new.columns else str(pid),
                "team_abbreviation": str(new["team_abbreviation"].iloc[0]) if "team_abbreviation" in new.columns else "",
                "original_minutes": round(orig_min, 1),
                "overridden_minutes": round(new_min, 1),
                "delta_minutes": round(new_min - orig_min, 1),
                "override_source": str(new["override_source"].iloc[0]) if "override_source" in new.columns else None,
            })
    return {"n_players_changed": len(changes), "changes": changes}
