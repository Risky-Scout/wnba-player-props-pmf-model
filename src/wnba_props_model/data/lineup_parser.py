"""Automated injury/lineup status parser for WNBA player props model (P4.2).

Parses BDL injury reports and infers pre-game lineup status for each player.
Critically, all inference is based only on PRE-GAME information — box scores
are never used to infer lineup status (enforced by leakage contract below).

LEAKAGE GUARD: ``confirmed_starter`` and ``lineup_confirmed`` must only derive
from the BDL player_injuries endpoint status field (which is a pre-game report),
never from the player_stats box score (``did_play``, ``minutes``, or ``started_proxy``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.constants import INJURY_STATUS_MAP


class LineupStatusParser:
    """Parse BDL injury data to produce per-player, per-game lineup status flags.

    Attributes produced (all pre-game safe):
    - confirmed_starter  : int — 1 probable starter, 0 uncertain, -1 confirmed out
    - lineup_confirmed   : int — 1 if status is definitive (out/inactive), 0 if uncertain
    - inferred_out       : int — 1 if inferred out from injury report

    Usage:
        parser = LineupStatusParser()
        status_df = parser.parse_injury_statuses(injuries_df)
        # status_df: player_id, game_id, confirmed_starter, lineup_confirmed, inferred_out
    """

    # Statuses that mean a player is definitively out
    _OUT_STATUSES = frozenset({"out", "inactive"})
    # Statuses where player is available/likely to play
    _AVAILABLE_STATUSES = frozenset({"available", "probable"})
    # Statuses with uncertainty
    _UNCERTAIN_STATUSES = frozenset({"questionable", "doubtful", "unknown"})

    def parse_injury_statuses(
        self,
        injuries_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Parse BDL injury report into lineup status flags.

        Parameters
        ----------
        injuries_df : DataFrame with at minimum columns:
            player_id, game_id, status

        Returns
        -------
        DataFrame with columns:
            player_id, game_id, confirmed_starter, lineup_confirmed, inferred_out
        """
        if injuries_df is None or injuries_df.empty:
            return pd.DataFrame(
                columns=["player_id", "game_id", "confirmed_starter",
                         "lineup_confirmed", "inferred_out"]
            )

        df = injuries_df.copy()
        required = {"player_id", "game_id", "status"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"injuries_df missing columns: {missing}")

        # Normalize status via the canonical injury status map
        df["_norm_status"] = (
            df["status"].fillna("unknown")
            .str.lower().str.strip()
            .map(lambda s: INJURY_STATUS_MAP.get(s, "unknown"))
        )

        # Assign flags
        def _flags(norm_status: str) -> dict[str, int]:
            if norm_status in self._OUT_STATUSES:
                return {"confirmed_starter": -1, "lineup_confirmed": 1, "inferred_out": 1}
            elif norm_status in self._AVAILABLE_STATUSES:
                return {"confirmed_starter": 0, "lineup_confirmed": 0, "inferred_out": 0}
            else:  # questionable, doubtful, unknown
                return {"confirmed_starter": 0, "lineup_confirmed": 0, "inferred_out": 0}

        flags_df = df["_norm_status"].apply(_flags).apply(pd.Series)
        result = pd.concat([
            df[["player_id", "game_id"]].reset_index(drop=True),
            flags_df.reset_index(drop=True),
        ], axis=1)

        # Deduplicate: for a given (player_id, game_id), take most severe status
        # (i.e. if any report says "out", use that)
        result = (
            result.sort_values("inferred_out", ascending=False)
            .drop_duplicates(subset=["player_id", "game_id"], keep="first")
            .reset_index(drop=True)
        )
        return result[["player_id", "game_id", "confirmed_starter",
                        "lineup_confirmed", "inferred_out"]]

    def infer_from_recent_starts(
        self,
        stats_df: pd.DataFrame,
        window: int = 5,
    ) -> pd.DataFrame:
        """Estimate starter probability from recent start rate.

        LEAKAGE GUARD: uses only the ``started_proxy`` column from prior games
        (shift-1 safe), never from the current game's box score.

        Parameters
        ----------
        stats_df : canonical player game stats DataFrame
        window   : rolling window for start rate

        Returns
        -------
        DataFrame: player_id, game_id, starter_rate_l{window}
        """
        if stats_df is None or stats_df.empty:
            return pd.DataFrame(columns=["player_id", "game_id", f"starter_rate_l{window}"])

        df = stats_df.sort_values(
            ["player_id", "game_date", "game_id"]
        ).reset_index(drop=True)

        if "started_proxy" not in df.columns:
            df["started_proxy"] = 0

        grp = df.groupby("player_id", sort=False)["started_proxy"]
        col_name = f"starter_rate_l{window}"
        df[col_name] = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        return df[["player_id", "game_id", col_name]].copy()


def apply_lineup_overrides(
    wide_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge lineup status into wide feature table and record p_dnp overrides.

    When ``lineup_confirmed == 1`` and ``confirmed_starter == -1`` (player is
    confirmed out), sets ``p_dnp_override = 1.0`` in the wide table.  The PMF
    engine reads this column and bypasses the model's DNP probability estimate.

    LEAKAGE CONTRACT: This function must only be called with pre-game injury data.
    It must never be called with box-score-derived status.

    Parameters
    ----------
    wide_df     : wide feature table
    injuries_df : BDL player_injuries table (pre-game reports only)

    Returns
    -------
    wide_df with added columns: confirmed_starter, lineup_confirmed, inferred_out,
    p_dnp_override (1.0 when confirmed out, else NaN)
    """
    parser = LineupStatusParser()
    status_df = parser.parse_injury_statuses(injuries_df)

    if status_df.empty:
        for col in ["confirmed_starter", "lineup_confirmed", "inferred_out", "p_dnp_override"]:
            if col not in wide_df.columns:
                wide_df[col] = np.nan
        return wide_df

    wide_df = wide_df.merge(
        status_df, on=["player_id", "game_id"], how="left"
    )
    # p_dnp_override: 1.0 when confirmed out, NaN otherwise (PMF engine treats NaN as no override)
    wide_df["p_dnp_override"] = np.where(
        (wide_df["lineup_confirmed"] == 1) & (wide_df["confirmed_starter"] == -1),
        1.0, np.nan,
    )
    return wide_df
