"""Point-in-time opportunity feature builder for Opportunity V2 (Tier-0 honest subset).

Produces one row per game_id x player_id x prediction_cutoff_utc with STRICTLY LAGGED features
(shift(1) then aggregate) so no target-game information can leak. Forward-only families
(availability/lineup/roster/tracking) are attached via strict as-of joins when snapshots are
provided and otherwise emitted as explicit ``*_missing`` flags -- never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .asof import assert_feature_time_purity, strict_asof_join
from .contracts import (
    CUTOFF_SOURCE_EXACT,
    CUTOFF_SOURCE_FALLBACK,
    STARTER_LABEL_PROXY,
    forbidden_market_columns,
)


@dataclass(frozen=True)
class OpportunityFeatureConfig:
    default_lead_minutes: int = 90
    availability_max_age_hours: float = 168.0
    lineup_max_age_hours: float = 48.0
    ewma_halflife_games: float = 6.0
    recent_windows: tuple[int, ...] = (3, 5, 10, 20)
    minimum_history_games: int = 3


# Per-minute rate source columns (numerator over minutes), all present in box.
_RATE_COLS = {
    "fga": "player_fga_per_min_ewma",
    "fg3a": "player_fg3a_per_min_ewma",
    "fta": "player_fta_per_min_ewma",
    "oreb": "player_oreb_chances_per_min_ewma",
    "dreb": "player_dreb_chances_per_min_ewma",
    "pts": "player_pts_per_min_ewma",
    "reb": "player_reb_per_min_ewma",
    "ast": "player_ast_per_min_ewma",
    "fg3m": "player_fg3m_per_min_ewma",
    "turnover": "player_turnover_per_min_ewma",
    "stl": "player_stl_per_min_ewma",
    "blk": "player_blk_per_min_ewma",
}


def _shift_ewma(df: pd.DataFrame, value: pd.Series, halflife: float) -> np.ndarray:
    """shift(1) within player then EWMA -- the mandated strictly-lagged pattern."""
    tmp = pd.DataFrame({"pid": df["player_id"].to_numpy(), "v": value.to_numpy()}, index=df.index)
    prior = tmp.groupby("pid")["v"].shift(1)
    return prior.groupby(tmp["pid"]).transform(lambda s: s.ewm(halflife=halflife, min_periods=1).mean()).to_numpy()


def _dnp_streak_prior(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("player_id").groups.items():
        idx = list(idx)
        streak = 0
        for j, i in enumerate(idx):
            out[df.index.get_loc(i)] = streak  # streak BEFORE this game (lagged)
            streak = 0 if bool(df.at[i, "did_play"]) else streak + 1
    return out


def build_opportunity_feature_frame(
    player_games: pd.DataFrame,
    games: pd.DataFrame,
    roster_intervals: pd.DataFrame | None,
    availability_snapshots: pd.DataFrame | None,
    lineup_snapshots: pd.DataFrame | None,
    player_tracking: pd.DataFrame | None,
    team_tracking: pd.DataFrame | None,
    quote_cutoffs: pd.DataFrame | None,
    config: OpportunityFeatureConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    req = ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "minutes",
           "did_play", "fga", "fg3a", "fta", "fg3m", "pts", "reb", "ast", "turnover",
           "stl", "blk", "oreb", "dreb", "position", "started_proxy"]
    missing = [c for c in req if c not in player_games.columns]
    if missing:
        raise KeyError(f"build_opportunity_feature_frame: player_games missing {missing}")

    pg = player_games.copy()
    # scheduled tip from games (authoritative tz-aware UTC); fall back to box date midnight UTC.
    tip = games[["game_id", "game_date"]].rename(columns={"game_date": "scheduled_tip_utc"})
    tip["scheduled_tip_utc"] = pd.to_datetime(tip["scheduled_tip_utc"], utc=True, errors="coerce")
    pg = pg.merge(tip, on="game_id", how="left")
    pg["scheduled_tip_utc"] = pg["scheduled_tip_utc"].fillna(
        pd.to_datetime(pg["game_date"], utc=True, errors="coerce"))
    pg["game_date"] = pd.to_datetime(pg["game_date"], utc=True, errors="coerce")
    pg = pg.sort_values(["player_id", "game_date", "scheduled_tip_utc"]).reset_index(drop=True)

    hl = config.ewma_halflife_games
    mins = pd.to_numeric(pg["minutes"], errors="coerce").fillna(0.0)
    appeared = pg["did_play"].astype(bool)

    # Player minutes / activity baseline (strictly lagged).
    pg["player_minutes_ewma"] = _shift_ewma(pg, mins.where(appeared), hl)
    pg["player_minutes_std_l10"] = (
        pd.DataFrame({"pid": pg["player_id"], "v": mins.where(appeared)})
        .groupby("pid")["v"].shift(1).groupby(pg["player_id"])
        .transform(lambda s: s.rolling(10, min_periods=2).std()).to_numpy())
    pg["player_active_rate_ewma"] = _shift_ewma(pg, appeared.astype(float), hl)
    pg["player_start_rate_ewma"] = _shift_ewma(pg, pg["started_proxy"].astype(float), hl)
    pg["player_dnp_streak_prior"] = _dnp_streak_prior(pg)
    pg["player_games_played_prior"] = (
        appeared.astype(int).groupby(pg["player_id"]).transform(lambda s: s.shift(1).cumsum()).fillna(0).to_numpy())
    last_date = pg.groupby("player_id")["game_date"].shift(1)
    pg["player_days_since_last_game"] = (pg["game_date"] - last_date).dt.total_seconds() / 86400.0
    pg["last_game_date_utc"] = last_date  # prior game -> temporal-purity source column

    # Per-minute rate families (numerator/minutes on appearances, strictly lagged).
    for src, feat in _RATE_COLS.items():
        rate = (pd.to_numeric(pg[src], errors="coerce") / mins.replace(0, np.nan)).where(appeared)
        pg[feat] = _shift_ewma(pg, rate, hl)
        pg[f"{feat}_missing"] = pg[feat].isna().to_numpy()
        pg[f"{feat}_support"] = pg["player_games_played_prior"].to_numpy()

    # Conversion inputs (only 3P is fully box-observable: fg3m / fg3a).
    pg["player_3p_pct_ewma"] = _shift_ewma(
        pg, (pd.to_numeric(pg["fg3m"], errors="coerce") /
             pd.to_numeric(pg["fg3a"], errors="coerce").replace(0, np.nan)).where(appeared), hl)

    # Role bucket + position.
    pg["role_bucket"] = np.where(pg["player_start_rate_ewma"].fillna(0) >= 0.5, "starter", "bench")
    pg["position"] = pg["position"].astype("string").fillna("UNK")

    # Team environment (strictly lagged, aggregated from team box).
    pg = _attach_team_environment(pg, hl)

    # Prediction cutoff.
    if quote_cutoffs is not None and len(quote_cutoffs):
        qc = quote_cutoffs.rename(columns={"quote_timestamp": "prediction_cutoff_utc"})
        keep = [c for c in ("game_id", "player_id", "prediction_cutoff_utc") if c in qc.columns]
        qc = qc[keep].dropna().drop_duplicates(["game_id", "player_id"])
        qc["prediction_cutoff_utc"] = pd.to_datetime(qc["prediction_cutoff_utc"], utc=True, errors="coerce")
        pg = pg.merge(qc, on=["game_id", "player_id"], how="left")
        pg["cutoff_source"] = np.where(pg["prediction_cutoff_utc"].notna(),
                                       CUTOFF_SOURCE_EXACT, CUTOFF_SOURCE_FALLBACK)
        fallback = pg["scheduled_tip_utc"] - pd.Timedelta(minutes=config.default_lead_minutes)
        pg["prediction_cutoff_utc"] = pg["prediction_cutoff_utc"].fillna(fallback)
    else:
        pg["prediction_cutoff_utc"] = pg["scheduled_tip_utc"] - pd.Timedelta(minutes=config.default_lead_minutes)
        pg["cutoff_source"] = CUTOFF_SOURCE_FALLBACK
    pg["proof_eligible"] = pg["cutoff_source"] == CUTOFF_SOURCE_EXACT
    pg["starter_label_quality"] = STARTER_LABEL_PROXY  # box start is a minutes proxy, not official

    source_ts_cols = ["last_game_date_utc"]
    # Optional forward snapshot joins (availability / lineup) via strict as-of.
    pg, av_cols = _attach_availability(pg, availability_snapshots, config)
    pg, lu_cols = _attach_lineup(pg, lineup_snapshots, config)
    source_ts_cols += av_cols + lu_cols

    # Temporal purity: every source timestamp <= cutoff.
    assert_feature_time_purity(pg, cutoff_col="prediction_cutoff_utc", source_timestamp_columns=source_ts_cols)

    model_feature_cols = [c for c in pg.columns if c.endswith("_ewma") or c.endswith("_prior")
                          or c in ("player_days_since_last_game", "player_minutes_std_l10")]
    forbidden = forbidden_market_columns(model_feature_cols)
    if forbidden:
        raise ValueError(f"build_opportunity_feature_frame: forbidden market features {forbidden}")

    manifest = {
        "schema_version": "opportunity_v2_features_v1",
        "rows": int(len(pg)),
        "proof_eligible_row_count": int(pg["proof_eligible"].sum()),
        "cutoff_policy_counts": pg["cutoff_source"].value_counts().to_dict(),
        "model_feature_columns": model_feature_cols,
        "source_timestamp_columns": source_ts_cols,
        "forbidden_market_columns_found": forbidden,
        "availability_snapshots_joined": bool(av_cols),
        "lineup_snapshots_joined": bool(lu_cols),
        "roster_intervals_available": roster_intervals is not None and len(roster_intervals) > 0,
        "player_tracking_available": player_tracking is not None and len(player_tracking) > 0,
        "team_tracking_available": team_tracking is not None and len(team_tracking) > 0,
    }
    return pg, manifest


def _attach_team_environment(pg: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Team possessions/attempts from box, strictly lagged per team-game."""
    team_game = (pg.groupby(["team_id", "game_id", "game_date"], as_index=False)
                 .agg(fga=("fga", "sum"), fg3a=("fg3a", "sum"), fta=("fta", "sum"),
                      oreb=("oreb", "sum"), turnover=("turnover", "sum")))
    team_game["possessions"] = (team_game["fga"] + 0.44 * team_game["fta"]
                                - team_game["oreb"] + team_game["turnover"])
    team_game = team_game.sort_values(["team_id", "game_date"])
    for col in ("possessions", "fga", "fg3a", "fta"):
        prior = team_game.groupby("team_id")[col].shift(1)
        team_game[f"team_{col}_ewma"] = (prior.groupby(team_game["team_id"])
                                         .transform(lambda s: s.ewm(halflife=halflife, min_periods=1).mean()))
    keep = ["team_id", "game_id"] + [f"team_{c}_ewma" for c in ("possessions", "fga", "fg3a", "fta")]
    return pg.merge(team_game[keep], on=["team_id", "game_id"], how="left")


def _attach_availability(pg: pd.DataFrame, snaps: pd.DataFrame | None,
                         config: OpportunityFeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    if snaps is None or len(snaps) == 0:
        pg["availability_snapshot_missing"] = True
        return pg, []
    joined = strict_asof_join(pg, snaps, by=["player_id"], suffix="availability",
                              max_age=pd.Timedelta(hours=config.availability_max_age_hours))
    joined["availability_snapshot_missing"] = ~joined["availability_matched"].astype(bool)
    return joined, ["availability_available_at_utc"]


def _attach_lineup(pg: pd.DataFrame, snaps: pd.DataFrame | None,
                   config: OpportunityFeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    if snaps is None or len(snaps) == 0:
        pg["lineup_snapshot_missing"] = True
        return pg, []
    joined = strict_asof_join(pg, snaps, by=["player_id", "game_id"], suffix="lineup",
                              max_age=pd.Timedelta(hours=config.lineup_max_age_hours))
    joined["lineup_snapshot_missing"] = ~joined["lineup_matched"].astype(bool)
    return joined, ["lineup_available_at_utc"]
