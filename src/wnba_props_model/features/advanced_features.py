"""Advanced feature engineering from BDL extended endpoints.

Produces 24 new feature columns for the player-game feature table:
  - Usage features (from player_season_advanced_stats, measure_type=usage)
  - Shot quality features (from player_shot_locations)
  - Injury context features (from player_injuries + usage data)
  - Opponent context features (from team_game_advanced_stats)
  - Standings / motivation features (from standings)
  - Four-factors features (from player_season_advanced_stats, measure_type=four_factors)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROLL_WINDOW = 10


# ---------------------------------------------------------------------------
# 2A. Usage features
# ---------------------------------------------------------------------------

def build_usage_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build USG% features.

    Reads wnba_player_season_advanced.parquet when processed_dir is given,
    otherwise fills with zeros (graceful degradation).
    """
    result = player_game_stats.copy()

    # Load season advanced stats
    season_adv: Optional[pd.DataFrame] = None
    if processed_dir is not None:
        p = Path(processed_dir) / "wnba_player_season_advanced.parquet"
        if p.exists():
            try:
                season_adv = pd.read_parquet(p)
            except Exception as exc:
                log.warning("Could not read %s: %s", p, exc)

    usg_col = None
    for candidate in ("usage_USG_PCT", "advanced_USG_PCT", "usage_pct", "usage_percentage"):
        if season_adv is not None and candidate in season_adv.columns:
            usg_col = candidate
            break

    if season_adv is not None and usg_col is not None:
        usage_df = season_adv[["player_id", "season", usg_col]].rename(
            columns={usg_col: "usg_pct_season"}
        ).drop_duplicates(["player_id", "season"])
        result = result.merge(usage_df, on=["player_id", "season"], how="left")
        result["usg_pct_season"] = result["usg_pct_season"].fillna(0.20)
    else:
        result["usg_pct_season"] = 0.20

    # Rename for downstream feature contract
    result["player_usage_pct"] = result["usg_pct_season"]

    # Rolling EWMA from game-level USG (if available)
    game_usg_col = next(
        (c for c in ["usg_pct_game", "usage_percentage", "advanced_USG_PCT"]
         if c in result.columns),
        None,
    )
    if game_usg_col:
        result["player_usage_pct_ewma10"] = (
            result.sort_values("game_date")
            .groupby("player_id")[game_usg_col]
            .transform(lambda x: x.ewm(span=_ROLL_WINDOW, min_periods=1).mean())
        )
    else:
        result["player_usage_pct_ewma10"] = result["player_usage_pct"]

    result["player_usage_pct_vs_avg"] = (
        result["player_usage_pct_ewma10"] - result["player_usage_pct"]
    )

    return result


# ---------------------------------------------------------------------------
# 2B. Four-factors features
# ---------------------------------------------------------------------------

def build_four_factors_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    result = player_game_stats.copy()

    season_adv: Optional[pd.DataFrame] = None
    if processed_dir is not None:
        p = Path(processed_dir) / "wnba_player_season_advanced.parquet"
        if p.exists():
            try:
                season_adv = pd.read_parquet(p)
            except Exception as exc:
                log.warning("Could not read %s: %s", p, exc)

    col_map = {
        "player_efg_pct": ["four_factors_EFG_PCT", "efg_pct"],
        "player_ft_rate": ["four_factors_FT_RATE", "ft_rate"],
        "player_tov_pct": ["four_factors_TOV_PCT", "tov_pct"],
    }

    if season_adv is not None:
        for feat_col, candidates in col_map.items():
            src = next((c for c in candidates if c in season_adv.columns), None)
            if src:
                ff_df = season_adv[["player_id", "season", src]].rename(
                    columns={src: feat_col}
                ).drop_duplicates(["player_id", "season"])
                result = result.merge(ff_df, on=["player_id", "season"], how="left")

    for feat_col in col_map:
        if feat_col not in result.columns:
            result[feat_col] = np.nan

    return result


# ---------------------------------------------------------------------------
# 2C. Shot quality features
# ---------------------------------------------------------------------------

def build_shot_quality_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    result = player_game_stats.copy()

    shot_feats: Optional[pd.DataFrame] = None
    if processed_dir is not None:
        p = Path(processed_dir) / "wnba_player_shot_locations.parquet"
        if p.exists():
            try:
                shot_df = pd.read_parquet(p)
                shot_feats = _compute_shot_quality_from_zones(shot_df)
            except Exception as exc:
                log.warning("Could not read/compute shot quality: %s", exc)

    shot_cols = [
        "player_pct_fg_restricted",
        "player_pct_fg_corner3",
        "player_pct_fg_midrange",
        "player_fg_pct_restricted",
        "shot_quality_score",
    ]

    if shot_feats is not None and "player_id" in shot_feats.columns:
        merge_on = ["player_id"]
        if "season" in shot_feats.columns and "season" in result.columns:
            merge_on.append("season")
        result = result.merge(shot_feats[merge_on + shot_cols], on=merge_on, how="left")
    else:
        for c in shot_cols:
            result[c] = np.nan

    return result


def _compute_shot_quality_from_zones(shot_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shot-location parquet into per-player shot quality metrics."""
    # The BDL shot_locations endpoint returns one row per player per zone
    # or embedded as stats dict. Handle both flat and nested formats.
    if "shot_zone" in shot_df.columns:
        # Flat format: one row per shot
        grp = shot_df.groupby(["player_id"]).apply(_zone_stats_from_flat).reset_index()
        return grp
    # Fallback: return empty
    return pd.DataFrame()


def _zone_stats_from_flat(g: pd.DataFrame) -> pd.Series:
    zone_col = "shot_zone"
    made_col = "shot_made" if "shot_made" in g.columns else None
    total_fga = len(g)
    if total_fga == 0:
        return pd.Series({c: np.nan for c in [
            "player_pct_fg_restricted", "player_pct_fg_corner3",
            "player_pct_fg_midrange", "player_fg_pct_restricted", "shot_quality_score",
        ]})
    restricted = g[g[zone_col].str.contains("Restricted|RA", case=False, na=False)]
    corner3 = g[g[zone_col].str.contains("Corner", case=False, na=False)]
    midrange = g[g[zone_col].str.contains("Mid", case=False, na=False)]
    pct_res = len(restricted) / total_fga
    pct_c3 = len(corner3) / total_fga
    pct_mid = len(midrange) / total_fga
    fg_pct_res = (
        restricted[made_col].fillna(False).astype(bool).mean()
        if made_col and len(restricted) > 0 else np.nan
    )
    shot_quality = pct_res * 1.2 + pct_c3 * 1.1 + (1 - pct_res - pct_c3 - pct_mid) * 1.0 + pct_mid * 0.5
    return pd.Series({
        "player_pct_fg_restricted": pct_res,
        "player_pct_fg_corner3": pct_c3,
        "player_pct_fg_midrange": pct_mid,
        "player_fg_pct_restricted": fg_pct_res,
        "shot_quality_score": shot_quality,
    })


# ---------------------------------------------------------------------------
# 2D. Injury context features
# ---------------------------------------------------------------------------

def build_injury_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    result = player_game_stats.copy()
    inj_cols = [
        "teammate_out_count",
        "teammate_questionable_count",
        "team_total_usage_of_out_players",
    ]

    injuries_df: Optional[pd.DataFrame] = None
    usage_lookup: dict = {}

    if processed_dir is not None:
        inj_path = Path(processed_dir) / "wnba_injuries.parquet"
        if inj_path.exists():
            try:
                injuries_df = pd.read_parquet(inj_path)
            except Exception as exc:
                log.warning("Could not read injuries: %s", exc)
        adv_path = Path(processed_dir) / "wnba_player_season_advanced.parquet"
        if adv_path.exists():
            try:
                adv = pd.read_parquet(adv_path)
                usg_col = next(
                    (c for c in ["usage_USG_PCT", "advanced_USG_PCT", "usage_pct"] if c in adv.columns),
                    None,
                )
                if usg_col:
                    usage_lookup = dict(zip(adv["player_id"], adv[usg_col].fillna(0.20)))
            except Exception as exc:
                log.warning("Could not read usage for injury features: %s", exc)

    if injuries_df is not None and "status" in injuries_df.columns:
        out_mask = injuries_df["status"].str.lower().isin(["out", "inactive"])
        q_mask = injuries_df["status"].str.lower().isin(["questionable", "doubtful"])
        out_df = injuries_df[out_mask]
        q_df = injuries_df[q_mask]

        def _count_out_teammates(row: pd.Series) -> pd.Series:
            if pd.isna(row.get("team_id")):
                return pd.Series([0, 0, 0.0])
            tid = row["team_id"]
            pid = row["player_id"]
            team_out = out_df[(out_df["team_id"] == tid) & (out_df["player_id"] != pid)]
            team_q = q_df[(q_df["team_id"] == tid) & (q_df["player_id"] != pid)]
            n_out = len(team_out)
            n_q = len(team_q)
            usg_sum = sum(usage_lookup.get(p, 0.20) for p in team_out["player_id"])
            return pd.Series([n_out, n_q, usg_sum])

        expanded = result.apply(_count_out_teammates, axis=1, result_type="expand")
        if expanded.shape[1] == 3:
            result["teammate_out_count"] = expanded[0]
            result["teammate_questionable_count"] = expanded[1]
            result["team_total_usage_of_out_players"] = expanded[2]
        else:
            for c in inj_cols:
                result[c] = 0.0
    else:
        for c in inj_cols:
            result[c] = 0.0

    return result


# ---------------------------------------------------------------------------
# 2E. Opponent context from team game advanced stats
# ---------------------------------------------------------------------------

def build_opponent_context_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    result = player_game_stats.copy()
    opp_cols = ["opp_def_rating_ewma10", "opp_pace_ewma10", "game_pace_predicted"]

    team_adv: Optional[pd.DataFrame] = None
    if processed_dir is not None:
        p = Path(processed_dir) / "wnba_team_game_advanced.parquet"
        if p.exists():
            try:
                team_adv = pd.read_parquet(p)
            except Exception as exc:
                log.warning("Could not read team_adv: %s", exc)

    if team_adv is not None and "team_id" in team_adv.columns:
        # Compute rolling EWMA per team
        team_adv = team_adv.sort_values(["team_id", "game_date"] if "game_date" in team_adv.columns else ["team_id"])
        if "defensive_rating" in team_adv.columns:
            team_adv["def_rtg_ewma10"] = (
                team_adv.groupby("team_id")["defensive_rating"]
                .transform(lambda x: x.ewm(span=_ROLL_WINDOW, min_periods=1).mean())
            )
        if "pace" in team_adv.columns:
            team_adv["pace_ewma10"] = (
                team_adv.groupby("team_id")["pace"]
                .transform(lambda x: x.ewm(span=_ROLL_WINDOW, min_periods=1).mean())
            )
        team_summary = (
            team_adv.sort_values("game_date" if "game_date" in team_adv.columns else "team_id")
            .groupby("team_id")
            .last()
            .reset_index()
        )
        # Merge opponent's stats (using opponent_team_id in player stats)
        opp_col = next((c for c in ["opponent_team_id", "opp_team_id"] if c in result.columns), None)
        if opp_col:
            opp_summary = team_summary[["team_id"] + [c for c in ["def_rtg_ewma10", "pace_ewma10"] if c in team_summary.columns]].rename(
                columns={"team_id": opp_col, "def_rtg_ewma10": "opp_def_rating_ewma10", "pace_ewma10": "opp_pace_ewma10"}
            )
            result = result.merge(opp_summary, on=opp_col, how="left")
            home_pace = team_summary.rename(columns={"team_id": "team_id_x", "pace_ewma10": "home_pace"})
            if "team_id" in result.columns and "pace_ewma10" in team_summary.columns:
                pace_map = team_summary.set_index("team_id")["pace_ewma10"].to_dict()
                result["own_pace_ewma10"] = result["team_id"].map(pace_map)
                opp_pace_map = {k: v for k, v in pace_map.items()}
                if "opp_pace_ewma10" in result.columns:
                    result["game_pace_predicted"] = (
                        result["own_pace_ewma10"].fillna(92.0) + result["opp_pace_ewma10"].fillna(92.0)
                    ) / 2
                else:
                    result["game_pace_predicted"] = 92.0
        else:
            for c in opp_cols:
                result[c] = np.nan
    else:
        for c in opp_cols:
            result[c] = np.nan

    for c in opp_cols:
        if c not in result.columns:
            result[c] = np.nan

    return result


# ---------------------------------------------------------------------------
# 2F. Standings / motivation features
# ---------------------------------------------------------------------------

def build_standings_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    result = player_game_stats.copy()
    stand_cols = ["team_playoff_seed", "team_games_behind", "season_phase"]

    standings: Optional[pd.DataFrame] = None
    if processed_dir is not None:
        p = Path(processed_dir) / "wnba_standings.parquet"
        if p.exists():
            try:
                standings = pd.read_parquet(p)
            except Exception as exc:
                log.warning("Could not read standings: %s", exc)

    if standings is not None and "team_id" in standings.columns:
        seed_col = next((c for c in ["conference_rank", "playoff_seed"] if c in standings.columns), None)
        gb_col = next((c for c in ["games_behind"] if c in standings.columns), None)
        merge_cols = ["team_id"]
        rename_map = {}
        if "season" in standings.columns:
            merge_cols.append("season")
        if seed_col:
            rename_map[seed_col] = "team_playoff_seed"
        if gb_col:
            rename_map[gb_col] = "team_games_behind"
        stand_sub = standings[merge_cols + list(rename_map.keys())].rename(columns=rename_map)
        result = result.merge(stand_sub, on=merge_cols, how="left")
    else:
        result["team_playoff_seed"] = np.nan
        result["team_games_behind"] = np.nan

    # Season phase: 0-0.2=early, 0.2-0.7=mid, 0.7-1.0=late
    if "game_number" in result.columns:
        result["season_phase"] = (result["game_number"] / 40.0).clip(0, 1)
    elif "game_date" in result.columns:
        result["season_phase"] = _estimate_season_phase(result)
    else:
        result["season_phase"] = 0.5

    return result


def _estimate_season_phase(df: pd.DataFrame) -> pd.Series:
    """Estimate season phase (0-1) from game_date within each season."""
    def _phase(g: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(g["game_date"])
        mn, mx = dates.min(), dates.max()
        rng = (mx - mn).total_seconds()
        if rng < 1:
            return pd.Series(0.5, index=g.index)
        return ((dates - mn).dt.total_seconds() / rng).clip(0, 1)

    if "season" in df.columns:
        return df.groupby("season", group_keys=False).apply(_phase)
    return _phase(df)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def build_all_advanced_features(
    player_game_stats: pd.DataFrame,
    processed_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Apply all 24 advanced feature columns to player_game_stats.

    Gracefully handles missing upstream data — fills with NaN/zeros.
    """
    df = player_game_stats.copy()
    steps = [
        build_usage_features,
        build_four_factors_features,
        build_shot_quality_features,
        build_injury_features,
        build_opponent_context_features,
        build_standings_features,
    ]
    for fn in steps:
        try:
            df = fn(df, processed_dir=processed_dir)
        except Exception as exc:
            log.warning("advanced_features.%s failed: %s", fn.__name__, exc)
    return df
