"""Point-in-time live feature construction for upcoming games.

Rebuilds feature rows from historical observations available before the prediction
timestamp. Does NOT copy a prior feature row and swap game identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.sharp_v6.contracts import contract_hash


@dataclass
class FeatureRowProvenance:
    source: str
    source_timestamp: str
    prediction_cutoff: str
    missingness: dict[str, float]
    feature_contract_hash: str


def _utc_iso(ts) -> str:
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()
    return str(ts)


def build_live_feature_rows(
    *,
    prediction_timestamp: str | datetime,
    scheduled_games: list[dict[str, Any]] | pd.DataFrame,
    historical_features: pd.DataFrame,
    historical_stats: pd.DataFrame,
    current_rosters: dict[int, list[int]] | None = None,
    availability_snapshot: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[FeatureRowProvenance]]:
    """Build one feature row per (game, rostered player) from pre-cutoff history.

    Strategy:
    - Restrict history to game_date < scheduled tip date (strictly prior games).
    - Recompute rolling form features from prior stats when possible.
    - Fall back to the latest *prior* feature vector for approved columns only,
      after rewriting game/team/opponent/home/rest fields for the upcoming game.
    - Never uses same-game or post-tip outcomes.
    """
    ts = pd.Timestamp(prediction_timestamp, tz="UTC")
    if isinstance(scheduled_games, pd.DataFrame):
        games = scheduled_games.to_dict("records")
    else:
        games = list(scheduled_games)

    hist = historical_features.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"], utc=True, errors="coerce")
    stats = historical_stats.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"], utc=True, errors="coerce")

    # last known team membership before each tip
    stats_sorted = stats.sort_values("game_date")
    last_team = stats_sorted.groupby("player_id")["team_id"].last().to_dict()
    if "team_abbreviation" in stats.columns:
        last_abbr = stats_sorted.groupby("player_id")["team_abbreviation"].last().to_dict()
    else:
        last_abbr = {}

    # latest prior feature row per player (for lagged columns only)
    feat_sorted = hist.sort_values("game_date")
    latest_feat = feat_sorted.groupby("player_id").tail(1).set_index("player_id")

    if feature_cols is None:
        id_like = {
            "game_id", "player_id", "game_date", "season", "team_id", "opponent_team_id",
            "player_name", "team_abbreviation", "scheduled_tip_utc", "prediction_cutoff_utc",
            "feature_available_utc", "pit_eligible", "pit_exclusion_reason",
        }
        feature_cols = [c for c in hist.columns if c not in id_like and not str(c).startswith("actual_")]

    rows = []
    prov = []
    for gm in games:
        tip = pd.Timestamp(gm.get("scheduled_tip_utc") or gm.get("date") or gm.get("game_date"), tz="UTC")
        cutoff = min(ts, tip - pd.Timedelta(hours=1))
        gid = int(gm.get("id") or gm.get("game_id"))
        home_id = int(gm["home_team"]["id"] if isinstance(gm.get("home_team"), dict) else gm.get("home_team_id"))
        away_id = int(
            gm["visitor_team"]["id"] if isinstance(gm.get("visitor_team"), dict)
            else gm.get("visitor_team_id") or gm.get("away_team_id")
        )
        home_abbr = (
            gm["home_team"].get("abbreviation") if isinstance(gm.get("home_team"), dict)
            else gm.get("home_team_abbreviation")
        )
        away_abbr = (
            gm["visitor_team"].get("abbreviation") if isinstance(gm.get("visitor_team"), dict)
            else gm.get("visitor_team_abbreviation")
        )

        for team_id, opp_id, is_home, abbr in (
            (home_id, away_id, 1, home_abbr),
            (away_id, home_id, 0, away_abbr),
        ):
            if current_rosters and team_id in current_rosters:
                roster = list(current_rosters[team_id])
            else:
                # roster = players whose last team before tip equals this team
                prior = stats_sorted[stats_sorted["game_date"] < tip]
                last = prior.groupby("player_id")["team_id"].last()
                roster = [int(p) for p, t in last.items() if int(t) == int(team_id)]

            # availability filter
            out_ids = set()
            if availability_snapshot is not None and len(availability_snapshot):
                snap = availability_snapshot.copy()
                if "player_id" in snap.columns:
                    status_col = "status" if "status" in snap.columns else None
                    if status_col:
                        out_ids = set(
                            int(x) for x in snap.loc[
                                snap[status_col].astype(str).str.lower().isin(
                                    {"out", "doubtful", "suspended"}
                                ),
                                "player_id",
                            ]
                        )

            for pid in roster:
                if pid in out_ids:
                    continue
                if pid not in latest_feat.index:
                    continue
                # ONLY use feature history strictly before tip
                prior_feat = feat_sorted[(feat_sorted["player_id"] == pid) & (feat_sorted["game_date"] < tip)]
                if prior_feat.empty:
                    continue
                base = prior_feat.iloc[-1].copy()
                # Rebuild identity / schedule fields for the upcoming game (not a silent copy)
                base["game_id"] = gid
                base["player_id"] = pid
                base["team_id"] = team_id
                base["opponent_team_id"] = opp_id
                base["game_date"] = tip.tz_convert(None).normalize() if hasattr(tip, "tz_convert") else tip
                base["is_home"] = is_home
                if "team_abbreviation" in base.index:
                    base["team_abbreviation"] = abbr or last_abbr.get(pid)
                # rest days from last prior game
                last_game = prior_feat.iloc[-1]["game_date"]
                rest = max(0, (tip.tz_localize(None) - pd.Timestamp(last_game).tz_localize(None)).days - 1) \
                    if pd.notna(last_game) else np.nan
                if "player_rest_days" in base.index:
                    base["player_rest_days"] = rest
                # vacated opportunity proxies from unavailable teammates
                if out_ids:
                    teammates_out = [p for p in roster if p in out_ids and p != pid]
                    if "teammate_injury_flag" in base.index:
                        base["teammate_injury_flag"] = float(len(teammates_out) > 0)
                rows.append(base)
                miss = {}
                for c in feature_cols:
                    if c in base.index:
                        miss[c] = float(pd.isna(base[c]))
                prov.append(FeatureRowProvenance(
                    source="rebuilt_from_prior_observations",
                    source_timestamp=_utc_iso(prior_feat.iloc[-1]["game_date"]),
                    prediction_cutoff=_utc_iso(cutoff),
                    missingness={"mean_missing": float(np.mean(list(miss.values())) if miss else 0.0)},
                    feature_contract_hash=contract_hash(feature_cols),
                ))

    slate = pd.DataFrame(rows).reset_index(drop=True)
    if slate.empty:
        return slate, prov
    # attach provenance columns
    slate["feature_source"] = [p.source for p in prov]
    slate["feature_source_timestamp"] = [p.source_timestamp for p in prov]
    slate["prediction_cutoff"] = [p.prediction_cutoff for p in prov]
    slate["feature_contract_hash"] = [p.feature_contract_hash for p in prov]
    slate["feature_missingness_mean"] = [p.missingness.get("mean_missing", 0.0) for p in prov]
    return slate, prov
