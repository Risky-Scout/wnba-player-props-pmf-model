#!/usr/bin/env python3
"""Honest data-availability audit for Opportunity V2 (directive section 41).

States, per feature family, exactly which game-specific opportunity signals can be built at the
historical prediction cutoff and which require a NEW historical source or FORWARD-only snapshot
collection. Writes artifacts/opportunity_v2/DATA_AVAILABILITY_AUDIT.json.

Context states:
  HISTORICAL_PREGAME_CONTEXT_COMPLETE  exact pregame context recoverable at every cutoff
  HISTORICAL_PREGAME_CONTEXT_PARTIAL   recoverable for some rows/dates only
  FORWARD_ONLY_CONTEXT                 not historical; append-only collection can start now
  NO_CONTEXT                           neither historical nor a live source exists yet
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import pandas as pd

OUT = "artifacts/opportunity_v2/DATA_AVAILABILITY_AUDIT.json"


def _load(p):
    return pd.read_parquet(p) if os.path.exists(p) else None


def _season_coverage(df):
    if df is None or "game_date" not in df.columns:
        return {}
    out = {}
    key = "season" if "season" in df.columns else None
    groups = df.groupby(key) if key else [("all", df)]
    for s, g in groups:
        gg = pd.to_datetime(g["game_date"], utc=True, errors="coerce")
        out[str(s)] = {
            "rows": int(len(g)),
            "min_date": str(gg.min().date()) if gg.notna().any() else None,
            "max_date": str(gg.max().date()) if gg.notna().any() else None,
            "game_dates": int(gg.dt.date.nunique()),
        }
    return out


def main() -> None:
    box = _load("data/processed/wnba_player_game_stats.parquet")
    inj = _load("data/processed/wnba_injuries.parquet")
    padv = _load("data/processed/wnba_player_advanced_stats.parquet")
    tadv = _load("data/processed/wnba_team_game_advanced.parquet")
    shots = _load("data/processed/wnba_player_shot_locations.parquet")

    injury_json = sorted(os.path.basename(x) for x in glob.glob("data/injuries/*.json"))
    tracking_cols_present = [c for c in ("touches", "potential_assists", "rebound_chances",
                                         "drives", "time_of_possession_seconds", "catch_shoot_3pa")
                            if padv is not None and c in padv.columns]

    families = {
        "player_baseline_box_rates": {
            "state": "HISTORICAL_PREGAME_CONTEXT_COMPLETE",
            "data_tier": 0,
            "evidence": "wnba_player_game_stats has minutes/fga/fg3a/fta/oreb/dreb/makes/did_play "
                        "for all rows; usable as strictly-lagged per-minute rates.",
            "buildable_features": [
                "player_minutes_ewma", "player_fga_per_min_ewma", "player_fg2a_per_min_ewma",
                "player_fg3a_per_min_ewma", "player_fta_per_min_ewma",
                "player_oreb_chances_per_min_ewma (proxy=oreb)", "player_dreb_chances_per_min_ewma (proxy=dreb)",
                "player_2p_pct_ewma", "player_3p_pct_ewma", "player_ft_pct_ewma", "player_active_rate_ewma",
            ],
            "proof_eligible": True,
        },
        "team_opponent_environment": {
            "state": "HISTORICAL_PREGAME_CONTEXT_COMPLETE",
            "data_tier": 0,
            "evidence": "Team possessions derivable from box (FGA+0.44*FTA-OREB+TOV); pace present in "
                        "wnba_team_game_advanced. Strictly lagged team/opponent aggregates buildable.",
            "buildable_features": [
                "team_possessions_ewma", "team_fga_ewma", "team_fg3a_ewma", "team_fta_ewma",
                "team_turnover_ewma", "team_fg_misses_ewma", "opponent_*_allowed_ewma",
                "team_rest_days", "is_home", "team_back_to_back",
            ],
            "proof_eligible": True,
        },
        "point_in_time_availability": {
            "state": "FORWARD_ONLY_CONTEXT",
            "data_tier": None,
            "evidence": f"wnba_injuries is LATEST-STATE only (rows={0 if inj is None else len(inj)}, "
                        f"single pull_timestamp, report_date null); injury JSON snapshots={injury_json} "
                        "are only recent forward pulls. No historical daily availability archive exists.",
            "buildable_features": [
                "availability_status_code (forward only)", "availability_snapshot_age_hours (forward only)",
                "reported_minutes_limit (forward only)",
            ],
            "proof_eligible": False,
            "remediation": "Begin append-only availability snapshot collection now "
                           "(scripts/build_opportunity_snapshots.py) for forward proof; do NOT "
                           "reconstruct historical pregame status from postgame DNP.",
        },
        "point_in_time_lineups": {
            "state": "NO_CONTEXT",
            "data_tier": None,
            "evidence": "No lineup snapshot table or source exists (no projected/confirmed starter "
                        "feed). started_proxy is a postgame minutes proxy, not a pregame lineup.",
            "buildable_features": [],
            "proof_eligible": False,
            "remediation": "Add a projected/confirmed-lineup source and collect append-only snapshots "
                           "forward; certification requires forward data.",
        },
        "roster_intervals": {
            "state": "NO_CONTEXT",
            "data_tier": None,
            "evidence": "Only current team is known (wnba_players.team_id). No transaction history / "
                        "valid_from/valid_to intervals -> teammate-at-cutoff cannot be reconstructed "
                        "historically without risking latest-team leakage.",
            "buildable_features": [],
            "proof_eligible": False,
            "remediation": "Build wnba_roster_intervals from a transaction/roster-history source; "
                           "until then vacated-opportunity shares are not historically certifiable.",
        },
        "vacated_role_and_opportunity_shares": {
            "state": "NO_CONTEXT",
            "data_tier": None,
            "evidence": "Requires BOTH point-in-time availability AND roster intervals, neither of "
                        "which is historically available. This is the distinctive V2 signal.",
            "buildable_features": [],
            "proof_eligible": False,
            "remediation": "Depends on availability (forward) + roster intervals (new source).",
        },
        "player_tracking_opportunity_tier2": {
            "state": "NO_CONTEXT",
            "data_tier": 2,
            "evidence": f"No player tracking columns present (found tracking cols: {tracking_cols_present}). "
                        "potential_assists, rebound_chances, touches, catch&shoot/pullup 3PA absent. "
                        f"Shot locations table has {0 if shots is None else len(shots)} rows with "
                        f"{0 if shots is None else int(pd.to_datetime(shots['game_date'],errors='coerce').notna().sum())} valid dates.",
            "buildable_features": [],
            "proof_eligible": False,
            "remediation": "Requires a tracking data source (WNBA tracking / play-by-play). Until then "
                           "ast/reb Tier-2 candidates CANNOT be built or certified; only Tier-0 proxies.",
        },
    }

    # Per-prop certifiability given the families above.
    prop_verdict = {
        "pts":  {"tier0_box_candidate": True,  "tier2_required": False, "historically_certifiable": "TIER0_ONLY",
                 "note": "Attempts (fg2a/fg3a/fta) + conversion from box, lagged. Overlaps existing structural."},
        "fg3m": {"tier0_box_candidate": True,  "tier2_required": False, "historically_certifiable": "TIER0_ONLY",
                 "note": "3PA + 3P% from box, lagged. Catch&shoot/pullup shares NOT available (NO_CONTEXT)."},
        "reb":  {"tier0_box_candidate": True,  "tier2_required": True,  "historically_certifiable": "TIER0_PROXY_ONLY",
                 "note": "True OREB/DREB CHANCES require tracking (NO_CONTEXT). Only oreb/dreb counts (proxy)."},
        "ast":  {"tier0_box_candidate": True,  "tier2_required": True,  "historically_certifiable": "TIER0_PROXY_ONLY",
                 "note": "Potential assists (Tier-2) NOT available (NO_CONTEXT). Only ast-per-min proxy."},
        "turnover": {"tier0_box_candidate": True, "tier2_required": True, "historically_certifiable": "TIER0_PROXY_ONLY",
                 "note": "Touches/time-of-possession (Tier-2) NOT available. Only tov-per-min proxy."},
        "stl":  {"tier0_box_candidate": True,  "tier2_required": True,  "historically_certifiable": "TIER0_PROXY_ONLY",
                 "note": "Opponent live-ball TO opportunities / matchup exposure NOT available."},
        "blk":  {"tier0_box_candidate": True,  "tier2_required": True,  "historically_certifiable": "TIER0_PROXY_ONLY",
                 "note": "Opponent rim-attempt exposure (tracking/pbp) NOT available."},
    }

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "headline": (
            "The game-specific opportunity signals that distinguish Opportunity V2 from the existing "
            "structural model (point-in-time availability/lineups, roster-at-cutoff, vacated role, and "
            "player/team TRACKING opportunity) DO NOT EXIST historically in this repository. Only "
            "Tier-0 box-score opportunity + team environment are historically recoverable. Therefore "
            "V2 can be built and OOF-measured as a Tier-0 candidate now, but its distinctive value "
            "requires (a) forward availability/lineup snapshot collection and (b) a new tracking "
            "source before any Tier-2 historical certification is possible."
        ),
        "box_score_coverage": _season_coverage(box),
        "injury_table": {
            "rows": 0 if inj is None else int(len(inj)),
            "is_historical_archive": False,
            "distinct_pull_dates": sorted(map(str, pd.to_datetime(inj["pull_timestamp_utc"], errors="coerce",
                                       utc=True).dt.date.dropna().unique())) if inj is not None else [],
            "forward_json_snapshots": injury_json,
        },
        "tracking_available": False,
        "roster_interval_history_available": False,
        "lineup_snapshot_source_available": False,
        "feature_families": families,
        "prop_certifiability": prop_verdict,
        "proof_eligible_historical_rows_tier2": 0,
        "recommended_path": [
            "Implement the point-in-time truth layer (DONE: contracts/snapshot_store/asof/audit).",
            "Start append-only forward collection of availability + lineup snapshots immediately.",
            "Build a Tier-0 OPP_V2 box-opportunity candidate now and OOF-measure vs P0 (development only).",
            "Do NOT certify Tier-2 (ast/reb/turnover/stl/blk) historically; mark those rows proof_eligible=false.",
            "Acquire a WNBA tracking / play-by-play source before Tier-2 modeling.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(audit, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("HEADLINE:", audit["headline"][:200], "...")
    for fam, v in families.items():
        print(f"  {fam:<38} {v['state']}")


if __name__ == "__main__":
    main()
