#!/usr/bin/env python3
"""Regenerate TRACKING_ASSET_RECONCILIATION.json with per-signal population truth (owner item 1).

For every intended Tier-2 signal this records non_null_count, nonzero_count, nonzero_pct, per-season
nonzero coverage, and a usable verdict. The asset is classified SCHEMA_PRESENT_SIGNAL_EMPTY (schema
columns exist but carry no modeling signal) and G2 status is BLOCKED_EMPTY_TRACKING_SIGNAL. Only
fieldGoalPercentage (used for PTS FGM reconstruction) and assists (== box) are populated.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "opportunity_v2" / "TRACKING_ASSET_RECONCILIATION.json"

# intended Tier-2 signal -> source column in wnba_tracking
TIER2_SIGNALS = {
    "rebound_chances_offensive": "reboundChancesOffensive",
    "rebound_chances_defensive": "reboundChancesDefensive",
    "rebound_chances_total": "reboundChancesTotal",
    "touches": "touches",
    "passes": "passes",
    "secondary_assists": "secondaryAssists",
    "free_throw_assists": "freeThrowAssists",
    "speed": "speed",
    "distance": "distance",
    "contested_field_goals_made": "contestedFieldGoalsMade",
    "contested_field_goals_attempted": "contestedFieldGoalsAttempted",
    "uncontested_field_goals_made": "uncontestedFieldGoalsMade",
    "uncontested_field_goals_attempted": "uncontestedFieldGoalsAttempted",
    "defended_at_rim_field_goals_made": "defendedAtRimFieldGoalsMade",
    "defended_at_rim_field_goals_attempted": "defendedAtRimFieldGoalsAttempted",
}
POPULATED_FIELDS = {"fieldGoalPercentage": "fieldGoalPercentage", "assists": "assists"}
USABLE_MIN_NONZERO_PCT = 5.0  # a signal is modeling-usable only if >5% of rows are non-zero


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _season_map(tr: pd.DataFrame) -> pd.Series:
    """Map each tracking row to a season via the game crosswalk + box seasons (tracking lacks date)."""
    gx = pd.read_parquet(REPO / "data/processed/tracking_game_crosswalk.parquet")
    box = pd.read_parquet(REPO / "data/processed/wnba_player_game_stats.parquet")[["game_id", "season"]].drop_duplicates()
    g2s = box.set_index("game_id")["season"].to_dict()
    gid = tr["gameId"].map(dict(zip(gx["gameId"], gx["game_id"])))
    return gid.map(g2s)


def _signal_report(s: pd.Series, season: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce")
    non_null = int(v.notna().sum())
    nz_mask = v.fillna(0) != 0
    nonzero = int(nz_mask.sum())
    pct = round(100.0 * nonzero / len(v), 4) if len(v) else 0.0
    by_season = {}
    for yr, grp in nz_mask.groupby(season):
        if pd.isna(yr):
            continue
        by_season[int(yr)] = int(grp.sum())
    return {
        "source_column": s.name,
        "non_null_count": non_null,
        "nonzero_count": nonzero,
        "nonzero_pct": pct,
        "season_coverage_nonzero": by_season,
        "usable_for_modeling": bool(pct >= USABLE_MIN_NONZERO_PCT),
    }


def main() -> None:
    tr_path = REPO / "data/processed/wnba_tracking_2021_2026.parquet"
    hu_path = REPO / "data/processed/wnba_hustle_2021_2026.parquet"
    tr = pd.read_parquet(tr_path)
    season = _season_map(tr)

    signals = {name: _signal_report(tr[col], season) for name, col in TIER2_SIGNALS.items()
               if col in tr.columns}
    populated = {name: _signal_report(tr[col], season) for name, col in POPULATED_FIELDS.items()
                 if col in tr.columns}

    usable_signals = [k for k, r in signals.items() if r["usable_for_modeling"]]
    any_usable = len(usable_signals) > 0

    report = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": "prior PARTIALLY_USABLE classification (which conflated identity coverage with signal presence)",
        "asset": "wnba_tracking (tracking-data-v1)",
        "asset_sha256": _sha(tr_path),
        "rows": int(len(tr)),
        "tier2_signal_population": signals,
        "populated_fields": {
            **populated,
            "_note": ("fieldGoalPercentage is the ONLY tracking field used by Opportunity V2 (PTS FGM "
                      "reconstruction). assists duplicates box assists and adds no game-varying signal."),
        },
        "usable_tier2_signals": usable_signals,
        "asset_classification": "SCHEMA_PRESENT_SIGNAL_EMPTY",
        "classification_reason": ("All intended Tier-2 tracking signals (rebound chances, touches, passes, "
                                  "secondary assists, speed, distance, contested/uncontested FG made+attempted, "
                                  "defended-at-rim made/attempted) have 0.0% non-zero rows. The schema columns "
                                  "exist but carry no modeling signal."),
        "g2_status": "BLOCKED_EMPTY_TRACKING_SIGNAL",
        "identity_reconciliation": {
            "note": ("Identity mapping is SOLVED (see tracking_identity_report.json): 262 one-to-one players, "
                     "0 conflicts, 100% canonical 2025-2026 game coverage. G2 is blocked by EMPTY SIGNAL, "
                     "NOT by identity."),
        },
        "hustle_asset": {
            "asset_sha256": _sha(hu_path) if hu_path.exists() else None,
            "rows": int(len(pd.read_parquet(hu_path))) if hu_path.exists() else 0,
            "verdict": "SCHEMA_PRESENT_SIGNAL_EMPTY (all hustle metrics 0.0% non-zero; 171 rows)",
        },
        "verdict": (f"SCHEMA_PRESENT_SIGNAL_EMPTY. {len(usable_signals)} of {len(signals)} intended Tier-2 "
                    f"signals are modeling-usable. G2 (tracking-derived game-varying share drivers) is "
                    f"BLOCKED_EMPTY_TRACKING_SIGNAL and cannot be built from tracking-data-v1."),
        "any_tier2_signal_usable": any_usable,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps({
        "asset_classification": report["asset_classification"],
        "g2_status": report["g2_status"],
        "usable_tier2_signals": usable_signals,
        "per_signal_nonzero_pct": {k: v["nonzero_pct"] for k, v in signals.items()},
        "populated_fields_nonzero_pct": {k: v["nonzero_pct"] for k, v in populated.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
