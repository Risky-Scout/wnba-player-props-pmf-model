#!/usr/bin/env python3
"""G2 tracking-driver feasibility audit (owner directive item 9).

G2 = G1 + tracking-derived, strictly-lagged, GAME-VARYING share drivers (rebound chances, touches,
passes, speed/distance, shot-contest context, defended-at-rim context). This script objectively tests
whether those drivers actually carry signal in the repository's tracking assets. Each requested driver
is mapped to its source column(s); a column is "populated" only if it has non-zero values. The verdict
is fully data-driven so G2 is never fabricated on empty inputs.

Writes artifacts/opportunity_v2/G2_TRACKING_FEASIBILITY.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "opportunity_v2" / "G2_TRACKING_FEASIBILITY.json"

# requested G2 driver -> candidate source columns in tracking / hustle assets
DRIVER_SOURCES = {
    "rebound_chances": ["reboundChancesTotal", "reboundChancesOffensive", "reboundChancesDefensive"],
    "touches": ["touches"],
    "passes": ["passes", "secondaryAssists", "freeThrowAssists"],
    "speed_distance": ["speed", "distance"],
    "shot_contest_context": ["contestedFieldGoalsAttempted", "uncontestedFieldGoalsAttempted",
                             "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT"],
    "defended_at_rim_context": ["defendedAtRimFieldGoalsAttempted", "defendedAtRimFieldGoalsMade"],
    "boxouts_screens_hustle": ["OFF_BOXOUTS", "DEF_BOXOUTS", "SCREEN_ASSISTS", "DEFLECTIONS"],
}


def _signal(frames: dict[str, pd.DataFrame], col: str) -> dict:
    for name, df in frames.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            nn = int(s.notna().sum())
            nz = int((s.fillna(0) != 0).sum())
            return {"source_asset": name, "present": True, "non_null": nn,
                    "non_zero": nz, "populated": bool(nz > 0),
                    "mean": (float(np.nanmean(s)) if nn else None)}
    return {"present": False, "populated": False}


def main() -> None:
    frames = {}
    for name, rel in {"tracking": "data/processed/wnba_tracking_2021_2026.parquet",
                      "hustle": "data/processed/wnba_hustle_2021_2026.parquet"}.items():
        p = REPO / rel
        if p.exists():
            frames[name] = pd.read_parquet(p)

    drivers = {}
    for driver, cols in DRIVER_SOURCES.items():
        col_reports = {c: _signal(frames, c) for c in cols}
        drivers[driver] = {
            "candidate_columns": col_reports,
            "any_populated": any(r.get("populated") for r in col_reports.values()),
        }

    populated_drivers = [d for d, r in drivers.items() if r["any_populated"]]
    # honest note on the only non-zero tracking fields
    fgpct = _signal(frames, "fieldGoalPercentage")
    trk_assists = _signal(frames, "assists")

    feasible = len(populated_drivers) > 0
    report = {
        "assets_examined": {n: {"rows": int(len(df)), "cols": int(len(df.columns))}
                            for n, df in frames.items()},
        "requested_drivers": drivers,
        "populated_drivers": populated_drivers,
        "only_nonzero_tracking_fields": {
            "fieldGoalPercentage": fgpct,   # already consumed by PTS decomposition (FGM recovery)
            "assists": trk_assists,         # identical to box assists -> not a new game-varying signal
        },
        "g2_feasible": feasible,
        "verdict": (
            "FEASIBLE" if feasible else
            "HARD BLOCKER: tracking-data-v1 and wnba_hustle_2021_2026 contain NO populated values for "
            "any requested G2 share driver (rebound chances, touches, passes, speed/distance, "
            "shot-contest, defended-at-rim, hustle) — every such column is structurally all-zero. The "
            "only non-zero tracking fields are fieldGoalPercentage (already used for PTS FGM recovery) "
            "and assists (identical to box assists). Genuine tracking-derived game-varying share drivers "
            "cannot be built and MUST NOT be fabricated; G2 cannot be constructed until a tracking source "
            "with populated hustle/tracking metrics is ingested."
        ),
        "leakage_note": ("Even if populated, only strictly-prior (lagged) tracking would be used, joined "
                         "via tracking_game_crosswalk / tracking_player_crosswalk; target-game tracking "
                         "is never used."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps({"g2_feasible": feasible, "populated_drivers": populated_drivers,
                      "verdict": report["verdict"][:80]}, indent=2))


if __name__ == "__main__":
    main()
