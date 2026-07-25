"""P11 - run the tiered identity crosswalk against the recovered canonical files.

Produces the crosswalk + rejection artifacts and a coverage report. Tracking uses
stats.nba.com game/person ids while the canonical tables use BDL ids, and the tracking extract
carries no game_date, so exact (date, team-set) and exact-id matching cannot bridge the two id
namespaces without a reviewed stats.nba.com<->BDL game-id map. This driver runs what is
possible, records the exact coverage, and labels the state honestly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "tracking"
TRACKING = REPO / "data" / "processed" / "wnba_tracking_2021_2026.parquet"
GAMES = REPO / "data" / "recovered_v2_preserved" / "wnba_games.parquet"
STATS = REPO / "data" / "recovered_v2_preserved" / "wnba_player_game_stats.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from wnba_props_model.data.identity_crosswalk import (
        CrosswalkCoverageError, build_game_crosswalk)

    if not (TRACKING.exists() and GAMES.exists()):
        print("[tracking-xwalk] inputs missing", flush=True)
        return 1
    tr = pd.read_parquet(TRACKING)
    games = pd.read_parquet(GAMES)
    prov_games = tr[["GAME_ID"]].rename(columns={"GAME_ID": "gameId"}).drop_duplicates()
    tr_ids = set(prov_games["gameId"].astype(str))
    canon_ids = set(games["game_id"].astype(str))
    id_overlap = len(tr_ids & canon_ids)

    # Attempt exact-id/date-team matching (no reviewed bridge available).
    status = "REAL_DATA_COVERAGE_RUN"
    coverage = 0.0
    conflicts = 0
    try:
        gcw = build_game_crosswalk(prov_games, games, min_coverage=0.0)  # no auto identical-id
        resolved = int((gcw["status"] == "RESOLVED").sum())
        coverage = resolved / max(1, len(gcw))
        conflicts = int((gcw["status"] == "CONFLICT_GAME").sum())
        gcw.to_parquet(OUT / "tracking_game_crosswalk.parquet", index=False)
        gcw[gcw["status"] != "RESOLVED"].to_csv(OUT / "unmatched_games.csv", index=False)
    except CrosswalkCoverageError as exc:
        status = "REAL_DATA_COVERAGE_FAILED"
        print(f"[tracking-xwalk] coverage gate: {exc}")

    blocked = id_overlap == 0 and coverage < 0.99
    if blocked:
        status = "REAL_DATA_COVERAGE_BLOCKED_NO_ID_BRIDGE"

    report = {
        "version": "tracking-identity-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tracking_games": len(tr_ids), "canonical_games": len(canon_ids),
        "exact_id_overlap": id_overlap,
        "id_namespaces": {"tracking": "stats.nba.com (10-digit)", "canonical": "BDL (short int)"},
        "tracking_has_game_date": bool("game_date" in tr.columns),
        "game_coverage": round(coverage, 4), "conflicts": conflicts,
        "gates": {"overall_row_weighted>=0.99": coverage >= 0.99,
                  "conflicts==0": conflicts == 0},
        "status": status,
        "blocker": (None if not blocked else
                    "tracking (stats.nba.com ids, no game_date) cannot be bridged to canonical "
                    "(BDL ids) without a reviewed stats.nba.com<->BDL game-id map or a schedule "
                    "carrying both ids + dates. Owner action: provide that bridge (or a dated "
                    "schedule). Non-blocking for the corrected baseline / low-cost candidates."),
        "label": "IMPLEMENTATION_COMPLETE / REAL_DATA_COVERAGE_BLOCKED_NO_ID_BRIDGE"
                 if blocked else "IMPLEMENTATION_COMPLETE / REAL_DATA_COVERAGE_RUN",
    }
    (OUT / "TRACKING_IDENTITY_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[tracking-xwalk] id_overlap={id_overlap} coverage={coverage:.3f} status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
