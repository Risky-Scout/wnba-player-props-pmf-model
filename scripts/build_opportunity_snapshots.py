#!/usr/bin/env python3
"""Orchestrate forward Opportunity V2 snapshot collection (directive section 8/9).

Currently drives availability-snapshot collection (via pull_pregame_context). Lineup collection is a
no-op until a projected/confirmed-lineup source exists (see DATA_AVAILABILITY_AUDIT.json) -- it is
NOT faked from postgame data. Roster-interval construction is likewise deferred to a real source.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injuries", default="data/processed/wnba_injuries.parquet")
    ap.add_argument("--availability-root", default="data/snapshots/availability")
    args = ap.parse_args()

    if not Path(args.injuries).exists():
        raise SystemExit(f"build_opportunity_snapshots: missing injuries table {args.injuries}")
    # Availability snapshots (real forward collection).
    sys.argv = ["pull_pregame_context.py", "--injuries", args.injuries,
                "--root", args.availability_root]
    runpy.run_path(str(Path(__file__).parent / "pull_pregame_context.py"), run_name="__main__")

    print("lineup snapshots: SKIPPED (no projected/confirmed-lineup source; forward-only when added)")
    print("roster intervals: SKIPPED (requires a transaction/roster-history source)")


if __name__ == "__main__":
    main()
