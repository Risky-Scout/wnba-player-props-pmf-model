#!/usr/bin/env python3
"""Fail-closed forward Opportunity V2 snapshot collector + coverage manifest (owner item 2).

Drives availability-snapshot collection from a current injuries pull. This is FORWARD collection: it
records availability AS PULLED NOW and never reconstructs pregame status from postgame results.

FAIL-CLOSED semantics (default). The job exits NON-ZERO on any of:
  * missing injuries response (injuries table absent) when games are scheduled;
  * zero injury rows when games are scheduled;
  * zero snapshot rows when games are scheduled;
  * any snapshot row missing a payload hash;
  * any invalid (unparseable) snapshot timestamp.
A coverage manifest is ALWAYS written (phase, game_date, games, players, injury_rows, availability_rows,
prediction_cutoff, source_timestamp, payload_hashes, failure_status).

``--scheduled-games`` is the count of games scheduled for ``--game-date`` (-1 = unknown, treated STRICT
i.e. fail-closed). Only an explicit ``--scheduled-games 0`` (no slate) relaxes the zero-row checks to a
clean ``ok_no_games`` exit 0. Lineup / roster-interval collection remains a no-op until a real
projected/confirmed-lineup source exists (never faked from postgame).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.opportunity.contracts import AVAILABILITY_REQUIRED, AVAILABILITY_UNIQUE
from wnba_props_model.opportunity.snapshot_store import append_snapshot_partition

REPO = Path(__file__).resolve().parent.parent


def _load_ppc():
    spec = importlib.util.spec_from_file_location("ppc", REPO / "scripts" / "pull_pregame_context.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injuries", default="data/processed/wnba_injuries.parquet")
    ap.add_argument("--availability-root", default="data/snapshots/availability")
    ap.add_argument("--coverage-manifest", default=None)
    ap.add_argument("--phase", default="manual")
    ap.add_argument("--game-date", default=None)
    ap.add_argument("--scheduled-games", type=int, default=-1,
                    help="games scheduled for --game-date; -1 = unknown (treated STRICT / fail-closed)")
    ap.add_argument("--no-fail-closed", action="store_true", default=False,
                    help="DEV ONLY: never use in scheduled/certified CI")
    args = ap.parse_args()

    fail_closed = not args.no_fail_closed
    now = pd.Timestamp(datetime.now(timezone.utc))
    game_date = args.game_date or now.date().isoformat()
    no_games = args.scheduled_games == 0  # explicit 0 => a genuinely gameless day

    manifest = {
        "phase": args.phase, "game_date": game_date, "games": int(args.scheduled_games),
        "players": 0, "injury_rows": 0, "availability_rows": 0,
        "prediction_cutoff": None, "source_timestamp": str(now),
        "payload_hashes": [], "payload_hash_count": 0,
        "failure_status": "ok", "fail_closed": fail_closed, "generated_utc": str(now),
    }

    def _finish(status: str, exit_code: int) -> None:
        manifest["failure_status"] = status
        if args.coverage_manifest:
            Path(args.coverage_manifest).parent.mkdir(parents=True, exist_ok=True)
            json.dump(manifest, open(args.coverage_manifest, "w"), indent=2, default=str)
        print(json.dumps({k: manifest[k] for k in
                          ("phase", "game_date", "games", "injury_rows",
                           "availability_rows", "failure_status")}, indent=2))
        raise SystemExit(exit_code)

    # 1) injuries response present?
    ip = Path(args.injuries)
    if not ip.exists():
        if no_games:
            _finish("ok_no_games_missing_injuries", 0)
        _finish("FAIL_missing_injuries_response", 1 if fail_closed else 0)

    inj = pd.read_parquet(ip)
    manifest["injury_rows"] = int(len(inj))
    if len(inj) == 0:
        if no_games:
            _finish("ok_no_games", 0)
        _finish("FAIL_zero_injury_rows", 1 if fail_closed else 0)

    # 2) build availability snapshot rows
    ppc = _load_ppc()
    frame = pd.DataFrame(ppc._rows_from_injuries(inj, now))
    manifest["availability_rows"] = int(len(frame))
    if len(frame) == 0:
        if no_games:
            _finish("ok_no_games", 0)
        _finish("FAIL_zero_snapshot_rows", 1 if fail_closed else 0)

    manifest["players"] = int(pd.Series(frame["player_id"]).nunique())
    # payload hashes present on every row?
    hashes = pd.Series(frame["payload_sha256"]).dropna().astype(str).tolist()
    if len(hashes) != len(frame):
        _finish("FAIL_missing_payload_hashes", 1 if fail_closed else 0)
    manifest["payload_hashes"] = hashes[:50]
    manifest["payload_hash_count"] = len(hashes)
    # timestamps valid?
    ts = pd.to_datetime(frame["pulled_at_utc"], utc=True, errors="coerce")
    if ts.isna().any():
        _finish("FAIL_invalid_timestamps", 1 if fail_closed else 0)
    manifest["prediction_cutoff"] = str(ts.max())

    # 3) append append-only partition
    written = append_snapshot_partition(frame, Path(args.availability_root),
                                        AVAILABILITY_REQUIRED, AVAILABILITY_UNIQUE)
    manifest["appended_partitions"] = [str(p) for p in written]
    print("lineup snapshots: SKIPPED (no projected/confirmed-lineup source; forward-only when added)")
    print("roster intervals: SKIPPED (requires a transaction/roster-history source)")
    _finish("ok", 0)


if __name__ == "__main__":
    main()
