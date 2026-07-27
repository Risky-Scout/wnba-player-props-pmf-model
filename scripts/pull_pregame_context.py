#!/usr/bin/env python3
"""Normalize the current injury pull into append-only availability snapshots (directive section 8/9).

This is FORWARD collection: it records the availability context AS PULLED NOW (with a pulled/available
timestamp) into data/snapshots/availability/. It never reconstructs historical pregame status from
postgame results. Unknown status stays 'unknown'. Each snapshot carries a deterministic payload hash.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.opportunity.contracts import (
    AVAILABILITY_REQUIRED,
    AVAILABILITY_UNIQUE,
    normalize_availability_status,
)
from wnba_props_model.opportunity.snapshot_store import append_snapshot_partition, payload_sha256


def _rows_from_injuries(inj: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    rows = []
    for _, r in inj.iterrows():
        pulled = pd.to_datetime(r.get("pull_timestamp_utc"), utc=True, errors="coerce")
        pulled = pulled if pd.notna(pulled) else now
        status_norm = normalize_availability_status(
            r.get("injury_status_normalized") or r.get("injury_status"))
        payload = {
            "player_id": r.get("player_id"), "team_id": r.get("team_id"),
            "status_raw": r.get("injury_status"), "status_normalized": status_norm,
            "description": r.get("injury_description"), "pulled_at_utc": str(pulled),
        }
        sid = payload_sha256(payload)
        rows.append({
            "snapshot_id": sid, "source": str(r.get("source") or "bdl_injuries"),
            "source_record_id": None, "pulled_at_utc": pulled, "available_at_utc": pulled,
            "effective_at_utc": None, "snapshot_date_utc": pulled.date(),
            "game_id": r.get("game_id"), "scheduled_tip_utc": None,
            "player_id": r.get("player_id"), "team_id": r.get("team_id"),
            "status_raw": r.get("injury_status"), "status_normalized": status_norm,
            "status_reason": r.get("injury_description"),
            "minutes_limit_reported": None, "is_expected_available_raw": None,
            "source_url_hash": None, "payload_sha256": sid,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--injuries", default="data/processed/wnba_injuries.parquet")
    ap.add_argument("--root", default="data/snapshots/availability")
    args = ap.parse_args()
    now = pd.Timestamp(datetime.now(timezone.utc))
    if not Path(args.injuries).exists():
        raise SystemExit(f"pull_pregame_context: injuries table not found: {args.injuries}")
    inj = pd.read_parquet(args.injuries)
    frame = pd.DataFrame(_rows_from_injuries(inj, now))
    if frame.empty:
        print("pull_pregame_context: no injury rows to snapshot")
        return
    written = append_snapshot_partition(frame, Path(args.root), AVAILABILITY_REQUIRED, AVAILABILITY_UNIQUE)
    print(json.dumps({"appended_partitions": [str(p) for p in written],
                      "rows": int(len(frame))}, indent=2))


if __name__ == "__main__":
    main()
