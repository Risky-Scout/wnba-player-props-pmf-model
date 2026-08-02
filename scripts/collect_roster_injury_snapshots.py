#!/usr/bin/env python3
"""Append-only prospective roster + injury snapshot collector.

Captures for each forecast timestamp:
  - /wnba/v1/players/active
  - scheduled-team player cross-check
  - /wnba/v1/player_injuries
  - source health, ingestion timestamp, raw payload hash

Raw payloads are written under gitignored data/ (or uploaded as workflow
artifacts). Never commit private snapshots to git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient


def _payload_hash(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _in_et_slate_window(commence: str, date_str: str) -> bool:
    try:
        tip = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lo = base.replace(hour=9)
    hi = (base + timedelta(days=1)).replace(hour=4)
    return lo <= tip <= hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--forecast-timestamp", default="")
    ap.add_argument("--out-dir", default="data/snapshots/roster_injury_forward")
    ap.add_argument(
        "--manifest-out",
        default="artifacts/phase2_repair/ROSTER_INJURY_SNAPSHOT_MANIFEST.json",
    )
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    forecast_ts = args.forecast_timestamp or now.isoformat()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    manifest: dict = {
        "forecast_timestamp": forecast_ts,
        "ingestion_timestamp_utc": now.isoformat(),
        "date": args.date,
        "status": "ok",
        "fail_open": True,
        "sources": {},
        "snapshot_ids": [],
        "append_only": True,
        "private_payloads_committed_to_git": False,
    }

    def _write_manifest() -> None:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    try:
        client = BDLClient()
    except BDLAPIError as exc:
        manifest["status"] = "no_bdl_api_key"
        manifest["error"] = str(exc)
        _write_manifest()
        print(f"[roster_injury] no BDL_API_KEY; fail-open at {args.manifest_out}")
        return

    next_day = (
        datetime.strptime(args.date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    out_dir = Path(args.out_dir) / f"snapshot_date_utc={args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _capture(source: str, fetch) -> None:
        try:
            payload = fetch()
            status = "ok"
            err = None
        except BDLAPIError as exc:
            payload = []
            status = "error"
            err = str(exc)[:200]
        ph = _payload_hash(payload)
        snap_id = f"{forecast_ts}|{source}|{ph}"
        path = out_dir / f"{source}_{stamp}_{ph[:12]}.json"
        path.write_text(json.dumps({
            "snapshot_id": snap_id,
            "forecast_timestamp": forecast_ts,
            "ingestion_timestamp_utc": now.isoformat(),
            "source": source,
            "payload_hash": ph,
            "n_rows": len(payload) if isinstance(payload, list) else None,
            "payload": payload,
        }, indent=2, default=str))
        manifest["sources"][source] = {
            "status": status,
            "error": err,
            "payload_hash": ph,
            "path": str(path),
            "n_rows": len(payload) if isinstance(payload, list) else None,
        }
        manifest["snapshot_ids"].append(snap_id)

    _capture(
        "players_active",
        lambda: client.list_endpoint("players/active", {"per_page": 100}),
    )
    _capture(
        "player_injuries",
        lambda: client.list_endpoint("player_injuries"),
    )

    def _games_and_roster_crosscheck():
        games = client.list_endpoint("games", {"dates": [args.date, next_day]})
        slate = [g for g in games if _in_et_slate_window(g.get("date"), args.date)]
        team_ids = set()
        for g in slate:
            home = g.get("home_team") or {}
            away = g.get("visitor_team") or g.get("away_team") or {}
            if home.get("id") is not None:
                team_ids.add(home["id"])
            if away.get("id") is not None:
                team_ids.add(away["id"])
        players = []
        for tid in sorted(team_ids):
            try:
                players.extend(
                    client.list_endpoint("players", {"team_ids[]": tid, "per_page": 100})
                )
            except BDLAPIError:
                continue
        return {
            "games": slate,
            "team_ids": sorted(team_ids),
            "scheduled_team_players": players,
        }

    _capture("scheduled_team_player_crosscheck", _games_and_roster_crosscheck)

    healthy = all(v.get("status") == "ok" for v in manifest["sources"].values())
    manifest["source_health"] = "ok" if healthy else "degraded"
    if not healthy and all(v.get("status") == "error" for v in manifest["sources"].values()):
        manifest["status"] = "all_sources_failed"
    _write_manifest()
    print(json.dumps({
        "date": args.date,
        "status": manifest["status"],
        "source_health": manifest.get("source_health"),
        "n_snapshots": len(manifest["snapshot_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
