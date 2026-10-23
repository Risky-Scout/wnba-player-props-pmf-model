#!/usr/bin/env python3
"""Forward availability collector (Path A) — BDL player_injuries + today's games.

Records player availability AS PULLED NOW (pre-tip), append-only, into a gitignored data
dir. This is genuine FORWARD collection: it never reconstructs pregame status from postgame
results. Each pull is stamped with ``pulled_at_utc`` and a deterministic payload hash so the
archive is idempotent and auditable.

Why this exists: the vacated-opportunity feature (see
``src/wnba_props_model/data/vacated_opportunity.py``) needs the set of players who were OUT
*before tip*. There is no historical pregame-availability archive to backfill from, so this
collector accrues that record going forward, one slate at a time.

Fail-open: no BDL_API_KEY, no games, or no injuries writes an honest manifest and exits 0.

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/collect_availability.py            # today UTC
  PYTHONPATH=$(pwd)/src python3 scripts/collect_availability.py --date 2026-07-28
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.data.availability_audit import (
    append_snapshot_manifest,
    assert_no_snapshot_overwrite,
    build_availability_audit,
    classify_endpoint_result,
    compute_coverage,
    payload_hash,
)
from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient, EndpointStatus
from wnba_props_model.data.normalize import normalize_injuries


def _payload_hash(row: dict) -> str:
    blob = json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _in_et_slate_window(commence: str, date_str: str) -> bool:
    """True if a game's UTC commence time falls in the Eastern game-day window.

    Mirrors OddsAPIClient.list_events_for_date: 09:00Z on the date through 04:00Z the
    next day, so late-ET tips (which BDL files under the next UTC date) are still counted.
    """
    try:
        tip = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True  # keep if unparseable rather than silently dropping
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lo = base.replace(hour=9)
    hi = (base + timedelta(days=1)).replace(hour=4)
    return lo <= tip <= hi


def _teams_playing(games: list[dict], date_str: str) -> list[dict]:
    out = []
    seen = set()
    for g in games:
        if not _in_et_slate_window(g.get("date"), date_str):
            continue
        gid = g.get("id")
        if gid in seen:
            continue
        seen.add(gid)
        home = g.get("home_team") or {}
        away = g.get("visitor_team") or g.get("away_team") or {}
        out.append({
            "game_id": gid,
            "status": g.get("status"),
            "home_team_id": home.get("id"),
            "home_team_abbr": home.get("abbreviation"),
            "away_team_id": away.get("id"),
            "away_team_abbr": away.get("abbreviation"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out-dir", default="data/snapshots/availability_forward")
    ap.add_argument("--manifest-out",
                    default="artifacts/availability/AVAILABILITY_MANIFEST.json")
    ap.add_argument("--audit-out",
                    default="artifacts/path_a/AVAILABILITY_COLLECTION_AUDIT.json")
    ap.add_argument("--snapshot-manifest-out",
                    default="artifacts/path_a/FORWARD_SNAPSHOT_MANIFEST.json")
    ap.add_argument("--prediction-cutoff", default=None,
                    help="Prediction cutoff UTC ISO (defaults to pull time = pregame).")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    ingestion_ts = now.isoformat()
    prediction_cutoff = args.prediction_cutoff or ingestion_ts
    manifest: dict = {
        "date": args.date,
        "pulled_at_utc": ingestion_ts,
        "n_games": 0,
        "teams_playing": [],
        "n_injury_rows": 0,
        "n_out_players": 0,
        "snapshot_path": None,
        "games_snapshot_path": None,
        "status": "ok",
        "fail_open": True,
    }

    def _write_manifest() -> None:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, default=str))

    def _write_audit(games_result, injuries_result, coverage, snapshot_paths, snap_hash):
        audit = build_availability_audit(
            date=args.date,
            source_timestamp_utc=ingestion_ts,
            ingestion_timestamp_utc=ingestion_ts,
            prediction_cutoff_utc=prediction_cutoff,
            games_result=games_result,
            injuries_result=injuries_result,
            coverage=coverage,
            snapshot_paths=snapshot_paths,
            snapshot_payload_hash=snap_hash,
        )
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_out).write_text(json.dumps(audit, indent=2, default=str))
        return audit

    try:
        client = BDLClient()
    except BDLAPIError as exc:
        # No key is an EXPLICIT auth failure — not "successful empty data".
        manifest["status"] = "no_bdl_api_key"
        manifest["error"] = str(exc)
        _write_manifest()
        fail = {"status": EndpointStatus.DOCUMENTED_AUTH_FAILED, "success": False,
                "n_rows": 0, "error": str(exc)[:300]}
        _write_audit(fail, fail, compute_coverage([], []), {}, None)
        print(f"[availability] no BDL_API_KEY; explicit auth-failure audit at {args.audit_out}")
        return

    # 1) Tonight's games (which teams are active). BDL files games under their UTC date,
    #    so a full Eastern evening slate spans two UTC dates — pull both and window-filter.
    next_day = (datetime.strptime(args.date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    games_error = None
    try:
        games = client.list_endpoint("games", {"dates": [args.date, next_day]})
    except BDLAPIError as exc:
        games = []
        games_error = str(exc)
        manifest["games_error"] = str(exc)[:200]
    teams = _teams_playing(games, args.date)
    manifest["n_games"] = len(teams)
    manifest["teams_playing"] = teams
    games_result = classify_endpoint_result(len(teams), games_error)

    # 2) Current injuries (the availability signal). Endpoint is current-state only.
    injuries_error = None
    try:
        inj_rows = client.list_endpoint("player_injuries")
    except BDLAPIError as exc:
        inj_rows = []
        injuries_error = str(exc)
        manifest["injuries_error"] = str(exc)[:200]
    inj = normalize_injuries(inj_rows)
    manifest["n_injury_rows"] = len(inj)
    injuries_result = classify_endpoint_result(len(inj), injuries_error)

    # 3) Build append-only snapshot rows (AS PULLED NOW; no postgame reconstruction).
    out_dir = Path(args.out_dir) / f"snapshot_date_utc={args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rows: list[dict] = []
    snapshot_paths: dict = {}
    if len(inj):
        active_team_ids = {t["home_team_id"] for t in teams} | {t["away_team_id"] for t in teams}
        for _, r in inj.iterrows():
            status_norm = str(r.get("injury_status_normalized") or "").lower()
            payload = {
                "date": args.date,
                "player_id": r.get("player_id"),
                "player_name": r.get("player_name"),
                "team_id": r.get("team_id"),
                "status_raw": r.get("injury_status"),
                "status_normalized": status_norm,
                "description": r.get("injury_description"),
                "pulled_at_utc": ingestion_ts,
            }
            row = dict(payload)
            row["is_out"] = status_norm in {"out", "inactive"}
            # Only mark "plays_today" when we could confirm the team is on tonight's slate.
            row["team_plays_today"] = (
                (r.get("team_id") in active_team_ids) if active_team_ids else None
            )
            row["payload_sha256"] = _payload_hash(payload)
            snapshot_rows.append(row)
        snap = pd.DataFrame(snapshot_rows)
        manifest["n_out_players"] = int(snap["is_out"].sum())
        snap_path = out_dir / f"injuries_{stamp}.parquet"
        assert_no_snapshot_overwrite(snap_path)   # append-only: never clobber
        snap.to_parquet(snap_path, index=False)
        manifest["snapshot_path"] = str(snap_path)
        snapshot_paths["injuries"] = str(snap_path)

    if teams:
        games_path = out_dir / f"games_{stamp}.parquet"
        assert_no_snapshot_overwrite(games_path)
        pd.DataFrame(teams).to_parquet(games_path, index=False)
        manifest["games_snapshot_path"] = str(games_path)
        snapshot_paths["games"] = str(games_path)

    coverage = compute_coverage(snapshot_rows, teams)
    snap_hash = payload_hash({"date": args.date, "rows": snapshot_rows, "teams": teams}) \
        if (snapshot_rows or teams) else None

    audit = _write_audit(games_result, injuries_result, coverage, snapshot_paths, snap_hash)
    manifest["status"] = audit["overall_status"]

    # 4) Forward snapshot manifest — strictly append-only (never overwrite prior slates).
    if snapshot_paths:
        append_snapshot_manifest(args.snapshot_manifest_out, {
            "date": args.date,
            "source_timestamp_utc": ingestion_ts,
            "ingestion_timestamp_utc": ingestion_ts,
            "prediction_cutoff_utc": prediction_cutoff,
            "snapshot_payload_hash": snap_hash,
            "snapshot_paths": snapshot_paths,
            "coverage": coverage,
            "overall_status": audit["overall_status"],
        })

    _write_manifest()
    print(json.dumps({
        "date": args.date,
        "n_games": manifest["n_games"],
        "n_injury_rows": manifest["n_injury_rows"],
        "n_out_players": manifest["n_out_players"],
        "games_status": games_result["status"],
        "injuries_status": injuries_result["status"],
        "overall_status": audit["overall_status"],
        "audit": args.audit_out,
        "snapshot_manifest": args.snapshot_manifest_out,
    }, indent=2))


if __name__ == "__main__":
    main()
