#!/usr/bin/env python3
"""Append-only prospective roster + injury snapshot collector.

Captures for each forecast timestamp:
  - /wnba/v1/players/active
  - scheduled-team player cross-check
  - /wnba/v1/player_injuries
  - source health, ingestion timestamp, raw payload hash

Source outcomes are classified explicitly. Endpoint failure is never treated
as a healthy empty injury/roster table.

When scheduled games exist:
  - roster-source failure fails the workflow (exit non-zero);
  - injury-source failure fails the availability gate (exit non-zero);
  - diagnostic artifacts may still be uploaded.

Raw payloads are written under gitignored data/ (or uploaded as workflow
artifacts). Never commit private snapshots to git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient

SUCCESS_POPULATED = "SUCCESS_POPULATED"
SUCCESS_EMPTY = "SUCCESS_EMPTY"
AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
ENTITLEMENT_FAILURE = "ENTITLEMENT_FAILURE"
RATE_LIMITED = "RATE_LIMITED"
ENDPOINT_FAILURE = "ENDPOINT_FAILURE"
PARSE_FAILURE = "PARSE_FAILURE"

FAILURE_STATUSES = frozenset(
    {
        AUTHENTICATION_FAILURE,
        ENTITLEMENT_FAILURE,
        RATE_LIMITED,
        ENDPOINT_FAILURE,
        PARSE_FAILURE,
    }
)

ROSTER_SOURCES = frozenset({"players_active", "scheduled_team_player_crosscheck"})
INJURY_SOURCES = frozenset({"player_injuries"})


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


def classify_bdl_error(exc: BaseException) -> str:
    """Map a BDL/client exception to an explicit source-health status."""
    msg = str(exc)
    m = msg.lower()
    code_match = re.search(r"\b([45]\d{2})\b", msg)
    code = int(code_match.group(1)) if code_match else None
    if (
        code == 401
        or "api key" in m
        or "bdl_api_key" in m
        or "unauthorized" in m
        or "authentication" in m
        or "is required" in m
    ):
        return AUTHENTICATION_FAILURE
    if code == 403 or "forbidden" in m or "entitlement" in m or "upgrade" in m:
        return ENTITLEMENT_FAILURE
    if code == 429 or "rate limit" in m or "too many requests" in m:
        return RATE_LIMITED
    if isinstance(exc, (json.JSONDecodeError, TypeError, ValueError, KeyError)):
        return PARSE_FAILURE
    if "parse" in m or "json" in m:
        return PARSE_FAILURE
    return ENDPOINT_FAILURE


def classify_payload(payload: Any, *, status_hint: str | None = None) -> str:
    if status_hint in FAILURE_STATUSES:
        return status_hint
    if isinstance(payload, list):
        return SUCCESS_POPULATED if len(payload) > 0 else SUCCESS_EMPTY
    if isinstance(payload, dict):
        # Cross-check payloads are dict wrappers.
        games = payload.get("games")
        players = payload.get("scheduled_team_players")
        if games is not None or players is not None:
            n_games = len(games) if isinstance(games, list) else 0
            n_players = len(players) if isinstance(players, list) else 0
            if n_games == 0 and n_players == 0:
                return SUCCESS_EMPTY
            return SUCCESS_POPULATED
        return SUCCESS_POPULATED if payload else SUCCESS_EMPTY
    return PARSE_FAILURE


def main() -> int:
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
    manifest: dict[str, Any] = {
        "forecast_timestamp": forecast_ts,
        "ingestion_timestamp_utc": now.isoformat(),
        "date": args.date,
        "status": "ok",
        "fail_open": False,
        "fail_closed": True,
        "sources": {},
        "snapshot_ids": [],
        "append_only": True,
        "private_payloads_committed_to_git": False,
        "scheduled_games_exist": False,
        "availability_gate": "pending",
        "workflow_gate": "pending",
    }

    def _write_manifest() -> None:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    try:
        client = BDLClient()
    except BDLAPIError as exc:
        status = classify_bdl_error(exc)
        manifest["status"] = status
        manifest["error"] = str(exc)
        manifest["availability_gate"] = "FAILED_NO_CLIENT"
        manifest["workflow_gate"] = "FAILED_NO_CLIENT"
        for src in ("players_active", "player_injuries", "scheduled_team_player_crosscheck"):
            manifest["sources"][src] = {
                "status": status,
                "error": str(exc)[:200],
                "payload_hash": None,
                "path": None,
                "n_rows": None,
                "healthy_empty": False,
            }
        _write_manifest()
        print(
            f"[roster_injury] client init failed ({status}); fail-closed at {args.manifest_out}",
            file=sys.stderr,
        )
        return 2

    next_day = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    out_dir = Path(args.out_dir) / f"snapshot_date_utc={args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _capture(source: str, fetch: Callable[[], Any]) -> str:
        payload: Any = None
        status_hint: str | None = None
        err: str | None = None
        try:
            payload = fetch()
        except BDLAPIError as exc:
            status_hint = classify_bdl_error(exc)
            err = str(exc)[:200]
            payload = {"_error": err, "_status": status_hint}
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            status_hint = PARSE_FAILURE
            err = str(exc)[:200]
            payload = {"_error": err, "_status": status_hint}
        except OSError as exc:
            status_hint = ENDPOINT_FAILURE
            err = str(exc)[:200]
            payload = {"_error": err, "_status": status_hint}

        status = classify_payload(payload, status_hint=status_hint)
        # Never treat a failure payload as SUCCESS_EMPTY.
        if status_hint in FAILURE_STATUSES:
            status = status_hint
        healthy_empty = status == SUCCESS_EMPTY
        ph = _payload_hash(payload)
        snap_id = f"{forecast_ts}|{source}|{ph}"
        path = out_dir / f"{source}_{stamp}_{ph[:12]}.json"
        path.write_text(
            json.dumps(
                {
                    "snapshot_id": snap_id,
                    "forecast_timestamp": forecast_ts,
                    "ingestion_timestamp_utc": now.isoformat(),
                    "source": source,
                    "status": status,
                    "payload_hash": ph,
                    "n_rows": len(payload) if isinstance(payload, list) else None,
                    "payload": payload,
                },
                indent=2,
                default=str,
            )
        )
        manifest["sources"][source] = {
            "status": status,
            "error": err,
            "payload_hash": ph,
            "path": str(path),
            "n_rows": len(payload) if isinstance(payload, list) else None,
            "healthy_empty": healthy_empty,
        }
        manifest["snapshot_ids"].append(snap_id)
        return status

    _capture(
        "players_active",
        lambda: client.list_endpoint("players_active", {"per_page": 100}),
    )
    _capture(
        "player_injuries",
        lambda: client.list_endpoint("player_injuries"),
    )

    def _games_and_roster_crosscheck() -> dict[str, Any]:
        games = client.list_endpoint("games", {"dates": [args.date, next_day]})
        slate = [g for g in games if _in_et_slate_window(g.get("date"), args.date)]
        team_ids: set[Any] = set()
        for g in slate:
            home = g.get("home_team") or {}
            away = g.get("visitor_team") or g.get("away_team") or {}
            if home.get("id") is not None:
                team_ids.add(home["id"])
            if away.get("id") is not None:
                team_ids.add(away["id"])
        players: list[Any] = []
        team_fetch_errors: list[str] = []
        for tid in sorted(team_ids):
            try:
                players.extend(
                    client.list_endpoint("players", {"team_ids[]": tid, "per_page": 100})
                )
            except BDLAPIError as exc:
                team_fetch_errors.append(f"team_id={tid}:{classify_bdl_error(exc)}")
        if team_ids and not players and team_fetch_errors:
            raise BDLAPIError(
                f"scheduled team player fetch failed for all teams: {team_fetch_errors[:5]}"
            )
        return {
            "games": slate,
            "team_ids": sorted(team_ids),
            "scheduled_team_players": players,
            "team_fetch_errors": team_fetch_errors,
        }

    cross_status = _capture("scheduled_team_player_crosscheck", _games_and_roster_crosscheck)
    cross = manifest["sources"]["scheduled_team_player_crosscheck"]
    scheduled_games_exist = False
    if cross_status in (SUCCESS_POPULATED, SUCCESS_EMPTY):
        try:
            cross_payload = json.loads(Path(cross["path"]).read_text()).get("payload") or {}
            games = cross_payload.get("games") or []
            scheduled_games_exist = len(games) > 0
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            scheduled_games_exist = False
    manifest["scheduled_games_exist"] = scheduled_games_exist

    roster_failed = any(
        manifest["sources"].get(s, {}).get("status") in FAILURE_STATUSES for s in ROSTER_SOURCES
    )
    injury_failed = any(
        manifest["sources"].get(s, {}).get("status") in FAILURE_STATUSES for s in INJURY_SOURCES
    )

    if injury_failed:
        manifest["availability_gate"] = "FAILED_INJURY_SOURCE"
    elif manifest["sources"].get("player_injuries", {}).get("status") == SUCCESS_EMPTY:
        # Empty injury list on a successful call is valid (no listed injuries).
        manifest["availability_gate"] = "PASSED_EMPTY_INJURY_LIST"
    else:
        manifest["availability_gate"] = "PASSED"

    if roster_failed and scheduled_games_exist:
        manifest["workflow_gate"] = "FAILED_ROSTER_SOURCE"
        manifest["status"] = "roster_source_failure"
    elif injury_failed and scheduled_games_exist:
        manifest["workflow_gate"] = "FAILED_AVAILABILITY_GATE"
        manifest["status"] = "injury_source_failure"
    elif roster_failed or injury_failed:
        manifest["workflow_gate"] = "FAILED_SOURCE"
        manifest["status"] = "source_failure"
    else:
        manifest["workflow_gate"] = "PASSED"
        manifest["status"] = "ok"

    any_failure = roster_failed or injury_failed
    all_failed = all(v.get("status") in FAILURE_STATUSES for v in manifest["sources"].values())
    if all_failed:
        manifest["status"] = "all_sources_failed"
        manifest["source_health"] = "failed"
    elif any_failure:
        manifest["source_health"] = "failed"
    else:
        manifest["source_health"] = "ok"

    # Explicit: never claim healthy-empty on a failed source.
    for src, meta in manifest["sources"].items():
        if meta.get("status") in FAILURE_STATUSES:
            meta["healthy_empty"] = False
            meta["treated_as_all_healthy"] = False

    _write_manifest()
    summary = {
        "date": args.date,
        "status": manifest["status"],
        "source_health": manifest.get("source_health"),
        "scheduled_games_exist": scheduled_games_exist,
        "availability_gate": manifest["availability_gate"],
        "workflow_gate": manifest["workflow_gate"],
        "n_snapshots": len(manifest["snapshot_ids"]),
        "sources": {k: v.get("status") for k, v in manifest["sources"].items()},
    }
    print(json.dumps(summary, indent=2))

    # Fail closed when scheduled games exist and roster/injury sources failed.
    if scheduled_games_exist and (roster_failed or injury_failed):
        return 1
    if all_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
