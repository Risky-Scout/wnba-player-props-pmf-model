"""Path A availability-collection audit + append-only forward-snapshot manifest.

Requirement 10: pregame availability snapshots are FORWARD-accrued and append-only. This
module provides the structured-failure classification and the append-only manifest writer
used by ``scripts/collect_availability.py`` so that:

  * a 403 / auth failure / empty response / unresolved identity is recorded with an EXPLICIT
    structured reason — never as "successful empty data";
  * every snapshot entry records source timestamp, ingestion timestamp, prediction cutoff,
    payload hash, and coverage;
  * the manifest is strictly append-only — an earlier snapshot entry is NEVER overwritten or
    removed, and writing refuses to clobber an existing snapshot file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wnba_props_model.data.bdl_client import EndpointStatus, classify_bdl_error


def classify_endpoint_result(n_rows: int, error: str | None) -> dict:
    """Classify one endpoint pull into a structured status.

    An error string is classified via ``classify_bdl_error`` (auth/bad-request/etc.). A
    clean pull with zero rows is DOCUMENTED_EMPTY and is explicitly NOT treated as a
    successful data collection (``success=False``) so downstream code can distinguish
    "genuinely nothing" from "we failed to collect".
    """
    if error:
        status = classify_bdl_error(error)
        return {"status": status, "success": False, "n_rows": int(n_rows), "error": error[:300]}
    if n_rows <= 0:
        return {"status": EndpointStatus.DOCUMENTED_EMPTY, "success": False, "n_rows": 0,
                "error": None}
    return {"status": EndpointStatus.DOCUMENTED_SUCCESS, "success": True,
            "n_rows": int(n_rows), "error": None}


def payload_hash(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compute_coverage(injury_rows: list[dict], teams_playing: list[dict]) -> dict:
    """Coverage of the pregame availability signal.

    Records identity resolution coverage (fraction of injury rows carrying a canonical
    player_id) and slate-linkage coverage (fraction of OUT players whose team could be
    matched to tonight's slate).
    """
    n = len(injury_rows)
    n_with_pid = sum(1 for r in injury_rows if r.get("player_id") is not None)
    out_rows = [r for r in injury_rows if r.get("is_out")]
    n_out = len(out_rows)
    n_out_slate_known = sum(1 for r in out_rows if r.get("team_plays_today") is not None)
    return {
        "n_injury_rows": n,
        "n_with_canonical_player_id": n_with_pid,
        "identity_resolution_rate": (round(n_with_pid / n, 4) if n else None),
        "n_out_players": n_out,
        "n_out_slate_linkage_known": n_out_slate_known,
        "slate_linkage_rate": (round(n_out_slate_known / n_out, 4) if n_out else None),
        "n_teams_playing": len(teams_playing),
    }


def build_availability_audit(
    *,
    date: str,
    source_timestamp_utc: str | None,
    ingestion_timestamp_utc: str,
    prediction_cutoff_utc: str,
    games_result: dict,
    injuries_result: dict,
    coverage: dict,
    snapshot_paths: dict,
    snapshot_payload_hash: str | None,
) -> dict:
    """Assemble AVAILABILITY_COLLECTION_AUDIT.json.

    ``overall_status`` is 'ok' only when BOTH endpoints succeeded with data; otherwise it
    carries the explicit failure/empty reasons (never a silent success).
    """
    failures = []
    for name, res in (("games", games_result), ("injuries", injuries_result)):
        if not res.get("success"):
            failures.append({"endpoint": name, "status": res.get("status"),
                             "error": res.get("error")})
    overall = "ok" if not failures else "degraded_or_failed"
    return {
        "schema_version": "path_a_availability_audit_v1",
        "path": "A",
        "date": date,
        "source_timestamp_utc": source_timestamp_utc,
        "ingestion_timestamp_utc": ingestion_timestamp_utc,
        "prediction_cutoff_utc": prediction_cutoff_utc,
        "forward_collection": True,
        "append_only": True,
        "endpoints": {"games": games_result, "injuries": injuries_result},
        "coverage": coverage,
        "snapshot_paths": snapshot_paths,
        "snapshot_payload_hash": snapshot_payload_hash,
        "overall_status": overall,
        "failure_reasons": failures,
        "note": (
            "Forward pregame availability accrual. A 403/auth/empty/unresolved result is "
            "recorded as an explicit structured failure — never as successful empty data."
        ),
    }


class SnapshotOverwriteError(RuntimeError):
    """Raised when an append-only snapshot write would clobber an existing file/entry."""


def append_snapshot_manifest(manifest_path: str | Path, entry: dict) -> dict:
    """Append one snapshot entry to the FORWARD_SNAPSHOT_MANIFEST.json (append-only).

    Never removes or mutates existing entries. If an entry with the same
    ``snapshot_payload_hash`` already exists, the new entry is skipped (idempotent) and the
    existing manifest is returned unchanged. Returns the full manifest dict.
    """
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            manifest = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            raise SnapshotOverwriteError(
                f"existing manifest {p} is unreadable; refusing to overwrite append-only history"
            )
        if not isinstance(manifest, dict) or "snapshots" not in manifest:
            raise SnapshotOverwriteError(
                f"existing manifest {p} has unexpected shape; refusing to overwrite"
            )
    else:
        manifest = {
            "schema_version": "path_a_forward_snapshot_manifest_v1",
            "append_only": True,
            "snapshots": [],
        }

    existing_hashes = {s.get("snapshot_payload_hash") for s in manifest["snapshots"]}
    h = entry.get("snapshot_payload_hash")
    if h is not None and h in existing_hashes:
        # Idempotent: identical snapshot already recorded — do not duplicate, do not overwrite.
        manifest["last_write_skipped_duplicate"] = True
    else:
        manifest["snapshots"].append(entry)
        manifest["last_write_skipped_duplicate"] = False
    manifest["n_snapshots"] = len(manifest["snapshots"])
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def assert_no_snapshot_overwrite(snapshot_path: str | Path) -> None:
    """Refuse to overwrite an existing snapshot data file (append-only discipline)."""
    if Path(snapshot_path).exists():
        raise SnapshotOverwriteError(
            f"snapshot {snapshot_path} already exists — forward snapshots are append-only "
            "and must never be overwritten"
        )
