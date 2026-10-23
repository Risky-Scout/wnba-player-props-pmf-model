"""Path A availability-collection audit tests (requirement 10).

Covers structured-failure classification (403/auth/empty/success are distinct — an empty
or failed pull is NEVER "successful empty data"), append-only forward-snapshot manifest
(no overwrite, idempotent by payload hash), the overwrite guard, and coverage.
"""
from __future__ import annotations

import json

import pytest

from wnba_props_model.data.availability_audit import (
    SnapshotOverwriteError,
    append_snapshot_manifest,
    assert_no_snapshot_overwrite,
    build_availability_audit,
    classify_endpoint_result,
    compute_coverage,
    payload_hash,
)
from wnba_props_model.data.bdl_client import EndpointStatus


# --------------------------------------------------------------------------- #
# Structured-failure classification
# --------------------------------------------------------------------------- #
def test_classify_success_with_rows():
    res = classify_endpoint_result(5, None)
    assert res["success"] is True
    assert res["status"] == EndpointStatus.DOCUMENTED_SUCCESS


def test_classify_empty_is_not_success():
    res = classify_endpoint_result(0, None)
    assert res["success"] is False
    assert res["status"] == EndpointStatus.DOCUMENTED_EMPTY


def test_classify_403_auth_failure_not_success():
    res = classify_endpoint_result(0, "403 Forbidden from ...")
    assert res["success"] is False
    assert res["status"] == EndpointStatus.DOCUMENTED_AUTH_FAILED


def test_classify_404_unavailable():
    res = classify_endpoint_result(0, "404 Not Found")
    assert res["status"] == EndpointStatus.DOCUMENTED_UNAVAILABLE
    assert res["success"] is False


def test_build_audit_marks_failure_when_endpoint_fails():
    ok = classify_endpoint_result(3, None)
    fail = classify_endpoint_result(0, "403 Forbidden")
    audit = build_availability_audit(
        date="2026-07-28", source_timestamp_utc="t", ingestion_timestamp_utc="t",
        prediction_cutoff_utc="t", games_result=ok, injuries_result=fail,
        coverage=compute_coverage([], []), snapshot_paths={}, snapshot_payload_hash=None,
    )
    assert audit["overall_status"] == "degraded_or_failed"
    assert any(f["endpoint"] == "injuries" for f in audit["failure_reasons"])
    assert audit["append_only"] is True and audit["forward_collection"] is True


def test_build_audit_ok_only_when_all_succeed():
    ok = classify_endpoint_result(3, None)
    audit = build_availability_audit(
        date="2026-07-28", source_timestamp_utc="t", ingestion_timestamp_utc="t",
        prediction_cutoff_utc="t", games_result=ok, injuries_result=ok,
        coverage=compute_coverage([], []), snapshot_paths={}, snapshot_payload_hash=None,
    )
    assert audit["overall_status"] == "ok"


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_coverage_identity_and_slate_rates():
    injury_rows = [
        {"player_id": 1, "is_out": True, "team_plays_today": True},
        {"player_id": None, "is_out": True, "team_plays_today": None},
        {"player_id": 3, "is_out": False, "team_plays_today": True},
    ]
    cov = compute_coverage(injury_rows, [{"game_id": 1}])
    assert cov["n_injury_rows"] == 3
    assert cov["n_with_canonical_player_id"] == 2
    assert cov["identity_resolution_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert cov["n_out_players"] == 2
    assert cov["slate_linkage_rate"] == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------- #
# Append-only forward snapshot manifest
# --------------------------------------------------------------------------- #
def test_manifest_is_append_only(tmp_path):
    mpath = tmp_path / "FORWARD_SNAPSHOT_MANIFEST.json"
    e1 = {"date": "2026-07-28", "snapshot_payload_hash": "h1", "snapshot_paths": {"x": "a"}}
    e2 = {"date": "2026-07-29", "snapshot_payload_hash": "h2", "snapshot_paths": {"x": "b"}}
    append_snapshot_manifest(mpath, e1)
    m = append_snapshot_manifest(mpath, e2)
    assert m["n_snapshots"] == 2
    hashes = [s["snapshot_payload_hash"] for s in m["snapshots"]]
    assert hashes == ["h1", "h2"]  # earlier entry preserved, never overwritten


def test_manifest_idempotent_on_duplicate_hash(tmp_path):
    mpath = tmp_path / "FORWARD_SNAPSHOT_MANIFEST.json"
    e1 = {"date": "2026-07-28", "snapshot_payload_hash": "h1", "snapshot_paths": {"x": "a"}}
    append_snapshot_manifest(mpath, e1)
    m = append_snapshot_manifest(mpath, dict(e1))  # same hash again
    assert m["n_snapshots"] == 1
    assert m["last_write_skipped_duplicate"] is True


def test_assert_no_snapshot_overwrite(tmp_path):
    p = tmp_path / "injuries.parquet"
    assert_no_snapshot_overwrite(p)  # does not exist -> ok
    p.write_text("x")
    with pytest.raises(SnapshotOverwriteError):
        assert_no_snapshot_overwrite(p)


def test_manifest_refuses_corrupt_existing(tmp_path):
    mpath = tmp_path / "FORWARD_SNAPSHOT_MANIFEST.json"
    mpath.write_text("{ not valid json")
    with pytest.raises(SnapshotOverwriteError):
        append_snapshot_manifest(mpath, {"snapshot_payload_hash": "h"})


def test_payload_hash_stable():
    a = payload_hash({"b": 1, "a": 2})
    b = payload_hash({"a": 2, "b": 1})
    assert a == b and len(a) == 64
