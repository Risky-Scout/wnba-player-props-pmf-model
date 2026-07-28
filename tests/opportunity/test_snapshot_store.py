"""Append-only snapshot storage tests for Opportunity V2."""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.opportunity.snapshot_store import (
    SnapshotContractError,
    append_snapshot_partition,
    canonicalize_utc,
    payload_sha256,
)


def _row(snapshot_id, status="out", pulled="2026-05-08T22:00:00Z"):
    return {
        "snapshot_id": snapshot_id,
        "source": "unit",
        "pulled_at_utc": pulled,
        "available_at_utc": pulled,
        "snapshot_date_utc": "2026-05-08",
        "player_id": 1,
        "team_id": 10,
        "status_raw": status,
        "status_normalized": status,
        "payload_sha256": payload_sha256({"snapshot_id": snapshot_id, "status": status}),
    }


REQ = ("snapshot_id", "source", "pulled_at_utc", "available_at_utc", "snapshot_date_utc",
       "player_id", "team_id", "status_raw", "status_normalized", "payload_sha256")


def test_canonicalize_utc_rejects_unparseable():
    good = canonicalize_utc(pd.Series(["2026-05-08T22:00:00Z", None]))
    assert str(good.dtype).endswith("UTC]")
    with pytest.raises(SnapshotContractError):
        canonicalize_utc(pd.Series(["not-a-timestamp"]))


def test_payload_sha256_is_deterministic_and_order_independent():
    a = payload_sha256({"x": 1, "y": 2})
    b = payload_sha256({"y": 2, "x": 1})
    assert a == b
    assert a != payload_sha256({"x": 1, "y": 3})


def test_append_then_read_roundtrip(tmp_path):
    frame = pd.DataFrame([_row("s1"), _row("s2")])
    written = append_snapshot_partition(frame, tmp_path, REQ, ("snapshot_id",))
    assert len(written) == 1
    got = pd.read_parquet(written[0])
    assert set(got["snapshot_id"]) == {"s1", "s2"}


def test_exact_duplicate_is_deduped_not_appended(tmp_path):
    append_snapshot_partition(pd.DataFrame([_row("s1")]), tmp_path, REQ, ("snapshot_id",))
    append_snapshot_partition(pd.DataFrame([_row("s1")]), tmp_path, REQ, ("snapshot_id",))
    got = pd.read_parquet(tmp_path / "snapshot_date_utc=2026-05-08" / "source=unit" / "snapshots.parquet")
    assert len(got) == 1  # identical identity + payload -> not duplicated


def test_conflicting_payload_same_identity_raises(tmp_path):
    append_snapshot_partition(pd.DataFrame([_row("s1", status="out")]), tmp_path, REQ, ("snapshot_id",))
    with pytest.raises(SnapshotContractError):
        # same snapshot_id, DIFFERENT payload -> immutable violation
        append_snapshot_partition(pd.DataFrame([_row("s1", status="available")]), tmp_path, REQ,
                                  ("snapshot_id",))


def test_new_identity_appends(tmp_path):
    append_snapshot_partition(pd.DataFrame([_row("s1")]), tmp_path, REQ, ("snapshot_id",))
    append_snapshot_partition(pd.DataFrame([_row("s2")]), tmp_path, REQ, ("snapshot_id",))
    got = pd.read_parquet(tmp_path / "snapshot_date_utc=2026-05-08" / "source=unit" / "snapshots.parquet")
    assert set(got["snapshot_id"]) == {"s1", "s2"}


def test_input_with_conflicting_duplicate_identity_raises(tmp_path):
    frame = pd.DataFrame([_row("s1", status="out"), _row("s1", status="available")])
    with pytest.raises(SnapshotContractError):
        append_snapshot_partition(frame, tmp_path, REQ, ("snapshot_id",))


def test_missing_required_column_raises(tmp_path):
    frame = pd.DataFrame([_row("s1")]).drop(columns=["status_raw"])
    with pytest.raises(SnapshotContractError):
        append_snapshot_partition(frame, tmp_path, REQ, ("snapshot_id",))
