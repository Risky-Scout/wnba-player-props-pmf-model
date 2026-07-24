"""A8: crash-safe, partitioned, append-only storage with dedup + SHA-256 manifests."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from wnba_props_model.data.quote_store import (
    read_kind,
    verify_partition_manifest,
    write_partition,
)


def _df(ids):
    return pd.DataFrame({"quote_id": ids, "sportsbook": ["fanduel"] * len(ids), "line": [1.5] * len(ids)})


def test_write_partition_layout_and_manifest(tmp_path):
    m = write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["a", "b"]), id_col="quote_id")
    part = tmp_path / "raw" / "date=2026-07-20" / "run=run1" / "part.parquet"
    assert part.exists()
    assert m["rows"] == 2 and len(m["sha256"]) == 64
    # Two retained manifest locations.
    assert (tmp_path / "raw" / "date=2026-07-20" / "run=run1" / "manifest.json").exists()
    assert (tmp_path / "_manifests" / "raw-2026-07-20-run1.json").exists()
    # Atomic pointer.
    ptr = json.loads((tmp_path / "raw" / "LATEST.json").read_text())
    assert ptr["run_id"] == "run1" and ptr["sha256"] == m["sha256"]
    assert verify_partition_manifest(tmp_path, "raw", "2026-07-20", "run1") is True


def test_in_batch_dedup(tmp_path):
    m = write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["a", "a", "b"]), id_col="quote_id")
    assert m["n_incoming"] == 3 and m["n_after_batch_dedup"] == 2


def test_cross_partition_dedup(tmp_path):
    write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["a", "b"]), id_col="quote_id")
    m2 = write_partition(tmp_path, "raw", "2026-07-21", "run2", _df(["b", "c"]), id_col="quote_id")
    assert m2["n_after_cross_partition_dedup"] == 1     # only 'c' is new
    allrows = read_kind(tmp_path, "raw")
    assert sorted(allrows["quote_id"]) == ["a", "b", "c"]


def test_never_rewrites_historical_partition(tmp_path):
    write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["a"]), id_col="quote_id")
    with pytest.raises(FileExistsError):
        write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["z"]), id_col="quote_id")


def test_pairs_kind_uses_pairs_filename(tmp_path):
    write_partition(tmp_path, "pairs", "2026-07-20", "run1", _df(["a"]), id_col="quote_id")
    assert (tmp_path / "pairs" / "date=2026-07-20" / "run=run1" / "pairs.parquet").exists()


def test_manifest_hash_detects_tampering(tmp_path):
    write_partition(tmp_path, "raw", "2026-07-20", "run1", _df(["a", "b"]), id_col="quote_id")
    part = tmp_path / "raw" / "date=2026-07-20" / "run=run1" / "part.parquet"
    _df(["a", "b", "tampered"]).to_parquet(part, index=False)   # corrupt the published partition
    assert verify_partition_manifest(tmp_path, "raw", "2026-07-20", "run1") is False
