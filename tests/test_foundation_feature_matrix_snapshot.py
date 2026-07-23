"""Foundation Lock tests for the feature-matrix snapshot + drift verifier.

Locked behavior:
  * committed snapshot is self-consistent (schema_hash == hash(ordered_schema)) and pins the
    live feature-contract hash;
  * verifier detects drift against a live parquet;
  * verifier fails closed when the parquet is required but absent, and reports an explicit
    DEFERRED (not a silent skip) when the parquet is legitimately absent in a clean checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_feature_matrix_snapshot.py"
SNAPSHOT = REPO / "artifacts" / "foundation_lock" / "feature_matrix_snapshot_v1.json"


def _mod():
    spec = importlib.util.spec_from_file_location("vfms", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_committed_snapshot_self_consistent():
    snap = json.loads(SNAPSHOT.read_text())
    assert snap["schema_version"] == 1
    m = _mod()
    assert m._schema_hash(snap["ordered_schema"]) == snap["schema_hash"]
    assert snap["feature_contract_hash"] == m._contract_hash()
    assert snap["row_count"] > 0 and snap["column_count"] > 0
    assert snap["game_date_min"] <= snap["game_date_max"]


def _synthetic_parquet(path: Path, rows: int = 40) -> None:
    rng = np.random.default_rng(1)
    pd.DataFrame({
        "player_id": rng.integers(1, 10, rows),
        "game_id": np.arange(rows),
        "game_date": pd.date_range("2025-05-01", periods=rows, freq="D"),
        "actual_pts": rng.integers(0, 30, rows),
        "f_x": rng.normal(0, 1, rows),
    }).to_parquet(path, index=False)


def test_build_then_verify_roundtrip(tmp_path):
    pq = tmp_path / "m.parquet"
    _synthetic_parquet(pq)
    snap = tmp_path / "snap.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "build", "--parquet", str(pq),
                        "--out", str(snap)], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    r2 = subprocess.run([sys.executable, str(SCRIPT), "verify", "--parquet", str(pq),
                        "--snapshot", str(snap), "--require-parquet"],
                        capture_output=True, text=True, cwd=str(REPO))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "SNAPSHOT PASS" in r2.stdout


def test_verify_detects_drift(tmp_path):
    pq = tmp_path / "m.parquet"
    _synthetic_parquet(pq)
    snap = tmp_path / "snap.json"
    subprocess.run([sys.executable, str(SCRIPT), "build", "--parquet", str(pq),
                    "--out", str(snap)], capture_output=True, text=True, cwd=str(REPO))
    obj = json.loads(snap.read_text())
    obj["row_count"] = obj["row_count"] + 1  # pretend the matrix changed silently
    snap.write_text(json.dumps(obj))
    r = subprocess.run([sys.executable, str(SCRIPT), "verify", "--parquet", str(pq),
                        "--snapshot", str(snap), "--require-parquet"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1
    assert "drift" in (r.stdout + r.stderr).lower()


def test_missing_parquet_fail_closed_when_required(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "verify",
                        "--parquet", str(tmp_path / "nope.parquet"),
                        "--snapshot", str(SNAPSHOT), "--require-parquet"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1
    assert "SNAPSHOT FAIL" in (r.stdout + r.stderr)


def test_missing_parquet_reports_deferred_not_silent(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "verify",
                        "--parquet", str(tmp_path / "nope.parquet"),
                        "--snapshot", str(SNAPSHOT)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0
    assert "DEFERRED" in r.stdout
    assert "self-consistency PASSED" in r.stdout


def test_contract_hash_drift_detected(tmp_path):
    snap = tmp_path / "snap.json"
    obj = json.loads(SNAPSHOT.read_text())
    obj["feature_contract_hash"] = hashlib.sha256(b"different").hexdigest()
    snap.write_text(json.dumps(obj))
    r = subprocess.run([sys.executable, str(SCRIPT), "verify",
                        "--parquet", str(tmp_path / "nope.parquet"), "--snapshot", str(snap)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1
    assert "feature_contract_hash drift" in (r.stdout + r.stderr)
