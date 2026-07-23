"""Foundation Lock tests for the surrogate feature-subset ablation runner.

Guards the corrected, FAIL-CLOSED contract (development-only exploration, no holdout scored):
  * chronology  - folds are strictly increasing; train_end < valid_end.
  * determinism - identical inputs -> byte-identical verdict.json and manifest (ex-timestamps).
  * feature-map consumption - each candidate uses exactly its mapped feature list.
  * no-holdout-leakage - reserved tail dates never appear in any scored fold.
  * fail-closed - raises on missing features, missing G0, missing prop map, maps-hash
    mismatch, and (for real runs) an unpinned maps hash.
  * scoping - artifacts are surrogate/dev-only and NOT promotion-eligible; the runner source
    contains no "holdout scored" claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "run_prop_ablation.py"
RESERVED = 20


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Small synthetic feature matrix (pts only) + maps with G0 + a strict subset."""
    d = tmp_path_factory.mktemp("abl")
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    rows = []
    for dt in dates:
        for _ in range(15):
            f = rng.normal(0, 1, 6)
            mu = np.clip(8 + 2 * f[0] + f[1], 0.5, None)
            rows.append({
                "game_date": dt, "player_id": int(rng.integers(1, 40)),
                "game_id": int(rng.integers(1, 100000)),
                "f_a": f[0], "f_b": f[1], "f_c": f[2], "f_d": f[3], "f_e": f[4], "f_f": f[5],
                "actual_pts": int(rng.poisson(mu)),
            })
    df = pd.DataFrame(rows)
    feats = d / "features.parquet"
    df.to_parquet(feats, index=False)

    maps = {"schema_version": 1, "candidates": {
        "G0":  {"pts": ["f_a", "f_b", "f_c", "f_d", "f_e", "f_f"]},
        "SUB": {"pts": ["f_a", "f_b", "f_c", "f_d"]},
    }}
    maps_p = d / "maps.json"
    maps_p.write_text(json.dumps(maps))
    return d, feats, maps_p, dates


def _run(outdir, feats, maps_p, *extra, check=True):
    r = subprocess.run(
        [sys.executable, str(RUNNER), "--features", str(feats), "--maps", str(maps_p),
         "--out-dir", str(outdir), "--reserved-tail-dates", str(RESERVED),
         "--no-require-maps-hash", *extra],
        capture_output=True, text=True, cwd=str(REPO))
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
        return json.loads((Path(outdir) / "ablation_verdict.json").read_text())
    return r


def test_chronology_and_scoping(synthetic, tmp_path):
    _, feats, maps_p, _ = synthetic
    v = _run(tmp_path / "o1", feats, maps_p)
    assert v["classification"] == "surrogate feature-subset ablation"
    assert v["promotion_eligible"] is False
    assert v["reserved_tail_dates"] == RESERVED
    folds = v["fold_dates"]
    assert len(folds) >= 2
    prev = None
    for fd in folds:
        assert fd["train_end"] < fd["valid_end"]
        if prev is not None:
            assert fd["train_end"] >= prev
        prev = fd["valid_end"]


def test_determinism_excluding_timestamps(synthetic, tmp_path):
    _, feats, maps_p, _ = synthetic
    _run(tmp_path / "a", feats, maps_p)
    _run(tmp_path / "b", feats, maps_p)
    v1 = json.loads((tmp_path / "a" / "ablation_verdict.json").read_text())
    v2 = json.loads((tmp_path / "b" / "ablation_verdict.json").read_text())
    assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)
    # Manifest determinism excluding generated timestamps.
    m1 = json.loads((tmp_path / "a" / "RUN_MANIFEST.json").read_text())
    m2 = json.loads((tmp_path / "b" / "RUN_MANIFEST.json").read_text())
    for m in (m1, m2):
        m.pop("created_utc", None)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


def test_feature_map_consumption(synthetic, tmp_path):
    _, feats, maps_p, _ = synthetic
    v = _run(tmp_path / "o2", feats, maps_p)
    res = v["results"]["pts"]
    assert res["all"]["G0"]["n_features"] == 6
    assert res["all"]["SUB"]["n_features"] == 4


def test_no_holdout_leakage(synthetic, tmp_path):
    _, feats, maps_p, dates = synthetic
    v = _run(tmp_path / "o3", feats, maps_p)
    first_reserved = str(pd.Timestamp(sorted(dates)[-RESERVED]).date())
    for fd in v["fold_dates"]:
        assert fd["valid_end"] <= first_reserved


def test_fail_closed_missing_feature(synthetic, tmp_path):
    d, feats, _, _ = synthetic
    bad = tmp_path / "maps_missing_feat.json"
    bad.write_text(json.dumps({"candidates": {
        "G0": {"pts": ["f_a", "f_b", "f_c", "f_d", "__absent_feature__"]}}}))
    r = _run(tmp_path / "x", feats, bad, check=False)
    assert r.returncode == 1
    assert "absent from the input" in (r.stdout + r.stderr)


def test_fail_closed_missing_g0(synthetic, tmp_path):
    _, feats, _, _ = synthetic
    bad = tmp_path / "maps_no_g0.json"
    bad.write_text(json.dumps({"candidates": {"SUB": {"pts": ["f_a", "f_b", "f_c", "f_d"]}}}))
    r = _run(tmp_path / "x", feats, bad, check=False)
    assert r.returncode == 1
    assert "G0 candidate is missing" in (r.stdout + r.stderr)


def test_fail_closed_missing_prop_map(synthetic, tmp_path):
    _, feats, _, _ = synthetic
    bad = tmp_path / "maps_missing_prop.json"
    bad.write_text(json.dumps({"candidates": {
        "G0":  {"pts": ["f_a", "f_b", "f_c", "f_d"], "reb": ["f_a", "f_b", "f_c", "f_d"]},
        "SUB": {"pts": ["f_a", "f_b", "f_c", "f_d"]}}}))  # SUB missing reb
    r = _run(tmp_path / "x", feats, bad, check=False)
    assert r.returncode == 1
    assert "missing prop map" in (r.stdout + r.stderr)


def test_fail_closed_maps_hash_mismatch(synthetic, tmp_path):
    _, feats, maps_p, _ = synthetic
    r = subprocess.run(
        [sys.executable, str(RUNNER), "--features", str(feats), "--maps", str(maps_p),
         "--out-dir", str(tmp_path / "x"), "--reserved-tail-dates", str(RESERVED),
         "--expected-maps-sha256", "0" * 64],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1
    assert "maps hash mismatch" in (r.stdout + r.stderr)


def test_fail_closed_requires_pinned_maps_hash_for_real_runs(synthetic, tmp_path):
    _, feats, maps_p, _ = synthetic
    # Without --no-require-maps-hash and without an expected hash -> fail closed.
    r = subprocess.run(
        [sys.executable, str(RUNNER), "--features", str(feats), "--maps", str(maps_p),
         "--out-dir", str(tmp_path / "x"), "--reserved-tail-dates", str(RESERVED)],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 1
    assert "lacks a pinned feature-map hash" in (r.stdout + r.stderr)


def test_runner_source_has_no_holdout_scored_claim():
    src = RUNNER.read_text()
    assert "no holdout scored" in src
    assert "surrogate feature-subset ablation" in src
    assert "holdout dates are scored" not in src
    assert "scored once for the winner" not in src
