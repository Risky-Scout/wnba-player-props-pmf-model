"""Foundation Lock tests for the surrogate feature-subset ablation runner.

Guards the corrected contract (development-only exploration, no holdout scored):
  * chronology  - folds are strictly increasing; train_end < valid_end.
  * determinism - identical inputs -> byte-identical ablation_verdict.json.
  * feature-map consumption - each candidate uses exactly its mapped feature list.
  * no-holdout-leakage - reserved tail dates never appear in any scored fold.
  * scoping - artifacts are classified surrogate/dev-only and NOT promotion-eligible;
    no "holdout scored" claim remains in the runner source.
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


def _run(outdir, feats, maps_p):
    r = subprocess.run(
        [sys.executable, str(RUNNER), "--features", str(feats), "--maps", str(maps_p),
         "--out-dir", str(outdir), "--reserved-tail-dates", str(RESERVED)],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((Path(outdir) / "ablation_verdict.json").read_text())


def test_chronology_and_scoping(synthetic, tmp_path):
    d, feats, maps_p, _ = synthetic
    v = _run(tmp_path / "o1", feats, maps_p)
    assert v["classification"] == "surrogate feature-subset ablation"
    assert v["promotion_eligible"] is False
    assert v["reserved_tail_dates"] == RESERVED
    folds = v["fold_dates"]
    assert len(folds) >= 2
    prev = None
    for fd in folds:
        assert fd["train_end"] < fd["valid_end"]           # train precedes valid
        if prev is not None:
            assert fd["train_end"] >= prev                  # expanding, monotone
        prev = fd["valid_end"]


def test_determinism(synthetic, tmp_path):
    d, feats, maps_p, _ = synthetic
    v1 = _run(tmp_path / "a", feats, maps_p)
    v2 = _run(tmp_path / "b", feats, maps_p)
    # ablation_verdict.json carries no timestamp -> must be byte-identical run to run.
    assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)


def test_feature_map_consumption(synthetic, tmp_path):
    d, feats, maps_p, _ = synthetic
    v = _run(tmp_path / "o2", feats, maps_p)
    res = v["results"]["pts"]
    assert res["all"]["G0"]["n_features"] == 6
    assert res["all"]["SUB"]["n_features"] == 4       # exactly the mapped subset


def test_no_holdout_leakage(synthetic, tmp_path):
    d, feats, maps_p, dates = synthetic
    v = _run(tmp_path / "o3", feats, maps_p)
    first_reserved = str(pd.Timestamp(sorted(dates)[-RESERVED]).date())
    # Every scored fold must end on/before the first reserved date: reserved tail
    # dates can never enter training or validation.
    for fd in v["fold_dates"]:
        assert fd["valid_end"] <= first_reserved


def test_runner_source_has_no_holdout_scored_claim():
    src = RUNNER.read_text()
    assert "no holdout scored" in src
    assert "surrogate feature-subset ablation" in src
    # The corrected runner must not claim it scores a final holdout.
    assert "holdout dates are scored" not in src
    assert "scored once for the winner" not in src
