"""Blocking tests for CLV tracking correctness (Plan P1.5).

Guards the audited defect: the CLV gate must use TRUE signed closing-line value
(price_clv / line_clv, which can be negative), never max(|edge_over|,|edge_under|)
(nonnegative by construction, which makes the gate pass trivially).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "verify_gates.py"


def _run_gate(parquet: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "clv-tracking", str(parquet), *extra],
        capture_output=True, text=True, cwd=str(REPO))


def _write(tmp_path, name, df) -> Path:
    p = tmp_path / name
    df.to_parquet(p, index=False)
    return p


def _dates(n, seed):
    r = np.random.default_rng(seed)
    return pd.to_datetime("2026-07-01") + pd.to_timedelta(r.integers(0, 25, n), "D")


def test_gate_fail_closed_without_signed_clv(tmp_path):
    # Only edge columns present (no price_clv/line_clv) -> gate must SKIP fail-closed,
    # never fall back to a nonnegative max-abs-edge proxy.
    n = 900
    r = np.random.default_rng(1)
    df = pd.DataFrame({
        "game_date": _dates(n, 1), "stat": r.choice(["pts", "reb", "ast"], n),
        "edge_over": r.normal(0, 0.05, n), "edge_under": r.normal(0, 0.05, n),
    })
    res = _run_gate(_write(tmp_path, "no_clv.parquet", df))
    assert res.returncode == 0
    out = (res.stdout + res.stderr).lower()
    assert "fail-closed" in out or "skipping gate" in out
    assert "gate ✓ pass" not in out and "all eligible stats pass" not in out


def test_gate_uses_signed_clv_and_can_hard_fail(tmp_path):
    # Signed price_clv with negative mean + >=300 rows/stat -> HARD FAIL (exit 1).
    # The old max-abs-edge proxy could never produce this.
    n = 1500
    r = np.random.default_rng(2)
    df = pd.DataFrame({
        "game_date": _dates(n, 2), "stat": r.choice(["pts", "reb", "ast"], n),
        "edge_over": r.normal(0, 0.05, n), "edge_under": r.normal(0, 0.05, n),
        "price_clv": r.normal(-0.02, 0.03, n),   # market moves against us on average
    })
    res = _run_gate(_write(tmp_path, "neg_clv.parquet", df), "--min-rows-per-stat", "100")
    assert res.returncode == 1
    assert "GATE FAIL" in (res.stdout + res.stderr)


def test_gate_passes_on_positive_signed_clv(tmp_path):
    n = 1500
    r = np.random.default_rng(3)
    df = pd.DataFrame({
        "game_date": _dates(n, 3), "stat": r.choice(["pts", "reb", "ast"], n),
        "price_clv": r.normal(0.03, 0.02, n),    # strongly positive CLV
    })
    res = _run_gate(_write(tmp_path, "pos_clv.parquet", df), "--min-rows-per-stat", "100")
    assert res.returncode == 0
    assert "pass" in (res.stdout + res.stderr).lower()


def test_no_max_abs_edge_clv_regression():
    # The nonnegative-by-construction CLV must never be reintroduced.
    src = GATE.read_text()
    assert '["edge_over", "edge_under"]].abs().max' not in src
    assert "price_clv" in src and "line_clv" in src


def test_score_daily_predictions_clv_is_signed_and_outcome_independent():
    # Contract check on the scorer's CLV construction: signed close-minus-open for the
    # selected side, and a fail-closed 1:1 closing-line join.
    src = (REPO / "scripts" / "score_daily_predictions.py").read_text()
    assert "close_side - open_side" in src            # signed price_clv
    assert "duplicated(subset=" in src                # fail-closed on many-to-many join


def test_report_date_cluster_ci_can_be_negative():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("gcr", REPO / "scripts" / "generate_clv_report.py")
    gcr = module_from_spec(spec); spec.loader.exec_module(gcr)
    r = np.random.default_rng(4); n = 600
    df = pd.DataFrame({"game_date": _dates(n, 4), "price_clv": r.normal(-0.02, 0.03, n)})
    lo, hi = gcr._date_cluster_ci(df, "price_clv", "game_date", n_boot=500)
    assert lo == lo and hi == hi and lo < hi        # finite, ordered
    assert lo < 0                                    # negative CLV is representable
    # single cluster -> undefined CI (not fabricated)
    one = pd.DataFrame({"game_date": ["2026-07-01"] * 10, "price_clv": [0.01] * 10})
    lo1, hi1 = gcr._date_cluster_ci(one, "price_clv", "game_date")
    assert lo1 != lo1 and hi1 != hi1                 # NaN
