"""P3 Defect 3 — corrected forecasting gate: randomized discrete PIT, two-sided
coverage (over- AND under-dispersion), midpoint-PIT NOT gated, pooled ECE NOT gated,
line-level calibration reported separately."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wnba_props_model.evaluation import forecasting as fc


def _pmf_json(arr):
    return json.dumps({str(i): float(v) for i, v in enumerate(arr) if v > 0})


def test_randomized_pit_deterministic_and_seed_varies():
    pmf = np.array([0.2, 0.5, 0.3])
    a = fc.randomized_pit(pmf, 1, "g|p|pts|m1")
    b = fc.randomized_pit(pmf, 1, "g|p|pts|m1")
    c = fc.randomized_pit(pmf, 1, "g|p|pts|m2")
    assert a == b            # deterministic for identical key
    assert a != c            # varies with the seed key (not degenerate)
    assert 0.2 <= a <= 0.7   # F(y-1)=0.2 .. F(y)=0.7


def test_ks_uniform_on_calibrated_is_not_tiny():
    # A maximally-uniform sample -> KS statistic ~0 -> large p (well above the 0.01 gate).
    u = np.linspace(0, 1, 500, endpoint=False) + 0.5 / 500
    d, p = fc.ks_uniform(u)
    assert d < 0.05 and p > 0.5
    # a clearly non-uniform (all-tiny) sample -> KS rejects
    _, p_bad = fc.ks_uniform(np.full(500, 0.02))
    assert p_bad < 0.01


def _build_df(pmfs, actuals, dates):
    return pd.DataFrame({
        "stat": ["pts"] * len(actuals), "pmf_json": [_pmf_json(p) for p in pmfs],
        "actual_outcome": actuals, "game_id": [f"g{i}" for i in range(len(actuals))],
        "player_id": [f"p{i%20}" for i in range(len(actuals))],
        "model_version": ["m1"] * len(actuals), "game_date": dates,
    })


def test_two_sided_coverage_fails_over_dispersed():
    # Wide PMFs (support 0..20 uniform-ish) but actuals concentrated near 10
    # => central intervals far too wide => over-coverage; the 50% interval covers ~100%.
    n = 400
    wide = np.ones(21) / 21.0
    actuals = [10] * n
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([wide] * n, actuals, dates), min_n=100, min_dates=20)
    c50 = r.coverage["0.5"]
    assert c50["materially_over"] and c50["fail"]
    assert not r.passed
    assert any("over-covers" in reason for reason in r.reasons)


def test_two_sided_coverage_fails_under_dispersed():
    # Narrow (near-point) PMFs at 10 but actuals spread widely => under-coverage.
    n = 400
    narrow = np.zeros(41); narrow[10] = 1.0
    rng = np.random.default_rng(1)
    actuals = list(rng.integers(0, 40, size=n))
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([narrow] * n, actuals, dates), min_n=100, min_dates=20)
    c90 = r.coverage["0.9"]
    assert c90["materially_under"] and c90["fail"]
    assert not r.passed


def test_midpoint_pit_and_pooled_ece_reported_but_not_gated():
    n = 400
    rng = np.random.default_rng(2)
    # reasonably-calibrated: actual ~ from the PMF
    base = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    actuals = [int(rng.choice(len(base), p=base)) for _ in range(n)]
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([base] * n, actuals, dates), min_n=100, min_dates=20)
    # both diagnostics computed
    assert r.pit_mid_ece == r.pit_mid_ece
    assert r.calib_ece_pooled == r.calib_ece_pooled
    # neither midpoint-PIT nor pooled ECE appears as a gate reason
    joined = " ".join(r.reasons)
    assert "mid" not in joined.lower() and "pooled" not in joined.lower()


def test_line_level_calibration_is_separate():
    rows = pd.DataFrame({
        "p_over": [0.6, 0.4, 0.55, 0.3, 0.7] * 10,
        "over_outcome": [1, 0, 1, 0, 1] * 10,
    })
    out = fc.line_level_threshold_calibration(rows)
    assert out["available"] and out["n_lines"] == 50
    assert "brier" in out and "log_loss" in out and "calibration_slope" in out
    # too few lines -> unavailable (not a pooled fallback)
    assert not fc.line_level_threshold_calibration(rows.head(5))["available"]


def test_insufficient_coverage_blocks():
    n = 60
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r = fc.evaluate_stat(_build_df([base] * n, [2] * n, ["2026-06-01"] * n),
                         min_n=300, min_dates=25)
    assert not r.passed
    assert any("insufficient" in reason for reason in r.reasons)
