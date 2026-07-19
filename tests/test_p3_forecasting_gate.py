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


# Task 1 corrected-gate regression tests -------------------------------------------------

def test_discrete_interval_calibrated_passes_when_inclusion_matches_mass():
    # A nominal-50% integer interval that CONTAINS ~80% PMF mass must PASS when empirical
    # inclusion is ~80% (residual ~ 0) — NOT fail merely for exceeding the nominal.
    base = np.array([0.05, 0.10, 0.70, 0.10, 0.05])  # concentrated at 2
    n = 3000
    rng = np.random.default_rng(11)
    actuals = [int(rng.choice(len(base), p=base)) for _ in range(n)]
    dates = [f"2026-{6 + (i // 900):02d}-{(i % 28) + 1:02d}" for i in range(n)]  # ~28 dates
    r = fc.evaluate_stat(_build_df([base] * n, actuals, dates), min_n=100, min_dates=20,
                         baseline=_baseline(crps=9, log_score=9, w80=99))
    c50 = r.coverage["0.5"]
    # the nominal-50% interval contains well over 50% mass, yet the residual is ~0 -> PASS
    assert c50["contained_mass"] > 0.5
    assert abs(c50["residual"]) < 0.05 and not c50["fail"]


def test_genuine_over_dispersion_fails():
    # Wide PMFs but actuals concentrated at the mode -> intervals contain far more mass
    # than actually lands in them -> residual strongly negative -> FAIL.
    n = 500
    wide = np.ones(21) / 21.0
    actuals = [10] * n
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([wide] * n, actuals, dates), min_n=100, min_dates=20,
                         baseline=_baseline(crps=9, log_score=9, w80=99))
    # over-dispersed: outcomes land inside the (too-wide) interval MORE than claimed -> residual > 0
    assert r.coverage["0.9"]["residual"] > 0.05 and r.coverage["0.9"]["fail"]
    assert not r.forecast_allowed


def test_genuine_under_dispersion_fails():
    # Near-point PMFs but actuals spread widely -> intervals contain little mass but many
    # outcomes land outside -> residual strongly positive -> FAIL.
    n = 500
    narrow = np.zeros(41); narrow[10] = 0.9; narrow[9] = 0.05; narrow[11] = 0.05
    rng = np.random.default_rng(1)
    actuals = list(rng.integers(0, 40, size=n))
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([narrow] * n, actuals, dates), min_n=100, min_dates=20,
                         baseline=_baseline(crps=9, log_score=9, w80=99))
    # under-dispersed/overconfident: outcomes escape the too-narrow interval -> residual < 0
    assert r.coverage["0.9"]["residual"] < -0.05 and r.coverage["0.9"]["fail"]
    assert not r.forecast_allowed


def test_matched_mass_sharpness_behaves():
    # matched_mass_width is monotonic in target mass and smaller for sharper PMFs.
    sharp = np.zeros(41); sharp[20] = 0.6; sharp[19] = 0.2; sharp[21] = 0.2
    broad = np.ones(41) / 41.0
    assert fc.matched_mass_width(sharp, 0.8) < fc.matched_mass_width(broad, 0.8)
    assert fc.matched_mass_width(sharp, 0.9) >= fc.matched_mass_width(sharp, 0.5)


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


def test_line_level_calibration_requires_150():
    rows = pd.DataFrame({
        "p_over": [0.6, 0.4, 0.55, 0.3, 0.7] * 40,       # 200 lines
        "over_outcome": [1, 0, 1, 0, 1] * 40,
    })
    out = fc.line_level_threshold_calibration(rows)   # default min_lines=150
    assert out["available"] and out["n_lines"] == 200
    assert "brier" in out and "log_loss" in out and "calibration_slope" in out
    # 50 lines -> UNAVAILABLE under the committed 150 minimum (not silently reduced to 30)
    assert not fc.line_level_threshold_calibration(rows.head(50))["available"]


def _baseline(crps, log_score, w80):
    return {"crps": crps, "log_score": log_score, "matched_width_80": w80}


def test_proper_score_gate_blocks_worse_than_baseline():
    n = 400
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    rng = np.random.default_rng(3)
    actuals = [int(rng.choice(len(base), p=base)) for _ in range(n)]
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    df = _build_df([base] * n, actuals, dates)
    # baseline strictly better (lower) CRPS/log -> model must fail proper-score gate
    r = fc.evaluate_stat(df, min_n=100, min_dates=20,
                         baseline=_baseline(crps=0.0, log_score=0.0, w80=99))
    assert any("worse than baseline" in reason for reason in r.reasons)
    assert not r.forecast_allowed


def test_sharpness_gate_blocks_overbroad():
    n = 400
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    rng = np.random.default_rng(4)
    actuals = [int(rng.choice(len(base), p=base)) for _ in range(n)]
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    df = _build_df([base] * n, actuals, dates)
    # tiny baseline width -> model intervals look far too broad -> sharpness fail
    r = fc.evaluate_stat(df, min_n=100, min_dates=20,
                         baseline=_baseline(crps=9, log_score=9, w80=0.1))
    assert any("too broad" in reason for reason in r.reasons)


def test_no_baseline_cannot_pass():
    n = 400
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    rng = np.random.default_rng(5)
    actuals = [int(rng.choice(len(base), p=base)) for _ in range(n)]
    dates = [f"2026-06-{(i % 30)+1:02d}" for i in range(n)]
    r = fc.evaluate_stat(_build_df([base] * n, actuals, dates), min_n=100, min_dates=20)
    assert not r.forecast_allowed
    assert any("no preregistered baseline" in reason for reason in r.reasons)


def test_three_independent_statuses_exist():
    n = 200
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r = fc.evaluate_stat(_build_df([base] * n, [2] * n, ["2026-06-01"] * n), min_n=300)
    # market/betting default False without real lines; all three attributes present
    assert hasattr(r, "forecast_allowed") and hasattr(r, "market_comparison_allowed")
    assert hasattr(r, "betting_recommendation_allowed")
    assert r.market_comparison_allowed is False and r.betting_recommendation_allowed is False


def test_out_of_support_scored_not_nan():
    # PMF supports 0..4 but actual=10 -> log_score finite (overflow), support_miss counted
    pmf = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    ls = fc.log_score(pmf, 10)
    assert np.isfinite(ls) and ls > 0
    ok, reason = fc.validate_pmf(pmf, 10)
    assert ok and reason == "support_miss"


def test_insufficient_coverage_blocks():
    n = 60
    base = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r = fc.evaluate_stat(_build_df([base] * n, [2] * n, ["2026-06-01"] * n),
                         min_n=300, min_dates=25)
    assert not r.passed
    assert any("insufficient" in reason for reason in r.reasons)
