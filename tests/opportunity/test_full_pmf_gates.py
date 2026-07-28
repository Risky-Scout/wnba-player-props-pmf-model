"""Full-PMF certification gates (owner directive item 6).

Each gate — normalization, tail-truncation, PMF-log-score noninferiority, CRPS noninferiority,
central-50%/90% coverage, and sharpness — must be able to INDEPENDENTLY fail a candidate. These
tests craft PMF/actual inputs that trip exactly one gate at a time and assert only that gate is False.
"""
from __future__ import annotations

import importlib.util
import json
from math import exp, factorial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "opp_eval", REPO / "scripts" / "evaluate_opportunity_oof.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EV = _mod()

# generous frozen-ish contract so only the intentionally-broken gate fails
CONTRACT = {
    "normalization_tolerance": 1e-6,
    "full_pmf_log_score_max": 5.0, "crps_max": 10.0,
    "full_pmf_log_score_noninferiority_tol": 0.05, "crps_noninferiority_tol": 0.05,
    "tail_bin_mass_max": 0.02, "out_of_support_frac_max": 0.01,
    "coverage_tol_50": 0.15, "coverage_tol_90": 0.10, "sharpness_max_width_90": 60.0,
}


def _poisson_pmf(mean, size=40):
    arr = np.array([exp(-mean) * mean ** i / factorial(i) for i in range(size)], float)
    return arr / arr.sum()


def _calibrated_sample(n=400, mean=6.0, seed=0):
    """Draw actuals from the SAME poisson the PMF encodes -> well-calibrated, sharp."""
    rng = np.random.default_rng(seed)
    pmf = _poisson_pmf(mean)
    actual = rng.poisson(mean, size=n).tolist()
    pmf_json = [json.dumps(pmf.tolist())] * n
    return pmf_json, actual


def test_all_gates_pass_on_calibrated_sharp_pmf():
    pj, a = _calibrated_sample()
    gates, meas = EV.full_pmf_certification(pj, a, CONTRACT)
    assert all(gates.values()), (gates, meas)


def test_normalization_gate_can_fail():
    pj, a = _calibrated_sample()
    bad = json.dumps((_poisson_pmf(6.0) * 1.5).tolist())  # sums to 1.5
    pj = [bad] + pj[1:]
    gates, _ = EV.full_pmf_certification(pj, a, CONTRACT)
    assert gates["pmf_normalization_ok"] is False


def test_tail_truncation_gate_can_fail():
    # PMF with huge mass piled on the final support bin -> truncated
    arr = np.zeros(10)
    arr[0] = 0.5
    arr[-1] = 0.5
    pj = [json.dumps(arr.tolist())] * 200
    a = [0] * 200
    gates, _ = EV.full_pmf_certification(pj, a, CONTRACT)
    assert gates["pmf_tail_truncation_ok"] is False


def test_log_score_noninferiority_gate_can_fail():
    pj, a = _calibrated_sample()
    ref = {"full_pmf_log_score": 0.01, "crps": 100.0}  # impossibly good reference LL
    gates, _ = EV.full_pmf_certification(pj, a, CONTRACT, reference=ref)
    assert gates["pmf_log_score_noninferiority_ok"] is False


def test_crps_noninferiority_gate_can_fail():
    pj, a = _calibrated_sample()
    ref = {"full_pmf_log_score": 100.0, "crps": 0.001}  # impossibly good reference CRPS
    gates, _ = EV.full_pmf_certification(pj, a, CONTRACT, reference=ref)
    assert gates["crps_noninferiority_ok"] is False


def test_coverage_gate_can_fail():
    # PMF concentrated at 6 but actuals always 0 -> central intervals never cover
    pj = [json.dumps(_poisson_pmf(6.0).tolist())] * 300
    a = [0] * 300
    gates, _ = EV.full_pmf_certification(pj, a, CONTRACT)
    assert gates["coverage_50_ok"] is False
    assert gates["coverage_90_ok"] is False


def test_sharpness_gate_can_fail():
    # near-uniform PMF over wide support -> central-90% width huge
    arr = np.ones(100) / 100.0
    pj = [json.dumps(arr.tolist())] * 300
    a = [50] * 300
    gates, meas = EV.full_pmf_certification(pj, a, CONTRACT)
    assert gates["sharpness_ok"] is False
    assert meas["mean_width_90"] > CONTRACT["sharpness_max_width_90"]
