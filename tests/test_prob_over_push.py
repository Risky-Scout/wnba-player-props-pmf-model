"""PR 1A B1: push-safe settled-probability API tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from wnba_props_model.models.market import (
    SettledProbabilities,
    UndefinedSettledProbabilityError,
    prob_over_from_pmf,
    settled_probabilities_from_pmf,
)

TOL = 1e-12


def _pmf(d, n=None):
    """Dense PMF array from {index: mass}."""
    n = n or (max(d) + 1)
    a = np.zeros(n)
    for k, v in d.items():
        a[k] = v
    return a


def test_half_line_no_push():
    # P(Y): 0->0.2, 1->0.3, 2->0.5. Line 1.5 -> over = P(Y>=2)=0.5, under=0.5.
    pmf = _pmf({0: 0.2, 1: 0.3, 2: 0.5})
    r = settled_probabilities_from_pmf(pmf, 1.5)
    assert r.p_push == 0.0
    assert r.p_over_unconditional == pytest.approx(0.5)
    assert r.p_under_unconditional == pytest.approx(0.5)
    assert r.p_over_settled == pytest.approx(0.5)
    assert r.p_under_settled == pytest.approx(0.5)
    assert r.p_over_settled + r.p_under_settled == pytest.approx(1.0)


def test_line_0_5():
    pmf = _pmf({0: 0.4, 1: 0.6})
    r = settled_probabilities_from_pmf(pmf, 0.5)
    assert r.p_push == 0.0
    assert r.p_over_settled == pytest.approx(0.6)
    assert r.p_under_settled == pytest.approx(0.4)


def test_integer_line_1_0_conditions_out_push():
    # 0->0.2, 1->0.3 (push), 2->0.5. Integer line 1.0.
    pmf = _pmf({0: 0.2, 1: 0.3, 2: 0.5})
    r = settled_probabilities_from_pmf(pmf, 1.0)
    assert r.p_push == pytest.approx(0.3)
    assert r.p_over_unconditional == pytest.approx(0.5)
    assert r.p_under_unconditional == pytest.approx(0.2)
    assert r.p_over_settled == pytest.approx(0.5 / 0.7)
    assert r.p_under_settled == pytest.approx(0.2 / 0.7)
    assert r.p_over_settled + r.p_under_settled == pytest.approx(1.0)


def test_integer_line_10_0():
    rng = np.random.default_rng(0)
    a = rng.random(25); a /= a.sum()
    r = settled_probabilities_from_pmf(a, 10.0)
    assert r.p_push == pytest.approx(float(a[10]))
    assert r.p_over_settled + r.p_under_settled == pytest.approx(1.0)


def test_quarter_line_is_non_integer():
    pmf = _pmf({9: 0.3, 10: 0.3, 11: 0.4})
    r = settled_probabilities_from_pmf(pmf, 10.25)
    assert r.p_push == 0.0
    assert r.p_over_unconditional == pytest.approx(0.4)   # P(Y>=11)
    assert r.p_under_unconditional == pytest.approx(0.6)  # P(Y<=10)
    assert r.p_over_settled + r.p_under_settled == pytest.approx(1.0)


def test_all_mass_on_push_raises():
    pmf = _pmf({10: 1.0})
    with pytest.raises(UndefinedSettledProbabilityError):
        settled_probabilities_from_pmf(pmf, 10.0)


def test_nonconsecutive_mapping_support():
    r = settled_probabilities_from_pmf({0: 0.5, 5: 0.5}, 2.5)
    assert r.p_under_unconditional == pytest.approx(0.5)
    assert r.p_over_unconditional == pytest.approx(0.5)


def test_numeric_string_keys():
    r = settled_probabilities_from_pmf({"0": 0.25, "3": 0.75}, 1.5)
    assert r.p_over_unconditional == pytest.approx(0.75)
    assert r.p_under_unconditional == pytest.approx(0.25)


def test_negative_pmf_value_raises():
    with pytest.raises(ValueError):
        settled_probabilities_from_pmf(np.array([0.5, -0.1, 0.6]), 0.5)


def test_nan_pmf_value_raises():
    with pytest.raises(ValueError):
        settled_probabilities_from_pmf(np.array([0.5, float("nan"), 0.5]), 0.5)


def test_material_sum_failure_raises():
    with pytest.raises(ValueError):
        settled_probabilities_from_pmf(np.array([0.5, 0.5, 0.5]), 0.5)  # sum 1.5


def test_minor_sum_drift_normalized():
    a = np.array([0.2, 0.3, 0.5]) * 1.0002  # tiny drift within tolerance
    r = settled_probabilities_from_pmf(a, 1.5)
    assert r.p_over_settled == pytest.approx(0.5, abs=1e-3)


def test_negative_line_raises():
    with pytest.raises(ValueError):
        settled_probabilities_from_pmf(np.array([0.5, 0.5]), -1.0)


def test_property_random_pmfs_settled_sum_to_one():
    rng = np.random.default_rng(20260723)
    for _ in range(500):
        n = int(rng.integers(3, 40))
        a = rng.random(n); a /= a.sum()
        line = float(rng.integers(0, n))          # integer line
        if a[int(line)] >= 1.0 - 1e-9:
            continue
        r = settled_probabilities_from_pmf(a, line)
        assert r.p_over_settled + r.p_under_settled == pytest.approx(1.0, abs=1e-9)
        assert r.p_over_unconditional + r.p_under_unconditional + r.p_push == pytest.approx(1.0, abs=1e-9)
        half = line + 0.5
        rh = settled_probabilities_from_pmf(a, half)
        assert rh.p_over_settled + rh.p_under_settled == pytest.approx(1.0, abs=1e-9)


def test_deprecated_wrapper_returns_unconditional():
    pmf = _pmf({0: 0.2, 1: 0.3, 2: 0.5})
    # Wrapper preserves historical unconditional P(Y>line) behavior.
    assert prob_over_from_pmf(pmf, 1.0) == pytest.approx(0.5)
    assert prob_over_from_pmf(pmf, 1.5) == pytest.approx(0.5)
    # For half-lines, unconditional == settled.
    assert prob_over_from_pmf(pmf, 1.5) == pytest.approx(
        settled_probabilities_from_pmf(pmf, 1.5).p_over_settled)
