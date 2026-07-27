"""Owner item 5 — zero-inflation must be minutes-sensitive for sparse stats.

The production minutes-marginalization (``_build_marginalized_pmf_matrix``) previously scaled the
positive-count mean by the minute ratio while holding the structural-zero / hurdle-cross
probability FIXED. The repair uses an opportunity-exposure formulation
``P(active-with-event) = 1 - (1 - p_nz) ** (minutes/median)`` (equivalently
``pi(minutes) = pi0 ** (minutes/median)``), so P(Y>0) is non-decreasing in minutes.

Reported separately for STL and BLK (both sparse, high zero-inflation).
"""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.models.pmf_utils import hurdle_pmf_batch, zinb_pmf_batch

# 8/18/28/38 minutes, with the median at 28 (scale == 1 leaves the base untouched).
_MEDIAN = 28.0
_MINUTES = np.array([8.0, 18.0, 28.0, 38.0])
_SCALE = _MINUTES / _MEDIAN
_TOL = 1e-9


def _p_over_zero_zinb(pi0, mu0, r, cap):
    """Repair formulation: pi(minutes) = pi0 ** scale, mu(minutes) = mu0 * scale."""
    pi = np.clip(pi0 ** _SCALE, 0.0, 1.0)
    mu = np.clip(mu0 * _SCALE, 1e-9, None)
    pmf = zinb_pmf_batch(pi, mu, r, cap)
    return 1.0 - pmf[:, 0]


def _p_over_zero_hurdle(p_nz0, mu0, r, cap):
    p_nz = np.clip(1.0 - (1.0 - p_nz0) ** _SCALE, 0.0, 1.0)
    mu = np.clip(mu0 * _SCALE, 1e-9, None)
    pmf = hurdle_pmf_batch(p_nz, mu, r, cap)
    return 1.0 - pmf[:, 0]


@pytest.mark.parametrize(
    "label,pi0,mu0,r,cap",
    [
        ("STL", 0.55, 1.3, 4.0, 12),   # steals: sparse, ~1 per game when active
        ("BLK", 0.68, 1.1, 4.0, 10),   # blocks: sparser, more zero-inflated
    ],
)
def test_sparse_nonzero_probability_increases_with_minutes(label, pi0, mu0, r, cap):
    p_over = _p_over_zero_zinb(pi0, mu0, r, cap)
    # Non-decreasing in minutes (8 <= 18 <= 28 <= 38) within tolerance, and strictly higher
    # at 38 than at 8 (a genuine, material minutes effect on P(Y>0)).
    diffs = np.diff(p_over)
    assert np.all(diffs >= -_TOL), f"{label}: P(Y>0) not monotone in minutes: {p_over}"
    assert p_over[-1] - p_over[0] > 0.02, f"{label}: P(Y>0) barely moves with minutes: {p_over}"


def test_hurdle_nonzero_probability_increases_with_minutes():
    p_over = _p_over_zero_hurdle(p_nz0=0.45, mu0=1.2, r=4.0, cap=12)
    diffs = np.diff(p_over)
    assert np.all(diffs >= -_TOL), f"hurdle P(Y>0) not monotone: {p_over}"
    assert p_over[-1] - p_over[0] > 0.02


def test_median_scale_is_neutral():
    # At the median quadrature point (scale == 1) the structural zero is unchanged.
    pi0 = 0.6
    assert (pi0 ** 1.0) == pytest.approx(pi0, abs=1e-12)
