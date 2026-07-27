"""STEP 4 regression tests: correct PMF-level DNP handling (active vs availability mixture).

Proves the post-hoc ``model_prob_over_final / (1 - p_dnp)`` shortcut is WRONG for an integer
line with push mass, and that active-PMF push-safe conditioning is correct; plus half-line,
active-zero vs DNP-zero, nonidentity calibration, DNP void, one-minute appearance, and the
unknown-settlement (undefined push denominator) case.
"""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.models.availability_pmf import (
    build_availability_conditioned_row,
    build_availability_mixture,
    invalid_posthoc_dednp_over,
    pmf_mean,
    recover_active_pmf,
    settle_over_from_active_pmf,
)
from wnba_props_model.models.market import (
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)


def _active():
    # A concrete active (conditional-on-play) count PMF over 0..8 with real push mass at 5.
    a = np.array([0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.12, 0.08, 0.05])
    return a / a.sum()


def test_mixture_forward_and_inverse_roundtrip():
    a = _active()
    for p_dnp in (0.0, 0.1, 0.25, 0.5):
        mix = build_availability_mixture(a, p_dnp)
        assert mix[0] >= a[0] * (1 - p_dnp)  # DNP mass added at 0
        rec = recover_active_pmf(mix, p_dnp)
        np.testing.assert_allclose(rec, a, atol=1e-9)


def test_posthoc_division_is_wrong_for_integer_line_push():
    a = _active()
    p_dnp = 0.20
    line = 5.0  # integer line WITH push mass at 5
    mix = build_availability_mixture(a, p_dnp)
    # Old pipeline "final" over prob = settled from the MIXTURE pmf.
    final_mixture = settled_probabilities_from_pmf(mix, line).p_over_settled
    shortcut = invalid_posthoc_dednp_over(final_mixture, p_dnp)
    correct = settle_over_from_active_pmf(a, line).p_over_settled
    # They must differ materially (different push denominators): shortcut is wrong.
    assert abs(shortcut - correct) > 1e-3
    # The correct value equals active P(>5)/(1-active push at 5).
    k = np.arange(a.size)
    expected = a[k > line].sum() / (1 - a[int(line)])
    assert abs(correct - expected) < 1e-12


def test_posthoc_division_only_agrees_for_half_line():
    a = _active()
    p_dnp = 0.20
    line = 5.5  # half line: no push
    mix = build_availability_mixture(a, p_dnp)
    final_mixture = settled_probabilities_from_pmf(mix, line).p_over_settled
    shortcut = invalid_posthoc_dednp_over(final_mixture, p_dnp)
    correct = settle_over_from_active_pmf(a, line).p_over_settled
    # For a half line with identity calibration the two coincide (why the shortcut looked ok).
    assert abs(shortcut - correct) < 1e-9


def test_active_zero_is_not_dnp_zero():
    a = _active()
    p_dnp = 0.3
    mix = build_availability_mixture(a, p_dnp)
    # mixture P(Y=0) includes DNP; active P(Y=0) does not.
    assert mix[0] > a[0]
    assert abs(mix[0] - (p_dnp + (1 - p_dnp) * a[0])) < 1e-12


def test_nonidentity_calibration_applied_after_active_settlement():
    a = _active()
    cal = lambda p: p ** 0.5  # monotone, nonlinear
    row = build_availability_conditioned_row(a, p_dnp=0.2, line=5.0, binary_calibrator=cal)
    settled = settle_over_from_active_pmf(a, 5.0).p_over_settled
    assert abs(row.model_prob_over_settled_from_active_pmf - settled) < 1e-12
    assert abs(row.model_prob_over_final - settled ** 0.5) < 1e-12
    # dividing this calibrated final by (1-p_dnp) would NOT recover the active settled prob.
    assert abs(invalid_posthoc_dednp_over(row.model_prob_over_final, 0.2) - settled) > 1e-3


def test_dnp_void_keeps_p_dnp_separate_from_over_under():
    a = _active()
    row = build_availability_conditioned_row(a, p_dnp=0.4, line=4.5)
    # settled over/under from active PMF are unaffected by p_dnp (void, not under).
    settled = settle_over_from_active_pmf(a, 4.5)
    assert abs(row.model_prob_over_settled_from_active_pmf - settled.p_over_settled) < 1e-12
    assert row.p_dnp == pytest.approx(0.4)
    assert row.sportsbook_settlement_basis == "active_pmf_push_safe_void_on_dnp"


def test_one_minute_appearance_still_uses_active_pmf():
    # A near-certain-DNP row (p_dnp high) still yields a well-defined settled over prob from the
    # active PMF (the tiny appearance branch), never a fabricated 0.5.
    a = _active()
    row = build_availability_conditioned_row(a, p_dnp=0.97, line=2.5)
    assert 0.0 <= row.model_prob_over_final <= 1.0
    assert abs(row.model_prob_over_settled_from_active_pmf
               - settle_over_from_active_pmf(a, 2.5).p_over_settled) < 1e-12


def test_unknown_settlement_undefined_push_denominator_raises():
    # Degenerate active PMF entirely on the integer line => push denominator collapses.
    degenerate = np.array([0.0, 0.0, 0.0, 1.0])  # all mass at 3
    with pytest.raises(UndefinedSettledProbabilityError):
        settle_over_from_active_pmf(degenerate, 3.0)


def test_mixture_mean_below_active_mean():
    a = _active()
    mix = build_availability_mixture(a, 0.25)
    assert pmf_mean(mix) < pmf_mean(a)  # DNP mass pulls the mixture mean down
