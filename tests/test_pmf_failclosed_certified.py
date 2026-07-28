"""Phase 3C - adversarial tests for the fail-closed certified PMF contract.

Every PMF defect must RAISE a structured PmfCertificationError in CERTIFIED mode (never silently
sanitised to uniform), while DIAGNOSTIC mode still repairs. Tiny float drift within the frozen
tolerance is permitted.
"""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.models.pmf_utils import (
    FROZEN_PMF_SUM_TOL,
    PmfCertificationError,
    sanitize_pmf_matrix,
    validate_pmf_matrix_certified,
)


def _good():
    return np.array([[0.2, 0.3, 0.5]], dtype=float)


def test_valid_pmf_passes_certified():
    validate_pmf_matrix_certified(_good())  # no raise


def test_tiny_float_drift_permitted():
    m = _good()
    m[0, 0] += FROZEN_PMF_SUM_TOL / 2  # within frozen tolerance
    validate_pmf_matrix_certified(m)


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m.__setitem__((0, 0), np.nan), "non_finite_mass"),
    (lambda m: m.__setitem__((0, 0), np.inf), "non_finite_mass"),
    (lambda m: m.__setitem__((0, 0), -np.inf), "non_finite_mass"),
    (lambda m: m.__setitem__((0, 0), -0.5), "negative_mass"),
    (lambda m: m.__imul__(0.0), "zero_mass_row"),
    (lambda m: m.__setitem__((0, 2), m[0, 2] - 0.01), "normalization_out_of_tolerance"),   # sum 0.99
    (lambda m: m.__setitem__((0, 2), m[0, 2] + 0.01), "normalization_out_of_tolerance"),   # sum 1.01
])
def test_each_defect_raises_certified(mutate, reason):
    m = _good()
    mutate(m)
    with pytest.raises(PmfCertificationError) as ei:
        validate_pmf_matrix_certified(m, context={"prop": "pts", "game_id": "g1", "player_id": "p1"})
    assert ei.value.reason == reason
    assert ei.value.context.get("prop") == "pts"


def test_excessive_tail_truncation_raises():
    with pytest.raises(PmfCertificationError) as ei:
        validate_pmf_matrix_certified(_good(), raw_upper_tail=np.array([0.05]))
    assert ei.value.reason == "excessive_tail_truncation"


def test_sanitize_certified_raises_but_diagnostic_repairs():
    bad = np.array([[np.nan, 0.0, 0.0]], dtype=float)
    with pytest.raises(PmfCertificationError):
        sanitize_pmf_matrix(bad, certified=True)
    # diagnostic mode still repairs (uniform) and reports rows fixed
    mat, n_fixed = sanitize_pmf_matrix(bad, certified=False)
    assert n_fixed >= 1
    assert np.isclose(mat.sum(), 1.0)


def test_structured_error_carries_magnitude_and_row():
    m = _good()
    m[0, 0] = -0.3
    with pytest.raises(PmfCertificationError) as ei:
        validate_pmf_matrix_certified(m)
    assert ei.value.magnitude is not None
    assert ei.value.row == 0
    assert ei.value.contract_version
