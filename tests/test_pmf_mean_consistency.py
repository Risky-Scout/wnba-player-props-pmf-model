"""PR 1A B6: shared PMF row-integrity validator."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wnba_props_model.models.pmf_utils import (
    negbinom_pmf_batch,
    validate_pmf_row_integrity,
)


def _row_from_pmf(pmf: np.ndarray) -> dict:
    k = np.arange(len(pmf), dtype=float)
    mean = float(np.dot(k, pmf))
    var = float(np.dot(k ** 2, pmf)) - mean ** 2
    return {
        "pmf_json": json.dumps({str(i): float(p) for i, p in enumerate(pmf) if p > 0}),
        "pmf_mean": mean,
        "pmf_variance": var,
    }


def test_consistent_row_passes_for_every_direct_prop():
    # Build a NB PMF per direct-prop-like mean; exported mean/var derived from the PMF.
    for mu in (0.4, 1.0, 2.5, 4.0, 8.0, 12.0, 20.0):
        pmf = negbinom_pmf_batch(np.array([mu]), 4.0, 60)[0]
        row = _row_from_pmf(pmf)
        validate_pmf_row_integrity(row)  # must not raise


def test_detached_mean_raises():
    pmf = negbinom_pmf_batch(np.array([4.0]), 4.0, 60)[0]
    row = _row_from_pmf(pmf)
    row["pmf_mean"] = row["pmf_mean"] + 1.5  # detached shift, PMF unchanged
    with pytest.raises(ValueError):
        validate_pmf_row_integrity(row)


def test_bad_sum_raises():
    row = {"pmf_json": json.dumps({"0": 0.5, "1": 0.9}), "pmf_mean": 0.64}
    with pytest.raises(ValueError):
        validate_pmf_row_integrity(row)


def test_negative_mass_raises():
    row = {"pmf_json": json.dumps({"0": 1.2, "1": -0.2}), "pmf_mean": -0.2}
    with pytest.raises(ValueError):
        validate_pmf_row_integrity(row)


def test_missing_mean_key_is_skipped():
    pmf = negbinom_pmf_batch(np.array([3.0]), 4.0, 60)[0]
    row = {"pmf_json": json.dumps({str(i): float(p) for i, p in enumerate(pmf) if p > 0})}
    validate_pmf_row_integrity(row)  # no mean key -> only structural checks
