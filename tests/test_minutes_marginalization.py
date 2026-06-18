"""Test F1: Minutes marginalization produces higher variance PMFs than point-estimate PMFs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.pmf_engine import _build_marginalized_pmf_matrix, _blend_with_dnp
from wnba_props_model.models.pmf_utils import negbinom_pmf_batch, pmf_mean_var


class _MockStatModel:
    """Minimal stat model stub for testing."""
    def __init__(self, r: float = 3.0):
        self.dispersion_r = r
        self._role_dispersion = None
        self._global_var = r
        def get_dispersion(role): return r  # noqa: ANN001, ANN202
        self.get_dispersion = get_dispersion


class TestMinutesMarginalization:
    """Marginalized PMF should have more variance than point-estimate PMF."""

    def _build_point_estimate_pmf(self, stat_means: np.ndarray, r: float, cap: int) -> np.ndarray:
        """Simple NegBinom point-estimate PMF for reference."""
        return negbinom_pmf_batch(stat_means, r, cap)

    def test_marginalized_variance_greater_than_point_estimate(self):
        n = 50
        rng = np.random.default_rng(42)
        stat_means = rng.uniform(0.5, 2.5, n)
        r = 3.0
        cap = 10

        # Build point-estimate PMF
        pmf_point = self._build_point_estimate_pmf(stat_means, r, cap)
        _, var_point = pmf_mean_var(pmf_point)

        # Build marginalized PMF: introduce deliberate spread around stat_means
        # Quadrature: 5 points with ±20% spread around median
        median_means = stat_means.copy()
        quant_mat = np.column_stack([
            median_means * 0.6,  # q10 (reduced minutes)
            median_means * 0.8,  # q25
            median_means,        # q50 (median)
            median_means * 1.2,  # q75
            median_means * 1.4,  # q90 (more minutes)
        ])
        quad_weights = np.array([0.10, 0.15, 0.50, 0.15, 0.10])

        stat_models = {"pts": _MockStatModel(r)}
        pmf_marg = _build_marginalized_pmf_matrix(
            "pts", quant_mat, quad_weights, None, None,
            stat_models, {}, cap
        )
        _, var_marg = pmf_mean_var(pmf_marg)

        # Marginalized PMFs must have strictly larger mean variance
        assert var_marg.mean() > var_point.mean() * 1.01, (
            f"Marginalized variance ({var_marg.mean():.4f}) should be > "
            f"point-estimate variance ({var_point.mean():.4f})"
        )

    def test_marginalized_pmf_sums_to_one(self):
        n = 20
        stat_means = np.full(n, 1.5)
        quant_mat = np.column_stack([stat_means * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]])
        quad_weights = np.array([0.10, 0.15, 0.50, 0.15, 0.10])
        stat_models = {"pts": _MockStatModel(3.0)}

        pmf = _build_marginalized_pmf_matrix(
            "pts", quant_mat, quad_weights, None, None,
            stat_models, {}, cap=10
        )
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)
        assert (pmf >= 0).all()

    def test_blend_with_dnp_inflates_zero(self):
        """_blend_with_dnp should increase P(k=0) in proportion to p_dnp."""
        n = 5
        pmf_base = negbinom_pmf_batch(np.full(n, 1.0), 3.0, 10)
        p_dnp = np.full(n, 0.2)

        pmf_blended = _blend_with_dnp(pmf_base, p_dnp)
        np.testing.assert_allclose(pmf_blended.sum(axis=1), 1.0, atol=1e-5)
        assert (pmf_blended[:, 0] > pmf_base[:, 0]).all(), "P(k=0) must increase after DNP blending"

    def test_blend_zero_dnp_unchanged(self):
        """With p_dnp=0, blended PMF equals original."""
        n = 5
        pmf_base = negbinom_pmf_batch(np.full(n, 1.0), 3.0, 10)
        p_dnp = np.zeros(n)
        pmf_blended = _blend_with_dnp(pmf_base, p_dnp)
        np.testing.assert_allclose(pmf_blended, pmf_base, atol=1e-10)
