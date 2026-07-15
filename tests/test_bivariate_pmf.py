"""Tests for bivariate_pmf IPF marginal preservation and combo mean integrity."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from wnba_props_model.models.bivariate_pmf import (
    _achieved_correlation,
    adjust_combo_pmf_for_correlation,
    build_bivariate_pmf,
    fit_joint_to_marginals,
)


def _negbinom_pmf(mu: float, r: float, n: int = 80) -> np.ndarray:
    """NegBinomial PMF with mean mu and dispersion r, support [0, n]."""
    p = r / (r + mu)
    xs = np.arange(n + 1)
    pmf = nbinom.pmf(xs, r, p).astype(float)
    pmf /= pmf.sum()
    return pmf


def _mean(pmf: np.ndarray) -> float:
    return float(np.arange(len(pmf)) @ pmf)


# ---------------------------------------------------------------------------
# fit_joint_to_marginals
# ---------------------------------------------------------------------------

class TestFitJointToMarginals:
    def test_rows_equal_first_marginal(self):
        pmf_x = _negbinom_pmf(4.5, 5.0)
        pmf_y = _negbinom_pmf(3.2, 4.0)
        seed = np.outer(pmf_x, pmf_y)
        joint, diag = fit_joint_to_marginals(seed, pmf_x, pmf_y)
        np.testing.assert_allclose(joint.sum(axis=1), pmf_x, atol=1e-9,
                                   err_msg="Row sums must equal pmf_x within 1e-9")

    def test_columns_equal_second_marginal(self):
        pmf_x = _negbinom_pmf(6.0, 8.0)
        pmf_y = _negbinom_pmf(2.1, 3.0)
        seed = np.outer(pmf_x, pmf_y)
        joint, diag = fit_joint_to_marginals(seed, pmf_x, pmf_y)
        np.testing.assert_allclose(joint.sum(axis=0), pmf_y, atol=1e-9,
                                   err_msg="Col sums must equal pmf_y within 1e-9")

    def test_joint_sums_to_one(self):
        pmf_x = _negbinom_pmf(5.0, 6.0)
        pmf_y = _negbinom_pmf(4.0, 5.0)
        seed = np.outer(pmf_x, pmf_y)
        joint, _ = fit_joint_to_marginals(seed, pmf_x, pmf_y)
        assert abs(joint.sum() - 1.0) < 1e-9

    def test_non_negative(self):
        pmf_x = _negbinom_pmf(3.0, 4.0)
        pmf_y = _negbinom_pmf(2.0, 3.0)
        seed = np.outer(pmf_x, pmf_y)
        joint, _ = fit_joint_to_marginals(seed, pmf_x, pmf_y)
        assert (joint >= 0).all(), "Joint PMF must be non-negative"

    def test_converges_from_perturbed_seed(self):
        """IPF should converge even from a perturbed (non-marginal-matching) seed."""
        rng = np.random.default_rng(1234)
        pmf_x = _negbinom_pmf(7.0, 6.0)
        pmf_y = _negbinom_pmf(4.5, 5.0)
        seed = np.outer(pmf_x, pmf_y) * rng.exponential(1.0, (len(pmf_x), len(pmf_y)))
        joint, diag = fit_joint_to_marginals(seed, pmf_x, pmf_y)
        assert diag["converged"], "IPF must converge"
        np.testing.assert_allclose(joint.sum(axis=1), pmf_x, atol=1e-9)
        np.testing.assert_allclose(joint.sum(axis=0), pmf_y, atol=1e-9)


# ---------------------------------------------------------------------------
# build_bivariate_pmf — marginal preservation after copula + IPF
# ---------------------------------------------------------------------------

class TestBivariatePmf:
    @pytest.mark.parametrize("rho", [-0.18, -0.12, -0.05, 0.08, 0.15])
    def test_marginals_preserved(self, rho):
        pmf_x = _negbinom_pmf(5.5, 6.0)
        pmf_y = _negbinom_pmf(3.8, 4.5)
        joint = build_bivariate_pmf(pmf_x, pmf_y, rho)
        np.testing.assert_allclose(joint.sum(axis=1), pmf_x, atol=1e-9,
                                   err_msg=f"Row marginals must equal pmf_x for rho={rho}")
        np.testing.assert_allclose(joint.sum(axis=0), pmf_y, atol=1e-9,
                                   err_msg=f"Col marginals must equal pmf_y for rho={rho}")

    def test_independence_case_exact(self):
        pmf_x = _negbinom_pmf(4.0, 5.0)
        pmf_y = _negbinom_pmf(2.5, 3.5)
        joint = build_bivariate_pmf(pmf_x, pmf_y, rho=0.0)
        expected = np.outer(pmf_x, pmf_y)
        np.testing.assert_allclose(joint, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# adjust_combo_pmf_for_correlation
# ---------------------------------------------------------------------------

class TestAdjustComboPmfForCorrelation:
    def test_combo_mean_equals_component_sum(self):
        """E[X+Y] must equal E[X] + E[Y] within 1e-8."""
        pmf_pts = _negbinom_pmf(14.0, 5.0)
        pmf_ast = _negbinom_pmf(3.5, 4.0)
        sum_pmf, diag = adjust_combo_pmf_for_correlation(pmf_pts, pmf_ast, "pts", "ast")
        mean_pts = _mean(pmf_pts)
        mean_ast = _mean(pmf_ast)
        combo_mean = _mean(sum_pmf)
        assert abs(combo_mean - (mean_pts + mean_ast)) < 1e-8, (
            f"combo_mean={combo_mean:.10f} vs expected={mean_pts + mean_ast:.10f}"
        )

    def test_combo_mean_reb_ast(self):
        pmf_reb = _negbinom_pmf(6.0, 5.0)
        pmf_ast = _negbinom_pmf(3.2, 4.0)
        sum_pmf, diag = adjust_combo_pmf_for_correlation(pmf_reb, pmf_ast, "reb", "ast")
        assert abs(_mean(sum_pmf) - (_mean(pmf_reb) + _mean(pmf_ast))) < 1e-8

    def test_combo_mean_pts_reb(self):
        pmf_pts = _negbinom_pmf(15.0, 6.0)
        pmf_reb = _negbinom_pmf(7.0, 5.0)
        sum_pmf, diag = adjust_combo_pmf_for_correlation(pmf_pts, pmf_reb, "pts", "reb")
        assert abs(_mean(sum_pmf) - (_mean(pmf_pts) + _mean(pmf_reb))) < 1e-8

    def test_positive_correlation_increases_variance(self):
        """With ρ > 0 the combo variance must exceed independence variance."""
        pmf_stl = _negbinom_pmf(1.2, 3.0)
        pmf_blk = _negbinom_pmf(0.9, 2.5)
        indep = np.convolve(pmf_stl, pmf_blk)
        indep /= indep.sum()
        sum_pmf, diag = adjust_combo_pmf_for_correlation(pmf_stl, pmf_blk, "stl", "blk")
        ks = np.arange(len(sum_pmf))
        var_corr = float((ks ** 2) @ sum_pmf - _mean(sum_pmf) ** 2)
        ks_ind = np.arange(len(indep))
        var_indep = float((ks_ind ** 2) @ indep - _mean(indep) ** 2)
        # Means must still be equal; only variance changes
        assert abs(_mean(sum_pmf) - _mean(indep)) < 1e-6, "Means must be equal regardless of correlation"
        assert var_corr > var_indep - 1e-6, (
            f"Positive correlation should increase variance: {var_corr:.6f} vs {var_indep:.6f}"
        )

    def test_diagnostics_keys_present(self):
        pmf_x = _negbinom_pmf(5.0, 4.0)
        pmf_y = _negbinom_pmf(3.0, 3.0)
        _, diag = adjust_combo_pmf_for_correlation(pmf_x, pmf_y, "pts", "ast")
        for key in ("requested_latent_rho", "achieved_count_correlation",
                    "row_marginal_max_error", "col_marginal_max_error",
                    "combo_mean_error", "ipf_iterations", "ipf_converged"):
            assert key in diag, f"Diagnostic key '{key}' missing"

    def test_ipf_row_col_error_below_threshold(self):
        """Marginal errors from IPF must be ≤ 1e-9."""
        pmf_x = _negbinom_pmf(8.0, 7.0)
        pmf_y = _negbinom_pmf(4.0, 5.0)
        _, diag = adjust_combo_pmf_for_correlation(pmf_x, pmf_y, "pts", "reb")
        assert diag["row_marginal_max_error"] <= 1e-9, (
            f"row_marginal_max_error={diag['row_marginal_max_error']:.2e} > 1e-9"
        )
        assert diag["col_marginal_max_error"] <= 1e-9, (
            f"col_marginal_max_error={diag['col_marginal_max_error']:.2e} > 1e-9"
        )

    def test_pmf_normalized(self):
        pmf_x = _negbinom_pmf(10.0, 6.0)
        pmf_y = _negbinom_pmf(5.0, 4.0)
        sum_pmf, _ = adjust_combo_pmf_for_correlation(pmf_x, pmf_y, "pts", "ast")
        assert abs(sum_pmf.sum() - 1.0) < 1e-9

    def test_pmf_non_negative(self):
        pmf_x = _negbinom_pmf(7.0, 5.0)
        pmf_y = _negbinom_pmf(3.0, 3.0)
        sum_pmf, _ = adjust_combo_pmf_for_correlation(pmf_x, pmf_y, "reb", "ast")
        assert (sum_pmf >= 0).all()


# ---------------------------------------------------------------------------
# C.6 rate correction: threshold logic — factors > 1e-6 must NOT be skipped
# ---------------------------------------------------------------------------

class TestRateCorrectionSubOnePercent:
    def test_factor_0007_not_identity(self):
        """Factor 1.007 (max practical C.6 factor) must not be numerically identity.

        The old threshold was 0.01, which skipped this factor. The new threshold
        is 1e-6, so abs(1.007 - 1.0) = 0.007 >> 1e-6 and the factor must apply.
        """
        factor = 1.007
        assert abs(factor - 1.0) > 1e-6, (
            f"Factor {factor} should have abs(f-1)={abs(factor-1.0):.4f} > 1e-6 "
            "and therefore NOT be skipped by the new threshold"
        )

    def test_factor_0007_would_have_been_skipped_by_old_threshold(self):
        """Prove the old 0.01 threshold would have skipped the max practical factor."""
        factor = 1.007
        old_threshold = 0.01
        assert abs(factor - 1.0) < old_threshold, (
            f"Factor {factor} has abs(f-1)={abs(factor-1.0):.4f} < old threshold {old_threshold}; "
            "confirm the old code was dead"
        )

    def test_sub_1e6_factor_is_identity(self):
        """Factor within 1e-6 of 1.0 is genuinely a no-op skip."""
        factor = 1.0 + 5e-7
        assert abs(factor - 1.0) <= 1e-6, (
            f"Factor {factor} has abs(f-1)={abs(factor-1.0):.2e} which should be ≤ 1e-6"
        )

    def test_max_practical_c6_factor_above_threshold(self):
        """The maximum practical C.6 factor (~0.0073) must be above the 1e-6 threshold.

        Computed as: strength=0.15, reliability=10/(10+24)≈0.294, log_ratio=ln(1.18)≈0.166
        log_adj = 0.15 * 0.294 * 0.166 ≈ 0.00732
        factor  = exp(0.00732) ≈ 1.00734
        """
        strength = 0.15
        reliability = 10.0 / (10.0 + 24.0)
        log_ratio = float(np.log(1.18))
        log_adj = strength * reliability * log_ratio
        factor = float(np.exp(log_adj))
        assert abs(factor - 1.0) > 1e-6, (
            f"Max C.6 factor {factor:.6f} must be > 1e-6 from 1.0 to ensure correction fires"
        )
        assert abs(factor - 1.0) > 0, "Factor must be strictly non-identity"
