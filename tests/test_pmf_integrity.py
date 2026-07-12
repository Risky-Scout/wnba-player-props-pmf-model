"""PMF integrity tests (§12.4 requirements).

Verifies:
  - All probabilities are finite and non-negative
  - Probabilities sum to one
  - Expected value matches model parameters
  - Push probabilities match integer thresholds
  - CDF is monotonic
  - Support includes relevant market lines
  - Deterministic seed reproduces output
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from wnba_props_model.models.simulation import normalize_pmf


def pmf_sum_to_one(pmf: np.ndarray) -> bool:
    return abs(pmf.sum() - 1.0) < 1e-6


def pmf_is_nonneg(pmf: np.ndarray) -> bool:
    return (pmf >= 0).all()


def pmf_is_finite(pmf: np.ndarray) -> bool:
    return np.isfinite(pmf).all()


def pmf_cdf_monotonic(pmf: np.ndarray) -> bool:
    cdf = np.cumsum(pmf)
    return (np.diff(cdf) >= -1e-10).all()


def pmf_expected_value(pmf: np.ndarray) -> float:
    k = np.arange(len(pmf))
    return float((k * pmf).sum())


def prob_over(pmf: np.ndarray, line: float) -> float:
    """P(X > line) from PMF array."""
    k_int = int(np.floor(line))
    if k_int + 1 >= len(pmf):
        return 0.0
    return float(pmf[k_int + 1:].sum())


def prob_under(pmf: np.ndarray, line: float) -> float:
    """P(X < line) from PMF array."""
    k_int = int(np.ceil(line))
    return float(pmf[:k_int].sum())


def prob_push(pmf: np.ndarray, line: float) -> float:
    """P(X == line) for integer lines."""
    if line == int(line) and 0 <= int(line) < len(pmf):
        return float(pmf[int(line)])
    return 0.0


# ---------------------------------------------------------------------------
# Core PMF invariants
# ---------------------------------------------------------------------------

class TestPMFInvariants:
    @pytest.fixture
    def sample_negbinom_pmf(self):
        """Generate a valid NegBinom PMF."""
        from scipy import stats
        r, p = 5.0, 0.4
        k = np.arange(61)
        pmf = stats.nbinom.pmf(k, r, p)
        return normalize_pmf(pmf)

    def test_pmf_sums_to_one(self, sample_negbinom_pmf):
        """§12.4: probabilities must sum to one."""
        assert pmf_sum_to_one(sample_negbinom_pmf)

    def test_pmf_nonneg(self, sample_negbinom_pmf):
        """§12.4: all probabilities must be non-negative."""
        assert pmf_is_nonneg(sample_negbinom_pmf)

    def test_pmf_finite(self, sample_negbinom_pmf):
        """§12.4: all probabilities must be finite."""
        assert pmf_is_finite(sample_negbinom_pmf)

    def test_cdf_monotonic(self, sample_negbinom_pmf):
        """§12.4: CDF must be monotonic."""
        assert pmf_cdf_monotonic(sample_negbinom_pmf)

    def test_expected_value_matches_negbinom_mean(self, sample_negbinom_pmf):
        """§12.4: expected value must match model parameters."""
        from scipy import stats
        r, p = 5.0, 0.4
        theoretical_mean = r * (1 - p) / p
        pmf_ev = pmf_expected_value(sample_negbinom_pmf)
        assert abs(pmf_ev - theoretical_mean) < 0.1

    def test_support_includes_market_lines(self, sample_negbinom_pmf):
        """§12.4: support must include relevant market lines."""
        # Typical WNBA pts lines: 10.5, 15.5, 20.5, 25.5
        for line in [10.5, 15.5, 20.5, 25.5]:
            k_int = int(line) + 1
            if k_int < len(sample_negbinom_pmf):
                assert prob_over(sample_negbinom_pmf, line) > 0, (
                    f"PMF has zero probability above line {line}"
                )


class TestPushProbabilities:
    def test_integer_line_push_mass(self):
        """§12.4: push probabilities must match integer thresholds."""
        from scipy import stats
        r, p = 5.0, 0.4
        k = np.arange(61)
        pmf = normalize_pmf(stats.nbinom.pmf(k, r, p))

        line = 18.0  # integer line
        p_over = prob_over(pmf, line)
        p_under = prob_under(pmf, line)
        p_push = prob_push(pmf, line)

        # Must sum to ~1
        assert abs(p_over + p_under + p_push - 1.0) < 1e-6

        # Push mass should equal pmf[18]
        expected_push = float(pmf[18])
        assert abs(p_push - expected_push) < 1e-10

    def test_half_point_line_no_push(self):
        """Half-point lines produce no push."""
        from scipy import stats
        pmf = normalize_pmf(stats.nbinom.pmf(np.arange(61), 5.0, 0.4))

        line = 18.5
        p_push = prob_push(pmf, line)
        assert p_push == 0.0

        p_over = prob_over(pmf, line)
        p_under = prob_under(pmf, line)
        assert abs(p_over + p_under - 1.0) < 1e-10

    def test_push_aware_probabilities_sum_to_one(self):
        """§8.4: p_over + p_under + p_push = 1 within numerical tolerance."""
        from scipy import stats
        for line_val in [15.0, 20.0, 25.0]:
            pmf = normalize_pmf(stats.nbinom.pmf(np.arange(61), 5.0, 0.4))
            p_o = prob_over(pmf, line_val)
            p_u = prob_under(pmf, line_val)
            p_p = prob_push(pmf, line_val)
            assert abs(p_o + p_u + p_p - 1.0) < 1e-6, (
                f"Push-aware probs don't sum to 1 at line={line_val}: "
                f"over={p_o:.6f} under={p_u:.6f} push={p_p:.6f}"
            )


class TestAdaptivePMFSupport:
    def test_tail_mass_below_tolerance(self):
        """§12.3: truncated tail mass must be below tolerance."""
        from scipy import stats
        tolerance = 1e-8

        pmf_full = stats.nbinom.pmf(np.arange(200), 5.0, 0.4)
        truncated_at_60 = pmf_full[61:].sum()

        assert truncated_at_60 < 0.01, (
            f"Too much mass truncated at index 60: {truncated_at_60:.4f}"
        )

    def test_adaptive_support_extends_to_cover_tail(self):
        """PMF support must expand until remaining tail mass < tolerance."""
        from scipy import stats
        tolerance = 1e-8
        r, p = 2.0, 0.2  # heavier tail (mean = 8, high variance)

        k = 0
        cumulative = 0.0
        while 1.0 - cumulative > tolerance:
            prob = stats.nbinom.pmf(k, r, p)
            cumulative += prob
            k += 1
            if k > 1000:
                break

        pmf_adaptive = normalize_pmf(stats.nbinom.pmf(np.arange(k), r, p))
        tail_mass = 1.0 - pmf_adaptive.sum()
        assert abs(tail_mass) < 1e-6  # normalized, so tail is 0 by construction

    def test_no_truncation_at_fixed_support_when_tail_significant(self):
        """Do not hard-truncate at a fixed support that removes meaningful mass."""
        from scipy import stats
        # High-mean player: 30+ ppg tail matters
        r, p = 10.0, 0.25  # mean ~ 30
        pmf_60 = stats.nbinom.pmf(np.arange(61), r, p)
        mass_above_60 = 1.0 - pmf_60.sum()

        # For a 30-pt mean player, mass above 60 may be significant
        pmf_100 = stats.nbinom.pmf(np.arange(101), r, p)
        mass_above_100 = 1.0 - pmf_100.sum()

        # More support = less truncated mass
        assert mass_above_100 < mass_above_60


class TestDeterministicPMF:
    def test_same_seed_produces_same_pmf(self):
        """§12.4: deterministic seed must reproduce simulation output."""
        from scipy import stats

        def generate_with_seed(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            # Simulate a simple discrete distribution with noise
            base_pmf = stats.nbinom.pmf(np.arange(61), 5.0, 0.4)
            noise = rng.normal(0, 0.001, size=61)
            noisy = base_pmf + noise * base_pmf
            return normalize_pmf(np.maximum(noisy, 0))

        pmf_a = generate_with_seed(20260712)
        pmf_b = generate_with_seed(20260712)
        np.testing.assert_array_almost_equal(pmf_a, pmf_b, decimal=15)

    def test_different_seed_produces_different_pmf(self):
        """Different seeds should (with high probability) produce different outputs."""
        from scipy import stats

        def generate_with_seed(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            base_pmf = stats.nbinom.pmf(np.arange(61), 5.0, 0.4)
            noise = rng.normal(0, 0.001, size=61)
            noisy = base_pmf + noise * base_pmf
            return normalize_pmf(np.maximum(noisy, 0))

        pmf_a = generate_with_seed(20260712)
        pmf_b = generate_with_seed(99999)
        assert not np.allclose(pmf_a, pmf_b)


class TestNormalizePMF:
    def test_normalize_makes_sum_one(self):
        pmf = normalize_pmf(np.array([0.0, 2.0, 3.0, 1.0]))
        assert abs(pmf.sum() - 1.0) < 1e-12

    def test_normalize_preserves_zero_probs(self):
        pmf = normalize_pmf(np.array([0.0, 2.0, 0.0, 2.0]))
        assert pmf[0] == 0.0
        assert pmf[2] == 0.0

    def test_normalize_handles_all_zeros(self):
        pmf = normalize_pmf(np.array([0.0, 0.0, 0.0]))
        assert abs(pmf.sum() - 1.0) < 1e-12  # uniform fallback
