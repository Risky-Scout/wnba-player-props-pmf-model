"""PMF primitive tests for Opportunity V2 (sections 23 & 34)."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.opportunity.pmf_builders import (
    beta_binomial_pmf,
    convolve_pmfs,
    marginal_beta_binomial_pmf,
    pmf_mean,
    pmf_variance,
    poisson_or_nbinom_pmf,
    settled_over_probability,
    weighted_mix_pmfs,
)

TOL = 1e-6


def test_all_pmfs_sum_to_one():
    for p in (poisson_or_nbinom_pmf(3.0, None), poisson_or_nbinom_pmf(5.0, 4.0),
              beta_binomial_pmf(8, 3, 5), marginal_beta_binomial_pmf(poisson_or_nbinom_pmf(4.0, None), 3, 5)):
        assert abs(p.sum() - 1.0) < 1e-12
        assert np.all(p >= 0) and np.all(np.isfinite(p))


def test_poisson_mean_matches():
    p = poisson_or_nbinom_pmf(3.7, None)
    assert abs(pmf_mean(p) - 3.7) < TOL


def test_nbinom_mean_and_overdispersion():
    mu, r = 6.0, 3.0
    p = poisson_or_nbinom_pmf(mu, r)
    assert abs(pmf_mean(p) - mu) < 1e-4
    # NB2 variance = mu + mu^2/r > mu (overdispersed vs Poisson)
    assert pmf_variance(p) > mu


def test_tail_mass_below_tolerance():
    p = poisson_or_nbinom_pmf(2.0, None, tail_tolerance=1e-10)
    assert abs(p.sum() - 1.0) < 1e-12


def test_reject_negative_and_nonfinite():
    with pytest.raises(ValueError):
        convolve_pmfs(np.array([0.5, -0.1, 0.6]))
    with pytest.raises(ValueError):
        weighted_mix_pmfs([np.array([np.nan, 1.0])], np.array([1.0]))


def test_convolution_sum_of_dice_like():
    # sum of two identical 0/1 fair coins -> [0.25, 0.5, 0.25]
    coin = np.array([0.5, 0.5])
    conv = convolve_pmfs(coin, coin)
    assert np.allclose(conv, [0.25, 0.5, 0.25])
    assert abs(pmf_mean(conv) - 1.0) < TOL


def test_weighted_mix_of_point_masses():
    a = np.array([1.0, 0.0, 0.0])   # mass at 0
    b = np.array([0.0, 0.0, 1.0])   # mass at 2
    mix = weighted_mix_pmfs([a, b], np.array([0.5, 0.5]))
    assert abs(pmf_mean(mix) - 1.0) < TOL
    assert abs(mix[0] - 0.5) < TOL and abs(mix[2] - 0.5) < TOL


# --- Beta-binomial monotonicity (section 34 PMF tests) ---------------------

def test_increasing_attempts_increases_expected_makes():
    m4 = pmf_mean(beta_binomial_pmf(4, 3, 5))
    m8 = pmf_mean(beta_binomial_pmf(8, 3, 5))
    assert m8 > m4


def test_marginal_bb_mean_equals_attempt_mean_times_p():
    # E[makes] = E[attempts] * alpha/(alpha+beta)
    attempts = poisson_or_nbinom_pmf(4.0, None)
    alpha, beta = 3.0, 5.0
    marg = marginal_beta_binomial_pmf(attempts, alpha, beta)
    expected = pmf_mean(attempts) * alpha / (alpha + beta)
    assert abs(pmf_mean(marg) - expected) < 1e-4


# --- Analytical mean tests (section 34) ------------------------------------

def _fixed_conversion_bb(attempts: int, p: float, strength: float = 1e6):
    # High-strength Beta centered at p behaves like Binomial(attempts, p).
    return beta_binomial_pmf(attempts, p * strength, (1 - p) * strength)


def test_analytical_fg3m():
    # 3PA=4, conversion mean 0.375 -> E[FG3M]=1.5
    pmf = _fixed_conversion_bb(4, 0.375)
    assert abs(pmf_mean(pmf) - 1.5) < 1e-3


def test_analytical_assists():
    # potential assists=10, conversion 0.60 -> E[AST]=6.0
    pmf = _fixed_conversion_bb(10, 0.60)
    assert abs(pmf_mean(pmf) - 6.0) < 1e-3


def test_analytical_rebounds():
    # rebound chances=12, conversion 0.50 -> E[REB]=6.0
    pmf = _fixed_conversion_bb(12, 0.50)
    assert abs(pmf_mean(pmf) - 6.0) < 1e-3


def test_analytical_points_convolution():
    # FG2A=8 p2=0.50 ; FG3A=4 p3=0.375 ; FTA=4 pft=0.80
    # PTS = 2*FG2M + 3*FG3M + FTM ; E = 2*4 + 3*1.5 + 3.2 = 15.7
    fg2m = _fixed_conversion_bb(8, 0.50)
    fg3m = _fixed_conversion_bb(4, 0.375)
    ftm = _fixed_conversion_bb(4, 0.80)
    # scale points: 2*FG2M -> stretch support by 2; 3*FG3M -> by 3
    def stretch(pmf, mult):
        out = np.zeros((pmf.size - 1) * mult + 1)
        out[:: mult] = pmf
        return out
    pts = convolve_pmfs(stretch(fg2m, 2), stretch(fg3m, 3), ftm)
    assert abs(pmf_mean(pts) - 15.7) < 1e-2


# --- push-safe settlement --------------------------------------------------

def test_integer_line_push_handling():
    # PMF: P(0)=0.2, P(1)=0.5, P(2)=0.3 ; line=1 -> push=0.5
    pmf = np.array([0.2, 0.5, 0.3])
    p_over, p_under, p_push = settled_over_probability(pmf, 1.0)
    assert abs(p_push - 0.5) < TOL
    assert abs(p_over - 0.3 / 0.5) < TOL
    assert abs(p_under - 0.2 / 0.5) < TOL
    assert abs((p_over + p_under) - 1.0) < TOL


def test_half_line_no_push():
    pmf = np.array([0.2, 0.5, 0.3])
    p_over, p_under, p_push = settled_over_probability(pmf, 1.5)
    assert p_push == 0.0
    assert abs(p_over - 0.3) < TOL
    assert abs(p_under - 0.7) < TOL
