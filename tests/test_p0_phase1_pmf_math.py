"""P0 Phase-1 adversarial tests: mixture mass, tails, moments, settlement parity."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.sharp_v6.distribution import (
    NORM_TOL,
    ConvolutionDistribution,
    CountDistribution,
    HurdleDistribution,
    MixtureDistribution,
    TiltedDistribution,
    ZeroInflatedDistribution,
    materialize_minutes_mixture,
    minutes_count_mixture,
)
from wnba_props_model.sharp_v6.models import mix_atoms


def _assert_valid_dist(d, *, line_int: float = 6.0, line_half: float = 6.5):
    m = d.materialize()
    assert np.all(np.isfinite(m.atoms))
    assert np.all(m.atoms >= -1e-15)
    assert m.normalization_error <= NORM_TOL
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL
    # CDF monotone
    cdf = np.cumsum(m.atoms)
    assert np.all(np.diff(cdf) >= -1e-15)
    # survival consistent
    for y in (0, 1, 5, m.support_max):
        assert d.survival(y) == pytest.approx(1.0 - d.cdf(y), abs=1e-10)
    assert np.isfinite(d.mean())
    assert d.variance() >= -1e-12
    s_int = d.settle_over_under(line_int)
    s_half = d.settle_over_under(line_half)
    assert s_int.p_push >= 0
    assert s_half.p_push == 0.0
    assert abs(s_int.p_over_win + s_int.p_under_win + s_int.p_push - 1.0) <= 1e-9


def test_wide_minutes_mixture_many_small_weights():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(49) * 0.25)
    d = minutes_count_mixture(0.85, 2.5, w)
    _assert_valid_dist(d)
    m = d.materialize(required_max=40)
    assert m.discarded_mixture_mass <= 1e-10
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL


def test_mass_previously_dropped_by_1e4_rule_is_retained():
    """Construct minutes weights where the old 1e-4 threshold drops measurable mass."""
    w = np.full(49, 1e-5)
    w[20:25] = 0.2
    w = w / w.sum()
    old_dropped = float(w[w <= 1e-4].sum())
    assert old_dropped > 1e-4  # defect would have dropped this
    atoms, ovf = mix_atoms(0.9, 3.0, w, 50)
    assert abs(float(atoms.sum()) + ovf - 1.0) <= NORM_TOL
    # Independent high-support check
    mix = minutes_count_mixture(0.9, 3.0, w)
    for y in range(51):
        assert atoms[y] == pytest.approx(mix.probability(y), abs=1e-12)


def test_high_dispersion_nb2_material_tail():
    d = CountDistribution(12.0, 1.5)
    m = d.materialize(tail_tolerance=1e-6)
    assert m.overflow_probability <= 1e-6
    # Exact atom beyond materialized support still available
    y = m.support_max + 5
    assert d.probability(y) > 0
    assert d.probability(y) == pytest.approx(
        np.exp(d.log_probability(y)), rel=1e-10, abs=1e-20
    )
    _assert_valid_dist(d)


def test_hurdle_material_positive_tail():
    base = CountDistribution(8.0, 1.8)
    h = HurdleDistribution(0.35, base)
    m = h.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL
    # Overflow must be p_pos * P_base(Y>K | Y>=1)
    K = m.support_max
    pos = 1.0 - base.probability(0)
    expected_ovf = 0.35 * base.survival(K) / pos
    assert m.overflow_probability == pytest.approx(expected_ovf, abs=1e-10)
    assert h.probability(0) == pytest.approx(0.65, abs=1e-12)
    # Analytic moments (not finite-atoms with zeroed tail)
    assert h.variance() == pytest.approx(
        __import__("wnba_props_model.sharp_v6.distribution", fromlist=["analytic_hurdle_variance"])
        .analytic_hurdle_variance(h),
        abs=1e-12,
    )
    _assert_valid_dist(h)


def test_zero_inflated_mass():
    base = CountDistribution(5.0, 4.0)
    zi = ZeroInflatedDistribution(0.4, base)
    m = zi.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL
    assert zi.probability(0) == pytest.approx(0.4 + 0.6 * base.probability(0), abs=1e-12)
    assert zi.mean() == pytest.approx(0.6 * base.mean(), abs=1e-12)
    _assert_valid_dist(zi)


def test_convolution_with_material_tail_component():
    a = CountDistribution(6.0, 1.5)  # heavy tail
    b = CountDistribution(4.0, 8.0)
    conv = ConvolutionDistribution(a, b, tail_tolerance=1e-6)
    m = conv.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL
    # Atom inside support equals full sum of contributing terms
    y = 10
    exact = sum(a.probability(i) * b.probability(y - i) for i in range(y + 1))
    assert conv.probability(y) == pytest.approx(exact, abs=1e-10)
    assert conv.mean() == pytest.approx(a.mean() + b.mean(), abs=1e-10)
    _assert_valid_dist(conv)


def test_tilted_nontrivial_transformed_tail():
    base = CountDistribution(6.0, 2.0)
    td = TiltedDistribution(base, theta_mean=0.12, theta_disp=0.3, theta_zero=0.2)
    m = td.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= NORM_TOL
    assert td._normalizer_exact is True
    # Tail atom transformed relative to mode
    r_tail = td.probability(30) / max(base.probability(30), 1e-300)
    r_mode = td.probability(6) / max(base.probability(6), 1e-300)
    assert abs(r_tail - r_mode) > 1e-6
    _assert_valid_dist(td)


def test_integer_line_push_settlement():
    d = CountDistribution(8.0, 6.0)
    s = d.settle_over_under(8.0)
    assert s.p_push == pytest.approx(d.probability(8), abs=1e-12)
    assert abs(s.p_over_win + s.p_under_win + s.p_push - 1.0) <= 1e-10


def test_half_point_settlement():
    d = CountDistribution(8.0, 6.0)
    s = d.settle_over_under(8.5)
    assert s.p_push == 0.0
    assert s.p_over_win == pytest.approx(d.survival(8), abs=1e-12)
    assert s.p_under_win == pytest.approx(d.cdf(8), abs=1e-12)


def test_exact_probability_above_materialized_support():
    d = CountDistribution(10.0, 2.0)
    m = d.materialize(tail_tolerance=1e-4)
    y = m.support_max + 7
    # Must not return aggregate overflow as the atom
    assert d.probability(y) < m.overflow_probability
    assert d.probability(y) > 0


def test_mean_variance_match_high_support_sum():
    d = MixtureDistribution(
        [CountDistribution(3.0, 4.0), CountDistribution(12.0, 2.0)],
        np.array([0.4, 0.6]),
    )
    K = 200
    atoms = np.array([d.probability(y) for y in range(K + 1)])
    ovf = d.survival(K)
    assert ovf < 1e-9
    ks = np.arange(K + 1)
    mean_sum = float(np.dot(ks, atoms))
    var_sum = float(np.dot((ks - mean_sum) ** 2, atoms))
    assert d.mean() == pytest.approx(mean_sum, rel=1e-6, abs=1e-6)
    assert d.variance() == pytest.approx(var_sum, rel=1e-5, abs=1e-5)


def test_sampling_moments_near_analytic():
    rng = np.random.default_rng(1)
    d = CountDistribution(7.0, 5.0)
    x = d.sample(200_000, rng)
    assert float(np.mean(x)) == pytest.approx(d.mean(), rel=0.02)
    assert float(np.var(x)) == pytest.approx(d.variance(), rel=0.05)


def test_oof_live_settlement_equality_same_distribution():
    """OOF array path and live distribution path must agree on settlement."""
    w = np.zeros(41)
    w[10:35] = np.linspace(0.01, 0.08, 25)
    w = w / w.sum()
    atoms, ovf = mix_atoms(0.7, 4.0, w, 60)
    dist = minutes_count_mixture(0.7, 4.0, w)
    for line in (4.5, 8.0, 12.5):
        # Array settlement (live delivery style)
        k = np.arange(atoms.size)
        A = float(atoms[k > line].sum()) + ovf
        B = float(atoms[k < line].sum())
        P = float(atoms[int(line)]) if float(line).is_integer() and 0 <= int(line) < atoms.size else 0.0
        s = dist.settle_over_under(line)
        assert A == pytest.approx(s.p_over_win, abs=1e-10)
        assert B == pytest.approx(s.p_under_win, abs=1e-10)
        assert P == pytest.approx(s.p_push, abs=1e-10)


def test_no_count_observation_clipped_in_probability():
    d = CountDistribution(20.0, 3.0)
    # Large outcome still has finite exact probability (not clipped to support max)
    assert 0 < d.probability(80) < 1
    assert np.isfinite(d.log_probability(80))


def test_invalid_weights_fail_closed():
    with pytest.raises(ValueError):
        MixtureDistribution([CountDistribution(1.0, 2.0)], np.array([np.nan]))
    with pytest.raises(ValueError):
        MixtureDistribution([CountDistribution(1.0, 2.0)], np.array([-0.1]))
    with pytest.raises(ValueError):
        MixtureDistribution([CountDistribution(1.0, 2.0)], np.array([0.0]))
    with pytest.raises(ValueError):
        minutes_count_mixture(1.0, 2.0, np.array([0.0, 0.0]))


def test_old_mix_atoms_defect_no_longer_present():
    """Regression: atoms + overflow must not exceed 1 (old renorm bug)."""
    rng = np.random.default_rng(2)
    w = rng.dirichlet(np.ones(49) * 0.3)
    atoms, ovf = mix_atoms(0.8, 2.0, w, 40)
    assert atoms.sum() <= 1.0 + 1e-12
    assert abs(float(atoms.sum()) + ovf - 1.0) <= NORM_TOL
    # Old code would have atoms.sum()==1 and ovf≈1-s_before > 0 → sum>1
    assert float(atoms.sum()) < 1.0 or ovf <= NORM_TOL


def test_mix_atoms_matches_mixture_distribution():
    w = np.linspace(0.01, 0.05, 40)
    w = w / w.sum()
    atoms, ovf = mix_atoms(1.1, 3.5, w, 55)
    mat = materialize_minutes_mixture(1.1, 3.5, w, 55, tail_tolerance=1.0)
    assert atoms == pytest.approx(mat.atoms, abs=1e-12)
    assert ovf == pytest.approx(mat.overflow_probability, abs=1e-12)


def test_hurdle_on_minutes_mixture_mass():
    w = np.zeros(41)
    w[15:30] = 1.0
    w = w / w.sum()
    base = minutes_count_mixture(0.6, 2.0, w)
    h = HurdleDistribution(0.55, base)
    atoms = np.array([h.probability(y) for y in range(50)])
    ovf = h.survival(49)
    assert abs(float(atoms.sum()) + ovf - 1.0) <= NORM_TOL
    _assert_valid_dist(h)


def test_inference_normalize_pmf_does_not_renorm():
    from wnba_props_model.sharp_v6.inference import InferenceError, _normalize_pmf

    atoms = np.array([0.5, 0.4])
    ovf = 0.1
    a, o = _normalize_pmf(atoms, ovf, mode="production", context="ok")
    assert a.sum() + o == pytest.approx(1.0, abs=1e-15)
    # Broken mass must fail closed — not silently renormalized
    with pytest.raises(InferenceError):
        _normalize_pmf(np.array([0.9, 0.9]), 0.2, mode="production", context="bad")
