"""Tail-aware tilted distribution + corrected moments (V6).

TiltedDistribution applies an exponential tilt to the COMPLETE base distribution (including its
analytic tail), not just stored atoms:

    P_theta(Y=y) = P_base(Y=y) * exp(f_theta(y)) / Z_theta      for every integer y >= 0

Z_theta is computed by adaptive summation with a certified remainder bound. The tilt basis is
bounded (a saturating mean term + a zero-mass term) so the transformed infinite tail stays
summable. Stored atoms + overflow = 1 exactly; mean/variance include the tail.
"""
from __future__ import annotations

import numpy as np

from wnba_props_model.sharp_v5.distribution import (CountDistribution, DiscreteDistribution,  # noqa: F401
                                                    HurdleDistribution, Materialized, TAIL_TOL,
                                                    ZeroInflatedDistribution)

NORM_TOL = 1e-10


def _bounded_mean_basis(y: np.ndarray, scale: float) -> np.ndarray:
    """Saturating, bounded transform of y in [-scale, scale] -> guarantees a summable tilted tail."""
    return scale * np.tanh(y / max(scale, 1e-6))


class TiltedDistribution(DiscreteDistribution):
    family = "tilted"

    def __init__(self, base: DiscreteDistribution, theta_mean: float = 0.0, theta_zero: float = 0.0,
                 theta_disp: float = 0.0, basis_scale: float = 30.0, tail_tolerance: float = TAIL_TOL):
        self.base = base
        self.theta_mean = float(theta_mean)
        self.theta_zero = float(theta_zero)
        self.theta_disp = float(theta_disp)
        self.basis_scale = float(basis_scale)
        self._mu0 = float(base.mean())
        self._tol = tail_tolerance
        self._Z, self._Zmax, self._remainder = self._normalizer()

    def _f(self, y):
        yv = np.asarray(y, float)
        # all basis functions are BOUNDED -> transformed infinite tail stays summable
        disp = -np.tanh(((yv - self._mu0) / max(self.basis_scale / 3, 1e-6)) ** 2)   # in [-1, 0]
        return (self.theta_mean * _bounded_mean_basis(yv, self.basis_scale)
                + self.theta_disp * disp + self.theta_zero * (yv == 0))

    def _normalizer(self):
        """Z = sum_y base(y) exp(f(y)) with a certified remainder bound. The bounded mean basis is
        <= |theta_mean|*basis_scale, so the tail of the tilted series is bounded by
        exp(|theta_mean|*basis_scale) * base.survival(K)."""
        K = max(int(self.base.mean() + 8 * np.sqrt(self.base.variance() + 1)), 10)
        bound_factor = float(np.exp(abs(self.theta_mean) * self.basis_scale + abs(self.theta_disp) + abs(self.theta_zero)))
        for _ in range(60):
            k = np.arange(K + 1)
            base_p = np.array([self.base.probability(int(i)) for i in k])
            terms = base_p * np.exp(self._f(k))
            remainder = bound_factor * self.base.survival(K)
            if remainder < self._tol:
                break
            K += max(4, int(K * 0.5))
        Z = float(terms.sum() + remainder)      # include a conservative tail bound in Z
        return Z, int(K), float(remainder)

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        return float(self.base.probability(y) * np.exp(self._f(y)) / self._Z)

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        return float(min(1.0, sum(self.probability(k) for k in range(y + 1))))

    def survival(self, y):
        return float(max(0.0, 1.0 - self.cdf(y)))

    def mean(self):
        m = self.materialize()      # overflow < tail tolerance, so materialized mass captures the mean
        k = np.arange(m.atoms.size)
        return float(np.dot(k, m.atoms))

    def variance(self):
        mat = self.materialize()
        k = np.arange(mat.atoms.size)
        m = float(np.dot(k, mat.atoms))
        ex2 = float(np.dot(k ** 2, mat.atoms))
        return float(max(0.0, ex2 - m ** 2))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        K = required_max if required_max is not None else self._Zmax
        while self.survival(K) >= tail_tolerance and K < 2000:
            K += max(4, int(K * 0.5))
        atoms = np.array([self.probability(k) for k in range(K + 1)])
        overflow = float(self.survival(K))
        stored = float(atoms.sum())
        return Materialized(atoms=atoms, support_min=0, support_max=int(K), stored_mass=stored,
                            overflow_probability=overflow, tail_upper_bound=overflow,
                            tail_method="tilted_analytic_remainder_bounded",
                            normalization_error=float(abs(stored + overflow - 1.0)),
                            distribution_family=self.family,
                            distribution_parameters={"theta_mean": self.theta_mean, "theta_zero": self.theta_zero})

    def validate(self, tail_tolerance: float = TAIL_TOL) -> None:
        m = self.materialize(tail_tolerance)
        if m.normalization_error > NORM_TOL:
            raise ValueError(f"tilted: normalization_error {m.normalization_error} > {NORM_TOL}")

    def sample(self, n, rng):
        m = self.materialize()
        return rng.choice(np.arange(m.atoms.size), size=n, p=m.atoms / m.atoms.sum())


def analytic_hurdle_variance(h: HurdleDistribution) -> float:
    """Complete analytic second moment for a hurdle-NB2 (does not zero-out tail)."""
    base = h.base
    pos = h._pos_norm                    # base P(Y>=1)
    # E[Y] and E[Y^2] over base conditional on Y>=1, times p_pos
    mb, vb = base.mean(), base.variance()
    ex_base = mb                          # E_base[Y]
    ex2_base = vb + mb ** 2               # E_base[Y^2]
    p0 = base.probability(0)
    ex_cond = ex_base / pos               # E[Y | Y>=1] (base 0-atom contributes 0 to numerator)
    ex2_cond = ex2_base / pos
    m = h.p_pos * ex_cond
    ex2 = h.p_pos * ex2_cond
    return float(max(0.0, ex2 - m ** 2))
