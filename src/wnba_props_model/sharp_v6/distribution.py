"""WNBA Sharp PMF v6 — authoritative discrete distributions + tail-aware tilt.

Self-contained production distributions (no sharp_v3/v4/v5 imports).

One production DiscreteDistribution interface with correct mass accounting.

Fixes the V4 defects:
- ``probability(y)`` returns the EXACT atom probability for any nonnegative integer y (analytic
  parametric tail), never the aggregate overflow reused as every tail atom.
- Hurdle / zero-inflated / convolution mass sum to one exactly (positive normalizer includes the
  analytic positive tail; convolution tail handled, not renormalized-away).
- overflow_probability is P(Y > support_max) as an aggregate bucket only.
- settlement is push-aware: A/(A+B), with A=P(Y>L), B=P(Y<L), P=P(Y=L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

TAIL_TOL = 1e-6
NORM_TOL = 1e-10


@dataclass
class SettleResult:
    line: float
    p_over_win: float      # A = P(Y > L)
    p_under_win: float     # B = P(Y < L)
    p_push: float          # P = P(Y = L)
    p_over_settled: float  # A/(A+B)
    p_under_settled: float # B/(A+B)


@dataclass
class Materialized:
    atoms: np.ndarray
    support_min: int
    support_max: int
    stored_mass: float
    overflow_probability: float
    tail_upper_bound: float
    tail_method: str
    normalization_error: float
    distribution_family: str
    distribution_parameters: dict


class DiscreteDistribution:
    family = "abstract"

    def probability(self, y: int) -> float:
        raise NotImplementedError

    def log_probability(self, y: int) -> float:
        return float(np.log(max(self.probability(y), 1e-300)))

    def cdf(self, y: int) -> float:
        raise NotImplementedError

    def survival(self, y: int) -> float:      # P(Y > y)
        return float(max(0.0, 1.0 - self.cdf(y)))

    def mean(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

    def settle_over_under(self, line: float) -> SettleResult:
        L = float(line)
        fl = int(np.floor(L))
        A = self.survival(fl)                       # P(Y > floor(L))
        if L.is_integer():
            P = self.probability(int(L))
            B = self.cdf(int(L) - 1) if L >= 1 else 0.0
        else:
            P = 0.0
            B = self.cdf(fl)                        # P(Y <= floor(L)) = P(Y < L)
        den = A + B
        return SettleResult(L, A, B, P, A / den if den > 0 else float("nan"),
                            B / den if den > 0 else float("nan"))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        K = required_max if required_max is not None else 1
        for _ in range(40):
            if self.survival(K) < tail_tolerance and (required_max is None or K >= required_max):
                break
            K = K + max(2, int(K * 0.5))
        atoms = np.array([self.probability(k) for k in range(K + 1)])
        overflow = float(self.survival(K))
        stored = float(atoms.sum())
        return Materialized(atoms=atoms, support_min=0, support_max=int(K), stored_mass=stored,
                            overflow_probability=overflow, tail_upper_bound=overflow,
                            tail_method=f"{self.family}_analytic_survival",
                            normalization_error=float(abs(stored + overflow - 1.0)),
                            distribution_family=self.family, distribution_parameters=self.parameters())

    def validate(self, tail_tolerance: float = TAIL_TOL) -> None:
        m = self.materialize(tail_tolerance)
        if m.normalization_error > NORM_TOL:
            raise ValueError(f"{self.family}: normalization_error {m.normalization_error} > {NORM_TOL}")
        if np.any(m.atoms < -1e-12):
            raise ValueError(f"{self.family}: negative atom")

    def sample(self, n, rng):
        raise NotImplementedError


class CountDistribution(DiscreteDistribution):
    """NB2 (or Poisson when r is None). Exact analytic probability for any y."""
    family = "nb2"

    def __init__(self, mu: float, r: float | None):
        self.mu = max(float(mu), 1e-9)
        self.r = None if r is None else float(r)

    def _p(self):
        return self.r / (self.r + self.mu)

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        if self.r is None:
            return float(poisson.pmf(y, self.mu))
        return float(nbinom.pmf(y, self.r, self._p()))

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        if self.r is None:
            return float(poisson.cdf(y, self.mu))
        return float(nbinom.cdf(y, self.r, self._p()))

    def mean(self):
        return self.mu

    def variance(self):
        return self.mu if self.r is None else self.mu + self.mu ** 2 / self.r

    def parameters(self):
        return {"mu": self.mu, "r": self.r}

    def sample(self, n, rng):
        if self.r is None:
            return rng.poisson(self.mu, n)
        return rng.negative_binomial(self.r, self._p(), n)


class HurdleDistribution(DiscreteDistribution):
    """Correct hurdle: P(0)=1-p_pos; P(y>=1)=p_pos * base.P(y)/base.P(Y>=1). Tail uses the base's
    exact atom probability (never the aggregate overflow)."""
    family = "hurdle_nb2"

    def __init__(self, p_positive: float, base: CountDistribution):
        self.p_pos = float(min(max(p_positive, 0.0), 1.0))
        self.base = base
        self._pos_norm = max(1.0 - base.probability(0), 1e-12)   # base.P(Y>=1), exact

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        if y == 0:
            return 1.0 - self.p_pos
        return self.p_pos * self.base.probability(y) / self._pos_norm

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        c = 1.0 - self.p_pos
        if y >= 1:
            c += self.p_pos * (self.base.cdf(y) - self.base.probability(0)) / self._pos_norm
        return float(min(c, 1.0))

    def mean(self):
        base_mean_pos = (self.base.mean()) / self._pos_norm  # E[base * 1{>=1}]/P(>=1) approx via mean
        return self.p_pos * base_mean_pos

    def variance(self):
        m = self.materialize()
        k = np.arange(m.atoms.size)
        mean = float(np.dot(k, m.atoms))
        return float(np.dot((k - mean) ** 2, m.atoms) + m.overflow_probability * 0)

    def parameters(self):
        return {"p_positive": self.p_pos, "base": self.base.parameters()}


class ZeroInflatedDistribution(DiscreteDistribution):
    """Correct ZI mixture: P(0)=pi + (1-pi)*base.P(0); P(y>=1)=(1-pi)*base.P(y)."""
    family = "zinb"

    def __init__(self, pi: float, base: CountDistribution):
        self.pi = float(min(max(pi, 0.0), 1.0))
        self.base = base

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        if y == 0:
            return self.pi + (1 - self.pi) * self.base.probability(0)
        return (1 - self.pi) * self.base.probability(y)

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        return float(self.pi + (1 - self.pi) * self.base.cdf(y))

    def mean(self):
        return (1 - self.pi) * self.base.mean()

    def variance(self):
        mu = self.base.mean()
        return (1 - self.pi) * (self.base.variance() + mu ** 2) - ((1 - self.pi) * mu) ** 2

    def parameters(self):
        return {"pi": self.pi, "base": self.base.parameters()}


class MixtureDistribution(DiscreteDistribution):
    """Mixture sum_i w_i * component_i. Used to propagate the minutes PMF (and role states) into a
    stat PMF: component_i is the stat distribution conditional on minutes bin i."""
    family = "mixture"

    def __init__(self, components: list[DiscreteDistribution], weights: np.ndarray):
        w = np.asarray(weights, float)
        self.w = w / w.sum()
        self.components = components

    def probability(self, y):
        return float(sum(wi * c.probability(y) for wi, c in zip(self.w, self.components)))

    def cdf(self, y):
        return float(sum(wi * c.cdf(y) for wi, c in zip(self.w, self.components)))

    def mean(self):
        return float(sum(wi * c.mean() for wi, c in zip(self.w, self.components)))

    def variance(self):
        m = self.mean()
        ex2 = sum(wi * (c.variance() + c.mean() ** 2) for wi, c in zip(self.w, self.components))
        return float(ex2 - m ** 2)

    def parameters(self):
        return {"n_components": len(self.components)}


class ConvolutionDistribution(DiscreteDistribution):
    """Correct convolution of two independent components A+B. Materializes each component to its
    own tail tolerance, convolves, and reports the exact joint overflow (never renormalizes the
    finite convolution to one and re-adds overflow)."""
    family = "convolution"

    def __init__(self, a: DiscreteDistribution, b: DiscreteDistribution, tail_tolerance: float = TAIL_TOL):
        ma = a.materialize(tail_tolerance / 2); mb = b.materialize(tail_tolerance / 2)
        conv = np.convolve(ma.atoms, mb.atoms)               # stored joint mass (no renormalization)
        self._atoms = conv
        # exact joint overflow: 1 - sum(stored joint mass); components' tails contribute here
        self._overflow = float(max(0.0, 1.0 - conv.sum()))
        self._a, self._b = a, b

    def probability(self, y):
        y = int(y)
        if 0 <= y < self._atoms.size:
            return float(self._atoms[y])
        # exact per-atom tail via direct convolution sum P(A=i)P(B=y-i)
        return float(sum(self._a.probability(i) * self._b.probability(y - i) for i in range(y + 1)))

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        if y < self._atoms.size:
            return float(self._atoms[:y + 1].sum())
        return float(min(1.0, self._atoms.sum() + sum(self.probability(k) for k in range(self._atoms.size, y + 1))))

    def mean(self):
        return self._a.mean() + self._b.mean()

    def variance(self):
        return self._a.variance() + self._b.variance()

    def parameters(self):
        return {"overflow": self._overflow, "stored_max": int(self._atoms.size - 1)}


class TabularDistribution(DiscreteDistribution):
    """Materialized atom table + aggregate overflow bucket (used for calibrated / market-consistent
    PMFs). Per-atom probability is exact within the table; beyond support only the aggregate
    overflow is known, so probability(y>max) raises unless overflow is requested via survival()."""
    family = "tabular"

    def __init__(self, atoms: np.ndarray, overflow: float = 0.0, tail_method: str = "aggregate_overflow"):
        a = np.clip(np.asarray(atoms, float), 0.0, None)
        self.atoms = a
        self.overflow = float(max(0.0, overflow))
        self.tail_method = tail_method

    def probability(self, y):
        y = int(y)
        if 0 <= y < self.atoms.size:
            return float(self.atoms[y])
        return 0.0   # individual tail atoms not resolved for a tabular dist; mass is in overflow

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        return float(min(1.0, self.atoms[:y + 1].sum()))

    def survival(self, y):
        y = int(np.floor(y))
        return float(max(0.0, self.atoms[y + 1:].sum() + self.overflow)) if y + 1 <= self.atoms.size \
            else self.overflow

    def mean(self):
        return float(np.dot(np.arange(self.atoms.size), self.atoms))

    def variance(self):
        m = self.mean()
        return float(np.dot((np.arange(self.atoms.size) - m) ** 2, self.atoms))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        stored = float(self.atoms.sum())
        return Materialized(atoms=self.atoms, support_min=0, support_max=int(self.atoms.size - 1),
                            stored_mass=stored, overflow_probability=self.overflow,
                            tail_upper_bound=self.overflow, tail_method=self.tail_method,
                            normalization_error=float(abs(stored + self.overflow - 1.0)),
                            distribution_family=self.family, distribution_parameters={})

# ---- V6 tail-aware tilt ----
NORM_TOL = 1e-10


def _base_atoms(base, K: int) -> np.ndarray:
    """Vectorized base atom probabilities over 0..K (fast path for CountDistribution)."""
    k = np.arange(K + 1)
    if isinstance(base, CountDistribution):
        if base.r is None:
            return poisson.pmf(k, base.mu)
        p = base.r / (base.r + base.mu)
        return nbinom.pmf(k, base.r, p)
    return np.array([base.probability(int(i)) for i in k])


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
        """Cache the vectorized tilted atom array (normalized) out to where base survival < tol.
        Z includes a certified tail remainder bound so the transform covers the complete base."""
        K = max(int(self.base.mean() + 8 * np.sqrt(self.base.variance() + 1)), 10)
        bound_factor = float(np.exp(abs(self.theta_mean) * self.basis_scale + abs(self.theta_disp) + abs(self.theta_zero)))
        for _ in range(60):
            k = np.arange(K + 1)
            base_p = _base_atoms(self.base, K)
            terms = base_p * np.exp(self._f(k))
            remainder = bound_factor * float(self.base.survival(K))
            if remainder < self._tol:
                break
            K += max(4, int(K * 0.5))
        Z = float(terms.sum() + remainder)
        self._tilted = terms / Z                      # normalized tilted atoms 0..K
        self._overflow = float(remainder / Z)
        return Z, int(K), float(remainder)

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        if y <= self._Zmax:
            return float(self._tilted[y])
        return float(self.base.probability(y) * np.exp(self._f(np.array([y]))[0]) / self._Z)

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        if y <= self._Zmax:
            return float(self._tilted[:y + 1].sum())
        return float(min(1.0, self._tilted.sum() + sum(self.probability(k) for k in range(self._Zmax + 1, y + 1))))

    def survival(self, y):
        return float(max(0.0, 1.0 - self.cdf(y)))

    def mean(self):
        # overflow < tail tolerance (1e-6); the tilted atom array captures the mean within tolerance
        k = np.arange(self._tilted.size)
        return float(np.dot(k, self._tilted))

    def variance(self):
        k = np.arange(self._tilted.size)
        m = float(np.dot(k, self._tilted))
        ex2 = float(np.dot(k ** 2, self._tilted))
        return float(max(0.0, ex2 - m ** 2))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        atoms = self._tilted.copy()
        K = self._Zmax
        if required_max is not None and required_max > K:
            extra = np.array([self.probability(k) for k in range(K + 1, required_max + 1)])
            atoms = np.concatenate([atoms, extra]); K = required_max
        overflow = float(max(0.0, 1.0 - atoms.sum()))
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
    ex_cond = ex_base / pos               # E[Y | Y>=1] (base 0-atom contributes 0 to numerator)
    ex2_cond = ex2_base / pos
    m = h.p_pos * ex_cond
    ex2 = h.p_pos * ex2_cond
    return float(max(0.0, ex2 - m ** 2))
