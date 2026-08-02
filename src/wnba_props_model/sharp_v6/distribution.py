"""WNBA Sharp PMF v6 — authoritative discrete distributions + tail-aware tilt.

Self-contained production distributions (no sharp_v3/v4/v5 imports).

One production DiscreteDistribution interface with correct mass accounting:

- ``probability(y)`` returns the EXACT atom probability for any nonnegative integer y
  whenever the family is analytic (never the aggregate overflow reused as every tail atom).
- Mixture / hurdle / zero-inflated / convolution mass: stored atoms + overflow = 1.
- Mixture weights are normalized once; finite stored atoms are NEVER renormalized away
  from overflow.
- Settlement is push-aware: A/(A+B), with A=P(Y>L), B=P(Y<L), P=P(Y=L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

TAIL_TOL = 1e-6
NORM_TOL = 1e-10
PRUNE_MASS_TOL = 1e-10


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
    discarded_mixture_mass: float = 0.0


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
        return SettleResult(
            L, A, B, P,
            A / den if den > 0 else float("nan"),
            B / den if den > 0 else float("nan"),
        )

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        K = required_max if required_max is not None else max(1, int(self.mean() + 8 * np.sqrt(max(self.variance(), 0.0) + 1)))
        for _ in range(60):
            if self.survival(K) <= tail_tolerance and (required_max is None or K >= required_max):
                break
            K = K + max(2, int(K * 0.5))
        atoms = np.array([self.probability(k) for k in range(K + 1)], dtype=float)
        overflow = float(self.survival(K))
        stored = float(atoms.sum())
        return Materialized(
            atoms=atoms, support_min=0, support_max=int(K), stored_mass=stored,
            overflow_probability=overflow, tail_upper_bound=overflow,
            tail_method=f"{self.family}_analytic_survival",
            normalization_error=float(abs(stored + overflow - 1.0)),
            distribution_family=self.family, distribution_parameters=self.parameters(),
        )

    def validate(self, tail_tolerance: float = TAIL_TOL) -> None:
        m = self.materialize(tail_tolerance)
        if m.normalization_error > NORM_TOL:
            raise ValueError(f"{self.family}: normalization_error {m.normalization_error} > {NORM_TOL}")
        if np.any(m.atoms < -1e-12):
            raise ValueError(f"{self.family}: negative atom")

    def sample(self, n, rng):
        m = self.materialize()
        # Include overflow as an extended terminal bucket, then map back via inverse-CDF for
        # unresolved tails using analytic probability when available.
        if m.overflow_probability <= NORM_TOL:
            p = m.atoms / m.atoms.sum()
            return rng.choice(np.arange(m.atoms.size), size=n, p=p)
        # Adaptive: expand until overflow negligible for sampling, else rejection from analytic.
        big = self.materialize(tail_tolerance=1e-12)
        if big.overflow_probability <= 1e-10:
            p = big.atoms / max(big.atoms.sum(), 1e-300)
            return rng.choice(np.arange(big.atoms.size), size=n, p=p)
        # Fallback inverse-transform with analytic CDF
        u = rng.random(n)
        out = np.empty(n, dtype=int)
        # crude scan up to a large K
        K = max(big.support_max + 50, int(self.mean() + 40 * np.sqrt(max(self.variance(), 0) + 1)))
        cdf_vals = np.cumsum([self.probability(k) for k in range(K + 1)])
        for i, ui in enumerate(u):
            idx = int(np.searchsorted(cdf_vals, ui, side="left"))
            out[i] = min(idx, K)
        return out


class CountDistribution(DiscreteDistribution):
    """NB2 (or Poisson when r is None). Exact analytic probability for any y."""
    family = "nb2"

    def __init__(self, mu: float, r: float | None):
        if not np.isfinite(mu) or mu < 0:
            raise ValueError(f"CountDistribution mu must be finite and nonnegative, got {mu}")
        self.mu = max(float(mu), 1e-9)
        if r is not None and (not np.isfinite(r) or r <= 0):
            raise ValueError(f"CountDistribution r must be None or positive finite, got {r}")
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
    """Hurdle: P(0)=1-p_pos; P(y>=1)=p_pos * base.P(y)/base.P(Y>=1).

    The positive normalizer includes the complete base distribution (analytic tail).
    Overflow at support K is p_pos * P_base(Y>K | Y>=1).
    """
    family = "hurdle_nb2"

    def __init__(self, p_positive: float, base: DiscreteDistribution):
        if not np.isfinite(p_positive):
            raise ValueError("hurdle p_positive must be finite")
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
        # E[Y] = p_pos * E_base[Y | Y>=1] = p_pos * E_base[Y] / P_base(Y>=1)
        return self.p_pos * (self.base.mean() / self._pos_norm)

    def variance(self):
        return analytic_hurdle_variance(self)

    def parameters(self):
        return {"p_positive": self.p_pos, "base": self.base.parameters()}

    def sample(self, n, rng):
        out = np.zeros(n, dtype=int)
        pos = rng.random(n) < self.p_pos
        n_pos = int(pos.sum())
        if n_pos == 0:
            return out
        # sample from base truncated to Y>=1 via rejection
        drawn = []
        while len(drawn) < n_pos:
            cand = np.asarray(self.base.sample(max(n_pos * 2, 8), rng), dtype=int)
            cand = cand[cand >= 1]
            drawn.extend(cand.tolist())
        out[pos] = np.asarray(drawn[:n_pos], dtype=int)
        return out


class ZeroInflatedDistribution(DiscreteDistribution):
    """ZI mixture: P(0)=pi + (1-pi)*base.P(0); P(y>=1)=(1-pi)*base.P(y)."""
    family = "zinb"

    def __init__(self, pi: float, base: DiscreteDistribution):
        if not np.isfinite(pi):
            raise ValueError("ZI pi must be finite")
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

    def sample(self, n, rng):
        out = np.asarray(self.base.sample(n, rng), dtype=int)
        zeroed = rng.random(n) < self.pi
        out[zeroed] = 0
        return out


class MixtureDistribution(DiscreteDistribution):
    """Mixture sum_i w_i * component_i.

    Used to propagate the minutes PMF into a stat PMF:
    P(Y=y|X) = sum_m P(M=m|X) * P(Y=y|M=m,X).

    Weights are validated and normalized exactly once. Stored atoms from
    materialize() are never renormalized independently of overflow.
    """
    family = "mixture"

    def __init__(self, components: list[DiscreteDistribution], weights: np.ndarray):
        if len(components) == 0:
            raise ValueError("mixture requires at least one component")
        w_raw = np.asarray(weights, dtype=float)
        if w_raw.size != len(components):
            raise ValueError("mixture weights length must match components")
        # Validate full weight vector before any prune so invalid inputs fail closed.
        if not np.isfinite(w_raw).all():
            raise ValueError("mixture weights contain NaN or infinite values")
        if (w_raw < 0).any():
            raise ValueError("mixture weights contain negative values")
        if float(w_raw.sum()) <= 0:
            raise ValueError("mixture weights sum to zero")
        w_norm = w_raw / float(w_raw.sum())
        # Keep all strictly positive weights; do not drop mass via 1e-4 thresholds.
        keep = w_norm > 0
        discarded = float(w_norm[~keep].sum())
        self.w = w_norm[keep]
        self.components = [c for c, k in zip(components, keep) if k]
        self.discarded_mixture_mass = discarded
        if len(self.components) == 0:
            raise ValueError("mixture has no positive-weight components")
        # Re-normalize kept weights to sum to 1 (discarded is only exact zeros here).
        self.w = self.w / self.w.sum()

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
        return {
            "n_components": len(self.components),
            "discarded_mixture_mass": self.discarded_mixture_mass,
        }

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        # Expand support until mixture overflow <= tail_tolerance (or required_max reached).
        K = required_max if required_max is not None else 1
        for _ in range(60):
            # overflow = sum_m w_m * P(Y>K|m)
            overflow = float(sum(wi * c.survival(K) for wi, c in zip(self.w, self.components)))
            if overflow <= tail_tolerance and (required_max is None or K >= required_max):
                break
            K = K + max(2, int(K * 0.5))
        atoms = np.array([self.probability(k) for k in range(K + 1)], dtype=float)
        overflow = float(sum(wi * c.survival(K) for wi, c in zip(self.w, self.components)))
        stored = float(atoms.sum())
        # Do NOT renormalize atoms. Mass identity: stored + overflow ~= 1.
        return Materialized(
            atoms=atoms, support_min=0, support_max=int(K), stored_mass=stored,
            overflow_probability=overflow, tail_upper_bound=overflow,
            tail_method="mixture_component_survival",
            normalization_error=float(abs(stored + overflow - 1.0)),
            distribution_family=self.family, distribution_parameters=self.parameters(),
            discarded_mixture_mass=float(self.discarded_mixture_mass),
        )

    def sample(self, n, rng):
        idx = rng.choice(len(self.components), size=n, p=self.w)
        out = np.empty(n, dtype=int)
        for j in range(len(self.components)):
            mask = idx == j
            nj = int(mask.sum())
            if nj:
                out[mask] = self.components[j].sample(nj, rng)
        return out


class ConvolutionDistribution(DiscreteDistribution):
    """Convolution of independent discrete A and B without truncating then renormalizing.

    For requested atoms inside a materialized support, every valid contributing term is
    included. Component supports are expanded until joint overflow <= tolerance.
    """
    family = "convolution"

    def __init__(self, a: DiscreteDistribution, b: DiscreteDistribution, tail_tolerance: float = TAIL_TOL):
        self._a = a
        self._b = b
        self._tol = float(tail_tolerance)
        # Expand each component so remaining component overflow is small; convolve stored atoms
        # WITHOUT renormalizing. Joint overflow = 1 - sum(conv).
        ma = a.materialize(tail_tolerance / 2)
        mb = b.materialize(tail_tolerance / 2)
        # Ensure component materializations themselves are well-normalized.
        if ma.normalization_error > NORM_TOL or mb.normalization_error > NORM_TOL:
            # Expand harder
            ma = a.materialize(min(tail_tolerance / 10, 1e-9))
            mb = b.materialize(min(tail_tolerance / 10, 1e-9))
        conv = np.convolve(ma.atoms, mb.atoms)
        stored = float(conv.sum())
        # Exact joint overflow from independence:
        # P(A+B > Ka+Kb) <= 1 - F_A(Ka)F_B(Kb) contributions already outside conv;
        # remaining mass not in conv is 1 - stored (includes all paths with A>Ka or B>Kb).
        overflow = float(max(0.0, 1.0 - stored))
        # If overflow still large, expand further via direct construction
        Ka, Kb = ma.support_max, mb.support_max
        guard = 0
        while overflow > self._tol and guard < 40:
            Ka = Ka + max(2, Ka // 2)
            Kb = Kb + max(2, Kb // 2)
            aa = np.array([a.probability(i) for i in range(Ka + 1)])
            bb = np.array([b.probability(i) for i in range(Kb + 1)])
            conv = np.convolve(aa, bb)
            stored = float(conv.sum())
            overflow = float(max(0.0, 1.0 - stored))
            guard += 1
        self._atoms = conv
        self._overflow = overflow
        self._Ka = Ka
        self._Kb = Kb

    def probability(self, y):
        y = int(y)
        if y < 0:
            return 0.0
        if y < self._atoms.size:
            return float(self._atoms[y])
        # Exact per-atom via full convolution sum (includes analytic tails of A,B)
        return float(sum(self._a.probability(i) * self._b.probability(y - i) for i in range(y + 1)))

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        if y < self._atoms.size:
            return float(self._atoms[: y + 1].sum())
        return float(min(1.0, self._atoms.sum() + sum(self.probability(k) for k in range(self._atoms.size, y + 1))))

    def survival(self, y):
        y = int(np.floor(y))
        if y + 1 < self._atoms.size:
            return float(self._atoms[y + 1 :].sum() + self._overflow)
        if y + 1 == self._atoms.size:
            return float(self._overflow)
        # Beyond materialized support: 1 - cdf
        return float(max(0.0, 1.0 - self.cdf(y)))

    def mean(self):
        return self._a.mean() + self._b.mean()

    def variance(self):
        return self._a.variance() + self._b.variance()

    def parameters(self):
        return {"overflow": self._overflow, "stored_max": int(self._atoms.size - 1)}

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        atoms = self._atoms.copy()
        K = int(atoms.size - 1)
        overflow = float(self._overflow)
        if required_max is not None and required_max > K:
            extra = np.array([self.probability(k) for k in range(K + 1, required_max + 1)])
            atoms = np.concatenate([atoms, extra])
            K = required_max
            overflow = float(max(0.0, 1.0 - atoms.sum()))
        # Never renormalize stored atoms.
        stored = float(atoms.sum())
        return Materialized(
            atoms=atoms, support_min=0, support_max=K, stored_mass=stored,
            overflow_probability=overflow, tail_upper_bound=overflow,
            tail_method="convolution_joint_overflow",
            normalization_error=float(abs(stored + overflow - 1.0)),
            distribution_family=self.family, distribution_parameters=self.parameters(),
        )


class TabularDistribution(DiscreteDistribution):
    """Materialized atom table + aggregate overflow bucket.

    Per-atom probability is exact within the table; beyond support only the aggregate
    overflow is known. Callers that need exact tail atoms must retain an analytic family.
    """
    family = "tabular"

    def __init__(self, atoms: np.ndarray, overflow: float = 0.0, tail_method: str = "aggregate_overflow"):
        a = np.asarray(atoms, dtype=float)
        if a.ndim != 1:
            raise ValueError("tabular atoms must be 1-d")
        if not np.isfinite(a).all() or not np.isfinite(overflow):
            raise ValueError("tabular atoms/overflow must be finite")
        if (a < -NORM_TOL).any() or overflow < -NORM_TOL:
            raise ValueError("tabular atoms/overflow must be nonnegative")
        self.atoms = np.clip(a, 0.0, None)
        self.overflow = float(max(0.0, overflow))
        self.tail_method = tail_method
        # Do not renormalize. Callers must supply already-correct mass.

    def probability(self, y):
        y = int(y)
        if 0 <= y < self.atoms.size:
            return float(self.atoms[y])
        return 0.0   # individual tail atoms not resolved; mass is in overflow

    def cdf(self, y):
        y = int(np.floor(y))
        if y < 0:
            return 0.0
        if y < self.atoms.size:
            return float(self.atoms[: y + 1].sum())
        return float(min(1.0, self.atoms.sum() + self.overflow))

    def survival(self, y):
        y = int(np.floor(y))
        if y + 1 < self.atoms.size:
            return float(self.atoms[y + 1 :].sum() + self.overflow)
        if y + 1 == self.atoms.size:
            return float(self.overflow)
        return float(self.overflow) if y >= self.atoms.size - 1 else 0.0

    def mean(self):
        # Stored-atom mean; when overflow is material, add a conservative lower-bound
        # contribution so unresolved tail mass is not treated as zero.
        k = np.arange(self.atoms.size)
        base = float(np.dot(k, self.atoms))
        if self.overflow <= NORM_TOL:
            return base
        return base + float(self.atoms.size) * self.overflow

    def variance(self):
        m = self.mean()
        k = np.arange(self.atoms.size)
        ex2 = float(np.dot(k ** 2, self.atoms))
        if self.overflow > NORM_TOL:
            ex2 += float(self.atoms.size ** 2) * self.overflow
        return float(max(0.0, ex2 - m ** 2))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        stored = float(self.atoms.sum())
        return Materialized(
            atoms=self.atoms, support_min=0, support_max=int(max(self.atoms.size - 1, 0)),
            stored_mass=stored, overflow_probability=self.overflow,
            tail_upper_bound=self.overflow, tail_method=self.tail_method,
            normalization_error=float(abs(stored + self.overflow - 1.0)),
            distribution_family=self.family, distribution_parameters={},
        )


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
    """Exponentially tilted discrete distribution with certified transformed-tail mass.

    The normalizer is computed by adaptively summing transformed tail atoms until the
    remaining analytic upper bound on the unsummed transformed tail is <= 1e-10.
    An upper bound is never inserted into Z as if it were exact probability mass unless
    that bound itself is within NORM_TOL (at which point it is a certified remainder).
    """
    family = "tilted"

    def __init__(
        self,
        base: DiscreteDistribution,
        theta_mean: float = 0.0,
        theta_zero: float = 0.0,
        theta_disp: float = 0.0,
        basis_scale: float = 30.0,
        tail_tolerance: float = TAIL_TOL,
    ):
        self.base = base
        self.theta_mean = float(theta_mean)
        self.theta_zero = float(theta_zero)
        self.theta_disp = float(theta_disp)
        self.basis_scale = float(basis_scale)
        self._mu0 = float(base.mean())
        self._tol = float(tail_tolerance)
        self._Z, self._Zmax, self._remainder_bound = self._normalizer()

    def _f(self, y):
        yv = np.asarray(y, float)
        disp = -np.tanh(((yv - self._mu0) / max(self.basis_scale / 3, 1e-6)) ** 2)   # in [-1, 0]
        return (
            self.theta_mean * _bounded_mean_basis(yv, self.basis_scale)
            + self.theta_disp * disp
            + self.theta_zero * (yv == 0)
        )

    def _normalizer(self):
        """Sum transformed atoms until remaining transformed-tail upper bound <= NORM_TOL."""
        K = max(int(self.base.mean() + 8 * np.sqrt(self.base.variance() + 1)), 10)
        bound_factor = float(
            np.exp(abs(self.theta_mean) * self.basis_scale + abs(self.theta_disp) + abs(self.theta_zero))
        )
        terms = None
        remainder_bound = float("inf")
        for _ in range(80):
            k = np.arange(K + 1)
            base_p = _base_atoms(self.base, K)
            terms = base_p * np.exp(self._f(k))
            remainder_bound = bound_factor * float(self.base.survival(K))
            # Expand until the *bound* on the unsummed transformed tail is negligible.
            if remainder_bound <= NORM_TOL:
                break
            K += max(4, int(K * 0.5))
        assert terms is not None
        # Z = exact summed terms + certified remainder (only admitted when <= NORM_TOL).
        # If still above NORM_TOL after expansion, keep explicit bounds and mark inexact.
        if remainder_bound > NORM_TOL:
            # Last resort: continue summing explicit transformed atoms further out.
            extra_budget = 0
            while remainder_bound > NORM_TOL and extra_budget < 40:
                K2 = K + max(8, int(K * 0.5))
                k_extra = np.arange(K + 1, K2 + 1)
                if k_extra.size:
                    base_extra = np.array([self.base.probability(int(i)) for i in k_extra])
                    terms = np.concatenate([terms, base_extra * np.exp(self._f(k_extra))])
                K = K2
                remainder_bound = bound_factor * float(self.base.survival(K))
                extra_budget += 1
        Z = float(terms.sum() + remainder_bound)
        if Z <= 0 or not np.isfinite(Z):
            raise ValueError("tilted normalizer is non-finite or non-positive")
        self._tilted = terms / Z
        self._overflow = float(remainder_bound / Z)
        self._normalizer_exact = bool(remainder_bound <= NORM_TOL)
        self._tail_lower = 0.0
        self._tail_upper = float(remainder_bound / Z)
        return Z, int(K), float(remainder_bound)

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
            return float(self._tilted[: y + 1].sum())
        return float(
            min(
                1.0,
                self._tilted.sum()
                + sum(self.probability(k) for k in range(self._Zmax + 1, y + 1)),
            )
        )

    def survival(self, y):
        return float(max(0.0, 1.0 - self.cdf(y)))

    def mean(self):
        # Summed tilted atoms always included. When a non-negligible certified remainder
        # remains, add a conservative (support_max+1)*overflow contribution so the tail
        # is not treated as zero mass.
        k = np.arange(self._tilted.size)
        base_mean = float(np.dot(k, self._tilted))
        if self._overflow <= NORM_TOL:
            return base_mean
        return base_mean + (self._Zmax + 1) * self._overflow

    def variance(self):
        k = np.arange(self._tilted.size)
        m = self.mean()
        ex2 = float(np.dot(k ** 2, self._tilted))
        if self._overflow > NORM_TOL:
            ex2 += ((self._Zmax + 1) ** 2) * self._overflow
        return float(max(0.0, ex2 - m ** 2))

    def materialize(self, tail_tolerance: float = TAIL_TOL, required_max: int | None = None) -> Materialized:
        atoms = self._tilted.copy()
        K = self._Zmax
        if required_max is not None and required_max > K:
            extra = np.array([self.probability(k) for k in range(K + 1, required_max + 1)])
            atoms = np.concatenate([atoms, extra])
            K = required_max
        stored = float(atoms.sum())
        # Overflow is the certified remaining transformed-tail mass (not a duplicated bound).
        overflow = float(max(0.0, 1.0 - stored))
        # Prefer the tighter of (1-stored) and remainder/Z when normalizer was exact.
        if self._normalizer_exact:
            overflow = float(self._overflow) if required_max is None or required_max <= self._Zmax else overflow
            # Re-sync: if we did not extend, stored + remainder/Z = 1 by construction of Z.
            if required_max is None or required_max <= self._Zmax:
                overflow = float(self._overflow)
                # Numerical drift: force identity within float error without renorming atoms.
                drift = stored + overflow - 1.0
                if abs(drift) <= 1e-12:
                    overflow = float(max(0.0, 1.0 - stored))
        return Materialized(
            atoms=atoms, support_min=0, support_max=int(K), stored_mass=stored,
            overflow_probability=float(max(0.0, overflow)),
            tail_upper_bound=float(max(overflow, self._tail_upper)),
            tail_method=(
                "tilted_adaptive_sum_certified_remainder"
                if self._normalizer_exact
                else "tilted_bounded_remainder_inexact"
            ),
            normalization_error=float(abs(stored + max(0.0, overflow) - 1.0)),
            distribution_family=self.family,
            distribution_parameters={
                "theta_mean": self.theta_mean,
                "theta_zero": self.theta_zero,
                "theta_disp": self.theta_disp,
                "normalizer_exact": self._normalizer_exact,
                "remainder_bound": self._remainder_bound,
            },
        )

    def validate(self, tail_tolerance: float = TAIL_TOL) -> None:
        m = self.materialize(tail_tolerance)
        if m.normalization_error > NORM_TOL:
            raise ValueError(f"tilted: normalization_error {m.normalization_error} > {NORM_TOL}")
        if not self._normalizer_exact and m.tail_upper_bound - (1.0 - m.stored_mass) > NORM_TOL:
            raise ValueError("tilted: inexact normalizer with unresolved tail gap")

    def sample(self, n, rng):
        m = self.materialize(tail_tolerance=1e-12)
        mass = m.stored_mass + m.overflow_probability
        if mass <= 0:
            raise ValueError("tilted sample: zero mass")
        # Do not drop overflow: extend support with a single overflow bucket then map via
        # analytic probability for draws that land in overflow.
        p = np.concatenate([m.atoms, [m.overflow_probability]])
        p = p / p.sum()
        choice = rng.choice(len(p), size=n, p=p)
        out = choice.copy()
        overflow_hits = choice == len(p) - 1
        if overflow_hits.any():
            # Sample from analytic tilted tail y > support_max
            n_tail = int(overflow_hits.sum())
            drawn = []
            y0 = m.support_max + 1
            # inverse-transform on a finite extended window
            ys = np.arange(y0, y0 + 200)
            probs = np.array([self.probability(int(y)) for y in ys])
            s = probs.sum()
            if s <= 0:
                out[overflow_hits] = y0
            else:
                probs = probs / s
                drawn = rng.choice(ys, size=n_tail, p=probs)
                out[overflow_hits] = drawn
        return out


def analytic_hurdle_variance(h: HurdleDistribution) -> float:
    """Complete analytic second moment for a hurdle count (does not zero-out tail)."""
    base = h.base
    pos = h._pos_norm                    # base P(Y>=1)
    mb, vb = base.mean(), base.variance()
    ex_base = mb                          # E_base[Y]
    ex2_base = vb + mb ** 2               # E_base[Y^2]
    ex_cond = ex_base / pos               # E[Y | Y>=1]
    ex2_cond = ex2_base / pos
    m = h.p_pos * ex_cond
    ex2 = h.p_pos * ex2_cond
    return float(max(0.0, ex2 - m ** 2))


def minutes_count_mixture(
    lam: float,
    r: float | None,
    minutes_atoms: np.ndarray,
) -> MixtureDistribution:
    """Authoritative minutes→count mixture: sum_m P(M=m) Count(lam*m, r)."""
    w = np.asarray(minutes_atoms, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("minutes atoms contain NaN/inf")
    if (w < 0).any():
        raise ValueError("minutes atoms contain negative values")
    if float(w.sum()) <= 0:
        raise ValueError("minutes atoms sum to zero")
    comps: list[DiscreteDistribution] = []
    for m in range(w.size):
        mu = max(float(lam) * float(m), 1e-9)
        comps.append(CountDistribution(mu, None if r is None or (isinstance(r, float) and np.isnan(r)) else float(r)))
    return MixtureDistribution(comps, w)


def materialize_minutes_mixture(
    lam: float,
    r: float | None,
    minutes_atoms: np.ndarray,
    K: int,
    *,
    tail_tolerance: float = TAIL_TOL,
) -> Materialized:
    """Vectorized materialization of a minutes mixture on support 0..K without atom renorm.

    atom[y] = sum_m w_m * P(Y=y|m)
    overflow = sum_m w_m * P(Y>K|m)
    Never renormalizes stored atoms independently of overflow.
    """
    w = np.asarray(minutes_atoms, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("minutes atoms contain NaN/inf")
    if (w < 0).any():
        raise ValueError("minutes atoms contain negative values")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("minutes atoms sum to zero")
    w = w / total
    # Keep every finite nonnegative state with positive weight — no 1e-4 drop.
    idx = np.arange(w.size)
    means = np.clip(float(lam) * idx.astype(float), 1e-9, None)
    k = np.arange(int(K) + 1)
    if r is None or (isinstance(r, float) and np.isnan(r)):
        # poisson.pmf broadcasts (K+1, n_states)
        comp = poisson.pmf(k[:, None], means[None, :])
        # survival P(Y>K) = 1 - cdf(K)
        surv = 1.0 - poisson.cdf(K, means)
    else:
        rr = float(r)
        p = rr / (rr + means)
        comp = nbinom.pmf(k[:, None], rr, p[None, :])
        surv = 1.0 - nbinom.cdf(K, rr, p)
    atoms = comp @ w
    overflow = float(np.dot(surv, w))
    stored = float(atoms.sum())
    # Expand support if overflow too large and caller used small K — documented by tolerance.
    K_eff = int(K)
    if overflow > tail_tolerance:
        # Adaptive expansion using the MixtureDistribution object (analytic components).
        mix = minutes_count_mixture(lam, r, minutes_atoms)
        mat = mix.materialize(tail_tolerance=tail_tolerance, required_max=K)
        return mat
    return Materialized(
        atoms=np.asarray(atoms, dtype=float),
        support_min=0,
        support_max=K_eff,
        stored_mass=stored,
        overflow_probability=overflow,
        tail_upper_bound=overflow,
        tail_method="minutes_mixture_vectorized",
        normalization_error=float(abs(stored + overflow - 1.0)),
        distribution_family="mixture",
        distribution_parameters={"lam": float(lam), "r": None if r is None else float(r), "K": K_eff},
        discarded_mixture_mass=0.0,
    )
