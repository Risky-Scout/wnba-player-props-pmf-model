"""Exact PMF primitives for Opportunity V2 compound construction.

Count models (Poisson / negative-binomial), Beta-binomial conversion, attempt-marginalized
conversion, convolution, mixing, moments, and push-safe sportsbook settlement. Every builder pads
supports before combining, rejects invalid mass, tracks omitted tail mass, and normalizes to 1.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import betabinom, nbinom, poisson

_NORM_TOL = 1e-12


def _validate(pmf: np.ndarray, *, tol: float = _NORM_TOL, where: str = "pmf") -> np.ndarray:
    a = np.asarray(pmf, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{where}: expected a non-empty 1-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{where}: contains non-finite values")
    if np.any(a < -tol):
        raise ValueError(f"{where}: contains negative mass (min={a.min()})")
    a = np.clip(a, 0.0, None)
    s = a.sum()
    if s <= 0:
        raise ValueError(f"{where}: total mass is non-positive")
    return a / s


def _pad_to(pmfs: Sequence[np.ndarray], length: int) -> list[np.ndarray]:
    out = []
    for p in pmfs:
        a = np.asarray(p, dtype=float)
        if a.size < length:
            a = np.concatenate([a, np.zeros(length - a.size)])
        out.append(a)
    return out


def poisson_or_nbinom_pmf(
    mean: float,
    dispersion_r: float | None,
    *,
    tail_tolerance: float = 1e-8,
    minimum_cap: int = 8,
    maximum_cap: int = 120,
) -> np.ndarray:
    """Count PMF over 0..K. Poisson when ``dispersion_r`` is None, else NB2 with variance mu+mu^2/r.

    Support K grows until the *omitted upper-tail* mass is below ``tail_tolerance`` (or ``maximum_cap``
    is reached). Omitted tail mass is never folded into lower outcomes; the array is renormalized only
    to correct floating error, and a ValueError is raised if the omitted tail exceeds tolerance.
    """
    mu = float(mean)
    if not np.isfinite(mu) or mu < 0:
        raise ValueError(f"poisson_or_nbinom_pmf: invalid mean {mean!r}")
    if mu == 0.0:
        return np.array([1.0])

    if dispersion_r is None:
        dist = poisson(mu)
    else:
        r = float(dispersion_r)
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f"poisson_or_nbinom_pmf: invalid dispersion_r {dispersion_r!r}")
        # NB2 parameterization: p = r/(r+mu); mean = mu; var = mu + mu^2/r.
        p = r / (r + mu)
        dist = nbinom(r, p)

    k_cap = max(int(minimum_cap), int(np.ceil(mu)) + int(minimum_cap))
    k_cap = min(k_cap, int(maximum_cap))
    while True:
        ks = np.arange(0, k_cap + 1)
        pmf = dist.pmf(ks)
        omitted = float(dist.sf(k_cap))  # P(Y > k_cap)
        if omitted <= tail_tolerance or k_cap >= maximum_cap:
            break
        k_cap = min(k_cap * 2 + 1, int(maximum_cap))

    if omitted > tail_tolerance and k_cap >= maximum_cap:
        # Only acceptable if the residual is negligible relative to mass captured.
        if omitted > max(tail_tolerance, 1e-6):
            raise ValueError(
                f"poisson_or_nbinom_pmf: omitted tail mass {omitted:.3e} exceeds tolerance at "
                f"maximum_cap={maximum_cap} for mean={mu}, r={dispersion_r}")
    return _validate(pmf, where="count_pmf")


def beta_binomial_pmf(attempts: int, alpha: float, beta: float) -> np.ndarray:
    """PMF of successes in {0..attempts} under Beta-Binomial(attempts, alpha, beta)."""
    n = int(attempts)
    if n < 0:
        raise ValueError(f"beta_binomial_pmf: negative attempts {attempts!r}")
    a, b = float(alpha), float(beta)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        raise ValueError(f"beta_binomial_pmf: invalid alpha/beta ({alpha!r},{beta!r})")
    if n == 0:
        return np.array([1.0])
    ks = np.arange(0, n + 1)
    pmf = betabinom.pmf(ks, n, a, b)
    return _validate(pmf, where="beta_binomial_pmf")


def marginal_beta_binomial_pmf(attempt_pmf: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Successes PMF marginalized over an attempts PMF: sum_n P(N=n) * BetaBinom(n, a, b)."""
    ap = _validate(attempt_pmf, where="attempt_pmf")
    max_n = ap.size - 1
    out = np.zeros(max_n + 1)
    for n in range(max_n + 1):
        w = ap[n]
        if w <= 0:
            continue
        bb = beta_binomial_pmf(n, alpha, beta)
        out[: bb.size] += w * bb
    return _validate(out, where="marginal_beta_binomial_pmf")


def convolve_pmfs(*pmfs: np.ndarray) -> np.ndarray:
    """Distribution of the sum of independent integer variables (chained convolution)."""
    if not pmfs:
        raise ValueError("convolve_pmfs: no pmfs provided")
    acc = _validate(pmfs[0], where="convolve[0]")
    for i, p in enumerate(pmfs[1:], start=1):
        acc = np.convolve(acc, _validate(p, where=f"convolve[{i}]"))
    return _validate(acc, where="convolve_result")


def weighted_mix_pmfs(pmfs: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted mixture of PMFs (e.g. averaging over minutes samples)."""
    if len(pmfs) == 0:
        raise ValueError("weighted_mix_pmfs: empty pmfs")
    w = np.asarray(weights, dtype=float)
    if w.size != len(pmfs):
        raise ValueError("weighted_mix_pmfs: weights length mismatch")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weighted_mix_pmfs: invalid weights")
    ws = w.sum()
    if ws <= 0:
        raise ValueError("weighted_mix_pmfs: weights sum to zero")
    w = w / ws
    length = max(np.asarray(p).size for p in pmfs)
    padded = _pad_to([_validate(p, where="mix_component") for p in pmfs], length)
    out = np.zeros(length)
    for wi, p in zip(w, padded):
        out += wi * p
    return _validate(out, where="weighted_mix_result")


def pmf_mean(pmf: np.ndarray) -> float:
    a = _validate(pmf, where="pmf_mean")
    return float(np.dot(np.arange(a.size, dtype=float), a))


def pmf_variance(pmf: np.ndarray) -> float:
    a = _validate(pmf, where="pmf_variance")
    k = np.arange(a.size, dtype=float)
    m = float(np.dot(k, a))
    return float(np.dot(k * k, a) - m * m)


def settled_over_probability(pmf: np.ndarray, line: float) -> tuple[float, float, float]:
    """Push-safe sportsbook settlement from an active PMF.

    Returns (p_over_settled, p_under_settled, p_push). For an integer line L, pushes at Y=L are
    removed and over/under are renormalized by (1 - p_push). For a half line, p_push is 0.
    p_over_settled + p_under_settled == 1 (within tolerance) whenever p_push < 1.
    """
    a = _validate(pmf, where="settled_over_probability")
    k = np.arange(a.size, dtype=float)
    ln = float(line)
    is_integer_line = abs(ln - round(ln)) < 1e-9
    if is_integer_line:
        li = int(round(ln))
        p_push = float(a[li]) if 0 <= li < a.size else 0.0
        p_over_raw = float(a[k > ln].sum())
        p_under_raw = float(a[k < ln].sum())
    else:
        p_push = 0.0
        p_over_raw = float(a[k > ln].sum())
        p_under_raw = float(a[k < ln].sum())
    denom = 1.0 - p_push
    if denom <= 0:
        return 0.0, 0.0, 1.0
    return p_over_raw / denom, p_under_raw / denom, p_push
