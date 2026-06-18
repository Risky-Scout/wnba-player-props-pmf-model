"""PMF math utilities for WNBA player-props model.

All distributions produce full discrete atom PMFs over non-negative integers.

Key rules (enforced everywhere):
- Support starts at 0
- All probabilities finite and non-negative
- PMF sums to 1 within 1e-6
- Tail-sum pricing: P(over L) = sum p(k) for k > L  (direct atom sum, no shortcuts)
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import gammaln as _gammaln
from scipy.stats import nbinom as scipy_nbinom
from scipy.stats import poisson as scipy_poisson


# ---------------------------------------------------------------------------
# Scalar PMF generators
# ---------------------------------------------------------------------------

def poisson_pmf(lam: float, cap: int) -> dict[int, float]:
    """Poisson PMF with rate lam, support 0..cap."""
    lam = max(float(lam), 1e-9)
    k = np.arange(0, cap + 1)
    probs = scipy_poisson.pmf(k, lam)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    probs = probs / total if total > 0 else _degenerate_at_zero(cap + 1)
    return {int(ki): float(pi) for ki, pi in zip(k, probs)}


def negbinom_pmf(mu: float, r: float, cap: int) -> dict[int, float]:
    """Negative Binomial PMF.

    Parameterisation: mean = mu, variance = mu + mu^2/r.
    r is the dispersion (smaller r = more overdispersion).
    """
    mu = max(float(mu), 1e-9)
    r = max(float(r), 1e-6)
    p = r / (r + mu)
    k = np.arange(0, cap + 1)
    probs = scipy_nbinom.pmf(k, r, p)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    probs = probs / total if total > 0 else _degenerate_at_zero(cap + 1)
    return {int(ki): float(pi) for ki, pi in zip(k, probs)}


def hurdle_pmf(
    p_nonzero: float,
    pos_mu: float,
    pos_r: float | None,
    cap: int,
) -> dict[int, float]:
    """Hurdle PMF.

    p0 mass sits at 0 (= 1 - p_nonzero).
    The positive tail uses a NegBinom or Poisson on k >= 1, renormalised to
    integrate to exactly p_nonzero.
    """
    p_nonzero = float(np.clip(p_nonzero, 0.0, 1.0))
    p0 = 1.0 - p_nonzero

    if p_nonzero < 1e-9:
        pmf: dict[int, float] = {k: 0.0 for k in range(cap + 1)}
        pmf[0] = 1.0
        return pmf

    pos_mu = max(float(pos_mu), 1e-9)
    k_pos = np.arange(1, cap + 1)

    if pos_r is not None:
        pos_r_f = max(float(pos_r), 1e-6)
        p = pos_r_f / (pos_r_f + pos_mu)
        pos_probs = scipy_nbinom.pmf(k_pos, pos_r_f, p)
    else:
        pos_probs = scipy_poisson.pmf(k_pos, pos_mu)

    pos_probs = np.clip(pos_probs, 0.0, None)
    pos_total = pos_probs.sum()
    if pos_total > 0:
        pos_probs = pos_probs / pos_total * p_nonzero  # scale to p_nonzero

    pmf = {0: float(p0)}
    for ki, pi in zip(k_pos, pos_probs):
        pmf[int(ki)] = float(pi)
    return pmf


# ---------------------------------------------------------------------------
# Vectorised batch generators (for performance with 100K+ rows)
# ---------------------------------------------------------------------------

def negbinom_pmf_batch(mus: np.ndarray, r: float, cap: int) -> np.ndarray:
    """Batch NegBinom PMF.  Returns shape (n, cap+1), each row sums to 1."""
    mus = np.clip(mus.astype(float), 1e-9, None)
    r = max(float(r), 1e-6)
    p = r / (r + mus)  # shape (n,)
    k = np.arange(cap + 1)  # shape (cap+1,)
    pmf_mat = scipy_nbinom.pmf(k[np.newaxis, :], r, p[:, np.newaxis])  # (n, cap+1)
    pmf_mat = np.clip(pmf_mat, 0.0, None)
    totals = pmf_mat.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return pmf_mat / totals


def poisson_pmf_batch(mus: np.ndarray, cap: int) -> np.ndarray:
    """Batch Poisson PMF. Returns shape (n, cap+1)."""
    mus = np.clip(mus.astype(float), 1e-9, None)
    k = np.arange(cap + 1)
    pmf_mat = scipy_poisson.pmf(k[np.newaxis, :], mus[:, np.newaxis])
    pmf_mat = np.clip(pmf_mat, 0.0, None)
    totals = pmf_mat.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return pmf_mat / totals


def hurdle_pmf_batch(
    p_nz: np.ndarray,
    pos_mus: np.ndarray,
    pos_r: float | None,
    cap: int,
) -> np.ndarray:
    """Batch hurdle PMF. Returns shape (n, cap+1), each row sums to 1."""
    p_nz = np.clip(p_nz.astype(float), 0.0, 1.0)
    pos_mus = np.clip(pos_mus.astype(float), 1e-9, None)
    n = len(p_nz)
    pmf_mat = np.zeros((n, cap + 1), dtype=float)
    pmf_mat[:, 0] = 1.0 - p_nz

    k_pos = np.arange(1, cap + 1)  # shape (cap,)
    if pos_r is not None:
        pos_r_f = max(float(pos_r), 1e-6)
        p_nb = pos_r_f / (pos_r_f + pos_mus)  # (n,)
        pos_mass = scipy_nbinom.pmf(k_pos[np.newaxis, :], pos_r_f,
                                    p_nb[:, np.newaxis])  # (n, cap)
    else:
        pos_mass = scipy_poisson.pmf(k_pos[np.newaxis, :],
                                     pos_mus[:, np.newaxis])  # (n, cap)

    pos_mass = np.clip(pos_mass, 0.0, None)
    pos_totals = pos_mass.sum(axis=1, keepdims=True)
    pos_totals = np.where(pos_totals > 0, pos_totals, 1.0)
    pmf_mat[:, 1:] = pos_mass / pos_totals * p_nz[:, np.newaxis]
    return pmf_mat


# ---------------------------------------------------------------------------
# Tail-sum pricing
# ---------------------------------------------------------------------------

def prob_over_from_pmf(pmf: dict[int, float], line: float) -> float:
    """P(outcome > line) using direct atom PMF sum.

    Works for integer and half-integer lines:
    - line=3.5 → P(Y > 3.5) = P(Y >= 4) = sum p(k) for k in {4,5,...}
    - line=3.0 → P(Y > 3.0) = P(Y >= 4) = sum p(k) for k in {4,5,...}
    - line=4.0 → P(Y > 4.0) = P(Y >= 5) = sum p(k) for k in {5,6,...}
    """
    return float(sum(float(p) for k, p in pmf.items() if int(k) > line))


def prob_over_from_row(pmf_array: np.ndarray, line: float, support_min: int = 0) -> float:
    """P(outcome > line) from a dense PMF array starting at support_min."""
    k_vals = np.arange(support_min, support_min + len(pmf_array))
    return float(pmf_array[k_vals > line].sum())


# ---------------------------------------------------------------------------
# PMF validation
# ---------------------------------------------------------------------------

def validate_pmf(pmf: dict[int, float], tol: float = 1e-6) -> None:
    """Raise ValueError if pmf is not a valid probability distribution."""
    if not pmf:
        raise ValueError("PMF is empty")
    if min(pmf.keys()) < 0:
        raise ValueError(f"PMF support below 0: min key = {min(pmf.keys())}")
    for k, p in pmf.items():
        if not np.isfinite(p):
            raise ValueError(f"Non-finite probability at k={k}: {p}")
        if p < -tol:
            raise ValueError(f"Negative probability at k={k}: {p}")
    total = sum(pmf.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"PMF sum = {total:.8f} (expected 1.0 ± {tol})")


def validate_pmf_matrix(pmf_mat: np.ndarray, tol: float = 1e-6) -> None:
    """Validate a batch of PMFs (shape n × support). Raise on any failure."""
    if not np.isfinite(pmf_mat).all():
        raise ValueError("PMF matrix contains non-finite values")
    if (pmf_mat < 0).any():
        raise ValueError("PMF matrix contains negative values")
    sums = pmf_mat.sum(axis=1)
    if (np.abs(sums - 1.0) > tol).any():
        bad = np.where(np.abs(sums - 1.0) > tol)[0]
        raise ValueError(f"PMF row(s) do not sum to 1: indices {bad[:5]}, sums {sums[bad[:5]]}")


# ---------------------------------------------------------------------------
# PMF summary statistics
# ---------------------------------------------------------------------------

def pmf_mean_var(pmf_mat: np.ndarray, support_start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for each PMF row. Returns (means, variances)."""
    k = np.arange(support_start, support_start + pmf_mat.shape[1])
    means = (pmf_mat * k[np.newaxis, :]).sum(axis=1)
    vars_ = (pmf_mat * (k[np.newaxis, :] - means[:, np.newaxis]) ** 2).sum(axis=1)
    return means, vars_


def pmf_pge(pmf_mat: np.ndarray, threshold: int) -> np.ndarray:
    """P(Y >= threshold) for each row. threshold is an absolute integer value."""
    if threshold <= 0:
        return np.ones(len(pmf_mat))
    if threshold >= pmf_mat.shape[1]:
        return np.zeros(len(pmf_mat))
    return pmf_mat[:, threshold:].sum(axis=1)


def pmf_matrix_to_json_list(pmf_mat: np.ndarray, n_digits: int = 8) -> list[str]:
    """Convert PMF matrix rows to JSON strings.  Keys are integer strings."""
    _, cap1 = pmf_mat.shape
    k_strs = [str(k) for k in range(cap1)]
    return [
        json.dumps({k_strs[k]: round(float(pmf_mat[i, k]), n_digits) for k in range(cap1)})
        for i in range(len(pmf_mat))
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _degenerate_at_zero(n: int) -> np.ndarray:
    arr = np.zeros(n)
    arr[0] = 1.0
    return arr


def zinb_pmf_batch(
    pi: np.ndarray,
    mu: np.ndarray,
    r: float,
    cap: int,
) -> np.ndarray:
    """Batch Zero-Inflated Negative Binomial PMF.

    P(k=0) = π + (1-π) * NB(0; μ, r)
    P(k>0) = (1-π) * NB(k; μ, r)

    Returns shape (n, cap+1), rows sum to 1.
    """
    n = len(pi)
    r = max(float(r), 1e-4)
    mu_safe = np.clip(mu, 1e-9, None)
    p_nb = r / (r + mu_safe)        # success prob for NegBinom

    k_arr = np.arange(cap + 1)
    log_r = np.log(r)

    # Compute log NB PMF for each k — vectorized over (n, cap+1)
    k_broad = k_arr[np.newaxis, :]                      # (1, cap+1)
    mu_broad = mu_safe[:, np.newaxis]                   # (n, 1)
    p_broad  = p_nb[:, np.newaxis]                      # (n, 1)

    log_nb = (
        _gammaln(k_broad + r) - _gammaln(r) - _gammaln(k_broad + 1)
        + r * np.log(p_broad) + k_broad * np.log1p(-p_broad + 1e-300)
    )
    nb_pmf = np.exp(np.clip(log_nb, -700, 0))
    nb_pmf = np.clip(nb_pmf, 0.0, None)

    pi_broad = pi[:, np.newaxis]                        # (n, 1)
    pmf_mat = (1.0 - pi_broad) * nb_pmf
    pmf_mat[:, 0] += pi_broad[:, 0]                    # add structural zero mass

    # Renormalize
    row_sums = pmf_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return pmf_mat / row_sums


def beta_binomial_pmf_batch(
    expected_n: np.ndarray,
    alpha: float,
    beta_param: float,
    cap: int,
) -> np.ndarray:
    """Batch Beta-Binomial PMF — convenience re-export from beta_binomial module.

    Returns shape (n, cap+1), each row sums to 1.
    """
    from wnba_props_model.models.beta_binomial import beta_binomial_pmf_batch as _bb  # noqa: PLC0415
    return _bb(expected_n, alpha, beta_param, cap)


def dispersion_from_moments(mean: float, var: float) -> float | None:
    """Negative Binomial dispersion r from mean and variance.

    Returns None if var <= mean (Poisson sufficient).
    """
    if var <= mean or mean <= 0:
        return None
    return float(mean ** 2 / (var - mean))
