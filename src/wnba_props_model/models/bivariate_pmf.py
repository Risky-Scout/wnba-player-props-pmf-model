"""Bivariate PMF for correlated combo props using Gaussian copula.

Motivation
----------
When building combo-prop PMFs (pts+ast, pts+reb, etc.), the independence
assumption of discrete convolution is wrong.  In WNBA data:

- pts & ast are *negatively* correlated within a game (high-scoring = shot-first
  offense; playmaking nights = fewer personal shot attempts)
- pts & reb are modestly positively correlated (active nights, more possessions)
- reb & ast are approximately independent

PenaltyBlog's Bivariate Poisson soccer model captures this by fitting a shared
covariance parameter.  The WNBA translation uses a Gaussian copula, which:

1. Is distribution-agnostic (works with NegBinom/Poisson marginals)
2. Has a single parameter ρ per pair (easy to estimate from data)
3. Produces a correct joint PMF P(X + Y > line) without Monte Carlo

Method
------
Given marginal PMFs P(X = j) and P(Y = k) with correlation ρ:

1. Compute CDFs F_X(j) and F_Y(k)
2. Convert to normal scores: u_j = F_X(j), v_k = F_Y(k)
3. Apply Gaussian copula: P(X=j, Y=k) ∝ P(X=j) * P(Y=k) * exp(copula_term)
   where copula_term adjusts the joint probability for correlation
4. Renormalize the joint matrix
5. Sum over {j,k : j+k > line} for over-probability

The copula density adjustment for Gaussian copula C_ρ(u,v) uses:
    c(u,v) = (1/sqrt(1-ρ²)) * exp(-(ρ²(φ_u² + φ_v²) - 2ρ*φ_u*φ_v) / (2(1-ρ²)))
where φ_u = Φ⁻¹(u), φ_v = Φ⁻¹(v), Φ⁻¹ is the normal quantile function.

Usage
-----
    from wnba_props_model.models.bivariate_pmf import (
        estimate_correlations,
        build_bivariate_pmf,
        bivariate_prob_over,
    )

    # Fit correlations from historical data once per season
    corr_map = estimate_correlations(features_wide_df)
    # {"pts_ast": -0.15, "pts_reb": 0.08, "reb_ast": -0.05, ...}

    # At inference time
    pts_pmf = np.array([...])   # marginal PMF for pts
    ast_pmf = np.array([...])   # marginal PMF for ast
    joint   = build_bivariate_pmf(pts_pmf, ast_pmf, rho=corr_map["pts_ast"])
    p_over  = bivariate_prob_over(joint, line=20.5)  # P(pts + ast > 20.5)
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Default correlations from 2022–2025 WNBA seasons (approximate baselines).
# These are overwritten by estimate_correlations() when training data is available.
_DEFAULT_CORRELATIONS: dict[str, float] = {
    "pts_ast":     -0.15,  # pts and ast negatively correlated
    "pts_reb":      0.07,  # modest positive (active nights)
    "reb_ast":     -0.05,  # approximately independent
    "pts_reb_ast": -0.10,  # composite: dominated by pts-ast term
    "stocks":       0.10,  # stl + blk: both defensive activity
}

# Minimum samples required to estimate a reliable correlation
_MIN_SAMPLES_FOR_CORR = 100

# Hard clamp on ρ to keep copula numerically stable
_MAX_ABS_RHO = 0.85


def _safe_norm_ppf(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normal quantile function with clipped input to avoid ±∞."""
    return stats.norm.ppf(np.clip(u, eps, 1.0 - eps))


def estimate_correlations(
    features_wide: "pd.DataFrame",
    pairs: Iterable[tuple[str, str]] | None = None,
    min_samples: int = _MIN_SAMPLES_FOR_CORR,
) -> dict[str, float]:
    """Estimate empirical Pearson correlations between stat pairs from wide table.

    Parameters
    ----------
    features_wide: wide feature DataFrame with actual_{stat} columns
    pairs:         stat pairs to estimate; defaults to canonical combo pairs
    min_samples:   minimum non-NaN rows to trust the estimate

    Returns
    -------
    dict mapping combo key (e.g. "pts_ast") to Pearson ρ, clipped to [-MAX, MAX].
    Missing pairs fall back to _DEFAULT_CORRELATIONS.
    """
    import pandas as pd

    if pairs is None:
        pairs = [
            ("pts", "ast"),
            ("pts", "reb"),
            ("reb", "ast"),
            ("stl", "blk"),
        ]

    corr_map = dict(_DEFAULT_CORRELATIONS)

    for s1, s2 in pairs:
        c1, c2 = f"actual_{s1}", f"actual_{s2}"
        if c1 not in features_wide.columns or c2 not in features_wide.columns:
            continue
        valid = features_wide[[c1, c2]].dropna()
        if len(valid) < min_samples:
            logger.debug("[bivariate_pmf] Too few samples for %s_%s correlation (%d < %d)",
                         s1, s2, len(valid), min_samples)
            continue
        rho = float(np.corrcoef(valid[c1].values, valid[c2].values)[0, 1])
        rho = float(np.clip(rho, -_MAX_ABS_RHO, _MAX_ABS_RHO))
        key = f"{s1}_{s2}" if (s1, s2) in [("pts", "ast"), ("pts", "reb"), ("reb", "ast")] else f"{s2}_{s1}"
        corr_map[key] = round(rho, 4)
        logger.debug("[bivariate_pmf] Estimated ρ(%s, %s) = %.4f from %d samples", s1, s2, rho, len(valid))

    return corr_map


def build_bivariate_pmf(
    pmf_x: np.ndarray,
    pmf_y: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Build joint PMF P(X=j, Y=k) using Gaussian copula density adjustment.

    Parameters
    ----------
    pmf_x: marginal PMF for X (array of probabilities summing to ~1)
    pmf_y: marginal PMF for Y (array of probabilities summing to ~1)
    rho:   Pearson correlation target for the bivariate distribution

    Returns
    -------
    2D array of shape (len(pmf_x), len(pmf_y)) representing the joint PMF.
    Sum is normalized to exactly 1.

    Notes
    -----
    When |ρ| < 0.02, the joint PMF is just the outer product (independence),
    avoiding numerical issues from the copula inversion.
    """
    pmf_x = np.asarray(pmf_x, dtype=float)
    pmf_y = np.asarray(pmf_y, dtype=float)

    # Normalize marginals
    if pmf_x.sum() < 1e-9 or pmf_y.sum() < 1e-9:
        return np.outer(pmf_x, pmf_y)
    pmf_x = pmf_x / pmf_x.sum()
    pmf_y = pmf_y / pmf_y.sum()

    # Independence case
    rho = float(np.clip(rho, -_MAX_ABS_RHO, _MAX_ABS_RHO))
    if abs(rho) < 0.02:
        return np.outer(pmf_x, pmf_y)

    nx, ny = len(pmf_x), len(pmf_y)

    # CDFs: F_X(j) = P(X ≤ j), F_Y(k) = P(Y ≤ k)
    # Use midpoint CDF F(j) ≈ CDF at j - 0.5 for discrete atoms (avoids boundary issues)
    cdf_x = np.cumsum(pmf_x)  # shape (nx,)
    cdf_y = np.cumsum(pmf_y)  # shape (ny,)

    # Compute previous CDF values (for atom-midpoint)
    cdf_x_prev = np.concatenate([[0.0], cdf_x[:-1]])
    cdf_y_prev = np.concatenate([[0.0], cdf_y[:-1]])

    # Midpoint CDF: average of upper and lower bound for each atom
    u = 0.5 * (cdf_x + cdf_x_prev)  # shape (nx,)
    v = 0.5 * (cdf_y + cdf_y_prev)  # shape (ny,)

    # Normal scores
    phi_u = _safe_norm_ppf(u)   # (nx,)
    phi_v = _safe_norm_ppf(v)   # (ny,)

    # Gaussian copula log-density correction:
    # log c(u, v; ρ) = -0.5*log(1-ρ²)
    #                  - (ρ²*(φ_u² + φ_v²) - 2ρ*φ_u*φ_v) / (2*(1-ρ²))
    r2 = rho ** 2
    denom = 2.0 * (1.0 - r2)

    # Outer products: (nx, ny)
    phi_u_sq = phi_u[:, None] ** 2   # (nx, 1)
    phi_v_sq = phi_v[None, :] ** 2   # (1, ny)
    cross     = phi_u[:, None] * phi_v[None, :]  # (nx, ny)

    log_copula = -0.5 * np.log(1.0 - r2) - (r2 * (phi_u_sq + phi_v_sq) - 2.0 * rho * cross) / denom

    # Clip to avoid numerical explosions in tails
    log_copula = np.clip(log_copula, -10.0, 10.0)

    # Joint PMF = outer product × copula density adjustment
    joint = np.outer(pmf_x, pmf_y) * np.exp(log_copula)
    joint = np.clip(joint, 0.0, None)

    total = joint.sum()
    if total < 1e-12:
        return np.outer(pmf_x, pmf_y)

    return joint / total


def bivariate_prob_over(
    joint_pmf: np.ndarray,
    line: float,
) -> float:
    """Compute P(X + Y > line) from a joint PMF matrix.

    Parameters
    ----------
    joint_pmf: 2D array P(X=j, Y=k) of shape (nx, ny)
    line:      threshold (e.g. 20.5)

    Returns
    -------
    float in [0, 1]: P(X + Y > line)
    """
    joint_pmf = np.asarray(joint_pmf, dtype=float)
    nx, ny = joint_pmf.shape
    total = 0.0
    for j in range(nx):
        for k in range(ny):
            if j + k > line:
                total += joint_pmf[j, k]
    return float(np.clip(total, 0.0, 1.0))


def adjust_combo_pmf_for_correlation(
    pmf_x: np.ndarray,
    pmf_y: np.ndarray,
    stat_x: str,
    stat_y: str,
    corr_map: dict[str, float] | None = None,
) -> np.ndarray:
    """Build a correlation-adjusted sum PMF P(X + Y = k).

    Unlike the plain convolution (assumes independence), this uses the
    bivariate copula joint PMF and then sums along anti-diagonals to get
    the marginal distribution of X + Y.

    Parameters
    ----------
    pmf_x, pmf_y: marginal PMFs
    stat_x, stat_y: stat names (used to look up ρ from corr_map)
    corr_map: dict of canonical pair key → ρ; defaults to _DEFAULT_CORRELATIONS

    Returns
    -------
    1D array: PMF of X + Y, length = len(pmf_x) + len(pmf_y) - 1
    """
    if corr_map is None:
        corr_map = _DEFAULT_CORRELATIONS

    key1 = f"{stat_x}_{stat_y}"
    key2 = f"{stat_y}_{stat_x}"
    rho  = corr_map.get(key1, corr_map.get(key2, 0.0))

    joint = build_bivariate_pmf(pmf_x, pmf_y, rho)
    nx, ny = joint.shape
    max_k  = nx + ny - 2
    sum_pmf = np.zeros(max_k + 1, dtype=float)
    for j in range(nx):
        for k in range(ny):
            if j + k <= max_k:
                sum_pmf[j + k] += joint[j, k]
    if sum_pmf.sum() > 1e-9:
        sum_pmf /= sum_pmf.sum()
    return sum_pmf
