"""Monotone distributional calibration (Section 6).

Calibrate the whole distribution (a monotone CDF transform), never independent per-line
isotonic calibrators (which can invert alternate lines, e.g. P(Y>10.5) < P(Y>11.5)). The
transform preserves nonnegative atoms, sum-to-one, a monotone CDF, structural identities, and
alternate-line monotonicity. Active-player outcomes are calibrated; p_dnp is calibrated
separately. When no prior out-of-fold calibration data exists, the fallback is identity.
"""
from __future__ import annotations

import numpy as np


def _atoms_from_cdf(cdf: np.ndarray) -> np.ndarray:
    cdf = np.clip(np.asarray(cdf, float), 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)      # enforce monotone non-decreasing CDF
    cdf[-1] = 1.0
    atoms = np.diff(np.concatenate([[0.0], cdf]))
    atoms = np.clip(atoms, 0.0, None)
    s = atoms.sum()
    return atoms / s if s > 0 else atoms


def identity_calibrate(pmf: np.ndarray) -> np.ndarray:
    """Fallback: revalidate + renormalize without changing shape."""
    a = np.clip(np.asarray(pmf, float), 0.0, None)
    s = a.sum()
    return a / s if s > 0 else a


def monotone_cdf_recalibrate(pmf: np.ndarray, cdf_link=None) -> np.ndarray:
    """Apply a monotone CDF link g:[0,1]->[0,1] (non-decreasing, g(0)=0,g(1)=1) to the PMF's CDF
    and return recalibrated atoms. ``cdf_link`` is fit out-of-fold from prior blocks; when None,
    identity is used (no calibration)."""
    a = identity_calibrate(pmf)
    if cdf_link is None:
        return a
    cdf = np.cumsum(a)
    new_cdf = np.array([float(cdf_link(c)) for c in cdf])
    return _atoms_from_cdf(new_cdf)


def assert_calibrated_pmf_valid(pmf: np.ndarray, tol: float = 1e-9) -> None:
    a = np.asarray(pmf, float)
    if not np.all(np.isfinite(a)):
        raise ValueError("calibrated pmf has non-finite atoms")
    if np.any(a < -tol):
        raise ValueError("calibrated pmf has negative atoms")
    if abs(a.sum() - 1.0) > 1e-6:
        raise ValueError(f"calibrated pmf sums to {a.sum()} != 1")
    cdf = np.cumsum(a)
    if np.any(np.diff(cdf) < -tol):
        raise ValueError("calibrated CDF is not monotone")


def calibrate_dnp(p_dnp: float, shift: float = 0.0) -> float:
    """Calibrate DNP probability separately (logit shift; identity when shift=0)."""
    p = min(max(float(p_dnp), 1e-6), 1 - 1e-6)
    if shift == 0.0:
        return p
    z = np.log(p / (1 - p)) + shift
    return float(1 / (1 + np.exp(-z)))
