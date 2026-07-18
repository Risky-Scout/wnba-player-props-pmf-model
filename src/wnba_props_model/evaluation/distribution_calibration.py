"""Full-distribution PMF calibrators (P3 final recovery).

All fits are walk-forward / pre-block only (no lookahead). Candidates:
  * monotone_cdf_recalibration  — PIT-CDF recalibration (Kuleshov-style) with shrinkage
    toward identity, recovering the PMF by adjacent CDF differences + a tail floor;
  * cdf_empirical_mixture        — calibrated PMF * w + prior conditional empirical PMF * (1-w);
  * hurdle_calibration           — separate P(Y=0) + positive-count conditional CDF (sparse stats);
  * hierarchical_empirical_pmf   — role/minutes-conditioned empirical residual PMF centered on
    the point forecast, shrunk to the pooled stat distribution (trustworthy fallback).
These repair arbitrary CDF-shape / tail errors that location(-and-scale) transport cannot.
"""
from __future__ import annotations

import numpy as np

from wnba_props_model.evaluation.forecasting import (
    _cdf, crps_discrete, log_score, pmf_to_array, randomized_pit,
)

PROB_FLOOR = 1e-4


def _clean(pmf: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Finite, nonnegative, floored, exactly-normalized PMF."""
    p = np.asarray(pmf, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        p = np.ones(len(p))
    # small floor so valid tail outcomes are not ~0 probability, then renormalize
    p = p + floor
    return p / p.sum()


# ---- monotone PIT-CDF recalibration -------------------------------------------------

def fit_pit_recalibration(pmfs, actuals, seed_keys):
    """Return sorted randomized-PIT values (the empirical PIT CDF R). R(q) = mean(pit<=q)."""
    pits = []
    for p, y, k in zip(pmfs, actuals, seed_keys):
        u = randomized_pit(p, int(y), k)
        if np.isfinite(u):
            pits.append(u)
    return np.sort(np.asarray(pits, dtype=float)) if pits else np.array([])


def _R(sorted_pits: np.ndarray, q: np.ndarray, shrink: float) -> np.ndarray:
    """Shrunk recalibration map: (1-shrink)*identity + shrink*empiricalCDF(pit)."""
    q = np.asarray(q, dtype=float)
    if sorted_pits.size == 0:
        return q
    emp = np.searchsorted(sorted_pits, q, side="right") / float(len(sorted_pits))
    return (1.0 - shrink) * q + shrink * emp


def apply_monotone_cdf_recalibration(pmf: np.ndarray, sorted_pits: np.ndarray,
                                     shrink: float) -> np.ndarray:
    """Recalibrated PMF: F'(k)=R(F(k)); pmf'[k]=F'(k)-F'(k-1); floored + normalized."""
    if pmf.size == 0:
        return pmf
    cdf = _cdf(pmf)
    fp = _R(sorted_pits, cdf, shrink)
    fp = np.maximum.accumulate(np.clip(fp, 0.0, 1.0))   # enforce monotone in [0,1]
    fp[-1] = 1.0
    newpmf = np.diff(np.concatenate([[0.0], fp]))
    return _clean(newpmf)


# ---- empirical PMFs / mixture -------------------------------------------------------

def empirical_pmf(actuals, max_support: int) -> np.ndarray:
    pmf = np.zeros(int(max_support) + 1)
    for a in actuals:
        ai = int(round(float(a)))
        if 0 <= ai <= max_support:
            pmf[ai] += 1
    return _clean(pmf) if pmf.sum() else pmf


def mixture_pmf(cal_pmf: np.ndarray, emp_pmf: np.ndarray, weight: float) -> np.ndarray:
    n = max(len(cal_pmf), len(emp_pmf))
    a = np.zeros(n); b = np.zeros(n)
    a[:len(cal_pmf)] = cal_pmf; b[:len(emp_pmf)] = emp_pmf
    return _clean(weight * a + (1.0 - weight) * b)


# ---- hurdle calibration (sparse stats) ----------------------------------------------

def hurdle_calibrate(pmf: np.ndarray, p0_target: float, pos_sorted_pits: np.ndarray,
                     shrink: float) -> np.ndarray:
    """Calibrate P(Y=0) to p0_target and recalibrate the positive-count conditional CDF."""
    if pmf.size == 0:
        return pmf
    p0 = float(np.clip(p0_target, PROB_FLOOR, 1 - PROB_FLOOR))
    pos = pmf[1:].copy()
    if pos.sum() <= 0:
        out = np.zeros(len(pmf)); out[0] = 1.0
        return _clean(out)
    pos = pos / pos.sum()
    pos_cal = apply_monotone_cdf_recalibration(pos, pos_sorted_pits, shrink)
    out = np.zeros(max(len(pmf), len(pos_cal) + 1))
    out[0] = p0
    out[1:len(pos_cal) + 1] = (1.0 - p0) * pos_cal
    return _clean(out)


# ---- hierarchical empirical residual fallback ---------------------------------------

def fit_residual_hist(train_df, stat_col="stat", role_col="role_bucket",
                      minbucket_col="_minbucket", min_cell: int = 30):
    """Per-(role, minutes-bucket) histogram of realized COUNTS at absolute level, plus a
    pooled fallback. We store absolute-outcome histograms (not round(point) residuals,
    which introduced a rounding-center bias); the point forecast is used only to select
    the conditioning cell + a fractional-shift correction. Returns dict cell->(vals, wts)."""
    d = train_df
    cells = {}
    vals_p, cnt_p = np.unique(d["actual_outcome"].astype(int).values, return_counts=True)
    cells["_pooled"] = (vals_p, cnt_p / cnt_p.sum(),
                        float((d["actual_outcome"].astype(float) - d["_point"].astype(float)).mean()))
    if role_col in d.columns and minbucket_col in d.columns:
        for (rb, mb), g in d.groupby([role_col, minbucket_col]):
            if len(g) < min_cell:
                continue
            v, c = np.unique(g["actual_outcome"].astype(int).values, return_counts=True)
            shift = float((g["actual_outcome"].astype(float) - g["_point"].astype(float)).mean())
            cells[f"{rb}|{mb}"] = (v, c / c.sum(), shift)
    return cells


def hierarchical_empirical_pmf(point: float, cell_key: str, cells: dict,
                               max_support: int, shrink_to_pooled: float = 0.3) -> np.ndarray:
    """Empirical count distribution for the conditioning cell, shifted so its mean tracks
    the player's point forecast (point + cell residual-bias), preserving player-level info
    without a rounding-center bias. Shrunk toward the pooled cell."""
    vp, wp, sp = cells["_pooled"]
    vc, wc, sc = cells.get(cell_key, cells["_pooled"])
    pmf = np.zeros(int(max_support) + 1)

    def _add(vals, wts, cell_shift, scale):
        # cell mean of vals:
        cell_mean = float((vals * wts).sum())
        # target mean = point + (cell residual bias); shift = target - cell_mean
        target_mean = float(point) + cell_shift
        shift = int(round(target_mean - cell_mean))
        for v, wt in zip(vals, wts):
            k = int(v) + shift
            if 0 <= k <= max_support:
                pmf[k] += scale * wt
    _add(vc, wc, sc, 1.0 - shrink_to_pooled)
    _add(vp, wp, sp, shrink_to_pooled)
    return _clean(pmf)


# ---- selection scoring --------------------------------------------------------------

def score_pmfs(pmfs, actuals) -> tuple[float, float]:
    """(mean CRPS, mean log score) — the pre-block selection objective."""
    c = [crps_discrete(p, int(y)) for p, y in zip(pmfs, actuals) if p.size]
    l = [log_score(p, int(y)) for p, y in zip(pmfs, actuals) if p.size]
    return (float(np.mean(c)) if c else float("inf"),
            float(np.mean(l)) if l else float("inf"))
