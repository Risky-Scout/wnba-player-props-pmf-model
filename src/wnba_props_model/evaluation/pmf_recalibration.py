"""Fold-safe PMF recalibration (P2 Challenger C, forecasting repair).

Corrects the two defects the raw OOF PMFs exhibit on the holdout — a systematic
negative mean bias (under-prediction) and interval mis-coverage — by shifting the
mean and scaling the dispersion of each PMF. Correction factors are estimated per
(stat, role) using ONLY folds strictly earlier than the fold being transformed
(walk-forward, no lookahead), with pooled fallback when a cell has too few games.
The transform preserves total probability mass and integer support.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wnba_props_model.evaluation.forecasting import pmf_to_array


def recalibrate_pmf(pmf: np.ndarray, delta: float, scale: float,
                    max_support: int = 80) -> np.ndarray:
    """Shift mean by `delta` and scale spread by `scale` about the mean, then
    redistribute mass onto the non-negative integer lattice."""
    if pmf.size == 0:
        return pmf
    k = np.arange(len(pmf))
    mean = float((k * pmf).sum())
    new_val = mean + scale * (k - mean) + delta
    new_val = np.clip(new_val, 0.0, max_support)
    out = np.zeros(max_support + 1)
    for p, v in zip(pmf, new_val):
        if p <= 0:
            continue
        lo = int(np.floor(v)); hi = min(lo + 1, max_support)
        frac = v - lo
        out[lo] += p * (1 - frac)
        out[hi] += p * frac
    s = out.sum()
    if s > 0:
        out /= s
    # trim trailing zeros for compactness
    nz = np.nonzero(out > 1e-12)[0]
    return out[: (nz[-1] + 1)] if len(nz) else out


def _fit_factors(train: pd.DataFrame, min_n: int = 40,
                 dispersion: bool = False) -> tuple[float, float]:
    """(delta, scale). delta = mean(actual - pmf_mean) corrects the systematic
    under-prediction. Dispersion scaling is OFF by default: on the holdout the raw
    PMF spread was already near-nominal, and rescaling by residual/model SD (which
    conflates irreducible noise with forecast spread) degraded interval coverage.
    Returns (0,1) when the training cell is too small."""
    if len(train) < min_n:
        return (0.0, 1.0)
    means = train["pmf_mean"].astype(float).values
    y = train["actual_outcome"].astype(float).values
    delta = float(np.mean(y - means))
    if not dispersion:
        return (delta, 1.0)
    resid_sd = float(np.std(y - means, ddof=1))
    model_sd = float(np.sqrt(np.mean(train["pmf_variance"].astype(float).clip(lower=1e-6))))
    scale = resid_sd / model_sd if model_sd > 1e-6 else 1.0
    # Allow meaningful sharpening (down to 0.4) since the raw PMFs are over-dispersed.
    return (delta, float(np.clip(scale, 0.4, 1.6)))


def fold_safe_pmf_recalibration(df: pd.DataFrame, role_col: str = "role_bucket",
                                use_role: bool = True, use_dispersion: bool = False) -> pd.Series:
    """Return a Series of recalibrated pmf_json aligned to df.index. For each fold
    (ordered by fold_id / date), correction factors are fit per (stat[, role]) on
    strictly-earlier folds only; pooled per-stat factors are the fallback."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    fold_key = "fold_id" if "fold_id" in d.columns else None
    if fold_key is not None:
        folds = sorted(d[fold_key].dropna().unique())
    else:
        folds = sorted(d["_date"].dropna().unique())
    out = pd.Series(index=d.index, dtype=object)

    for i, f in enumerate(folds):
        cur = d[d[fold_key] == f] if fold_key else d[d["_date"] == f]
        prior = d[d[fold_key] < f] if fold_key else d[d["_date"] < f]
        for stat, g in cur.groupby("stat"):
            pri_stat = prior[prior["stat"] == stat]
            pooled = _fit_factors(pri_stat, dispersion=use_dispersion)
            for idx, row in g.iterrows():
                delta, scale = pooled
                if use_role and role_col in g.columns:
                    pri_cell = pri_stat[pri_stat[role_col] == row[role_col]]
                    cell = _fit_factors(pri_cell, dispersion=use_dispersion)
                    if cell != (0.0, 1.0):
                        delta, scale = cell
                pmf = pmf_to_array(row["pmf_json"])
                cal = recalibrate_pmf(pmf, delta, scale)
                out.at[idx] = json.dumps({str(k): float(round(v, 8))
                                          for k, v in enumerate(cal) if v > 1e-9})
    # first fold (no prior) keeps raw
    out = out.fillna(d["pmf_json"])
    return out
