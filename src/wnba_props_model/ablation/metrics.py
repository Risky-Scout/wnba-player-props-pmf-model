"""Scoring primitives for the feature-ablation study.

These MIRROR the frozen market-superiority evaluator
(``scripts/evaluate_opportunity_oof.py``) so binary-prop metrics are directly
comparable to the promotion contract, and add proper count-prop scores
(Poisson deviance, discrete PMF log-score, discrete CRPS) for the outcome-only
props (stl / blk / tov) that books do not quote.

Every function is import-safe and unit tested. The paired date-cluster
bootstrap and Holm correction are copied verbatim (behaviourally) from the
evaluator so p-values in this study use the identical procedure.
"""
from __future__ import annotations

import json

import numpy as np

try:  # AUC / calibration slope need sklearn; degrade gracefully otherwise
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    LogisticRegression = None

_EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, float), _EPS, 1 - _EPS)


def log_loss(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((_clip(p) - np.asarray(y, float)) ** 2))


def auc(y, p) -> float:
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    if roc_auc_score is None:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for AUC scoring")
    return float(roc_auc_score(y, np.asarray(p, float)))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = _clip(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Logistic re-fit of outcome on logit(p): returns (intercept, slope)."""
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2 or LogisticRegression is None:
        return float("nan"), float("nan")
    x = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


# --------------------------------------------------------------------------- #
# count-prop proper scores (no market comparison possible)
# --------------------------------------------------------------------------- #
def poisson_deviance(y, mu) -> float:
    """Mean unit Poisson deviance: 2*(y*log(y/mu) - (y-mu)), with 0*log0 := 0."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), _EPS, None)
    term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(np.mean(2.0 * term))


def pmf_log_score(pmf_list, actual) -> float:
    """Mean negative log probability the count PMF assigns to the realized count."""
    scores = []
    for arr, a in zip(pmf_list, actual):
        if arr is None or a is None or not np.isfinite(a):
            continue
        arr = np.asarray(arr, float)
        k = int(round(float(a)))
        pk = arr[k] if 0 <= k < arr.size else 0.0
        scores.append(-np.log(max(pk, _EPS)))
    return float(np.mean(scores)) if scores else float("nan")


def crps_discrete(pmf_list, actual) -> float:
    """Mean discrete CRPS = sum_k (CDF(k) - 1{actual<=k})^2 over the PMF support."""
    vals = []
    for arr, a in zip(pmf_list, actual):
        if arr is None or a is None or not np.isfinite(a):
            continue
        arr = np.asarray(arr, float)
        cdf = np.cumsum(arr)
        k = np.arange(arr.size)
        step = (k >= int(round(float(a)))).astype(float)
        vals.append(float(np.sum((cdf - step) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# paired date-cluster bootstrap + Holm (mirrors evaluate_opportunity_oof)
# --------------------------------------------------------------------------- #
def paired_bootstrap(y, model_p, ref_p, dates, iters: int, seed: int):
    """Paired date-cluster bootstrap. Returns (ci_ll, ci_brier, p_ll, p_brier).

    p_ll = P(model no better than ref on log-loss); p_brier analogous. Resampling
    clusters by unique date so within-date correlation is respected.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    model_p = np.asarray(model_p, float)
    ref_p = np.asarray(ref_p, float)
    # cluster on integer codes (robust to datetime64 vs python-datetime comparison)
    codes = pd.factorize(pd.Series(dates))[0]
    uniq = np.unique(codes)
    by_date = {int(d): np.where(codes == d)[0] for d in uniq}
    dll, dbs = [], []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_date[int(d)] for d in pick])
        yy = y[idx]
        dll.append(log_loss(yy, model_p[idx]) - log_loss(yy, ref_p[idx]))
        dbs.append(brier(yy, model_p[idx]) - brier(yy, ref_p[idx]))
    dll = np.asarray(dll)
    dbs = np.asarray(dbs)
    ci_ll = np.percentile(dll, [2.5, 97.5]).tolist()
    ci_bs = np.percentile(dbs, [2.5, 97.5]).tolist()
    p_ll = float(np.mean(dll >= 0.0))
    p_brier = float(np.mean(dbs >= 0.0))
    return ci_ll, ci_bs, p_ll, p_brier


def holm(pvals: dict) -> dict:
    """Holm step-down adjustment across a family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def parse_pmf_json(js):
    if js is None or (isinstance(js, float) and np.isnan(js)):
        return None
    return np.asarray(json.loads(js), float)
