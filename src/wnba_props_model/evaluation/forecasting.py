"""Forecasting-quality diagnostics for discrete PMF predictions (P2 Phase 3).

Pure, deterministic functions to evaluate whether the PMF forecasting model is
trustworthy independent of any betting result: bias, MAE, RMSE, CRPS, PIT
uniformity, central-interval coverage (with binomial-compatibility tests),
calibration error by probability bucket, tail calibration, and a per-stat launch
gate. Coverage is judged by whether nominal coverage lies inside the binomial
uncertainty interval, not by the point estimate alone.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


def pmf_to_array(pmf_json) -> np.ndarray:
    obj = json.loads(pmf_json) if isinstance(pmf_json, str) else pmf_json
    if obj is None:
        return np.array([])
    if isinstance(obj, dict):
        n = max(int(k) for k in obj) + 1
        a = np.zeros(n)
        for k, v in obj.items():
            a[int(k)] = float(v)
        return a
    return np.asarray(obj, dtype=float)


def _cdf(pmf: np.ndarray) -> np.ndarray:
    return np.clip(np.cumsum(pmf), 0.0, 1.0)


def crps_discrete(pmf: np.ndarray, y: int) -> float:
    """CRPS for a discrete distribution on 0..K: sum_k (CDF(k) - 1{y<=k})^2."""
    if pmf.size == 0:
        return float("nan")
    cdf = _cdf(pmf)
    k = np.arange(len(pmf))
    step = (k >= y).astype(float)
    return float(np.sum((cdf - step) ** 2))


def mid_pit(pmf: np.ndarray, y: int) -> float:
    """Randomized/mid PIT value: CDF(y-1) + 0.5 * P(y). Uniform on [0,1] if calibrated."""
    if pmf.size == 0 or y < 0:
        return float("nan")
    below = float(pmf[:y].sum()) if y > 0 else 0.0
    at = float(pmf[y]) if y < len(pmf) else 0.0
    return below + 0.5 * at


def central_interval(pmf: np.ndarray, cover: float) -> tuple[int, int]:
    """Smallest central interval [lo,hi] with coverage >= `cover` (equal-tailed)."""
    cdf = _cdf(pmf)
    alpha = (1.0 - cover) / 2.0
    lo = int(np.searchsorted(cdf, alpha))
    hi = int(np.searchsorted(cdf, 1.0 - alpha))
    lo = min(lo, len(pmf) - 1)
    hi = min(hi, len(pmf) - 1)
    return lo, hi


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def _ece(pred: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    """Expected calibration error over pooled (predicted prob, binary outcome)."""
    if len(pred) == 0:
        return float("nan"), []
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(pred[m].mean()); acc = float(y[m].mean()); w = int(m.sum())
        ece += (w / len(pred)) * abs(conf - acc)
        table.append({"bin": b, "n": w, "pred": round(conf, 4), "obs": round(acc, 4)})
    return float(ece), table


@dataclass
class StatForecastResult:
    stat: str = ""
    n: int = 0
    bias: float = float("nan")
    bias_se: float = float("nan")
    bias_ok: bool = False
    mae: float = float("nan")
    rmse: float = float("nan")
    crps: float = float("nan")
    pit_ece: float = float("nan")           # deviation of mid-PIT hist from uniform
    coverage: dict = field(default_factory=dict)   # {nominal: {emp, lo, hi, ok}}
    calib_ece: float = float("nan")
    tail_cov90_ok: bool = False
    passed: bool = False
    reasons: list = field(default_factory=list)


def evaluate_stat(df, *, coverage_levels=(0.5, 0.8, 0.9),
                  bias_z: float = 3.0, max_calib_ece: float = 0.06,
                  max_pit_ece: float = 0.06) -> StatForecastResult:
    """Forecasting diagnostics + gate for one stat. df needs pmf_json, actual_outcome."""
    r = StatForecastResult(stat=str(df["stat"].iloc[0]) if len(df) else "", n=int(len(df)))
    if df.empty:
        r.reasons.append("no rows"); return r
    pmfs = [pmf_to_array(p) for p in df["pmf_json"]]
    y = df["actual_outcome"].astype(float).values
    means = np.array([float((np.arange(len(p)) * p).sum()) if p.size else np.nan for p in pmfs])
    valid = ~np.isnan(means) & ~np.isnan(y)
    pmfs = [p for p, v in zip(pmfs, valid) if v]
    y = y[valid].astype(int)
    means = means[valid]
    r.n = int(len(y))
    if r.n == 0:
        r.reasons.append("no valid rows"); return r

    err = means - y
    r.bias = float(err.mean()); r.mae = float(np.abs(err).mean())
    r.rmse = float(np.sqrt((err ** 2).mean()))
    r.bias_se = float(err.std(ddof=1) / math.sqrt(r.n)) if r.n > 1 else float("nan")
    r.bias_ok = bool(abs(r.bias) <= bias_z * r.bias_se) if r.bias_se == r.bias_se else False
    r.crps = float(np.mean([crps_discrete(p, int(yi)) for p, yi in zip(pmfs, y)]))

    # mid-PIT uniformity (ECE of the PIT histogram vs uniform)
    pit = np.array([mid_pit(p, int(yi)) for p, yi in zip(pmfs, y)])
    pit = pit[~np.isnan(pit)]
    nb = 10
    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, nb + 1))
    r.pit_ece = float(np.abs(hist / max(len(pit), 1) - 1.0 / nb).sum() / 2.0)

    # central-interval coverage + binomial compatibility
    for cl in coverage_levels:
        inside = 0
        widths = []
        for p, yi in zip(pmfs, y):
            lo, hi = central_interval(p, cl)
            widths.append(hi - lo)
            if lo <= yi <= hi:
                inside += 1
        emp = inside / r.n
        clo, chi = wilson_ci(inside, r.n)
        r.coverage[str(cl)] = {"empirical": round(emp, 4), "ci_lo": round(clo, 4),
                               "ci_hi": round(chi, 4), "nominal": cl,
                               "compatible": bool(clo <= cl <= chi),
                               "mean_width": round(float(np.mean(widths)), 3)}

    # pooled calibration ECE over P(Y >= k) across integer thresholds
    preds, outs = [], []
    for p, yi in zip(pmfs, y):
        cdf = _cdf(p)
        for k in range(1, len(p)):
            preds.append(1.0 - cdf[k - 1])   # P(Y >= k)
            outs.append(1.0 if yi >= k else 0.0)
    r.calib_ece, _ = _ece(np.array(preds), np.array(outs))

    r.tail_cov90_ok = bool(r.coverage.get("0.9", {}).get("compatible", False))

    # ---- launch gate ----
    if not r.bias_ok:
        r.reasons.append(f"bias {r.bias:+.2f} not within {bias_z}·SE ({r.bias_se:.3f})")
    if not r.coverage.get("0.8", {}).get("compatible", False):
        r.reasons.append("80% interval coverage incompatible with nominal")
    if not r.tail_cov90_ok:
        r.reasons.append("90% interval coverage incompatible with nominal")
    if not (r.calib_ece == r.calib_ece and r.calib_ece <= max_calib_ece):
        r.reasons.append(f"threshold calibration ECE {r.calib_ece:.3f} > {max_calib_ece}")
    if not (r.pit_ece == r.pit_ece and r.pit_ece <= max_pit_ece):
        r.reasons.append(f"PIT non-uniformity {r.pit_ece:.3f} > {max_pit_ece}")
    r.passed = len(r.reasons) == 0
    return r
