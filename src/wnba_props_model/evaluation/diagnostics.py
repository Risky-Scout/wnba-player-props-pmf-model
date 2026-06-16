from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wnba_props_model.models.market import binary_logloss, brier, ignorance_score_binary, prob_over_from_pmf
from wnba_props_model.models.simulation import normalize_pmf


def pmf_nll(pmf: np.ndarray, outcome: int) -> float:
    arr = normalize_pmf(pmf)
    y = int(np.clip(outcome, 0, len(arr) - 1))
    return float(-math.log(max(arr[y], 1e-12)))


def rps(pmf: np.ndarray, outcome: int) -> float:
    arr = normalize_pmf(pmf)
    y = int(np.clip(outcome, 0, len(arr) - 1))
    cdf = np.cumsum(arr)
    truth = (np.arange(len(arr)) >= y).astype(float)
    return float(np.mean((cdf - truth) ** 2))


def randomized_pit_values(pmfs: list[np.ndarray], outcomes: list[int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = []
    for pmf, y0 in zip(pmfs, outcomes):
        arr = normalize_pmf(pmf)
        y = int(np.clip(y0, 0, len(arr) - 1))
        vals.append(arr[:y].sum() + rng.uniform() * arr[y])
    return np.asarray(vals)


def ks_uniform(u: np.ndarray) -> float:
    x = np.sort(np.asarray(u, dtype=float))
    if len(x) == 0:
        return float("nan")
    ecdf_hi = np.arange(1, len(x) + 1) / len(x)
    ecdf_lo = np.arange(0, len(x)) / len(x)
    return float(max(np.max(np.abs(ecdf_hi - x)), np.max(np.abs(x - ecdf_lo))))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    obs = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - obs[mask].mean())
    return float(ece)


def _over_half_line_ece(pmfs: list[np.ndarray], outcomes: list[int]) -> float:
    """ECE on binary over/under at the natural median line (mean - 0.5).

    Computes P(Y > line) from each PMF vs. empirical hit rate per probability
    bin. Per PenaltyBlog methodology: ECE < 0.03 per stat is the production gate.
    """
    if not pmfs:
        return float("nan")
    lines = [max(0.0, float(np.dot(np.arange(len(p)), normalize_pmf(p))) - 0.5) for p in pmfs]
    model_probs = np.array([prob_over_from_pmf(normalize_pmf(p), l) for p, l in zip(pmfs, lines)])
    hits = np.array([1.0 if float(y) > l else 0.0 for y, l in zip(outcomes, lines)])
    return expected_calibration_error(model_probs, hits)


def calibration_report(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows require columns: stat, role_bucket, pmf (np.ndarray), outcome.

    Returns per-stat/role metrics including:
    - nll: Negative log-likelihood (primary full-PMF calibration metric)
    - ignorance_score: Log Loss in bits (PenaltyBlog primary binary metric)
    - rps: Ranked Probability Score (secondary)
    - ece: Expected Calibration Error on binary over/under (gate: < 0.03)
    - pit_ks: KS statistic vs. Uniform[0,1] (gate: < 0.075)
    """
    records = []
    for (stat, role), g in rows.groupby(["stat", "role_bucket"], dropna=False):
        pmfs = list(g["pmf"])
        y = g["outcome"].astype(int).tolist()
        means = np.array([np.dot(np.arange(len(p)), normalize_pmf(p)) for p in pmfs])
        vars_ = np.array([
            np.dot((np.arange(len(p)) - m) ** 2, normalize_pmf(p))
            for p, m in zip(pmfs, means)
        ])
        pit = randomized_pit_values(pmfs, y)
        nll_vals = [pmf_nll(p, yy) for p, yy in zip(pmfs, y)]
        rps_vals = [rps(p, yy) for p, yy in zip(pmfs, y)]
        ign_vals = [ignorance_score_binary(
            prob_over_from_pmf(normalize_pmf(p), max(0.0, float(np.dot(np.arange(len(p)), normalize_pmf(p))) - 0.5)),
            1 if float(yy) > max(0.0, float(np.dot(np.arange(len(p)), normalize_pmf(p))) - 0.5) else 0,
        ) for p, yy in zip(pmfs, y)]
        records.append({
            "stat": stat,
            "role_bucket": role,
            "n": len(g),
            "nll": float(np.mean(nll_vals)),
            "ignorance_score": float(np.mean(ign_vals)),
            "rps": float(np.mean(rps_vals)),
            "ece": _over_half_line_ece(pmfs, y),
            "mean_error": float(np.mean(means - np.asarray(y))),
            "variance_error": float(np.mean(vars_) - np.var(y)),
            "pit_mean": float(pit.mean()),
            "pit_std": float(pit.std(ddof=1)) if len(pit) > 1 else float("nan"),
            "pit_ks": ks_uniform(pit),
        })
    return pd.DataFrame(records)


def build_event_market_loss_rows(market_rows: pd.DataFrame) -> pd.DataFrame:
    """Score model PMFs against no-vig market probabilities.

    Required columns: pmf, line, outcome, market_prob_over_no_vig.
    """
    out = market_rows.copy()
    out["model_prob_over"] = [prob_over_from_pmf(p, line) for p, line in zip(out["pmf"], out["line"])]
    out["hit_result"] = (out["outcome"].astype(float) > out["line"].astype(float)).astype(int)
    out["is_push"] = (out["outcome"].astype(float) == out["line"].astype(float))
    out = out[~out["is_push"]].copy()
    out["model_event_logloss"] = [binary_logloss(p, y) for p, y in zip(out["model_prob_over"], out["hit_result"])]
    out["market_event_logloss"] = [binary_logloss(p, y) for p, y in zip(out["market_prob_over_no_vig"], out["hit_result"])]
    out["model_ignorance_score"] = [ignorance_score_binary(p, y) for p, y in zip(out["model_prob_over"], out["hit_result"])]
    out["market_ignorance_score"] = [ignorance_score_binary(p, y) for p, y in zip(out["market_prob_over_no_vig"], out["hit_result"])]
    out["model_brier"] = [brier(p, y) for p, y in zip(out["model_prob_over"], out["hit_result"])]
    out["market_brier"] = [brier(p, y) for p, y in zip(out["market_prob_over_no_vig"], out["hit_result"])]
    out["event_logloss_delta"] = out["model_event_logloss"] - out["market_event_logloss"]
    out["ignorance_score_delta"] = out["model_ignorance_score"] - out["market_ignorance_score"]
    out["brier_delta"] = out["model_brier"] - out["market_brier"]
    return out


def bootstrap_ucb95(x: np.ndarray, reps: int = 2000, seed: int = 20260512) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if len(arr) == 0:
        return float("nan"), float("nan")
    boot = np.empty(reps)
    for j in range(reps):
        boot[j] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(arr.mean()), float(np.quantile(boot, 0.95))


def market_superiority_report(loss_rows: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    records = []
    for (stat, role), g in loss_rows.groupby(["stat", "role_bucket"], dropna=False):
        mean_log, ucb_log = bootstrap_ucb95(g["event_logloss_delta"].to_numpy())
        mean_ign, ucb_ign = bootstrap_ucb95(g["ignorance_score_delta"].to_numpy()) if "ignorance_score_delta" in g else (float("nan"), float("nan"))
        mean_brier, ucb_brier = bootstrap_ucb95(g["brier_delta"].to_numpy())
        records.append({
            "stat": stat,
            "role_bucket": role,
            "n": len(g),
            "event_logloss_delta_mean": mean_log,
            "event_logloss_delta_ucb95": ucb_log,
            "ignorance_score_delta_mean": mean_ign,
            "ignorance_score_delta_ucb95": ucb_ign,
            "brier_delta_mean": mean_brier,
            "brier_delta_ucb95": ucb_brier,
            "eligible": len(g) >= min_rows,
            "certified_pass": (
                len(g) >= min_rows
                and ucb_log < -0.0025
                and ucb_brier < -0.0010
            ),
        })
    return pd.DataFrame(records)
