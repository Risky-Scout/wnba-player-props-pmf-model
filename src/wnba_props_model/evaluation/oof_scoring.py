"""OOF PMF scoring utilities for Stage 5.

Computes NLL, RPS, mean error, variance ratio and related diagnostics
from a long OOF PMF DataFrame.

Works with JSON-string PMF columns (pmf_json) as stored in parquet.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-row scoring functions
# ---------------------------------------------------------------------------

def nll_from_pmf_json(pmf_json: str, actual: int | float, eps: float = 1e-15) -> float:
    """Negative log-likelihood for one PMF and observed outcome."""
    pmf = json.loads(pmf_json)
    k = str(int(actual))
    p = float(pmf.get(k, 0.0))
    return -math.log(max(p, eps))


def rps_from_pmf_json(pmf_json: str, actual: int | float, cap: int) -> float:
    """Ranked Probability Score (discrete CRPS) for one PMF.

    RPS = sum_{k=0}^{cap} (F(k) - G(k))^2
    where F(k) = P(Y <= k) [model CDF], G(k) = I(actual <= k) [step CDF].
    """
    pmf = json.loads(pmf_json)
    k_arr = np.arange(cap + 1)
    probs = np.array([float(pmf.get(str(k), 0.0)) for k in k_arr])
    F = np.cumsum(probs)
    G = (k_arr >= int(actual)).astype(float)  # G(k) = 1 iff actual <= k
    return float(np.sum((F - G) ** 2))


def exact_outcome_prob(pmf_json: str, actual: int | float) -> float:
    """p(actual) from the PMF."""
    pmf = json.loads(pmf_json)
    return float(pmf.get(str(int(actual)), 0.0))


# ---------------------------------------------------------------------------
# Batch scoring of a PMF DataFrame
# ---------------------------------------------------------------------------

def score_oof_dataframe(
    pmf_df: pd.DataFrame,
    caps: dict[str, int],
    calibration_only: bool = False,
) -> dict[str, Any]:
    """Compute full OOF scoring metrics for all stats and role breakdowns.

    Args:
        pmf_df: Long OOF PMF DataFrame with columns:
            stat, actual_outcome, pmf_json, pmf_mean, pmf_variance,
            p0, calibration_eligible, projected_minutes_bucket,
            role_status, role_uncertainty_bucket (optional role cols).
        caps: PMF support cap per stat.
        calibration_only: If True, score only calibration_eligible rows.

    Returns:
        Nested dict: {"by_stat": {...}, "by_stat_role_bucket": {...}, ...}
    """
    if calibration_only:
        df = pmf_df[pmf_df["calibration_eligible"] == True].copy()  # noqa: E712
    else:
        df = pmf_df.copy()

    if df.empty:
        return {"error": "No rows to score"}

    results: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_calibration_eligible": int((pmf_df["calibration_eligible"] == True).sum()),  # noqa: E712
        "n_prior_only": int((pmf_df["oof_prediction_type"] == "prior_only").sum()),
        "n_model_oof": int((pmf_df["oof_prediction_type"] == "model_oof").sum()),
        "by_stat": {},
    }

    for stat, sub in df.groupby("stat"):
        cap = caps.get(stat, 20)
        actuals = sub["actual_outcome"].values.astype(float)
        pmf_means = sub["pmf_mean"].values.astype(float)
        pmf_vars = sub["pmf_variance"].values.astype(float)
        p0_arr = sub["p0"].values.astype(float)

        nlls = np.array([
            nll_from_pmf_json(pj, a) for pj, a in zip(sub["pmf_json"], actuals)
        ])
        rps_arr = np.array([
            rps_from_pmf_json(pj, a, cap) for pj, a in zip(sub["pmf_json"], actuals)
        ])
        exact_probs = np.array([
            exact_outcome_prob(pj, a) for pj, a in zip(sub["pmf_json"], actuals)
        ])

        actual_mean = float(np.mean(actuals))
        actual_var = float(np.var(actuals))
        pmf_mean_avg = float(np.mean(pmf_means))
        pmf_var_avg = float(np.mean(pmf_vars))

        results["by_stat"][stat] = {
            "n": int(len(sub)),
            "n_calibration_eligible": int((sub["calibration_eligible"] == True).sum()),  # noqa: E712
            "n_prior_only": int((sub["oof_prediction_type"] == "prior_only").sum()),
            "pmf_nll_mean": float(np.mean(nlls)),
            "pmf_rps_mean": float(np.mean(rps_arr)),
            "mean_actual": actual_mean,
            "mean_pmf": pmf_mean_avg,
            "mean_error": round(pmf_mean_avg - actual_mean, 4),
            "actual_variance": actual_var,
            "mean_pmf_variance": pmf_var_avg,
            "variance_ratio": round(pmf_var_avg / actual_var, 4) if actual_var > 0 else None,
            "empirical_zero_rate": float(np.mean(actuals == 0)),
            "mean_predicted_p0": float(np.mean(p0_arr)),
            "p0_vs_zero_delta": round(float(np.mean(p0_arr)) - float(np.mean(actuals == 0)), 4),
            "exact_outcome_prob_mean": float(np.mean(exact_probs)),
        }

    # ---- Role-level breakdowns -------------------------------------------
    role_cols = [c for c in ["projected_minutes_bucket", "role_status",
                              "role_uncertainty_bucket"] if c in pmf_df.columns]
    if role_cols:
        results["by_stat_role_bucket"] = {}
        for (stat, role_val), grp in df.groupby(["stat", role_cols[0]]):
            cap = caps.get(stat, 20)
            actuals = grp["actual_outcome"].values.astype(float)
            pmf_means = grp["pmf_mean"].values.astype(float)
            key = f"{stat}|{role_cols[0]}={role_val}"
            results["by_stat_role_bucket"][key] = {
                "n": int(len(grp)),
                "mean_actual": float(np.mean(actuals)),
                "mean_pmf": float(np.mean(pmf_means)),
                "mean_error": round(float(np.mean(pmf_means)) - float(np.mean(actuals)), 4),
                "pmf_nll_mean": float(np.mean([
                    nll_from_pmf_json(pj, a) for pj, a in zip(grp["pmf_json"], actuals)
                ])),
            }

    return results
