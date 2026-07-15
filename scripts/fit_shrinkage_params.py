"""Fit optimal K_BASE for each stat by minimizing ECE on OOF PMF data.

K_BASE controls the strength of Bayesian shrinkage toward the league prior.
Higher K = more shrinkage (predictions pulled toward mean).
Lower K = less shrinkage (predictions rely more on player history).

This script finds the K that minimizes Expected Calibration Error on the
OOF validation set — meaning K is tuned to maximize prediction accuracy,
not set by hand.

Methodology:
  For each stat, sweep K in [0.5, 30.0]. For each K, re-apply shrinkage
  to OOF PMF means and compute ECE from the resulting P(over line) vs actuals.
  Store the K minimizing ECE in shrinkage_params.json.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import norm

OOF_PATH = Path("data/oof/oof_player_stat_pmfs.parquet")
LEAGUE_PRIORS_PATH = Path("artifacts/models/league_priors.json")
OUT_PATH = Path("artifacts/models/shrinkage_params.json")

BASE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMBO_STATS = ["pts_reb", "pts_ast", "reb_ast", "pts_reb_ast", "stocks"]
COMBO_K_FLOOR = 5.0


def compute_ece(p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean |bin_accuracy - bin_confidence|."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def shrink_mean(raw_mean: float, prior_mean: float, k: float) -> float:
    """Bayesian shrinkage: blend raw_mean toward prior_mean with strength k."""
    return (raw_mean + k * prior_mean) / (1.0 + k)


def evaluate_k(k: float, stat_oof: pd.DataFrame, prior_mean: float) -> float:
    """Apply shrinkage at strength k and return ECE."""
    if stat_oof.empty or "pmf_mean" not in stat_oof.columns:
        return 999.0
    shrunk = stat_oof["pmf_mean"].apply(
        lambda m: shrink_mean(float(m), prior_mean, k)
    )
    if "line" not in stat_oof.columns or "actual" not in stat_oof.columns:
        return 999.0
    std = float(stat_oof["pmf_mean"].std())
    if std < 0.1:
        return 999.0
    p_over = 1.0 - norm.cdf(
        stat_oof["line"].values, loc=shrunk.values, scale=std
    )
    y_true = (stat_oof["actual"].values > stat_oof["line"].values).astype(float)
    return compute_ece(p_over, y_true)


def main() -> None:
    if not OOF_PATH.exists():
        print(f"OOF file not found at {OOF_PATH}. Run build_oof_pmfs.py first.")
        return

    oof = pd.read_parquet(OOF_PATH)
    print(f"Loaded OOF: {len(oof)} rows, columns: {list(oof.columns[:10])}")

    # Normalise column names: accept both 'actual_outcome' and 'actual'
    if "actual_outcome" in oof.columns and "actual" not in oof.columns:
        oof = oof.rename(columns={"actual_outcome": "actual"})

    # Load league priors
    priors: dict = {}
    if LEAGUE_PRIORS_PATH.exists():
        with open(LEAGUE_PRIORS_PATH) as f:
            priors = json.load(f)

    # Load existing shrinkage params (to preserve shin_z_optimal and other keys)
    existing: dict = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)

    params = dict(existing)

    stat_col = "stat" if "stat" in oof.columns else None
    if stat_col is None:
        print("OOF missing 'stat' column, cannot fit K_BASE")
        return

    k_results: dict = {}
    for stat in BASE_STATS:
        stat_oof = oof[oof[stat_col] == stat].copy()
        if len(stat_oof) < 50:
            print(
                f"{stat}: insufficient OOF rows ({len(stat_oof)}), skipping"
            )
            k_results[stat] = {
                "k_base_fitted": None,
                "n_rows": len(stat_oof),
                "ece_at_k": None,
            }
            continue

        prior_val = priors.get(stat)
        if isinstance(prior_val, dict):
            prior_mean = float(prior_val.get("mean", stat_oof["pmf_mean"].mean()))
        elif prior_val is not None:
            prior_mean = float(prior_val)
        else:
            prior_mean = float(stat_oof["pmf_mean"].mean())

        result = minimize_scalar(
            lambda k: evaluate_k(k, stat_oof, prior_mean),
            bounds=(0.5, 30.0),
            method="bounded",
        )
        k_opt = round(float(result.x), 2)
        ece_opt = round(float(result.fun), 5)
        print(
            f"{stat}: K_BASE_optimal={k_opt}, ECE={ece_opt:.4f} "
            f"(n={len(stat_oof)})"
        )
        k_results[stat] = {
            "k_base_fitted": k_opt,
            "n_rows": len(stat_oof),
            "ece_at_k": ece_opt,
        }

    # Combo stats: always floor at COMBO_K_FLOOR
    for stat in COMBO_STATS:
        k_results[stat] = {
            "k_base_fitted": COMBO_K_FLOOR,
            "n_rows": 0,
            "note": "floored at combo minimum",
        }

    params["k_base_fitted"] = k_results
    params["_generated_at"] = pd.Timestamp.now().isoformat()

    # Preserve shin_z_optimal from previous run if present and not overwritten
    if "shin_z_optimal" not in params:
        params["shin_z_optimal"] = 0.15

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Shrinkage params written to {OUT_PATH}")


if __name__ == "__main__":
    main()
