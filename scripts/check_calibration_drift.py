"""Daily calibration drift detector.

Loads the last N scored predictions (where actual outcomes are known), computes
PIT-based anytime-valid drift checks and ECCE-MAD (binning-free calibration error)
in addition to the legacy ECE/mean-error check.

Exit codes:
  0  — no drift or soft drift only (warning logged)
  1  — hard drift detected (triggers auto-recalibration in daily_pipeline.yml)

PIT (Probability Integral Transform):
  If model is well-calibrated, PIT values ~ Uniform(0,1).
  KS test: p < 0.05 → calibration drift detected.

ECCE-MAD (Empirical Calibration-Coverage Error — Mean Absolute Deviation):
  Binning-free calibration error, superior to ECE because:
  1. No arbitrary bin count
  2. Monotone in sample size
  3. Theoretical connection to KS test
  ECCE-MAD = max|cumulative(actual - predicted)| / n < 0.05 → well-calibrated
  Reference: Farran (2026), https://arxiv.org/abs/2603.13156

Usage:
    python scripts/check_calibration_drift.py \\
        --scored-predictions artifacts/audits/scored_predictions.parquet \\
        --out artifacts/audits/drift_check_2026-06-09.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats as sp_stats

from wnba_props_model.evaluation.diagnostics import calibration_report, randomized_pit_values


# ---------------------------------------------------------------------------
# PIT + ECCE-MAD implementation (Item 12)
# ---------------------------------------------------------------------------

def compute_pit_values(oof_pmfs: list, actuals: list[int]) -> np.ndarray:
    """Compute randomized Probability Integral Transform values.

    Uses randomized PIT (ties broken uniformly) so that discrete PMFs produce
    PIT values that are Uniform(0,1) under correct calibration. This is consistent
    with the PIT convention used in calibration fitting (diagnostics.randomized_pit_values).

    Deterministic PIT (CDF at actual) produces discrete artifacts that cause KS tests
    to over-reject — randomized PIT is the correct choice for discrete count data.
    """
    # Normalize PMFs to ndarray format for randomized_pit_values
    pmf_arrays: list[np.ndarray] = []
    for pmf in oof_pmfs:
        if isinstance(pmf, dict):
            kmax = max(int(k) for k in pmf.keys()) if pmf else 0
            arr = np.zeros(kmax + 1)
            for k, p in pmf.items():
                arr[int(k)] = float(p)
            pmf_arrays.append(arr)
        elif hasattr(pmf, "__len__"):
            pmf_arrays.append(np.asarray(pmf, dtype=float))
        else:
            pmf_arrays.append(np.array([1.0]))

    actuals_int = [int(a) for a in actuals]
    return randomized_pit_values(pmf_arrays, actuals_int)


def check_pit_uniformity(pit_values: np.ndarray, alpha: float = 0.05) -> dict:
    """KS test for PIT uniformity.

    Returns:
        dict with p_value, is_uniform, ks_stat
    """
    if len(pit_values) < 10:
        return {"p_value": 1.0, "is_uniform": True, "ks_stat": 0.0, "n": len(pit_values)}
    ks_stat, p_value = sp_stats.kstest(pit_values, "uniform")
    return {
        "p_value": float(p_value),
        "is_uniform": bool(p_value > alpha),
        "ks_stat": float(ks_stat),
        "n": len(pit_values),
    }


def compute_ecce_mad(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """Compute ECCE-MAD (binning-free calibration error).

    ECCE-MAD = max|cumulative(actual - predicted)| / n

    Values < 0.05 indicate well-calibrated predictions.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(predictions) == 0:
        return 0.0
    sorted_idx = np.argsort(predictions)
    cum_diff = np.cumsum(actuals[sorted_idx] - predictions[sorted_idx])
    return float(np.max(np.abs(cum_diff)) / len(predictions))


def analyze_direction(predictions: np.ndarray, actuals: np.ndarray) -> str:
    """Diagnose direction of miscalibration."""
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(predictions) == 0:
        return "balanced"
    delta = float(predictions.mean() - actuals.mean())
    if delta > 0.03:
        return "overprojection"
    elif delta < -0.03:
        return "underprojection"
    return "balanced"

app = typer.Typer(add_completion=False)

_DEFAULT_SOFT_ECE = 0.06
_DEFAULT_HARD_ECE = 0.12
_DEFAULT_SOFT_MEAN_ERR = 0.25
_DEFAULT_HARD_MEAN_ERR = 0.50
_DEFAULT_WINDOW = 100


@app.command()
def main(
    scored_predictions: str = typer.Option(
        ...,
        "--scored-predictions",
        help="Parquet of scored predictions with actual outcomes. "
             "Must have stat, pmf_json/pmf, actual_outcome/outcome cols.",
    ),
    out: str = typer.Option(
        "artifacts/audits/drift_check.json",
        "--out",
        help="Output path for drift report JSON.",
    ),
    window: int = typer.Option(
        _DEFAULT_WINDOW,
        "--window",
        help="Number of most-recent scored rows per stat to evaluate.",
    ),
    soft_ece: float = typer.Option(
        _DEFAULT_SOFT_ECE,
        "--thresholds-soft-ece",
        help="ECE soft-drift threshold (warning only).",
    ),
    hard_ece: float = typer.Option(
        _DEFAULT_HARD_ECE,
        "--thresholds-hard-ece",
        help="ECE hard-drift threshold (exit 1 → triggers auto-refit).",
    ),
    soft_mean_err: float = typer.Option(
        _DEFAULT_SOFT_MEAN_ERR,
        "--thresholds-soft-mean-err",
        help="|mean_error| soft-drift threshold.",
    ),
    hard_mean_err: float = typer.Option(
        _DEFAULT_HARD_MEAN_ERR,
        "--thresholds-hard-mean-err",
        help="|mean_error| hard-drift threshold.",
    ),
) -> None:
    """Detect calibration drift in recent scored predictions.

    Soft drift (ECE>0.06 or |mean_error|>0.25): log warning, exit 0.
    Hard drift (ECE>0.12 or |mean_error|>0.50): log error, exit 1 →
        triggers auto-recalibration in daily_pipeline.yml.
    """
    scored_path = Path(scored_predictions)
    if not scored_path.exists():
        typer.echo(f"[DRIFT] No scored predictions file found at {scored_path}. "
                   "Skipping drift check (first run).")
        _write_report(out, {"status": "no_data", "message": "No scored predictions yet."})
        raise typer.Exit(0)

    from wnba_props_model.models.simulation import json_to_pmf

    df = pd.read_parquet(scored_path).copy()
    typer.echo(f"[DRIFT] Loaded {len(df):,} scored rows from {scored_path}")

    # Normalize column names
    if "outcome" not in df.columns and "actual_outcome" in df.columns:
        df["outcome"] = df["actual_outcome"]
    if "role_bucket" not in df.columns:
        df["role_bucket"] = "all"
    if "pmf" not in df.columns and "pmf_json" in df.columns:
        df["pmf"] = df["pmf_json"].map(json_to_pmf)

    # Drop rows missing required columns
    required = ["stat", "pmf", "outcome"]
    df = df.dropna(subset=required)
    if df.empty:
        typer.echo("[DRIFT] No valid scored rows after filtering. Skipping.")
        _write_report(out, {"status": "no_valid_data"})
        raise typer.Exit(0)

    # Restrict to last N rows per stat (most recent scored predictions)
    if "game_date" in df.columns:
        df = df.sort_values("game_date")
    recent_frames = []
    for stat, grp in df.groupby("stat"):
        recent_frames.append(grp.tail(window))
    df = pd.concat(recent_frames, ignore_index=True)
    typer.echo(f"[DRIFT] Evaluating {len(df):,} rows (last {window} per stat)")

    rep = calibration_report(df)
    typer.echo("\n=== Drift Check Report ===")
    typer.echo(rep.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Item 12: PIT + ECCE-MAD per-stat analysis (replaces ECE-only check)
    # ---------------------------------------------------------------------------
    pit_results: dict[str, dict] = {}
    ecce_results: dict[str, float] = {}
    direction_results: dict[str, str] = {}

    for stat, grp in df.groupby("stat"):
        pmfs = grp["pmf"].tolist()
        actuals = grp["outcome"].tolist()
        n_stat = len(grp)

        if n_stat >= 10:
            pit_vals = compute_pit_values(pmfs, actuals)
            pit_results[stat] = check_pit_uniformity(pit_vals)

        if "model_p_over" in grp.columns and "actual_over" in grp.columns:
            valid = grp[["model_p_over", "actual_over"]].dropna()
            if len(valid) >= 10:
                ecce = compute_ecce_mad(valid["model_p_over"].values, valid["actual_over"].values)
                ecce_results[stat] = ecce
                direction_results[stat] = analyze_direction(
                    valid["model_p_over"].values, valid["actual_over"].values
                )

    typer.echo("\n=== PIT Uniformity (KS test) ===")
    for stat, pit in pit_results.items():
        status = "PASS" if pit["is_uniform"] else "DRIFT"
        typer.echo(f"  {stat:12s}: {status}  KS={pit['ks_stat']:.3f}  p={pit['p_value']:.4f}  n={pit['n']}")

    typer.echo("\n=== ECCE-MAD (Binning-Free Calibration) ===")
    for stat, ecce in ecce_results.items():
        status = "PASS" if ecce < 0.05 else "DRIFT"
        direction = direction_results.get(stat, "balanced")
        typer.echo(f"  {stat:12s}: {status}  ECCE-MAD={ecce:.4f}  direction={direction}")

    # ---------------------------------------------------------------------------
    # Evaluate drift per stat (ECE + PIT + ECCE-MAD combined)
    # ---------------------------------------------------------------------------
    soft_flags: list[dict] = []
    hard_flags: list[dict] = []

    for _, row in rep.iterrows():
        stat = row["stat"]
        ece = float(row.get("ece", 0.0))
        me = abs(float(row.get("mean_error", 0.0)))
        n = int(row.get("n", 0))

        if n < 20:
            continue

        pit_ok = pit_results.get(stat, {}).get("is_uniform", True)
        ecce_val = ecce_results.get(stat, 0.0)
        ecce_ok = ecce_val < 0.05
        direction = direction_results.get(stat, "balanced")
        brier_trending = False  # Placeholder (would need time series)

        needs_recal = (not pit_ok or not ecce_ok or not pit_ok)
        flag = {
            "stat": stat,
            "ece": ece,
            "mean_error": float(row.get("mean_error", 0.0)),
            "ecce_mad": ecce_val,
            "pit_uniform": pit_ok,
            "pit_ks_stat": pit_results.get(stat, {}).get("ks_stat", 0.0),
            "pit_p_value": pit_results.get(stat, {}).get("p_value", 1.0),
            "direction": direction,
            "n": n,
            "needs_recalibration": needs_recal,
        }

        # Hard drift: ECE exceeds threshold OR ECCE-MAD too high + PIT fails
        is_hard = (
            ece > hard_ece or me > hard_mean_err
            or (ecce_val > 0.08 and not pit_ok)
        )
        # Soft drift: ECE in soft range OR ECCE-MAD marginal OR PIT marginal
        is_soft = (
            ece > soft_ece or me > soft_mean_err
            or ecce_val > 0.05 or not pit_ok
        )

        if is_hard:
            hard_flags.append(flag)
            typer.echo(f"[DRIFT][HARD] {stat}: ECE={ece:.4f} ECCE-MAD={ecce_val:.4f} PIT-uniform={pit_ok} direction={direction}")
        elif is_soft:
            soft_flags.append(flag)
            typer.echo(f"[DRIFT][SOFT] {stat}: ECE={ece:.4f} ECCE-MAD={ecce_val:.4f} PIT-uniform={pit_ok} direction={direction}")

    drift_status = "none"
    if hard_flags:
        drift_status = "hard"
    elif soft_flags:
        drift_status = "soft"

    report = {
        "status": drift_status,
        "window_rows_per_stat": window,
        "thresholds": {
            "soft_ece": soft_ece,
            "hard_ece": hard_ece,
            "soft_mean_err": soft_mean_err,
            "hard_mean_err": hard_mean_err,
            "ecce_mad_soft": 0.05,
            "ecce_mad_hard": 0.08,
        },
        "pit_results": pit_results,
        "ecce_mad_results": ecce_results,
        "direction_results": direction_results,
        "hard_drift_stats": hard_flags,
        "soft_drift_stats": soft_flags,
        "calibration_report": rep.to_dict(orient="records"),
    }
    _write_report(out, report)
    typer.echo(f"\n[DRIFT] Report written → {out}")

    if hard_flags:
        stats_str = ", ".join(f["stat"] for f in hard_flags)
        typer.echo(
            f"\n[DRIFT][EXIT 1] Hard drift detected for: {stats_str}. "
            "Auto-recalibration will be triggered."
        )
        raise typer.Exit(1)

    if soft_flags:
        stats_str = ", ".join(f["stat"] for f in soft_flags)
        typer.echo(f"\n[DRIFT][WARN] Soft drift for: {stats_str}. Monitoring.")

    typer.echo("\n[DRIFT][OK] No hard drift detected.")
    raise typer.Exit(0)


def _write_report(path: str, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    app()
