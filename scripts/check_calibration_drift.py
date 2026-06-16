"""Daily calibration drift detector.

Loads the last N scored predictions (where actual outcomes are known), computes
rolling ECE and mean_error per stat, flags soft/hard drift, and writes a JSON
report.

Exit codes:
  0  — no drift or soft drift only (warning logged)
  1  — hard drift detected (ECE > hard threshold for any stat); triggers
       auto-recalibration step in daily_pipeline.yml

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

from wnba_props_model.evaluation.diagnostics import calibration_report

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

    # Evaluate drift per stat
    soft_flags: list[dict] = []
    hard_flags: list[dict] = []

    for _, row in rep.iterrows():
        stat = row["stat"]
        ece = row.get("ece", 0.0)
        me = abs(row.get("mean_error", 0.0))
        n = int(row.get("n", 0))

        if n < 20:
            continue  # not enough data to flag

        flag = {"stat": stat, "ece": float(ece), "mean_error": float(row.get("mean_error", 0.0)), "n": n}

        if ece > hard_ece or me > hard_mean_err:
            hard_flags.append(flag)
            typer.echo(f"[DRIFT][HARD] {stat}: ECE={ece:.4f} mean_error={flag['mean_error']:.3f} n={n}")
        elif ece > soft_ece or me > soft_mean_err:
            soft_flags.append(flag)
            typer.echo(f"[DRIFT][SOFT] {stat}: ECE={ece:.4f} mean_error={flag['mean_error']:.3f} n={n}")

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
        },
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
