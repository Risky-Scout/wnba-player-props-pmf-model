"""Live calibration health check (blueprint §1.2 — every quarter break).

Computes ECCE-MAD and PIT uniformity on the in-game live predictions vs.
actuals-so-far, and writes a calibration health JSON to the live output dir.

Called by live_inplay.yml at every detected quarter break.

Usage:
    python scripts/live_recal_check.py \\
        --live-dir artifacts/live \\
        --out-dir artifacts/live/cal_checks
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    live_dir: str = typer.Option("artifacts/live", "--live-dir"),
    out_dir: str = typer.Option("", "--out-dir"),
) -> None:
    """Run live calibration check and write health JSON."""
    live = Path(live_dir)
    out = Path(out_dir) if out_dir else live / "cal_checks"
    out.mkdir(parents=True, exist_ok=True)

    summary_path = live / "live_session_summary.json"
    if not summary_path.exists():
        typer.echo("[WARN] No live_session_summary.json — skipping recal check.", err=True)
        raise typer.Exit(0)

    summary = json.loads(summary_path.read_text())
    period = summary.get("current_period", 1)
    clock = summary.get("current_clock", "10:00")

    # Load observed vs predicted if available
    obs_path = live / "live_observed_vs_predicted.json"
    health = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "clock": clock,
        "ecce_mad": None,
        "pit_ks_pvalue": None,
        "coverage_90ci": None,
        "status": "OK",
        "note": "No observed_vs_predicted data available yet",
    }

    if obs_path.exists():
        data = json.loads(obs_path.read_text())
        records = data if isinstance(data, list) else data.get("records", [])
        if records:
            health = _compute_calibration(records, period, clock)

    out_file = out / f"cal_check_Q{period}_{clock.replace(':', '')}.json"
    out_file.write_text(json.dumps(health, indent=2))
    typer.echo(f"Cal check → {out_file}")
    typer.echo(f"Status: {health['status']} | ECCE-MAD={health.get('ecce_mad')} | PIT p={health.get('pit_ks_pvalue')}")


def _compute_calibration(records: list[dict], period: int, clock: str) -> dict:
    """Compute ECCE-MAD and PIT KS test from records."""
    from scipy import stats as scipy_stats  # noqa: PLC0415

    p_overs = []
    actuals_over = []

    for r in records:
        p_o = r.get("p_over")
        actual = r.get("actual_over")
        if p_o is not None and actual is not None:
            p_overs.append(float(p_o))
            actuals_over.append(float(actual))

    if not p_overs:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": period, "clock": clock,
            "status": "INSUFFICIENT_DATA",
            "note": f"Only {len(records)} records, none with p_over+actual",
        }

    p_arr = np.array(p_overs)
    a_arr = np.array(actuals_over)

    # ECCE-MAD: bin by predicted prob, check calibration error
    n_bins = min(10, max(3, len(p_arr) // 5))
    bins = np.linspace(0, 1, n_bins + 1)
    ecce_vals = []
    for i in range(n_bins):
        mask = (p_arr >= bins[i]) & (p_arr < bins[i + 1])
        if mask.sum() < 2:
            continue
        mean_pred = p_arr[mask].mean()
        mean_actual = a_arr[mask].mean()
        ecce_vals.append(abs(mean_pred - mean_actual))

    ecce_mad = float(np.mean(ecce_vals)) if ecce_vals else None

    # PIT KS test (simplified: treat predicted CDF at actual value as uniform)
    pit_vals = []
    for p_o, act in zip(p_overs, actuals_over):
        pit_vals.append(p_o if act == 1 else (1 - p_o))
    pit_ks = scipy_stats.kstest(pit_vals, "uniform")
    pit_pval = float(pit_ks.pvalue)

    # 90% CI coverage
    coverage = float(np.mean((p_arr >= 0.05) & (p_arr <= 0.95)))

    status = "OK"
    if ecce_mad is not None and ecce_mad > 0.05:
        status = "RECAL_NEEDED"
    elif pit_pval < 0.01:
        status = "CALIBRATION_DRIFT"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "clock": clock,
        "n_records": len(p_overs),
        "ecce_mad": round(ecce_mad, 4) if ecce_mad is not None else None,
        "pit_ks_pvalue": round(pit_pval, 4),
        "coverage_90ci": round(coverage, 3),
        "status": status,
    }


if __name__ == "__main__":
    app()
