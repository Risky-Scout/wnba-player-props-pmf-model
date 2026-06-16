"""Stage 6 — Apply per-stat role-aware calibrators to a PMF parquet.

Usage:
    python scripts/apply_calibrators.py \\
        --pmfs data/model_outputs/stage4_baseline/player_stat_pmfs.parquet \\
        --cal-dir artifacts/models/calibration \\
        --out data/model_outputs/stage6_calibrated/player_stat_pmfs.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.pipeline.calibrate import apply_calibrators

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pmfs: str = typer.Option(..., help="Path to uncalibrated PMF parquet."),
    cal_dir: str = typer.Option("artifacts/models/calibration", help="Directory with fitted calibrator .pkl files."),
    out: str = typer.Option(..., help="Output path for calibrated PMF parquet."),
) -> None:
    """Apply isotonic calibrators to PMFs and write calibrated output."""
    pmfs_df = pd.read_parquet(pmfs)
    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows from {pmfs}")

    calibrated = apply_calibrators(pmfs_df, cal_dir=cal_dir)

    n_cal = calibrated["is_calibrated"].sum()
    n_total = len(calibrated)
    typer.echo(f"Calibrated {n_cal:,}/{n_total:,} rows ({100 * n_cal / max(n_total, 1):.1f}%)")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(out_path, index=False)
    typer.echo(f"Wrote calibrated PMFs → {out_path}")

    uncal_stats = calibrated[~calibrated["is_calibrated"]]["stat"].unique().tolist()
    if uncal_stats:
        typer.echo(f"[WARN] No calibrator found for stats: {uncal_stats}")


if __name__ == "__main__":
    app()
