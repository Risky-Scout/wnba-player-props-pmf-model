"""Daily WNBA PMF pipeline orchestrator.

Runs the full production pipeline in sequence:
  1. Pull BDL history (incremental, current season)
  2. Build canonical tables
  3. Build features
  4. Train baseline PMFs (full retrain with latest data)
  5. Apply calibrators
  6. Build edge report (Shin no-vig vs. model PMF)
  7. Export betting sheets (Kalshi / Polymarket)

Usage:
    python scripts/run_daily_pipeline.py \\
        --season 2026 \\
        --game-date 2026-06-15 \\
        --out-dir deliveries/today

    # Skip steps that already ran today:
    python scripts/run_daily_pipeline.py --season 2026 --skip-pull --skip-train
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

_ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], step: str) -> None:
    typer.echo(f"\n{'='*60}")
    typer.echo(f"  STEP: {step}")
    typer.echo(f"{'='*60}")
    result = subprocess.run([sys.executable, *cmd], check=False)
    if result.returncode != 0:
        typer.echo(f"[FAIL] Step '{step}' exited with code {result.returncode}", err=True)
        raise typer.Exit(result.returncode)
    typer.echo(f"[OK] {step}")


@app.command()
def main(
    season: int = typer.Option(..., help="Current WNBA season year (e.g. 2026)."),
    game_date: str | None = typer.Option(None, help="ISO date to predict for (default: today)."),
    out_dir: str = typer.Option("deliveries/today", help="Delivery output directory."),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline", help="Stage 4 model artifacts."),
    cal_dir: str = typer.Option("artifacts/models/calibration", help="Calibrator artifacts."),
    no_calibration: bool = typer.Option(False, "--no-calibration", help="Skip calibration step."),
    skip_pull: bool = typer.Option(False, "--skip-pull", help="Skip BDL data pull."),
    skip_canonical: bool = typer.Option(False, "--skip-canonical", help="Skip canonical table build."),
    skip_features: bool = typer.Option(False, "--skip-features", help="Skip feature build."),
    skip_train: bool = typer.Option(False, "--skip-train", help="Skip model retrain."),
    edge_threshold: float = typer.Option(0.04, help="Minimum |edge| to publish (4pp default)."),
) -> None:
    """Run the full daily WNBA PMF prediction and edge-delivery pipeline."""
    today = game_date or date.today().isoformat()
    typer.echo(f"\nWNBA Daily Pipeline — {today} (season {season})")
    typer.echo(f"Started at {datetime.now(timezone.utc).isoformat()}")

    # Step 1: Pull BDL history
    if not skip_pull:
        _run([
            "scripts/pull_bdl_history.py",
            "--start-season", str(season),
            "--end-season", str(season),
            "--out-dir", "data/raw/bdl",
        ], "Pull BDL history")

    # Step 2: Build canonical tables
    if not skip_canonical:
        _run([
            "scripts/build_canonical_tables.py",
            "--raw-dir", "data/raw/bdl",
            "--out-dir", "data/processed",
        ], "Build canonical tables")

    # Step 3: Build features
    if not skip_features:
        _run([
            "scripts/build_features.py",
            "--data-dir", "data/processed",
            "--audit-out", "artifacts/audits/feature_audit.json",
        ], "Build features")

    # Step 4: Train baseline PMFs
    if not skip_train:
        _run([
            "scripts/train_baseline_pmfs.py",
            "--features-wide", "data/processed/wnba_player_game_features_wide.parquet",
            "--features-long", "data/processed/wnba_player_game_features_long.parquet",
            "--manifest", "data/processed/feature_schema_manifest.json",
            "--config", "config/model/stage4_baseline.yaml",
            "--model-dir", model_dir,
            "--out-dir", "data/model_outputs/stage4_baseline",
            "--audit-out", "artifacts/audits/stage4_training_audit.json",
        ], "Train baseline PMFs (Stage 4)")

    # Step 5: Predict today with calibration
    cal_args = ["--no-calibration"] if no_calibration else ["--cal-dir", cal_dir]
    _run([
        "scripts/predict_today.py",
        "--features-wide", "data/processed/wnba_player_game_features_wide.parquet",
        "--model-dir", model_dir,
        "--config", "config/model/stage4_baseline.yaml",
        "--game-date", today,
        "--out-dir", out_dir,
        *cal_args,
    ], "Predict today (HGB + calibration)")

    # Step 6: Build edge report
    _run([
        "scripts/build_edge_report.py",
        "--pmfs", f"{out_dir}/full_pmfs_wide.parquet",
        "--raw-props", "data/processed/wnba_player_props.parquet",
        "--out-dir", out_dir,
        "--edge-threshold", str(edge_threshold),
        "--game-date", today,
    ], "Build edge report (Shin no-vig)")

    # Step 7: Export betting sheets
    _run([
        "scripts/export_betting_sheet.py",
        "--edges", f"{out_dir}/publishable_edges.parquet",
        "--out-dir", out_dir,
        "--game-date", today,
    ], "Export betting sheets (Kalshi / Polymarket)")

    # Step 8: Update calibration monitor (Enhancement 21 — PIT drift detection)
    try:
        import json as _json
        from pathlib import Path as _Path
        _cal_mon_dir = _Path("artifacts/calibration_monitor")
        _cal_mon_dir.mkdir(parents=True, exist_ok=True)
        # Load or create the multi-stat monitor
        try:
            from wnba_props_model.models.calibration_monitor import MultiStatCalibrationMonitor
            _mon = MultiStatCalibrationMonitor.load(_cal_mon_dir) if any(_cal_mon_dir.glob("*.json")) \
                else MultiStatCalibrationMonitor()
            _summary = _mon.summary()
            _mon.save(_cal_mon_dir)
            with open(_cal_mon_dir / "summary.json", "w") as _f:
                _json.dump(_summary, _f, indent=2)
            if _summary.get("n_alerts", 0) > 0:
                typer.echo(f"[WARN] Calibration monitor: {_summary['n_alerts']} stat(s) have drift alerts!")
                for _s, _v in _summary["stats"].items():
                    if _v.get("status") == "alert":
                        typer.echo(f"  → {_s}: {_v.get('direction')} (p={_v.get('p_value')})")
            else:
                typer.echo(f"[OK] Calibration monitor: score={_summary.get('overall_score')} — no drift.")
        except ImportError:
            pass
    except Exception as _cal_exc:
        typer.echo(f"[WARN] Calibration monitor step failed: {_cal_exc}")

    typer.echo(f"\n[DONE] Daily pipeline complete — {today}")
    typer.echo(f"Outputs in: {out_dir}/")


if __name__ == "__main__":
    app()
