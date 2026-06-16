"""Generate calibrated PMF predictions for today's WNBA slate.

Uses the Stage 4 HGB engine + Stage 6 isotonic calibrators (if available).

Usage:
    python scripts/predict_today.py \\
        --features-wide data/processed/wnba_player_game_features_wide.parquet \\
        --model-dir artifacts/models/stage4_baseline \\
        --cal-dir artifacts/models/calibration \\
        --raw-props data/processed/player_props.parquet \\
        --out-dir deliveries/today
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.pipeline.deliver import write_delivery
from wnba_props_model.pipeline.predict import predict_player_pmfs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    features_wide: str = typer.Option(..., help="Wide feature parquet from build_features.py."),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline", help="Stage 4 HGB artifact directory."),
    config: str = typer.Option("config/model/stage4_baseline.yaml", help="Stage 4 YAML config."),
    cal_dir: str | None = typer.Option("artifacts/models/calibration", help="Calibrator directory; None to skip."),
    no_calibration: bool = typer.Option(False, "--no-calibration", help="Skip calibration application."),
    raw_props: str | None = typer.Option(None, help="BDL player props parquet for edge calculation."),
    out_dir: str = typer.Option("deliveries/today", help="Delivery output directory."),
    game_date: str | None = typer.Option(None, help="ISO date filter (YYYY-MM-DD); predicts only this date."),
) -> None:
    """Predict today's WNBA player stat PMFs and compute market edges."""
    features_df = pd.read_parquet(features_wide)

    if game_date:
        if "game_date" in features_df.columns:
            filtered = features_df[features_df["game_date"].astype(str) == game_date].copy()
            typer.echo(f"Filtered to game_date={game_date}: {len(filtered):,} rows")
            if not filtered.empty:
                features_df = filtered
            else:
                typer.echo(
                    f"[WARN] 0 rows for game_date={game_date}. "
                    "Using all rows from input (slate forward-dated features)."
                )
                # Slate files have game_date already set to the target date so no filter needed

    if features_df.empty:
        typer.echo(f"[WARN] No player rows to predict — no games on {game_date}. Exiting.")
        raise typer.Exit(0)

    typer.echo(f"Generating PMFs for {len(features_df):,} player-game rows...")

    apply_cal = not no_calibration
    effective_cal_dir = cal_dir if apply_cal else None

    pmfs = predict_player_pmfs(
        feature_df=features_df,
        model_dir=model_dir,
        config_path=config,
        cal_dir=effective_cal_dir,
        apply_calibration=apply_cal,
    )
    typer.echo(f"Generated {len(pmfs):,} PMF rows (stats × players × games)")
    n_cal = pmfs["is_calibrated"].sum() if "is_calibrated" in pmfs.columns else 0
    typer.echo(f"Calibrated: {n_cal:,}/{len(pmfs):,} rows")

    props_df = pd.read_parquet(raw_props) if raw_props else None
    paths = write_delivery(pmfs, out_dir, props_df)
    for k, v in paths.items():
        typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
