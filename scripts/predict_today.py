from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.pipeline.predict import build_features_for_prediction, predict_player_pmfs
from wnba_props_model.pipeline.deliver import write_delivery

app = typer.Typer(add_completion=False)


@app.command()
def main(
    recent_player_stats: str = typer.Option(...),
    games: str | None = typer.Option(None),
    raw_props: str | None = typer.Option(None),
    model_dir: str = typer.Option("artifacts/models/player_props"),
    out_dir: str = typer.Option("deliveries/today/wizard_of_odds"),
    draws: int = typer.Option(50000),
):
    stats = pd.read_parquet(recent_player_stats)
    games_df = pd.read_parquet(games) if games else None
    features = build_features_for_prediction(stats, games_df)
    # In real production, pass only projected active rows for today's slate.
    pmfs = predict_player_pmfs(features, model_dir=model_dir, draws=draws)
    props = pd.read_parquet(raw_props) if raw_props else None
    paths = write_delivery(pmfs, out_dir, props)
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
