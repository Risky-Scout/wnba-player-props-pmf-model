from __future__ import annotations

from pathlib import Path

import typer
import pandas as pd

from wnba_props_model.features.build_features import build_game_total_training_table
from wnba_props_model.models.game_totals import GAME_TOTAL_FEATURES, GameTotalsModel

app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: str = typer.Option(..., help="Normalized games parquet"),
    out_dir: str = typer.Option("artifacts/models/game_totals"),
):
    g = pd.read_parquet(games)
    train = build_game_total_training_table(g)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "training_table.parquet", index=False)
    model = GameTotalsModel(features=[f for f in GAME_TOTAL_FEATURES if f in train.columns]).fit(train.dropna(subset=["game_total"]))
    model.save(str(out / "game_totals_model.pkl"))
    typer.echo(out / "game_totals_model.pkl")


if __name__ == "__main__":
    app()
