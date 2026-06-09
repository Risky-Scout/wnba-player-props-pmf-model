from __future__ import annotations

import typer

from wnba_props_model.pipeline.train import train_player_models

app = typer.Typer(add_completion=False)


@app.command()
def main(
    player_stats: str = typer.Option(..., help="Parquet file from BDL player_stats normalized table."),
    games: str | None = typer.Option(None, help="Parquet file from BDL games normalized table."),
    out_dir: str = typer.Option("artifacts/models/player_props"),
):
    paths = train_player_models(player_stats, games, out_dir)
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
