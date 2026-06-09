from __future__ import annotations

import typer

from wnba_props_model.pipeline.oof import build_walk_forward_oof_pmfs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    player_stats: str = typer.Option(...),
    games: str | None = typer.Option(None),
    out_path: str = typer.Option("data/processed/oof_pmfs.parquet"),
    min_training_days: int = typer.Option(365),
    window_days: int = typer.Option(28),
    draws: int = typer.Option(5000),
):
    path = build_walk_forward_oof_pmfs(
        player_stats_path=player_stats,
        games_path=games,
        out_path=out_path,
        min_training_days=min_training_days,
        window_days=window_days,
        draws=draws,
    )
    typer.echo(path)


if __name__ == "__main__":
    app()
