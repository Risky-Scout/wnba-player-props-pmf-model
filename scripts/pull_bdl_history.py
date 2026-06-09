from __future__ import annotations

import typer

from wnba_props_model.data.ingest import pull_season_history

app = typer.Typer(add_completion=False)


@app.command()
def main(
    start_season: int = typer.Option(...),
    end_season: int = typer.Option(...),
    out_dir: str = typer.Option("data/raw/bdl"),
):
    seasons = list(range(start_season, end_season + 1))
    paths = pull_season_history(seasons, out_dir=out_dir)
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
