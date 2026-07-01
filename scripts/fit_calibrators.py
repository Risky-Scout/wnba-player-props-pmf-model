from __future__ import annotations

import typer

from wnba_props_model.pipeline.calibrate import fit_calibrators as fit

app = typer.Typer(add_completion=False)


@app.command()
def main(
    oof_pmfs: str = typer.Option(...),
    out_dir: str = typer.Option("artifacts/models/calibration"),
    props_parquet: str = typer.Option("", help="Optional path to historical player props parquet for beta calibrator line joining."),
):
    paths = fit(oof_pmfs, out_dir, props_parquet_path=props_parquet if props_parquet else None)
    for stat, path in paths.items():
        typer.echo(f"{stat}: {path}")


if __name__ == "__main__":
    app()
