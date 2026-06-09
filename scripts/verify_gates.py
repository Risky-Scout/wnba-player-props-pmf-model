from __future__ import annotations

import pandas as pd
import typer

from wnba_props_model.evaluation.diagnostics import calibration_report, market_superiority_report

app = typer.Typer(add_completion=False)


@app.command()
def calibration(oof_scored: str):
    df = pd.read_parquet(oof_scored)
    rep = calibration_report(df)
    print(rep.to_string(index=False))
    bad = rep[(rep["pit_ks"] > 0.075) | (rep["mean_error"].abs() > 0.15)]
    raise typer.Exit(1 if len(bad) else 0)


@app.command()
def market(loss_rows: str):
    df = pd.read_parquet(loss_rows)
    rep = market_superiority_report(df)
    print(rep.to_string(index=False))
    eligible = rep[rep["eligible"]]
    raise typer.Exit(1 if len(eligible) and not eligible["certified_pass"].all() else 0)


if __name__ == "__main__":
    app()
