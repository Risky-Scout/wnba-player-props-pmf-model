"""Build the scored-rows table consumed by evaluate_market_superiority.py.

Given a candidate's OOF PMFs and executable closing quotes, emit one row per settled
(player, prop, line) with the model's P(over) and the no-vig market P(over), tagged with a
candidate label and a chronological selection/test split. This turns real model output +
real book prices into the evaluator's expected input, so the market-superiority proof runs
on genuine data (not the synthetic self-test).

Output columns: game_date, game_id, player_id, prop, candidate, split, actual, line,
                model_prob_over, market_prob_over_no_vig
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation.forecasting import pmf_to_array  # noqa: E402

app = typer.Typer(add_completion=False)


def _p_over(pmf_json: str, line: float) -> float:
    a = pmf_to_array(pmf_json)
    if a is None or not len(a):
        return float("nan")
    k = int(np.ceil(line))               # .5 lines: over means actual >= ceil(line)
    return float(a[k:].sum()) if k < len(a) else 0.0


@app.command()
def main(oof: str = typer.Option(..., "--oof", help="Calibrated OOF PMFs parquet."),
         quotes: str = typer.Option(..., "--quotes", help="Executable closing quotes parquet."),
         out: str = typer.Option("artifacts/market_feature_proof/scored_candidates.parquet", "--out"),
         candidate: str = typer.Option("G0_current", "--candidate"),
         selection_frac: float = typer.Option(0.6, "--selection-frac",
                                              help="Earliest fraction of dates = selection; rest = test.")) -> None:
    oof_df = pd.read_parquet(oof)
    oof_df = oof_df[oof_df["actual_outcome"].notna() & oof_df["pmf_json"].notna()].copy()
    for c in ("game_id", "player_id"):
        oof_df[c] = oof_df[c].astype(str)
    q = pd.read_parquet(quotes)
    for c in ("game_id", "player_id"):
        q[c] = q[c].astype(str)
    # closing no-vig market prob per (game, player, stat, line)
    q = q[q["market_prob_over_no_vig"].notna()] if "market_prob_over_no_vig" in q.columns else q
    grp = (q.groupby(["game_id", "player_id", "stat", "line"], as_index=False)
           .agg(market_prob_over_no_vig=("market_prob_over_no_vig", "mean")))

    key = ["game_id", "player_id", "stat"]
    om = oof_df[key + ["game_date", "pmf_json", "actual_outcome"]].drop_duplicates(key)
    df = grp.merge(om, on=key, how="inner")
    df["model_prob_over"] = [_p_over(pj, ln) for pj, ln in zip(df["pmf_json"], df["line"])]
    df = df[(df["model_prob_over"] > 1e-9) & (df["model_prob_over"] < 1 - 1e-9)
            & df["market_prob_over_no_vig"].notna()].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["prop"] = df["stat"]
    df["candidate"] = candidate
    df["actual"] = df["actual_outcome"].astype(float)

    dates = np.sort(df["game_date"].unique())
    cut = dates[int(len(dates) * selection_frac)] if len(dates) > 2 else dates[0]
    df["split"] = np.where(df["game_date"] < cut, "selection", "test")

    cols = ["game_date", "game_id", "player_id", "prop", "candidate", "split",
            "actual", "line", "model_prob_over", "market_prob_over_no_vig"]
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(out_p, index=False)
    typer.echo(f"[scored] {len(df)} rows -> {out_p}")
    typer.echo(f"[scored] props: {dict(df['prop'].value_counts())}")
    typer.echo(f"[scored] split cutoff {cut}: "
               f"selection={int((df['split'] == 'selection').sum())} test={int((df['split'] == 'test').sum())}")


if __name__ == "__main__":
    app()
