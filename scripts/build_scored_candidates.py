"""Build the scored-rows table consumed by evaluate_market_superiority.py.

Given a candidate's OOF PMFs and executable closing quotes, emit one row per settled
(player, prop, line) with the model's push-safe FINAL P(over) and the no-vig market P(over),
tagged with a candidate label and a chronological selection/test split.

PR 1A source-of-truth: the model probability is produced ONLY by the sole creator
build_probability_lineage (the same function delivery uses) - never reconstructed from
pmf_json here. Output uses model_prob_over_final; a binary-ineligible (all-push) row is
dropped rather than assigned a fabricated value.

NON-PROMOTABLE until PR 1B: quote identity, book (non-)averaging, quote selection,
closing-line matching, and settlement policy are NOT yet corrected here (they remain PR 1B
work); evidence built from this table is not promotion evidence.

Output columns: game_date, game_id, player_id, prop, candidate, split, actual, line,
                model_prob_over_final, model_prob_push, market_prob_over_no_vig
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402
from wnba_props_model.models.probability_lineage import build_probability_lineage  # noqa: E402
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)


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
    # closing no-vig market prob per (game, player, stat, line).
    # NOTE (PR 1B): book averaging / quote identity is unchanged here and remains PR 1B work.
    q = q[q["market_prob_over_no_vig"].notna()] if "market_prob_over_no_vig" in q.columns else q
    grp = (q.groupby(["game_id", "player_id", "stat", "line"], as_index=False)
           .agg(market_prob_over_no_vig=("market_prob_over_no_vig", "mean")))

    key = ["game_id", "player_id", "stat"]
    _om_cols = key + ["game_date", "pmf_json", "actual_outcome"]
    if "role_bucket" in oof_df.columns:
        _om_cols.append("role_bucket")
    om = oof_df[_om_cols].drop_duplicates(key)
    df = grp.merge(om, on=key, how="inner")

    # SINGLE SOURCE: the delivered final probability, produced by build_probability_lineage
    # (identity binary calibration in 1A), at the exact quote line. No pmf reconstruction.
    finals: list[float] = []
    pushes: list[float] = []
    for _, r in df.iterrows():
        lin = build_probability_lineage(
            final_pmf=json_to_pmf(r["pmf_json"]), line=float(r["line"]),
            prop=str(r["stat"]), role=str(r.get("role_bucket", "all")),
            probability_track="pure_forecast")
        finals.append(np.nan if lin.model_prob_over_final is None else float(lin.model_prob_over_final))
        pushes.append(float(lin.model_prob_push))
    df[FINAL_PROBABILITY_COLUMN] = finals
    df["model_prob_push"] = pushes
    # Drop binary-ineligible (all-push) and degenerate rows; never fabricate a value.
    df = df[df[FINAL_PROBABILITY_COLUMN].notna()
            & (df[FINAL_PROBABILITY_COLUMN] > 1e-9) & (df[FINAL_PROBABILITY_COLUMN] < 1 - 1e-9)
            & df["market_prob_over_no_vig"].notna()].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["prop"] = df["stat"]
    df["candidate"] = candidate
    df["actual"] = df["actual_outcome"].astype(float)

    dates = np.sort(df["game_date"].unique())
    cut = dates[int(len(dates) * selection_frac)] if len(dates) > 2 else dates[0]
    df["split"] = np.where(df["game_date"] < cut, "selection", "test")

    cols = ["game_date", "game_id", "player_id", "prop", "candidate", "split",
            "actual", "line", FINAL_PROBABILITY_COLUMN, "model_prob_push",
            "market_prob_over_no_vig"]
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(out_p, index=False)
    typer.echo(f"[scored] {len(df)} rows -> {out_p} (NON-PROMOTABLE until PR 1B quote/settlement fix)")
    typer.echo(f"[scored] props: {dict(df['prop'].value_counts())}")
    typer.echo(f"[scored] split cutoff {cut}: "
               f"selection={int((df['split'] == 'selection').sum())} test={int((df['split'] == 'test').sum())}")


if __name__ == "__main__":
    app()
