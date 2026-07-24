"""Assemble the market-superiority evaluator input from the P1 historical archive.

Bridges three sources into the one table `evaluate_market_superiority.py` scores:

  * MARKET  — p1_closing_consensus.parquet (no-vig closing P(over) at the modal line);
  * OUTCOME — OOF table `actual_outcome` (the realized stat), keyed to the same game;
  * MODEL   — P(over) at the closing line from the OOF PMF, then fold-safe calibrated
              (walk-forward, no lookahead) so it is production-equivalent.

Output columns match the evaluator contract:
  game_date, prop, candidate, split, actual, line, model_prob_over_final,
  market_prob_over_no_vig

so that the moment the archive exists you get the per-prop log-loss / Brier / AUC
verdict vs. the market. With --run-eval it invokes the evaluator directly.

Usage:
    python3 scripts/build_market_superiority_input.py \\
        --closing artifacts/p1/p1_closing_consensus.parquet \\
        --oof data/oof/oof_player_stat_pmfs.parquet \\
        --out artifacts/p1/market_superiority_input.parquet \\
        --run-eval
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402

app = typer.Typer(add_completion=False)
KEYS = ["game_id", "player_id", "stat"]


def _pmf_array(pmf_json) -> np.ndarray:
    """Decode a stored PMF (json string / list / dict / value-prob pairs) to an array."""
    obj = json.loads(pmf_json) if isinstance(pmf_json, str) else pmf_json
    if obj is None:
        return np.array([])
    if isinstance(obj, dict):
        n = max(int(k) for k in obj.keys()) + 1
        arr = np.zeros(n)
        for k, v in obj.items():
            arr[int(k)] = float(v)
        return arr
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 2:
        n = int(arr[:, 0].max()) + 1
        out = np.zeros(n)
        for val, prob in arr:
            out[int(val)] = float(prob)
        return out
    return arr


def _chronological_split(dates: pd.Series, split_date: str | None, test_frac: float) -> pd.Series:
    """Return a 'selection'/'test' label per row. Earlier games = selection (used to
    pick a candidate), later games = untouched forward test (the proof period)."""
    d = pd.to_datetime(dates, errors="coerce")
    if split_date:
        cut = pd.Timestamp(split_date)
    else:
        uniq = np.sort(d.dropna().unique())
        if len(uniq) < 2:
            # everything to test; selection stays empty (prove-only)
            return pd.Series(["test"] * len(dates), index=dates.index)
        cut = pd.Timestamp(uniq[int(len(uniq) * (1.0 - test_frac))])
    return np.where(d < cut, "selection", "test")


@app.command()
def build(
    closing: str = typer.Option("artifacts/p1/p1_closing_consensus.parquet", "--closing",
                                help="P1 closing consensus (market no-vig P(over) + line)."),
    oof: str = typer.Option("data/oof/oof_player_stat_pmfs.parquet", "--oof",
                            help="OOF PMFs with actual_outcome + pmf_json (model + outcome)."),
    out: str = typer.Option("artifacts/p1/market_superiority_input.parquet", "--out"),
    split_date: str = typer.Option("", "--split-date",
                                   help="YYYY-MM-DD; games before = selection, on/after = test."),
    test_frac: float = typer.Option(0.4, "--test-frac",
                                    help="If no --split-date, fraction of latest dates used as test."),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate",
                                   help="Fold-safe (walk-forward) calibration of model P(over)."),
    candidate: str = typer.Option("production", "--candidate"),
    run_eval: bool = typer.Option(False, "--run-eval", help="Invoke the evaluator (prove mode)."),
    eval_output_dir: str = typer.Option("artifacts/market_feature_proof/from_archive", "--eval-output-dir"),
) -> None:
    closing_p, oof_p = Path(closing), Path(oof)
    if not closing_p.exists():
        typer.echo(f"[FATAL] closing consensus not found: {closing_p}. Run the P1 backfill first.", err=True)
        raise typer.Exit(1)
    if not oof_p.exists():
        typer.echo(f"[FATAL] OOF PMFs not found: {oof_p}. Regenerate (build_oof_pmfs.py) or fetch it.", err=True)
        raise typer.Exit(1)

    cc = pd.read_parquet(closing_p)
    oofd = pd.read_parquet(oof_p)
    for req, df, nm in [(["game_id", "player_id", "stat", "line", "market_prob_over_no_vig"], cc, "closing"),
                        (["game_id", "player_id", "stat", "actual_outcome", "pmf_json"], oofd, "oof")]:
        miss = [c for c in req if c not in df.columns]
        if miss:
            typer.echo(f"[FATAL] {nm} table missing columns: {miss}", err=True)
            raise typer.Exit(1)

    hm.assert_no_lookahead(oofd)  # refuse leaky OOF folds
    oofd = oofd.dropna(subset=["actual_outcome", "pmf_json"]).copy()
    if oofd.duplicated(subset=KEYS).any():
        typer.echo("[FATAL] duplicate OOF keys (game_id,player_id,stat).", err=True)
        raise typer.Exit(1)

    for k in KEYS:
        cc[k] = cc[k].astype("string")
        oofd[k] = oofd[k].astype("string")

    gd_col = "game_date" if "game_date" in oofd.columns else None
    keep = KEYS + ["actual_outcome", "pmf_json"] + ([gd_col] if gd_col else [])
    df = cc.merge(oofd[keep], on=KEYS, how="inner")
    if df.empty:
        typer.echo("[FATAL] no (game,player,stat) overlap between closing consensus and OOF. "
                   "Check that both cover the same games (stale/mismatched inputs?).", err=True)
        raise typer.Exit(1)

    if gd_col is None:
        df["game_date"] = pd.to_datetime(df["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Model P(over) at the exact closing line, then fold-safe calibrate (no lookahead).
    df["raw_p_over"] = [
        hm.p_over_conditional(_pmf_array(pj), float(ln))
        for pj, ln in zip(df["pmf_json"], df["line"])
    ]
    df = df[df["raw_p_over"].notna()].copy()
    df["over_outcome"] = (df["actual_outcome"].astype(float) > df["line"].astype(float)).astype(int)
    if calibrate:
        model_p_over = hm.fold_safe_calibrated_prob_over(
            df, prob_col="raw_p_over", outcome_col="over_outcome").to_numpy()
    else:
        model_p_over = df["raw_p_over"].to_numpy()
    split = np.asarray(_chronological_split(df["game_date"], split_date or None, test_frac))

    # Build the evaluator input as a NEW frame (never a post-lineage mutation of an
    # existing decision-grade column): the delivered probability's SOLE creator is the
    # production lineage, not this offline evaluation assembler.
    result = pd.DataFrame({
        "game_date": df["game_date"].to_numpy(),
        "prop": df["stat"].to_numpy(),
        "candidate": candidate,
        "split": split,
        "actual": df["actual_outcome"].astype(float).to_numpy(),
        "line": df["line"].astype(float).to_numpy(),
        "model_prob_over_final": model_p_over,
        "market_prob_over_no_vig": df["market_prob_over_no_vig"].astype(float).to_numpy(),
    })
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_p, index=False)

    by_split = result["split"].value_counts().to_dict()
    typer.echo(f"[market-input] wrote {len(result):,} rows -> {out_p}")
    typer.echo(f"  splits: {by_split} | props: {sorted(result['prop'].unique())}")
    typer.echo(f"  calibrated={calibrate} | date range {result['game_date'].min()} .. {result['game_date'].max()}")

    if run_eval:
        cmd = [sys.executable, str(Path(__file__).parent / "evaluate_market_superiority.py"),
               "--input", str(out_p), "--output-dir", eval_output_dir, "--mode", "prove"]
        typer.echo(f"[market-input] running verdict: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    else:
        typer.echo("Next: python3 scripts/evaluate_market_superiority.py "
                   f"--input {out_p} --mode prove")


if __name__ == "__main__":
    app()
