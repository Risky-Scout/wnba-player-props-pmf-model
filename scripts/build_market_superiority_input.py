"""Assemble the market-superiority evaluator input from the P1 historical archive.

Bridges three sources into the one table `evaluate_market_superiority.py` scores:

  * MARKET  — p1_closing_consensus.parquet (no-vig closing P(over) at the modal line);
  * OUTCOME — OOF table `actual_outcome` (the realized stat), keyed to the same game;
  * MODEL   — model_prob_over_final computed by the SAME function production delivery
              uses (build_probability_lineage): push-safe settled P(over) -> binary
              calibration registry (identity unless a policy file is present). This
              guarantees EVALUATED == DEPLOYED: the proof scores the exact shipped
              probability, so a PASS is directly promotable.

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
from wnba_props_model.models.binary_probability_calibration import (  # noqa: E402
    BinaryCalibrationRegistry,
)
from wnba_props_model.models.probability_lineage import build_probability_lineage  # noqa: E402
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

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
    calibration_policy: str = typer.Option(
        "config/binary_calibration_policy_v1.json", "--calibration-policy",
        help="Binary calibration policy file (the SAME one delivery loads). Absent -> "
             "identity, matching current production. Guarantees evaluated == deployed."),
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
    # Carry role_bucket + model_version so the lineage matches delivery's inputs exactly.
    optional = [c for c in ("role_bucket", "model_version") if c in oofd.columns]
    keep = KEYS + ["actual_outcome", "pmf_json"] + ([gd_col] if gd_col else []) + optional
    df = cc.merge(oofd[keep], on=KEYS, how="inner")
    if df.empty:
        typer.echo("[FATAL] no (game,player,stat) overlap between closing consensus and OOF. "
                   "Check that both cover the same games (stale/mismatched inputs?).", err=True)
        raise typer.Exit(1)

    if gd_col is None:
        df["game_date"] = pd.to_datetime(df["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")

    # MODEL P(over) — produced by the SAME function delivery uses (build_probability_lineage):
    # push-safe settled P(over) -> binary calibration registry (identity unless a policy file
    # is present) -> model_prob_over_final. Evaluated == deployed. The PMF is decoded with the
    # same json_to_pmf delivery uses. Binary-ineligible (all-push) rows have no defined P(over)
    # and are dropped (they cannot be scored and are never shipped).
    registry = BinaryCalibrationRegistry.from_policy(calibration_policy or None)
    roles = (df["role_bucket"].astype(str) if "role_bucket" in df.columns
             else pd.Series(["all"] * len(df), index=df.index))
    versions = (df["model_version"] if "model_version" in df.columns
                else pd.Series([None] * len(df), index=df.index))
    finals: list[float | None] = []
    for pj, ln, stat_, role_, ver_ in zip(df["pmf_json"], df["line"], df["stat"], roles, versions):
        lineage = build_probability_lineage(
            final_pmf=json_to_pmf(pj),
            line=float(ln),
            prop=str(stat_),
            role=str(role_) if role_ is not None else "all",
            binary_calibration_registry=registry,
            structural_model_id=(str(ver_) if ver_ not in (None, "None", "") else None),
            probability_track="pure_forecast",
        )
        finals.append(lineage.model_prob_over_final)
    df["_final"] = finals
    n_before = len(df)
    df = df[df["_final"].notna()].copy()
    n_ineligible = n_before - len(df)
    if df.empty:
        typer.echo("[FATAL] no binary-eligible rows after lineage (all rows all-push?).", err=True)
        raise typer.Exit(1)
    model_p_over = df["_final"].astype(float).to_numpy()
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
    typer.echo(f"  probability=delivery-lineage(parity) | calibration={registry.status} "
               f"| binary_ineligible_dropped={n_ineligible} "
               f"| date range {result['game_date'].min()} .. {result['game_date'].max()}")

    typer.echo("[market-input] NOTE: this closing-consensus archive is DEVELOPMENT_DIAGNOSTIC / "
               "NOT_EXACT_QUOTE_PROOF / NOT_UNTOUCHED / NOT_PROMOTION_ELIGIBLE. Promotion proof "
               "requires atomic decision-time same-book quotes and a frozen split manifest (W0.7/W12).")
    if run_eval:
        # Diagnostic only -> AUDIT mode (never prove; prove requires frozen split + candidate
        # manifests). Pass the delivered-probability column explicitly and PROPAGATE failure.
        cmd = [sys.executable, str(Path(__file__).parent / "evaluate_market_superiority.py"),
               "--input", str(out_p), "--output-dir", eval_output_dir, "--mode", "audit",
               "--model-prob-col", "model_prob_over_final"]
        typer.echo(f"[market-input] running DIAGNOSTIC audit: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        typer.echo("Next (diagnostic): python3 scripts/evaluate_market_superiority.py "
                   f"--input {out_p} --mode audit --model-prob-col model_prob_over_final")


if __name__ == "__main__":
    app()
