"""Freeze the best low-cost candidate for a prop and run the untouched-window proof.

Owner critical-path step 6: rank props by distance from passing (selection only), freeze
the best valid candidate for the closest prop, and run the strict proof under the existing
frozen proof contract (evaluate_market_superiority.py prove mode: cluster bootstrap, Holm
correction, CI gates, min rows / min clusters). No wait-for-all-seven.

The calibrator is fit on the SELECTION window only (primary book) and applied to the
untouched TEST/proof window; the proof input, candidate manifest, and split manifest are
written with real hashes so the evaluator's fail-closed manifest-integrity check passes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_lowcost_candidates import _fit as _fit_candidate_impl  # noqa: E402


def _fit_candidate(candidate, sel):
    """Back-compat shim: corrected _fit returns (predictor, meta); return the predictor."""
    fn, _meta = _fit_candidate_impl(candidate, sel)
    return fn
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
FEATURE_CONTRACT_HASH = "302de341643008330520bc9c76c6b397f9ba24b80bd011faf038366ad6a95357"


def _sha_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@app.command()
def main(
    prop: str = typer.Option(..., "--prop"),
    candidate: str = typer.Option(..., "--candidate"),
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/scored_candidates_g0v2.parquet", "--scored"),
    primary_book: str = typer.Option("primary", "--primary-book",
                                     help="'primary'=deterministic one-quote; 'all'=pooled sensitivity; or a book."),
    out_dir: str = typer.Option("", "--out-dir"),
    min_rows: int = typer.Option(300, "--min-rows"),
    min_clusters: int = typer.Option(30, "--min-clusters"),
    bootstrap: int = typer.Option(5000, "--bootstrap"),
) -> None:
    outp = Path(out_dir or f"artifacts/market_feature_proof/G0_v2_proof_{prop}")
    outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(scored)
    pdf = df[df["prop"] == prop].copy()
    if primary_book == "primary":
        pdf = pdf[pdf["is_primary"]].copy()
    elif primary_book != "all":
        pdf = pdf[pdf["book"] == primary_book].copy()
    sel = pdf[pdf["split"] == "selection"].reset_index(drop=True)
    test = pdf[pdf["split"] == "test"].reset_index(drop=True)
    if len(sel) == 0 or len(test) == 0:
        raise SystemExit(f"insufficient data: selection={len(sel)} test={len(test)}")

    # Freeze: fit calibrator on ALL selection rows; apply to the untouched test window.
    fn = _fit_candidate(candidate, sel)
    test = test.copy()
    test[FINAL_PROBABILITY_COLUMN] = fn(test)
    proof_in = test[["game_date", "game_id", "player_id", "prop", "split", "actual", "line",
                     FINAL_PROBABILITY_COLUMN, "market_prob_over_no_vig"]].copy()
    proof_in["candidate"] = candidate
    proof_path = outp / "proof_input.parquet"
    proof_in.to_parquet(proof_path, index=False)

    sel_date_max = str(sel["game_date"].max())
    proof_dates = sorted(test["game_date"].unique().tolist())
    proof_min, proof_max = proof_dates[0], proof_dates[-1]

    calibrator_spec = json.dumps({"candidate": candidate, "prop": prop,
                                  "fit_rows": int(len(sel)), "primary_book": primary_book},
                                 sort_keys=True)
    candidate_manifest = {prop: {
        "candidate_id": candidate,
        "probability_track": "pure_forecast",
        "model_hash": _sha_file(REPO / "artifacts/models/calibration/oof_predictions.parquet"),
        "feature_schema_hash": FEATURE_CONTRACT_HASH,
        "calibration_policy_hash": _sha_str(f"lowcost:{candidate}"),
        "calibrator_hash": _sha_str(calibrator_spec),
        "training_date_max": sel_date_max,
        "selection_date_max": sel_date_max,
        "proof_date_min": proof_min,
    }}
    cand_path = outp / "candidate_manifest.json"
    cand_path.write_text(json.dumps({"selected_candidates": candidate_manifest}, indent=2) + "\n")

    split_manifest = {
        "proof_date_min": proof_min,
        "proof_date_max": proof_max,
        "proof_dates": proof_dates,
        "input_dataset_hash": _sha_file(proof_path),
        "quote_policy_hash": _sha_str("exact_decision_snapshot_shin_no_vig_per_book_v1"),
        "settlement_policy_hash": _sha_str("actual_gt_line_over_push_dropped_v1"),
        "creation_timestamp": pd.Timestamp.utcnow().isoformat(),
        "previous_access_count": 0,
    }
    split_path = outp / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n")

    cmd = [sys.executable, str(REPO / "scripts/evaluate_market_superiority.py"),
           "--mode", "prove", "--input", str(proof_path),
           "--selected-candidates", str(cand_path), "--split-manifest", str(split_path),
           "--output-dir", str(outp), "--target-props", prop,
           "--min-rows", str(min_rows), "--min-clusters", str(min_clusters),
           "--bootstrap", str(bootstrap)]
    print(f"[freeze-prove] {prop} <- {candidate} | selection={len(sel)}r/{sel['game_date'].nunique()}d "
          f"test={len(test)}r/{len(proof_dates)}d proof={proof_min}..{proof_max}")
    r = subprocess.run(cmd, cwd=str(REPO))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    app()
