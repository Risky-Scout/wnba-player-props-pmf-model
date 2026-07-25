"""Phase 2/9: AST first-edge reassessment under CORRECTED nested rolling-origin CV.

Uses the ONE canonical deterministic scored-row artifact and the leakage-safe expanding-window
nested CV (see build_lowcost_candidates). Reports, for each AST candidate, BOTH eligibility
gates explicitly:

    proper_score_selection_eligible : aggregate outer-fold logloss<market AND brier<market,
                                      monotone-deployable, acceptable ECE, no catastrophic fold
    strict_auc_selection_eligible   : additionally aggregate outer-fold AUC > market AUC

Candidate A-map: A0 market identity | A1 = C1_platt | A2 = C4_blend | A3 = C5_role_blend |
A4 = regularized feature residual (BLOCKED_NO_FEATURE_MATRIX). Freezes exactly one AST
candidate ONLY if it is proper_score_selection_eligible, and records the strict-AUC gate
outcome transparently (a proper-score-track freeze is NOT described as strict-gate-ready).
If none qualifies, no freeze is created and the exact deficit is reported.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lowcost_candidates import _fit, _nested_outer_eval  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"
FEATURE_CONTRACT_HASH = "302de341643008330520bc9c76c6b397f9ba24b80bd011faf038366ad6a95357"
A_MAP = {"A1": "C1_platt", "A2": "C4_blend", "A3": "C5_role_blend"}
ECE_MARGIN = 0.03


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _gates(r: dict) -> dict:
    proper = bool(r["cand_logloss"] < r["market_logloss"] and r["cand_brier"] < r["market_brier"]
                  and r["cand_ece"] <= r["market_ece"] + ECE_MARGIN
                  and r["monotone_deployable"] and r["worst_fold_logloss_delta"] < 0.05)
    strict_auc = bool(proper and (r["cand_auc"] - r["market_auc"]) > 0)
    return {"proper_score_selection_eligible": proper, "strict_auc_selection_eligible": strict_auc}


@app.command()
def main(
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet", "--scored"),
    quote_policy: str = typer.Option("config/book_quote_priority_v1.json", "--quote-policy"),
    out_dir: str = typer.Option("", "--out-dir"),
    prop: str = typer.Option("ast", "--prop", help="Prop to reassess/freeze (ast or reb)."),
) -> None:
    prop = prop.lower()
    P = prop.upper()
    out_dir = out_dir or f"artifacts/market_feature_proof/{P}_sprint"
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(scored)
    ast = df[df["prop"] == prop].sort_values("game_date").reset_index(drop=True)

    rows = []
    for a_id, c_id in A_MAP.items():
        r = _nested_outer_eval(ast, c_id, ECE_MARGIN)
        if r is None or r.get("skipped"):
            rows.append({"a_id": a_id, "candidate": c_id, "status": "NO_FOLDS_OR_SKIPPED"}); continue
        g = _gates(r)
        rows.append({"a_id": a_id, "candidate": c_id, "status": "EVALUATED",
                     "logloss_delta": r["logloss_delta_vs_market"], "brier_delta": r["brier_delta_vs_market"],
                     "cand_auc": r["cand_auc"], "market_auc": r["market_auc"],
                     "auc_delta": r["cand_auc"] - r["market_auc"], "cand_ece": r["cand_ece"],
                     "cal_slope": r["cal_slope"], "monotone_deployable": r["monotone_deployable"],
                     "worst_fold_logloss_delta": r["worst_fold_logloss_delta"], **g})
    # A0 market identity baseline (reference) + A4 blocked
    rows.append({"a_id": "A0", "candidate": "market_identity", "status": "BASELINE"})
    rows.append({"a_id": "A4", "candidate": "regularized_feature_residual",
                 "status": "BLOCKED_NO_FEATURE_MATRIX"})
    res = pd.DataFrame(rows)
    res.to_csv(outp / f"{P}_A0_A4_METRICS.csv", index=False)

    ev = [r for r in rows if r.get("status") == "EVALUATED"]
    proper = [r for r in ev if r["proper_score_selection_eligible"]]
    (outp / f"{P}_A0_A4_METRICS.json").write_text(json.dumps(
        {"cv": "corrected nested expanding-window rolling-origin (leakage-safe)",
         "scored_artifact": "PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet",
         "gates": {"proper_score": "logloss<market AND brier<market AND monotone AND ece ok AND no catastrophic fold",
                   "strict_auc": "proper_score AND cand_auc>market_auc"},
         "records": res.replace({np.nan: None}).to_dict("records")}, indent=2) + "\n")

    print(f"[{P} reassessment, corrected nested CV]")
    for r in ev:
        print(f"  {r['a_id']}={r['candidate']:14s} dLL={r['logloss_delta']:+.5f} "
              f"dBrier={r['brier_delta']:+.5f} auc_delta={r['auc_delta']:+.4f} "
              f"proper={r['proper_score_selection_eligible']} strict_auc={r['strict_auc_selection_eligible']}")

    if not proper:
        (outp / f"{P}_FREEZE_DECISION.json").write_text(json.dumps(
            {"frozen": False, "reason": f"no {prop} candidate is proper_score_selection_eligible under "
             "corrected nested rolling-origin CV", "records": res.replace({np.nan: None}).to_dict("records")},
            indent=2) + "\n")
        print(f"\n[{P} freeze] NO candidate qualifies (proper-score) — none frozen.")
        return

    best = sorted(proper, key=lambda r: r["logloss_delta"])[0]
    c_id = best["candidate"]
    # Deployment calibrator refit on ALL development dates (selection perf stays nested outer-fold).
    fit_fn, meta = _fit(c_id, ast) if c_id != "C4_blend" else (None, {})
    frozen_params = {"candidate_id": c_id, "a_id": best["a_id"]}
    if c_id == "C4_blend":
        alphas = np.linspace(0, 1, 41)
        y = ast["outcome_over"].to_numpy(int)
        pm = ast[FINAL_PROBABILITY_COLUMN].to_numpy(float); pk = ast["market_prob_over_no_vig"].to_numpy(float)
        from build_lowcost_candidates import _ll
        alpha = float(min(alphas, key=lambda a: _ll(y, a * pm + (1 - a) * pk)))
        frozen_params.update({"form": "convex_blend a*model+(1-a)*market", "alpha_model_weight": alpha})
    elif c_id == "C1_platt":
        frozen_params.update({"form": "platt", **{k: meta[k] for k in meta if k != "monotone"}})
    else:
        frozen_params.update({"form": c_id})

    now = pd.Timestamp.now(tz="UTC")
    policy_sha = _sha_file(Path(quote_policy))
    canon = Path(scored)
    supersedes = ("AST_FIRST_EDGE_FREEZE.json (v1, INVALIDATED_TEMPORAL_CV_LEAKAGE)"
                  if prop == "ast" else "none (first freeze for this prop)")
    freeze = {
        "version": f"{prop}-first-edge-freeze-v2",
        "supersedes": supersedes,
        "prop": prop, "candidate_id": c_id, "a_id": best["a_id"],
        "track": "proper_score" if not best["strict_auc_selection_eligible"] else "strict",
        "proper_score_selection_eligible": best["proper_score_selection_eligible"],
        "strict_auc_selection_eligible": best["strict_auc_selection_eligible"],
        "frozen_calibrator_params": frozen_params,
        "cv": "corrected nested expanding-window rolling-origin (leakage-safe)",
        "selection_metrics_outer_fold": {k: best[k] for k in (
            "logloss_delta", "brier_delta", "auc_delta", "cand_ece", "cal_slope",
            "worst_fold_logloss_delta")},
        # Required freeze hashes (raw file SHA-256 for policy; canonical + manifest + code).
        "quote_policy_file_sha256": policy_sha,
        "settlement_policy_hash": hashlib.sha256(b"actual_gt_line_over_push_dropped_v1").hexdigest(),
        "model_hash": _sha_file(REPO / "artifacts/models/calibration/oof_predictions.parquet"),
        "feature_hash": FEATURE_CONTRACT_HASH,
        "calibrator_hash": hashlib.sha256(json.dumps(frozen_params, sort_keys=True).encode()).hexdigest(),
        "candidate_code_sha256": _sha_file(REPO / "scripts/build_lowcost_candidates.py"),
        "canonical_scored_row_sha256": _sha_file(canon),
        "nested_fold_manifest_sha256": _sha_file(G0 / "ROLLING_ORIGIN_FOLD_MANIFEST.json"),
        "training_date_max": str(ast["game_date"].max()),
        "development_rows": int(len(ast)),
        "freeze_timestamp_utc": now.isoformat(),
        "prospective_proof_start_utc": now.isoformat(),
        "prospective_proof_rule": ("NEW game dates strictly AFTER freeze; deterministic one-quote; "
                                   ">=5000 cluster bootstrap; Holm; upper-95%-CI(model-market)<0 for "
                                   "BOTH log loss and Brier. AUC gate applied separately."),
        "caveat": ("proper-score-track freeze: beats exact market on log loss & Brier under corrected "
                   "nested CV but does NOT beat market AUC (strict-AUC gate not met). Not strict-gate-"
                   "ready. Certification depends solely on the prospective proof; this freeze does NOT certify."),
    }
    fp = REPO / f"artifacts/candidate_freeze/{P}_FIRST_EDGE_FREEZE_V2.json"
    fp.write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"\n[{P} freeze v2] candidate={c_id} track={freeze['track']} "
          f"proper={best['proper_score_selection_eligible']} strict_auc={best['strict_auc_selection_eligible']}")
    print(f"[{P} freeze v2] wrote {fp.relative_to(REPO)} (policy_sha={policy_sha[:12]}…)")


if __name__ == "__main__":
    app()
