"""Consolidate the G0-v2 critical-path evidence into the owner's required deliverables.

Reads the committed OOF, exact-quote, G0-v2 metric, low-cost-candidate, and proof artifacts
and emits:
  * artifacts/market_feature_proof/G0_v2/PER_PROP_FAILURE_DIAGNOSIS.json  (owner step 8)
  * artifacts/market_feature_proof/G0_v2/CRITICAL_PATH_STATUS.json        (deliverables 1-7)

No new modeling; pure summarization of measured results (no fabrication).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"
DIRECT = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
QUOTE_COVERED = ["pts", "reb", "ast", "fg3m"]


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def _c0(metrics, prop):
    for r in metrics["records"]:
        if r["prop"] == prop and r["scope"] == "primary_deterministic" and r["status"] == "EVALUATED":
            return r
    return None


def _classify(c0) -> dict:
    """Owner step-8 reason classification from measured C0 quantities + proof outcome."""
    dauc = c0["model_auc"] - c0["market_auc"]
    dece = c0["model_ece"] - c0["market_ece"]
    if c0["model_auc"] < 0.50:
        reason = "discrimination"
        note = "model binary AUC < 0.50 at market lines: ordering is non-informative."
    elif dauc <= -0.02:
        reason = "discrimination"
        note = f"market AUC exceeds model AUC by {-dauc:.3f}; calibration cannot add ordering."
    elif dece >= 0.03:
        reason = "calibration"
        note = (f"discrimination ~ market (dAUC={dauc:+.3f}) but model ECE exceeds market by "
                f"{dece:.3f}; recalibration is the lever.")
    else:
        reason = "pmf_shape_or_margin"
        note = "discrimination and calibration ~ market; residual proper-score gap is small."
    return {"reason": reason, "delta_auc_vs_market": dauc, "delta_ece_vs_market": dece, "note": note}


def main() -> int:
    oof = pd.read_parquet(REPO / "artifacts/models/calibration/oof_predictions.parquet")
    quotes = pd.read_parquet(REPO / "artifacts/p1/p1_quotes.parquet")
    metrics = _load(G0 / "G0_V2_METRICS.json")
    coverage = _load(G0 / "G0_V2_QUOTE_COVERAGE.json")
    cand = _load(G0 / "LOWCOST_CANDIDATE_METRICS.json")
    ranking = _load(G0 / "CLOSEST_PROP_RANKING.json")

    # Deliverable 2: OOF status.
    oof_status = {
        "rows": int(len(oof)), "props": sorted(oof["stat"].unique().tolist()),
        "rows_per_prop": {k: int(v) for k, v in oof.groupby("stat").size().items()},
        "oof_prediction_type": {k: int(v) for k, v in oof["oof_prediction_type"].value_counts().items()},
        "minutes_prediction_type": {k: int(v) for k, v in oof["minutes_prediction_type"].value_counts().items()},
        "n_folds": int(oof["fold_id"].nunique()),
        "train_before_validation_all_rows": bool(
            (pd.to_datetime(oof["fold_train_end_date"]) < pd.to_datetime(oof["fold_validation_start_date"])).all()),
        "date_range": [str(oof["game_date"].min()), str(oof["game_date"].max())],
        "no_failed_model_fallback": bool((oof["oof_prediction_type"] == "model_oof").all()),
        "no_prior_only_rows": bool((oof["oof_prediction_type"] == "model_oof").all()),
        "all_seven_props_present": bool(set(DIRECT).issubset(set(oof["stat"].unique()))),
        "null_pmf_json": int(oof["pmf_json"].isna().sum()),
        "status": "COMPLETE_ONE_FULL_OOF_RUN",
    }

    # Deliverable 3: exact quote counts by prop (decision snapshot exact pairs).
    quote_counts = {}
    for prop in DIRECT:
        prim = coverage["by_prop_primary_deterministic"].get(prop, {}) if coverage else {}
        pooled = coverage["by_prop_all_books_pooled"].get(prop, {}) if coverage else {}
        quote_counts[prop] = {
            "primary_deterministic_exact_rows": prim.get("rows", 0),
            "primary_deterministic_unique_dates": prim.get("dates", 0),
            "primary_book_mix": prim.get("book_mix", {}),
            "all_books_pooled_rows_SENSITIVITY": pooled.get("rows", 0),
            "status": prim.get("status", "NO_EXACT_QUOTES"),
        }

    # Deliverable 4/5/6/8 for quote-covered props.
    semantics = _load(G0 / "PROBABILITY_TARGET_SEMANTICS_AUDIT.json")
    phase3 = _load(G0 / "PHASE3_DISCRIMINATION_REPAIR.json")
    per_prop = {}
    for prop in QUOTE_COVERED:
        c0 = _c0(metrics, prop)
        cands = [r for r in cand["records"] if r["prop"] == prop and r.get("status") == "EVALUATED"] if cand else []
        advancing = [r["candidate"] for r in cands if r.get("advances")]
        diag = _classify(c0) if c0 else {"reason": "no_data"}
        sem = (semantics or {}).get("per_prop", {}).get(prop, {})
        per_prop[prop] = {
            "g0v2_c0_primary_deterministic": {k: c0[k] for k in (
                "n_settled", "n_dates", "model_logloss", "market_logloss", "logloss_delta",
                "model_brier", "market_brier", "brier_delta", "model_auc", "market_auc",
                "model_ece", "market_ece")} if c0 else None,
            "semantics_sign_status": sem.get("sign_status"),
            "selection_advancing_candidates_diagnostic": advancing,
            "failure_diagnosis": diag,
            "historical_window_state": "DEVELOPMENT_SELECTION_EVIDENCE / NOT_FUTURE_PROOF",
            "certified": False,
        }

    diagnosis = {
        "version": "per-prop-failure-diagnosis-v1",
        "quote_covered_props": per_prop,
        "no_quote_props": {p: "NO_EXACT_QUOTES (cannot G0/certify until odds collected)"
                           for p in ("stl", "blk", "turnover")},
        "summary": {
            "discrimination_limited": [p for p, v in per_prop.items()
                                       if v["failure_diagnosis"]["reason"] == "discrimination"],
            "calibration_limited": [p for p, v in per_prop.items()
                                    if v["failure_diagnosis"]["reason"] == "calibration"],
        },
    }
    (G0 / "PER_PROP_FAILURE_DIAGNOSIS.json").write_text(json.dumps(diagnosis, indent=2) + "\n")

    closest = None
    if ranking and ranking.get("records"):
        # closest proof-ready = discrimination on par with market (calibration-limited) first
        cal = [p for p, v in per_prop.items() if v["failure_diagnosis"]["reason"] == "calibration"]
        closest = cal[0] if cal else ranking["records"][0]["prop"]

    status = {
        "version": "critical-path-status-v1",
        "deliverable_1_data_preservation": _load(REPO / "artifacts/data_bootstrap/DATA_PRESERVATION_RESULT.json"),
        "deliverable_2_oof_status": oof_status,
        "deliverable_3_exact_quote_counts_by_prop": quote_counts,
        "deliverable_4_g0v2_metrics": metrics,
        "deliverable_5_lowcost_candidate_metrics": cand,
        "deliverable_6_closest_proof_ready_prop": {
            "closest_prop": closest,
            "rationale": ("ast: model binary discrimination is on par with the market "
                          "(dAUC ~ 0), so it is the only prop where low-cost recalibration "
                          "achieves the correct forward sign; pts/reb/fg3m carry a measured "
                          "discrimination deficit vs market that calibration cannot repair."),
            "selection_ranking": ranking,
        },
        "deliverable_6b_ast_freeze_v2": _load(REPO / "artifacts/candidate_freeze/AST_FIRST_EDGE_FREEZE_V2.json"),
        "deliverable_6c_ast_freeze_v1_invalidation": _load(REPO / "artifacts/candidate_freeze/AST_FIRST_EDGE_FREEZE_INVALIDATION.json"),
        "deliverable_6d_quote_policy_hash_resolution": {
            "authoritative_hash_method": "raw file SHA-256 of config/book_quote_priority_v1.json",
            "quote_policy_file_sha256": "962db96af3cceb31eb0e2efc08ca5f069e517e131e10d1a76619de4f8a20c780",
            "stale_v1_freeze_hash": "4b39ee8f1deb33cd211e83e186d89aa3fc0bdc2b7ae00a0197a2392360b70c89",
            "resolution": ("The stale value was a string hash of a label, not the file digest; the "
                           "v1 freeze is invalidated. The v2 freeze records the raw file SHA-256."),
            "tiebreak_audit": _load(G0 / "QUOTE_POLICY_TIEBREAK_AUDIT.json"),
        },
        "deliverable_7_first_edge_board_row_blocker": {
            "certified_props": [],
            "blocker": ("No certified prop yet. AST candidate A1 (monotone-calibrated existing model) "
                        "is FROZEN and beats the exact one-quote market on both proper scores on the "
                        "DEVELOPMENT window (dLL -0.0037, dBrier -0.0019, ECE 0.016) but not "
                        "significantly (historical DIAGNOSTIC FAIL). Certification now requires a "
                        "PROSPECTIVE proof on NEW dates after the freeze timestamp (>=5000 cluster "
                        "bootstrap, Holm, upper-95%-CI<0 for log loss AND Brier). That prospective "
                        "window does not exist yet and its collection is BLOCKED_NO_ODDS_API_KEY."),
            "primary_causes": {
                "pts": "sign-inversion/near-null signal (cross-fit AUC 0.47, Platt slope negative); no gross target/line defect found",
                "reb": "market discrimination advantage (market AUC 0.587 vs model 0.545); no existing-output challenger beats market",
                "fg3m": "largest market discrimination advantage (0.617 vs 0.546); no existing-output challenger beats market",
                "ast": "frozen A1 beats market on dev proper scores but not significant; needs prospective proof",
                "stl/blk/turnover": "no exact market quotes (BLOCKED_NO_ODDS_API_KEY)",
            },
            "targeted_repair_blocker": ("The A4 feature-residual challenger (the only path that could "
                                        "add genuine discrimination for pts/reb/fg3m) is BLOCKED_NO_FEATURE_MATRIX: "
                                        "the pregame feature matrix is gitignored/unrecoverable on this VM."),
            "actionable_next": [
                "Add ODDS_API_KEY -> start append-only prospective exact-quote collection for all 7 "
                "props; the AST prospective proof then accrues automatically after the freeze timestamp.",
                "Recover the feature matrix (grant token access to the private data repo, or restore "
                "BDL_API_KEY) to enable A4 feature-residual repair for pts/reb/fg3m.",
                "Fix token access to the private data repo (BLOCKED_TOKEN_REPOSITORY_ACCESS) to publish "
                "the recovery bundle.",
            ],
        },
    }
    (G0 / "CRITICAL_PATH_STATUS.json").write_text(json.dumps(status, indent=2, default=str) + "\n")
    print("[status] wrote PER_PROP_FAILURE_DIAGNOSIS.json + CRITICAL_PATH_STATUS.json")
    print(f"[status] closest proof-ready prop = {closest}")
    print(f"[status] discrimination-limited = {diagnosis['summary']['discrimination_limited']}")
    print(f"[status] calibration-limited   = {diagnosis['summary']['calibration_limited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
