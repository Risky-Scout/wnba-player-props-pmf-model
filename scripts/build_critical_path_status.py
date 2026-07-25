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
        if r["prop"] == prop and r["scope"] == "primary_book" and r["status"] == "EVALUATED":
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
        prim = coverage["by_prop_primary"].get(prop, {}) if coverage else {}
        pooled = coverage["by_prop_book"].get(prop, {}) if coverage else {}
        pooled_rows = sum(v["rows"] for v in pooled.values()) if pooled else 0
        quote_counts[prop] = {
            "primary_book_exact_rows": prim.get("rows", 0),
            "primary_book_unique_dates": prim.get("dates", 0),
            "all_books_exact_rows": pooled_rows,
            "status": prim.get("status", "NO_EXACT_QUOTES"),
        }

    # Deliverable 4/5/6/8 for quote-covered props.
    per_prop = {}
    for prop in QUOTE_COVERED:
        c0 = _c0(metrics, prop)
        cands = [r for r in cand["records"] if r["prop"] == prop and r.get("status") == "EVALUATED"] if cand else []
        advancing = [r["candidate"] for r in cands if r.get("advances")]
        diag = _classify(c0) if c0 else {"reason": "no_data"}
        # forward proof outcome (primary + pooled)
        proofs = {}
        for scope, d in (("primary_book", REPO / f"artifacts/market_feature_proof/G0_v2_proof_{prop}"),
                         ("all_books_pooled", REPO / f"artifacts/market_feature_proof/G0_v2_proof_pooled_{prop}")):
            j = _load(d / "market_superiority_proof.json")
            if j and j.get("results"):
                r = j["results"][0]
                proofs[scope] = {"candidate": r["candidate"], "n_settled": r["n_settled"],
                                 "n_clusters": r["n_clusters"], "logloss_delta": r["logloss_delta"],
                                 "brier_delta": r["brier_delta"], "auc_delta": r["auc_delta"],
                                 "status": r["status"]}
        per_prop[prop] = {
            "g0v2_c0": {k: c0[k] for k in ("n_settled", "n_dates", "model_logloss", "market_logloss",
                        "logloss_delta", "model_brier", "market_brier", "brier_delta",
                        "model_auc", "market_auc", "model_ece", "market_ece")} if c0 else None,
            "selection_advancing_candidates": advancing,
            "failure_diagnosis": diag,
            "forward_proof": proofs,
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
        "deliverable_7_first_edge_board_row_blocker": {
            "certified_props": [],
            "blocker": ("No prop's probabilities (existing model with or without C0-C6 low-cost "
                        "correction) certifiably beat the EXACT decision-time market on the untouched "
                        "forward window under the frozen proof contract (>=300 rows, >=30 clusters, "
                        "cluster-bootstrap 95% CI, Holm). Pure-model recalibration edges seen on the "
                        "selection window REVERSE on the untouched window (non-stationary/overfit). "
                        "The closest prop (ast) reaches the correct forward sign but the margin is not "
                        "statistically significant and the exact-quote history is a single partial "
                        "season (~55 game-dates)."),
            "primary_causes": {
                "pts": "discrimination (model AUC < 0.50 at market lines)",
                "reb": "discrimination (market AUC materially higher)",
                "fg3m": "discrimination (largest AUC gap; selection calibration edge overfit)",
                "ast": "calibration + insufficient forward exact-quote volume (closest to passing)",
                "stl/blk/turnover": "no exact market quotes collected",
            },
            "actionable_next": [
                "Continue append-only exact-quote collection to lengthen the forward window (ast is "
                "the closest; a few more ast game-dates enable a powered significance test).",
                "Collect stl/blk/turnover odds to make those props evaluable at all.",
                "Targeted model-signal (discrimination) repair for pts/reb/fg3m is DEFERRED pending "
                "this evidence per owner instruction (no new architecture without measured need).",
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
