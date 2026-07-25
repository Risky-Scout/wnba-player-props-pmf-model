"""Phase 3: targeted discrimination-repair diagnosis for PTS / REB / FG3M.

Derives the per-prop diagnosis from the CORRECTED nested rolling-origin candidate metrics
(LOWCOST_CANDIDATE_METRICS.json) and the probability/target semantics audit. No leaky CV.
The A4 feature-residual learner remains BLOCKED_NO_FEATURE_MATRIX (pregame feature matrix
unrecoverable on this VM), so no new decision-time feature signal can be introduced here.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main() -> int:
    cand = _load(G0 / "LOWCOST_CANDIDATE_METRICS.json")
    sem = _load(G0 / "PROBABILITY_TARGET_SEMANTICS_AUDIT.json")
    a4 = _load(G0 / "PHASE3_A4_FEATURE_RESIDUAL.json")
    a4_props = (a4 or {}).get("props", {})
    a4_status = (a4.get("a4_status") if a4 else
                 "BLOCKED_NO_FEATURE_MATRIX (pregame feature matrix unrecoverable on this VM)")
    recs = {(r["prop"], r["candidate"]): r for r in (cand["records"] if cand else [])
            if r.get("status") == "EVALUATED"}
    out = {"version": "phase3-discrimination-repair-v3",
           "cv": "corrected nested expanding-window rolling-origin",
           "a4_status": a4_status,
           "props": {}}
    for prop in ["pts", "reb", "fg3m"]:
        prop_recs = [r for (p, _), r in recs.items() if p == prop]
        advancing = [r["candidate"] for r in prop_recs if r.get("advances")]
        s = (sem or {}).get("per_prop", {}).get(prop, {})
        # market AUC advantage from C0 (raw model) row
        c0 = recs.get((prop, "C0_identity"), {})
        a4r = a4_props.get(prop, {})
        a4_eval = a4r.get("status") == "EVALUATED"
        if a4_eval:
            next_repair = (
                "A4 regularized feature residual FIT (feature matrix recovered): "
                f"cand AUC {a4r['cand_auc']:.3f} vs base {a4r['base_model_auc']:.3f} "
                f"vs market {a4r['market_auc']:.3f}; "
                f"beats_market={a4r['proper_score_selection_eligible']}, "
                f"adds_discrimination_vs_market={a4r['auc_delta_vs_market'] > 0}. "
                + ("Repair does NOT beat the exact market on the leakage-safe nested CV; prop "
                   "remains NOT promotion-eligible (no certified edge)."
                   if not a4r["proper_score_selection_eligible"] else
                   "Repair beats market on proper scores; eligible for a prospective proof."))
        else:
            next_repair = ("A4 regularized feature residual (existing pregame features) — "
                           "BLOCKED_NO_FEATURE_MATRIX until feature matrix is recovered/rebuilt")
        out["props"][prop] = {
            "any_existing_output_challenger_beats_market": bool(advancing),
            "advancing_candidates": advancing,
            "raw_model_auc": c0.get("cand_auc"), "market_auc": c0.get("market_auc"),
            "auc_deficit_vs_market": (None if not c0 else round(c0["market_auc"] - c0["cand_auc"], 4)),
            "semantics_sign_status": s.get("sign_status"),
            "a4_feature_residual": (a4r if a4_eval else None),
            "diagnosis": (
                "sign-inversion/near-null signal at market lines; no gross target/line defect found "
                "(under=1-over, prob at quote line, pushes excluded, identities clean)"
                if s.get("sign_status") == "SIGN_INVERSION_DIAGNOSTIC" else
                "market discrimination advantage; recalibration/market-blend cannot add ordering"),
            "next_repair": next_repair,
        }
    (G0 / "PHASE3_DISCRIMINATION_REPAIR.json").write_text(json.dumps(out, indent=2) + "\n")
    for prop, r in out["props"].items():
        print(f"{prop:5s} beats_market={r['any_existing_output_challenger_beats_market']} "
              f"auc_deficit={r['auc_deficit_vs_market']} sign={r['semantics_sign_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
