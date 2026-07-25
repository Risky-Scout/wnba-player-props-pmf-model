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
    recs = {(r["prop"], r["candidate"]): r for r in (cand["records"] if cand else [])
            if r.get("status") == "EVALUATED"}
    out = {"version": "phase3-discrimination-repair-v2",
           "cv": "corrected nested expanding-window rolling-origin",
           "a4_status": "BLOCKED_NO_FEATURE_MATRIX (pregame feature matrix unrecoverable on this VM)",
           "props": {}}
    for prop in ["pts", "reb", "fg3m"]:
        prop_recs = [r for (p, _), r in recs.items() if p == prop]
        advancing = [r["candidate"] for r in prop_recs if r.get("advances")]
        s = (sem or {}).get("per_prop", {}).get(prop, {})
        # market AUC advantage from C0 (raw model) row
        c0 = recs.get((prop, "C0_identity"), {})
        out["props"][prop] = {
            "any_existing_output_challenger_beats_market": bool(advancing),
            "advancing_candidates": advancing,
            "raw_model_auc": c0.get("cand_auc"), "market_auc": c0.get("market_auc"),
            "auc_deficit_vs_market": (None if not c0 else round(c0["market_auc"] - c0["cand_auc"], 4)),
            "semantics_sign_status": s.get("sign_status"),
            "diagnosis": (
                "sign-inversion/near-null signal at market lines; no gross target/line defect found "
                "(under=1-over, prob at quote line, pushes excluded, identities clean)"
                if s.get("sign_status") == "SIGN_INVERSION_DIAGNOSTIC" else
                "market discrimination advantage; recalibration/market-blend cannot add ordering"),
            "next_repair": ("A4 regularized feature residual (existing pregame features) — "
                            "BLOCKED_NO_FEATURE_MATRIX until feature matrix is recovered/rebuilt"),
        }
    (G0 / "PHASE3_DISCRIMINATION_REPAIR.json").write_text(json.dumps(out, indent=2) + "\n")
    for prop, r in out["props"].items():
        print(f"{prop:5s} beats_market={r['any_existing_output_challenger_beats_market']} "
              f"auc_deficit={r['auc_deficit_vs_market']} sign={r['semantics_sign_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
