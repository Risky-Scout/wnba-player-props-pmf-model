"""Phase 3: targeted discrimination-repair diagnostics for PTS / REB / FG3M.

Runs the same A0-A3 challenger battery (market identity / monotone-calibrated model /
strongly-shrunk convex logit blend / role+line-band hierarchical blend) on the primary
deterministic one-quote development window per prop. The A4 feature-residual learner is
BLOCKED here: the pregame feature matrix is gitignored and unrecoverable on this VM, so no
new decision-time feature signal can be introduced without data recovery.

No freeze (only one AST candidate is frozen in Phase 2). This documents whether ANY
low-cost, existing-output challenger achieves incremental discrimination beyond the market;
if not, targeted feature/minutes/tracking work (requiring the feature matrix) is justified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ast_first_edge_sprint import _crossfit, _date_folds, _ll, _metrics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"


def run_prop(df, prop, k=5) -> dict:
    dev = df[(df["prop"] == prop) & df["is_primary"]].reset_index(drop=True)
    import numpy as np
    y = dev["outcome_over"].to_numpy(int)
    pk = dev["market_prob_over_no_vig"].to_numpy(float)
    recs = []
    for kind in ["A0", "A1", "A2", "A3"]:
        pred = _crossfit(dev, kind, k)
        m = _metrics(dev, pred, y, pk); mask = m.pop("mask")
        worst = -np.inf
        for fold in _date_folds(dev["game_date"].to_numpy(), k):
            fm = np.array([d in fold for d in dev["game_date"].to_numpy()]) & mask
            if fm.sum() >= 15:
                worst = max(worst, _ll(y[fm], pred[fm]) - _ll(y[fm], pk[fm]))
        recs.append({"candidate": kind, "n": m["n"],
                     "logloss_delta_vs_market": m["logloss"] - m["market_logloss"],
                     "brier_delta_vs_market": m["brier"] - m["market_brier"],
                     "ece": m["ece"], "auc": m["auc"], "market_auc": m["market_auc"],
                     "cal_slope": m["cal_slope"], "worst_fold_logloss_delta": float(worst),
                     "beats_market_both": bool(m["logloss"] < m["market_logloss"]
                                               and m["brier"] < m["market_brier"])})
    recs.append({"candidate": "A4_feature_residual", "status": "BLOCKED_NO_FEATURE_MATRIX"})
    best = min((r for r in recs if r.get("beats_market_both")),
               key=lambda r: r["logloss_delta_vs_market"], default=None)
    return {"prop": prop, "n_dev": int(len(dev)), "records": recs,
            "any_existing_output_challenger_beats_market": best is not None,
            "best_challenger": (best["candidate"] if best else None)}


def main() -> int:
    df = pd.read_parquet(G0 / "scored_candidates_g0v2.parquet")
    out = {"version": "phase3-discrimination-repair-v1",
           "scope": "primary_deterministic_one_quote (DEVELOPMENT/SELECTION; NOT future proof)",
           "a4_status": "BLOCKED_NO_FEATURE_MATRIX (pregame feature matrix unrecoverable on this VM)",
           "props": {}}
    for prop in ["pts", "reb", "fg3m"]:
        out["props"][prop] = run_prop(df, prop)
    (G0 / "PHASE3_DISCRIMINATION_REPAIR.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    for prop, r in out["props"].items():
        print(f"\n== {prop} (n_dev={r['n_dev']}) best_existing_output_challenger="
              f"{r['best_challenger']} ==")
        for rec in r["records"]:
            if rec.get("status"):
                print(f"  {rec['candidate']}: {rec['status']}"); continue
            print(f"  {rec['candidate']}: dLL={rec['logloss_delta_vs_market']:+.5f} "
                  f"dBrier={rec['brier_delta_vs_market']:+.5f} auc={rec['auc']:.3f} "
                  f"mkt_auc={rec['market_auc']:.3f} slope={rec['cal_slope']:+.3f} "
                  f"beats_both={rec['beats_market_both']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
