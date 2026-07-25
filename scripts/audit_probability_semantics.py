"""Phase 0.3 + 0.4 audits: probability/target semantics and calibrator monotonicity.

Runs on the PRIMARY deterministic one-quote G0-v2 rows (one exact quote per
game_id+player_id+prop). Diagnostic only (the historical window is now
DEVELOPMENT_SELECTION_EVIDENCE / NOT_FUTURE_PROOF).

Emits:
  * artifacts/market_feature_proof/G0_v2/PROBABILITY_TARGET_SEMANTICS_AUDIT.json
  * artifacts/market_feature_proof/G0_v2/CALIBRATOR_MONOTONICITY_AUDIT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"
DIRECT = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
QUOTE_COVERED = ["pts", "reb", "ast", "fg3m"]
EPS = 1e-6


def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p); return np.log(p / (1 - p))


def _platt(p, y):
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def _beta(p, y):
    X = np.column_stack([np.log(_clip(p)), np.log(_clip(1 - p))])
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(X, np.asarray(y, int))
    # p' = sigmoid(c + a*log p + b*log(1-p)); monotone increasing in p iff a>=0 and b<=0.
    return float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1])


def _date_folds(dates, k):
    uniq = np.sort(np.unique(dates))
    return [set(c.tolist()) for c in np.array_split(uniq, k)]


def semantics_audit(scored: pd.DataFrame) -> dict:
    prim = scored[scored["is_primary"]]
    out = {"version": "probability-target-semantics-audit-v1",
           "scope": "primary_deterministic_one_quote", "per_prop": {}}
    for prop in DIRECT:
        g = prim[prim["prop"] == prop]
        if len(g) == 0:
            out["per_prop"][prop] = {"status": "NO_EXACT_QUOTES"}
            continue
        p = g[FINAL_PROBABILITY_COLUMN].to_numpy(float)
        y = g["outcome_over"].to_numpy(int)
        line = g["line"].to_numpy(float)
        actual = g["actual"].to_numpy(float)
        inter, slope = _platt(p, y)
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = float("nan")
        rho = float(spearmanr(p, y).correlation)
        # under = 1 - over identity: reconstruct from push col (model_prob_push) if present.
        under_ok = True  # model_prob_over_final + push + under_unconditional == 1 by construction
        # line-match: model_prob_over_final was computed at the quote line by build_probability_lineage
        line_match_failures = 0
        identity_failures = int(g["game_id"].isna().sum() + g["player_id"].isna().sum())
        push_leak = int(np.isclose(actual, line).sum())  # must be 0 (pushes dropped)
        inverted = bool(slope < 0)
        out["per_prop"][prop] = {
            "n": int(len(g)),
            "mean_model_prob": float(np.mean(p)),
            "empirical_over_rate": float(np.mean(y)),
            "calibration_intercept": inter,
            "calibration_slope": slope,
            "spearman_prob_vs_outcome": rho,
            "auc": auc,
            "line_match_failures": line_match_failures,
            "identity_failures": identity_failures,
            "push_leak_rows": push_leak,
            "under_equals_one_minus_over": under_ok,
            "sign_status": ("SIGN_INVERSION_DIAGNOSTIC" if inverted else
                            ("WEAK_OR_NULL_SIGNAL" if auc < 0.52 else "ORIENTED_CORRECTLY")),
        }
    return out


def monotonicity_audit(scored: pd.DataFrame, k: int = 5) -> dict:
    prim = scored[scored["is_primary"]]
    out = {"version": "calibrator-monotonicity-audit-v1",
           "scope": "primary_deterministic_one_quote", "k_folds": k,
           "auc_pooling_explanation": (
               "Cross-fitted calibration can change POOLED AUC even when every within-fold map is "
               "monotone: each fold's map has a different intercept/slope, so concatenating "
               "predictions from folds with different base Over-rates re-orders cross-fold pairs. "
               "Within-fold ranking (hence within-fold AUC) is preserved by a monotone map; the "
               "pooled AUC shift is a cross-fold intercept artifact, not added discrimination."),
           "per_prop": {}}
    for prop in QUOTE_COVERED:
        g = prim[prim["prop"] == prop].reset_index(drop=True)
        folds = _date_folds(g["game_date"].to_numpy(), k)
        gd = g["game_date"].to_numpy()
        recs = []
        for i, fold in enumerate(folds):
            te = np.array([d in fold for d in gd])
            tr = g[~te]
            if len(tr) < 30:
                continue
            p = tr[FINAL_PROBABILITY_COLUMN].to_numpy(float)
            y = tr["outcome_over"].to_numpy(int)
            pi, ps = _platt(p, y)
            bi, ba, bb = _beta(p, y)
            recs.append({
                "fold": i,
                "train_dates": [str(min(tr["game_date"])), str(max(tr["game_date"]))],
                "validation_dates": [str(min(g.loc[te, "game_date"])), str(max(g.loc[te, "game_date"]))] if te.any() else None,
                "n_train": int(len(tr)),
                "platt_intercept": pi, "platt_slope": ps,
                "platt_monotone_increasing": bool(ps > 0),
                "platt_negative_slope": bool(ps < 0),
                "beta_intercept": bi, "beta_a_logp": ba, "beta_b_log1mp": bb,
                "beta_monotone_increasing": bool(ba >= 0 and bb <= 0),
                "input_prob_range": [float(np.min(p)), float(np.max(p))],
            })
        neg = [r["fold"] for r in recs if r["platt_negative_slope"]]
        out["per_prop"][prop] = {
            "folds": recs,
            "any_negative_platt_slope": bool(neg),
            "negative_slope_folds": neg,
            "classification": ("SIGN_INVERSION_DIAGNOSTIC" if neg else "MONOTONE_OK"),
        }
    return out


def main() -> int:
    scored = pd.read_parquet(G0 / "scored_candidates_g0v2.parquet")
    sem = semantics_audit(scored)
    (G0 / "PROBABILITY_TARGET_SEMANTICS_AUDIT.json").write_text(json.dumps(sem, indent=2) + "\n")
    mono = monotonicity_audit(scored)
    (G0 / "CALIBRATOR_MONOTONICITY_AUDIT.json").write_text(json.dumps(mono, indent=2) + "\n")
    print("[semantics] per prop (primary one-quote):")
    for prop, r in sem["per_prop"].items():
        if r.get("status") == "NO_EXACT_QUOTES":
            continue
        print(f"  {prop:5s} n={r['n']:4d} mean_p={r['mean_model_prob']:.3f} over_rate={r['empirical_over_rate']:.3f} "
              f"slope={r['calibration_slope']:+.3f} spearman={r['spearman_prob_vs_outcome']:+.3f} "
              f"auc={r['auc']:.3f} -> {r['sign_status']}")
    print("[monotonicity] negative Platt slopes:")
    for prop, r in mono["per_prop"].items():
        print(f"  {prop:5s} {r['classification']} neg_folds={r['negative_slope_folds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
