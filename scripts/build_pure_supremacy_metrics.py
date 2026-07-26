"""DIAGNOSTIC_POSTHOC_DEDNP_PLATT -- development-only probe (NOT a production candidate).

OWNER AUDIT CORRECTION: the post-hoc ``model_prob_over_final / (1 - p_dnp)`` de-DNP shortcut is
INVALID as a production fix (exact only for a half-line, pre-calibration, exact DNP-zero mixture;
wrong for integer lines with push mass and not the inverse of a nonlinear binary calibration).
The correct production DNP handling lives in ``models/availability_pmf.py`` (active PMF ->
push-safe settled probability -> calibration). This script is retained ONLY as a labelled
diagnostic and its numbers are:

    DEVELOPMENT_SELECTION_ONLY / UPSTREAM_PURITY_UNVERIFIED / NOT_PRODUCTION_WIRED /
    NOT_PROOF_ELIGIBLE / NOT_CERTIFIED

Upstream purity is UNVERIFIED: the consumed OOF was produced by a config with
market_prior_lambda>0 and a CLV head, so the probabilities already contained market information;
a downstream column guard cannot remove upstream market leakage. Regenerating a pure OOF requires
the raw feature matrix + provider API (absent in this environment).

It reports, under grouped expanding-window nested rolling-origin CV, the diagnostic de-DNP+Platt
metrics AND the REAL selection contract verdict (STEP 9): a prop passes ONLY when it is pure,
    dLL<0, dBrier<0, model_auc>=market_auc, AND the date-cluster bootstrap upper-95% CI of BOTH
    deltas < 0. The old point-estimate-only win label has been removed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.evaluation.rolling_origin import (  # noqa: E402
    all_chronology_pass,
    expanding_window_folds,
    fold_manifest,
)
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402
from wnba_props_model.models.pure_model_contract import (  # noqa: E402
    assert_pure_feature_columns,
)

app = typer.Typer(add_completion=False)
EPS = 1e-6
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
QUOTE_COVERED = ["pts", "reb", "ast", "fg3m"]
MIN_TRAIN_DATES = 15
VAL_BLOCK_DATES = 3
ECE_MARGIN = 0.03
WORST_FOLD_LL_MAX = 0.05
MARKET_COL = "market_prob_over_no_vig"
PURE_ALLOWED = {FINAL_PROBABILITY_COLUMN, "p_dnp", "outcome_over", "game_date"}


def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = _clip(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(y, p):
    return float(np.mean((_clip(p) - np.asarray(y, float)) ** 2))


def _ece(y, p, nb=10):
    y = np.asarray(y, int)
    p = _clip(p)
    ed = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, ed[1:-1]), 0, nb - 1)
    n = len(p)
    e = 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def de_dnp(p_over_blended, p_dnp):
    p = np.asarray(p_over_blended, float)
    d = np.clip(np.asarray(p_dnp, float), 0.0, 1 - EPS)
    return np.clip(p / (1.0 - d), EPS, 1 - EPS)


def _fit(name, tr):
    if name == "P0_identity":
        return lambda d: _clip(d[FINAL_PROBABILITY_COLUMN].to_numpy(float)), True
    if name == "P1_deDNP_platt":
        base = de_dnp(tr[FINAL_PROBABILITY_COLUMN].to_numpy(float), tr["p_dnp"].to_numpy(float))
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(base).reshape(-1, 1),
                                                            tr["outcome_over"].to_numpy(int))
        slope = float(lr.coef_[0][0])

        def _p(d):
            b = de_dnp(d[FINAL_PROBABILITY_COLUMN].to_numpy(float), d["p_dnp"].to_numpy(float))
            return _clip(lr.predict_proba(_logit(b).reshape(-1, 1))[:, 1])
        return _p, slope >= 0
    raise ValueError(name)


def nested_eval(pdf, name):
    pure = pdf[[c for c in pdf.columns if c in PURE_ALLOWED]].copy()
    assert_pure_feature_columns(pure.columns, context=f"pure_supremacy:{name}")
    d = pure["game_date"].to_numpy()
    folds = expanding_window_folds(d, min_train_dates=MIN_TRAIN_DATES, val_block_dates=VAL_BLOCK_DATES)
    if not folds:
        return None
    pred = np.full(len(pdf), np.nan)
    mono = True
    per_fold = []
    for f in folds:
        tr = pure[np.isin(d, list(f.train_dates))]
        va = pure[np.isin(d, list(f.val_dates))]
        if len(tr) < 30 or len(va) == 0:
            continue
        fn, ok = _fit(name, tr)
        mono = mono and ok
        idx = np.where(np.isin(d, list(f.val_dates)))[0]
        pv = fn(va)
        pred[idx] = pv
        yv = pdf["outcome_over"].to_numpy(int)[idx]
        pkv = pdf[MARKET_COL].to_numpy(float)[idx]
        per_fold.append(_ll(yv, pv) - _ll(yv, pkv))
    m = np.isfinite(pred)
    if m.sum() == 0:
        return None
    y = pdf["outcome_over"].to_numpy(int)[m]
    pk = pdf[MARKET_COL].to_numpy(float)[m]
    pr = pred[m]
    worst = float(max(per_fold)) if per_fold else float("nan")
    return {
        "candidate": name, "n": int(m.sum()), "n_folds": len(per_fold),
        "model_logloss": _ll(y, pr), "market_logloss": _ll(y, pk),
        "logloss_delta": _ll(y, pr) - _ll(y, pk),
        "model_brier": _brier(y, pr), "market_brier": _brier(y, pk),
        "brier_delta": _brier(y, pr) - _brier(y, pk),
        "model_auc": _auc(y, pr), "market_auc": _auc(y, pk),
        "model_ece": _ece(y, pr), "market_ece": _ece(y, pk),
        "worst_fold_logloss_delta": worst, "monotone": bool(mono),
        "_oof_pred": pred, "_mask": m, "_dates": pdf["game_date"].to_numpy(),
    }


def _bootstrap_ci(res, pdf, n_boot=5000, seed=20260726):
    """Paired date-cluster bootstrap of (model - market) LL and Brier deltas."""
    m = res["_mask"]
    pr = res["_oof_pred"][m]
    sub = pdf[m].reset_index(drop=True)
    y = sub["outcome_over"].to_numpy(int)
    pk = sub[MARKET_COL].to_numpy(float)
    dates = sub["game_date"].to_numpy()
    uniq = np.array(sorted(set(dates)))
    by = {u: np.where(dates == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    dll, dbr = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        dll.append(_ll(y[idx], pr[idx]) - _ll(y[idx], pk[idx]))
        dbr.append(_brier(y[idx], pr[idx]) - _brier(y[idx], pk[idx]))
    dll, dbr = np.array(dll), np.array(dbr)
    return {
        "logloss_delta_ci95": [float(np.percentile(dll, 2.5)), float(np.percentile(dll, 97.5))],
        "brier_delta_ci95": [float(np.percentile(dbr, 2.5)), float(np.percentile(dbr, 97.5))],
        "logloss_upper95_below_zero": bool(np.percentile(dll, 97.5) < 0),
        "brier_upper95_below_zero": bool(np.percentile(dbr, 97.5) < 0),
    }


def real_selection_contract(res, ci) -> dict:
    """STEP 9 selection contract (development pre-proof screen). Returns PASS/FAIL + reasons."""
    reasons = []
    if not (res["logloss_delta"] < 0):
        reasons.append(f"dLL>=0 ({res['logloss_delta']:+.5f})")
    if not (res["brier_delta"] < 0):
        reasons.append(f"dBrier>=0 ({res['brier_delta']:+.5f})")
    if not (res["model_auc"] >= res["market_auc"]):
        reasons.append(f"AUC<market ({res['model_auc']:.3f}<{res['market_auc']:.3f})")
    if not ci["logloss_upper95_below_zero"]:
        reasons.append(f"LL CI upper>=0 {ci['logloss_delta_ci95']}")
    if not ci["brier_upper95_below_zero"]:
        reasons.append(f"Brier CI upper>=0 {ci['brier_delta_ci95']}")
    return {"selection_contract_pass": len(reasons) == 0, "fail_reasons": reasons}


def load_joined(scored_path, oof_path):
    scored = pd.read_parquet(scored_path)
    oof = pd.read_parquet(oof_path).rename(columns={"stat": "prop"})
    for df in (scored, oof):
        df["game_id"] = df["game_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
    key = ["game_id", "player_id", "prop"]
    m = scored.merge(oof[key + ["p_dnp"]].drop_duplicates(key), on=key, how="left")
    m["p_dnp"] = m["p_dnp"].fillna(0.0)
    return m


@app.command()
def main(
    scored: str = typer.Option(
        "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet", "--scored"),
    oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet", "--oof"),
    out_dir: str = typer.Option("artifacts/pure_supremacy", "--out-dir"),
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    m = load_joined(scored, oof)

    manifests = {}
    for prop in QUOTE_COVERED:
        pdf = m[m["prop"] == prop]
        folds = expanding_window_folds(pdf["game_date"].to_numpy(),
                                       min_train_dates=MIN_TRAIN_DATES, val_block_dates=VAL_BLOCK_DATES)
        manifests[prop] = {"folds": fold_manifest(folds, lambda ds: int(pdf["game_date"].isin(ds).sum())),
                           "all_chronology_pass": all_chronology_pass(folds)}

    per_prop = {}
    csv_rows = []
    for prop in DIRECT_PROPS:
        if prop not in QUOTE_COVERED:
            per_prop[prop] = {"status": "NO_EXACT_QUOTES",
                              "note": "market comparison blocked until same-book quotes collected"}
            continue
        pdf = m[m["prop"] == prop].sort_values("game_date").reset_index(drop=True)
        before = nested_eval(pdf, "P0_identity")
        after = nested_eval(pdf, "P1_deDNP_platt")
        ci = _bootstrap_ci(after, pdf) if after else None
        contract = real_selection_contract(after, ci) if (after and ci) else None
        pub = lambda r: {k: v for k, v in r.items() if not k.startswith("_")} if r else None
        per_prop[prop] = {
            "before_P0": pub(before),
            "diagnostic_after_deDNP_platt": pub(after),
            "diagnostic_after_bootstrap_ci": ci,
            "real_selection_contract": contract,
            "labels": ["DIAGNOSTIC_POSTHOC_DEDNP_PLATT", "DEVELOPMENT_SELECTION_ONLY",
                       "UPSTREAM_PURITY_UNVERIFIED", "NOT_PRODUCTION_WIRED",
                       "NOT_PROOF_ELIGIBLE", "NOT_CERTIFIED"],
        }
        for tag, r in [("BEFORE_P0", before), ("DIAGNOSTIC_AFTER_dednp_platt", after)]:
            if r:
                csv_rows.append({"prop": prop, "stage": tag, **{k: r[k] for k in (
                    "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                    "model_brier", "market_brier", "brier_delta", "model_auc", "market_auc",
                    "model_ece", "market_ece", "worst_fold_logloss_delta", "monotone")}})

    passing = [p for p in QUOTE_COVERED
               if per_prop[p]["real_selection_contract"]
               and per_prop[p]["real_selection_contract"]["selection_contract_pass"]]
    (outp / "PURE_SUPREMACY_OOF_METRICS.json").write_text(json.dumps({
        "version": "diagnostic-posthoc-dednp-platt-v2",
        "status_labels": ["DIAGNOSTIC_POSTHOC_DEDNP_PLATT", "DEVELOPMENT_SELECTION_ONLY",
                          "UPSTREAM_PURITY_UNVERIFIED", "NOT_PRODUCTION_WIRED",
                          "NOT_PROOF_ELIGIBLE", "NOT_CERTIFIED"],
        "invalid_shortcut_note": ("post-hoc model_prob_over_final/(1-p_dnp) is NOT a production "
                                  "fix; correct DNP handling is models/availability_pmf.py"),
        "upstream_purity": "UNVERIFIED (consumed OOF built with market_prior_lambda>0 and CLV head)",
        "downstream_column_guard_only": sorted(PURE_ALLOWED),
        "cv": "grouped expanding-window nested rolling-origin (max(train)<min(val))",
        "real_selection_contract": ("pure AND dLL<0 AND dBrier<0 AND model_auc>=market_auc AND "
                                    "bootstrap upper-95% CI of BOTH deltas < 0"),
        "fold_manifest": manifests, "per_prop": per_prop,
        "props_passing_real_selection_contract": passing,
    }, indent=2) + "\n")
    pd.DataFrame(csv_rows).to_csv(outp / "PURE_SUPREMACY_OOF_METRICS.csv", index=False)

    print("\n=== DIAGNOSTIC de-DNP+Platt (NOT PRODUCTION) vs exact no-vig market (nested CV) ===")
    print(f"{'prop':5s} {'stage':10s} {'n':>4s} {'mLL':>7s} {'kLL':>7s} {'dLL':>8s} {'dBr':>8s} "
          f"{'mAUC':>6s} {'kAUC':>6s} {'mECE':>6s}")
    for prop in QUOTE_COVERED:
        for tag, key in [("BEFORE", "before_P0"), ("DIAG", "diagnostic_after_deDNP_platt")]:
            r = per_prop[prop][key]
            if r:
                print(f"{prop:5s} {tag:10s} {r['n']:>4d} {r['model_logloss']:>7.4f} {r['market_logloss']:>7.4f} "
                      f"{r['logloss_delta']:>+8.4f} {r['brier_delta']:>+8.4f} {r['model_auc']:>6.3f} "
                      f"{r['market_auc']:>6.3f} {r['model_ece']:>6.3f}")
        c = per_prop[prop]["real_selection_contract"]
        if c:
            print(f"      -> REAL selection contract: {'PASS' if c['selection_contract_pass'] else 'FAIL'} "
                  f"{c['fail_reasons']}")
    print(f"\nPROPS PASSING REAL SELECTION CONTRACT: {passing or 'none'}")
    print("stl/blk/turnover: NO_EXACT_QUOTES (market comparison blocked on odds collection)")


if __name__ == "__main__":
    app()
