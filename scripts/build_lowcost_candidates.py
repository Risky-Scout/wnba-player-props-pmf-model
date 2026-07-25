"""Low-cost probability-correction candidates C0-C6 (owner critical-path step 5/8).

CORRECTED CV: grouped EXPANDING-WINDOW rolling-origin (leakage-safe). Every outer fold
satisfies max(train game_date) < min(validation game_date); each candidate is fit ONLY on
the outer training period (hyperparameters nested-selected inside the outer training dates)
and scored exactly once on the untouched outer validation block. Aggregate = concatenation
of outer-validation predictions (never trained on future). This replaces the prior
leave-block-out cross-fit (which trained on future date blocks).

Candidates:
  C0 identity | C1 Platt | C2 Beta | C3 isotonic (support-gated) |
  C4 convex market blend | C5 prop-role blend | C6 regularized market residual

Calibration constraints (deploy-eligibility):
  * Platt slope must be >= 0 in every outer fold (a negative slope is a SIGN_INVERSION
    diagnostic, never deployed);
  * Beta must be monotone increasing over the observed input domain;
  * isotonic skipped when inner training support is insufficient;
  * probabilities kept in the frozen epsilon range.

Reads the ONE canonical artifact PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.evaluation.rolling_origin import (  # noqa: E402
    all_chronology_pass,
    expanding_window_folds,
    fold_manifest,
    nested_select,
)
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

app = typer.Typer(add_completion=False)
EPS = 1e-6
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
QUOTE_COVERED = ["pts", "reb", "ast", "fg3m"]
ISO_MIN_SUPPORT = 200
MIN_TRAIN_DATES = 15
VAL_BLOCK_DATES = 3


def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p); return np.log(p / (1 - p))


def _ll(y, p):
    p = _clip(p); return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(y, p):
    return float(np.mean((_clip(p) - np.asarray(y, float)) ** 2))


def _ece(y, p, nb=10):
    y = np.asarray(y, int); p = _clip(p)
    ed = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(p, ed[1:-1]), 0, nb - 1)
    n = len(p); e = 0.0
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


def _cal_slope(y, p):
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, int))
        return float(lr.coef_[0][0])
    except Exception:  # noqa: BLE001
        return float("nan")


# ---- candidate fit on a train frame -> (predictor, meta) ; meta.monotone flags deploy-eligibility
def _fit(name, tr, alpha=None):
    y = tr["outcome_over"].to_numpy(int)
    pm = tr[FINAL_PROBABILITY_COLUMN].to_numpy(float)
    pk = tr["market_prob_over_no_vig"].to_numpy(float)
    meta = {"monotone": True}
    if name == "C0_identity":
        return (lambda d: _clip(d[FINAL_PROBABILITY_COLUMN].to_numpy(float))), meta
    if name == "C1_platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(pm).reshape(-1, 1), y)
        meta["platt_slope"] = float(lr.coef_[0][0]); meta["monotone"] = bool(lr.coef_[0][0] >= 0)
        return (lambda d: _clip(lr.predict_proba(_logit(d[FINAL_PROBABILITY_COLUMN]).reshape(-1, 1))[:, 1])), meta
    if name == "C2_beta":
        X = np.column_stack([np.log(_clip(pm)), np.log(_clip(1 - pm))])
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, y)
        a, b = float(lr.coef_[0][0]), float(lr.coef_[0][1])
        meta["beta_a"], meta["beta_b"] = a, b; meta["monotone"] = bool(a >= 0 and b <= 0)
        return (lambda d: _clip(lr.predict_proba(np.column_stack([
            np.log(_clip(d[FINAL_PROBABILITY_COLUMN])), np.log(_clip(1 - d[FINAL_PROBABILITY_COLUMN]))]))[:, 1])), meta
    if name == "C3_isotonic":
        if len(tr) < ISO_MIN_SUPPORT:
            meta["skipped"] = True
            return None, meta
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pm, y)
        return (lambda d: _clip(iso.predict(_clip(d[FINAL_PROBABILITY_COLUMN].to_numpy(float))))), meta
    if name == "C4_blend":
        a = alpha if alpha is not None else 0.5
        return (lambda d: _clip(a * d[FINAL_PROBABILITY_COLUMN].to_numpy(float)
                                + (1 - a) * d["market_prob_over_no_vig"].to_numpy(float))), meta
    if name == "C5_role_blend":
        alphas = np.linspace(0, 1, 21)
        by_role = {}
        for role, g in tr.groupby("role_bucket"):
            yy = g["outcome_over"].to_numpy(int)
            by_role[role] = min(alphas, key=lambda a: _ll(
                yy, a * g[FINAL_PROBABILITY_COLUMN].to_numpy(float)
                + (1 - a) * g["market_prob_over_no_vig"].to_numpy(float)))
        ga = min(alphas, key=lambda a: _ll(y, a * pm + (1 - a) * pk))

        def _ap(d):
            a = d["role_bucket"].map(lambda r: by_role.get(r, ga)).to_numpy(float)
            return _clip(a * d[FINAL_PROBABILITY_COLUMN].to_numpy(float)
                         + (1 - a) * d["market_prob_over_no_vig"].to_numpy(float))
        return _ap, meta
    if name == "C6_market_residual":
        X = np.column_stack([_logit(pm), _logit(pk)])
        lr = LogisticRegression(C=1.0, solver="lbfgs").fit(X, y)
        return (lambda d: _clip(lr.predict_proba(np.column_stack([
            _logit(d[FINAL_PROBABILITY_COLUMN]), _logit(d["market_prob_over_no_vig"])]))[:, 1])), meta
    raise ValueError(name)


CANDIDATES = ["C0_identity", "C1_platt", "C2_beta", "C3_isotonic",
              "C4_blend", "C5_role_blend", "C6_market_residual"]


def _nested_outer_eval(pdf, name, ece_margin):
    """Expanding-window outer folds; nested alpha selection for C4; leakage-safe."""
    dates = pdf["game_date"].to_numpy()
    folds = expanding_window_folds(dates, min_train_dates=MIN_TRAIN_DATES, val_block_dates=VAL_BLOCK_DATES)
    if not folds:
        return None
    pred = np.full(len(pdf), np.nan)
    monotone_all = True
    per_fold = []
    d = pdf["game_date"].to_numpy()
    for f in folds:
        tr = pdf[np.isin(d, list(f.train_dates))]
        va = pdf[np.isin(d, list(f.val_dates))]
        if len(tr) < 30 or len(va) == 0:
            continue
        alpha = None
        if name == "C4_blend":
            grid = list(np.linspace(0, 1, 21))
            def _score(a, itr, iva):
                t = tr[np.isin(tr["game_date"].to_numpy(), list(itr))]
                v = tr[np.isin(tr["game_date"].to_numpy(), list(iva))]
                if len(t) < 20 or len(v) == 0:
                    return np.inf
                fn, _ = _fit("C4_blend", t, alpha=a)
                return _ll(v["outcome_over"].to_numpy(int), fn(v))
            alpha = nested_select(list(f.train_dates), param_grid=grid, score_fn=_score,
                                  min_train_dates=max(8, MIN_TRAIN_DATES // 2), val_block_dates=VAL_BLOCK_DATES)
        fn, meta = _fit(name, tr, alpha=alpha)
        if fn is None:
            return {"skipped": True}
        if not meta.get("monotone", True):
            monotone_all = False
        idx = np.where(np.isin(d, list(f.val_dates)))[0]
        pv = fn(va)
        pred[idx] = pv
        yv = va["outcome_over"].to_numpy(int); pkv = va["market_prob_over_no_vig"].to_numpy(float)
        per_fold.append({"fold_id": f.fold_id, "n_val": int(len(va)),
                         "logloss_delta": _ll(yv, pv) - _ll(yv, pkv),
                         "brier_delta": _brier(yv, pv) - _brier(yv, pkv)})
    m = np.isfinite(pred)
    if m.sum() == 0:
        return None
    y = pdf["outcome_over"].to_numpy(int)[m]
    pk = pdf["market_prob_over_no_vig"].to_numpy(float)[m]
    pr = pred[m]
    worst_ll = max((r["logloss_delta"] for r in per_fold), default=np.nan)
    worst_br = max((r["brier_delta"] for r in per_fold), default=np.nan)
    return {
        "n": int(m.sum()), "n_folds": len(per_fold),
        "cand_logloss": _ll(y, pr), "market_logloss": _ll(y, pk),
        "cand_brier": _brier(y, pr), "market_brier": _brier(y, pk),
        "cand_ece": _ece(y, pr), "market_ece": _ece(y, pk),
        "cand_auc": _auc(y, pr), "market_auc": _auc(y, pk),
        "cal_slope": _cal_slope(y, pr),
        "worst_fold_logloss_delta": float(worst_ll), "worst_fold_brier_delta": float(worst_br),
        "monotone_deployable": bool(monotone_all),
        "logloss_delta_vs_market": _ll(y, pr) - _ll(y, pk),
        "brier_delta_vs_market": _brier(y, pr) - _brier(y, pk),
    }


@app.command()
def main(
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet", "--scored"),
    out_dir: str = typer.Option("artifacts/market_feature_proof/G0_v2", "--out-dir"),
    ece_margin: float = typer.Option(0.03, "--ece-margin"),
) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(scored)

    # Emit the outer-fold manifest (per prop) + chronology proof.
    manifests = {}
    for prop in QUOTE_COVERED:
        pdf = df[df["prop"] == prop]
        folds = expanding_window_folds(pdf["game_date"].to_numpy(),
                                       min_train_dates=MIN_TRAIN_DATES, val_block_dates=VAL_BLOCK_DATES)
        man = fold_manifest(folds, lambda ds: int(pdf["game_date"].isin(ds).sum()))
        manifests[prop] = {"folds": man, "all_chronology_pass": all_chronology_pass(folds)}
    (outp / "ROLLING_ORIGIN_FOLD_MANIFEST.json").write_text(json.dumps(
        {"version": "rolling-origin-fold-manifest-v1", "min_train_dates": MIN_TRAIN_DATES,
         "val_block_dates": VAL_BLOCK_DATES, "invariant": "max(train_date) < min(validation_date)",
         "per_prop": manifests}, indent=2) + "\n")

    rows = []
    for prop in DIRECT_PROPS:
        pdf = df[df["prop"] == prop].sort_values("game_date").reset_index(drop=True)
        if len(pdf) == 0:
            rows.append({"prop": prop, "candidate": "-", "status": "NO_EXACT_QUOTES"}); continue
        for name in CANDIDATES:
            r = _nested_outer_eval(pdf, name, ece_margin)
            if r is None:
                rows.append({"prop": prop, "candidate": name, "status": "NO_FOLDS"}); continue
            if r.get("skipped"):
                rows.append({"prop": prop, "candidate": name, "status": "SKIPPED_INSUFFICIENT_SUPPORT"}); continue
            advances = bool(r["cand_logloss"] < r["market_logloss"]
                            and r["cand_brier"] < r["market_brier"]
                            and r["cand_ece"] <= r["market_ece"] + ece_margin
                            and r["monotone_deployable"]
                            and r["worst_fold_logloss_delta"] < 0.05)
            rows.append({"prop": prop, "candidate": name, "status": "EVALUATED", **r, "advances": advances})
    res = pd.DataFrame(rows)
    res.to_csv(outp / "LOWCOST_CANDIDATE_METRICS.csv", index=False)
    (outp / "LOWCOST_CANDIDATE_METRICS.json").write_text(json.dumps(
        {"cv": "grouped expanding-window rolling-origin (nested); leakage-safe",
         "scored_artifact": "PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet",
         "advancement_rule": ("aggregate outer-fold logloss<market AND brier<market AND "
                              "ece<=market+margin AND monotone_deployable AND worst_fold_logloss_delta<0.05"),
         "records": res.replace({np.nan: None}).to_dict("records")}, indent=2) + "\n")

    ev = res[res["status"] == "EVALUATED"]
    print(ev[["prop", "candidate", "n", "n_folds", "logloss_delta_vs_market",
              "brier_delta_vs_market", "cand_ece", "cal_slope", "monotone_deployable",
              "worst_fold_logloss_delta", "advances"]].to_string(index=False))

    # closest-prop ranking (corrected, nested)
    summ = []
    for prop in sorted(ev["prop"].unique()):
        g = ev[ev["prop"] == prop]
        adv = g[g["advances"]]
        best = (adv.sort_values("cand_logloss").iloc[0] if len(adv)
                else g.sort_values("logloss_delta_vs_market").iloc[0])
        summ.append({"prop": prop, "has_advancing_candidate": bool(len(adv) > 0),
                     "best_candidate": best["candidate"],
                     "best_logloss_delta_vs_market": float(best["logloss_delta_vs_market"]),
                     "best_brier_delta_vs_market": float(best["brier_delta_vs_market"]),
                     "distance_from_passing": float(max(best["logloss_delta_vs_market"], 0)
                                                    + max(best["brier_delta_vs_market"], 0))})
    sdf = pd.DataFrame(summ).sort_values(["has_advancing_candidate", "distance_from_passing"],
                                         ascending=[False, True])
    sdf.to_csv(outp / "CLOSEST_PROP_RANKING.csv", index=False)
    (outp / "CLOSEST_PROP_RANKING.json").write_text(json.dumps(
        {"cv": "nested expanding-window rolling-origin", "records": sdf.to_dict("records")}, indent=2) + "\n")
    print("\n[closest-prop ranking, corrected nested CV]")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    app()
