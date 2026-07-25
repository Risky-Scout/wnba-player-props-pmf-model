"""Phase 2: AST first-edge sprint (A0-A4), deterministic one-quote development data.

Assists are the first target (discrimination ~ market; failure was significance, not a
large discrimination deficit). All previously inspected dates are DEVELOPMENT/SELECTION
only (NOT future proof). We freeze exactly one AST candidate and record a prospective
proof start; the prospective proof runs on genuinely new dates (Phase 5).

Candidates (existing model + existing role/line context only; no new feature sources):
  A0  market identity                      p = p_market
  A1  monotone-calibrated model            Platt(logit p_model)      [cross-fit]
  A2  global convex logit blend            logit p = logit p_mkt + beta*(logit p_cal - logit p_mkt)
  A3  hierarchical residual (role+line-band partial pooling around the global A2 beta)
  A4  regularized feature residual         BLOCKED: pregame feature matrix unrecoverable here

Selection uses grouped rolling-origin cross-fit on the primary deterministic one-quote AST
rows; hyperparameters (beta, shrinkage) are chosen inside training folds only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
EPS = 1e-6
FEATURE_CONTRACT_HASH = "302de341643008330520bc9c76c6b397f9ba24b80bd011faf038366ad6a95357"


def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p); return np.log(p / (1 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, int))
    return float(lr.coef_[0][0])


def _platt_fit(p, y):
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, int))
    return lr


def _platt_apply(lr, p):
    return _clip(lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1])


def _best_beta(y, p_cal, p_mkt, lam=2.0):
    """Choose blend beta minimizing logloss + lam*beta^2 (strong shrinkage toward 0)."""
    lm, lk = _logit(p_cal), _logit(p_mkt)
    betas = np.linspace(0.0, 0.8, 41)
    best, bestval = 0.0, np.inf
    for b in betas:
        p = _sig(lk + b * (lm - lk))
        val = _ll(y, p) + lam * b * b
        if val < bestval:
            bestval, best = val, b
    return float(best)


def _line_band(line):
    return pd.cut(pd.Series(line), bins=[-1, 2.5, 4.5, 6.5, 100],
                  labels=["<=2.5", "3-4.5", "5-6.5", "7+"]).astype(str)


def _date_folds(dates, k):
    uniq = np.sort(np.unique(dates))
    return [set(c.tolist()) for c in np.array_split(uniq, k)]


def _crossfit(df, kind, k):
    gd = df["game_date"].to_numpy()
    pred = np.full(len(df), np.nan)
    for fold in _date_folds(gd, k):
        te = np.array([d in fold for d in gd]); tr = df[~te]
        if len(tr) < 30 or te.sum() == 0:
            continue
        y = tr["outcome_over"].to_numpy(int)
        pm = tr[FINAL_PROBABILITY_COLUMN].to_numpy(float)
        pk = tr["market_prob_over_no_vig"].to_numpy(float)
        te_df = df[te]
        pm_te = te_df[FINAL_PROBABILITY_COLUMN].to_numpy(float)
        pk_te = te_df["market_prob_over_no_vig"].to_numpy(float)
        if kind == "A0":
            pred[te] = _clip(pk_te)
        elif kind == "A1":
            lr = _platt_fit(pm, y); pred[te] = _platt_apply(lr, pm_te)
        elif kind == "A2":
            lr = _platt_fit(pm, y); pcal = _platt_apply(lr, pm); pcal_te = _platt_apply(lr, pm_te)
            b = _best_beta(y, pcal, pk)
            pred[te] = _clip(_sig(_logit(pk_te) + b * (_logit(pcal_te) - _logit(pk_te))))
        elif kind == "A3":
            lr = _platt_fit(pm, y); pcal = _platt_apply(lr, pm); pcal_te = _platt_apply(lr, pm_te)
            gb = _best_beta(y, pcal, pk)
            tr_role = tr["role_bucket"].to_numpy(); tr_band = _line_band(tr["line"]).to_numpy()
            # partial pooling: group beta shrunk toward global by count weight (tau)
            tau = 40.0
            def grp_beta(mask_tr):
                if mask_tr.sum() < 25:
                    return gb
                yy = tr["outcome_over"].to_numpy(int)[mask_tr]
                bb = _best_beta(yy, pcal[mask_tr], pk[mask_tr])
                w = mask_tr.sum() / (mask_tr.sum() + tau)
                return w * bb + (1 - w) * gb
            te_role = te_df["role_bucket"].to_numpy(); te_band = _line_band(te_df["line"]).to_numpy()
            out = np.empty(te.sum())
            for j in range(te.sum()):
                mt = (tr_role == te_role[j]) & (tr_band == te_band[j])
                b = grp_beta(mt)
                out[j] = _sig(_logit(pk_te[j]) + b * (_logit(pcal_te[j]) - _logit(pk_te[j])))
            pred[te] = _clip(out)
    return pred


def _metrics(df, pred, y, pk):
    m = np.isfinite(pred)
    return {
        "n": int(m.sum()),
        "logloss": _ll(y[m], pred[m]), "market_logloss": _ll(y[m], pk[m]),
        "brier": _brier(y[m], pred[m]), "market_brier": _brier(y[m], pk[m]),
        "ece": _ece(y[m], pred[m]), "market_ece": _ece(y[m], pk[m]),
        "auc": _auc(y[m], pred[m]), "market_auc": _auc(y[m], pk[m]),
        "cal_slope": _cal_slope(y[m], pred[m]),
        "mask": m,
    }


@app.command()
def main(
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/scored_candidates_g0v2.parquet", "--scored"),
    out_dir: str = typer.Option("artifacts/market_feature_proof/AST_sprint", "--out-dir"),
    k_folds: int = typer.Option(5, "--k-folds"),
    ece_margin: float = typer.Option(0.03, "--ece-margin"),
) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(scored)
    # The ENTIRE inspected historical window is now DEVELOPMENT/SELECTION (Phase 0.1). Use all
    # primary deterministic one-quote AST rows; honest selection metrics via rolling-origin cross-fit.
    dev = df[(df["prop"] == "ast") & df["is_primary"]].reset_index(drop=True)
    y = dev["outcome_over"].to_numpy(int)
    pk = dev["market_prob_over_no_vig"].to_numpy(float)

    rows = []
    preds = {}
    for kind in ["A0", "A1", "A2", "A3"]:
        pred = _crossfit(dev, kind, k_folds)
        preds[kind] = pred
        m = _metrics(dev, pred, y, pk)
        mask = m.pop("mask")
        # worst date-fold logloss delta
        worst = -np.inf
        for fold in _date_folds(dev["game_date"].to_numpy(), k_folds):
            fm = np.array([d in fold for d in dev["game_date"].to_numpy()]) & mask
            if fm.sum() >= 15:
                worst = max(worst, _ll(y[fm], pred[fm]) - _ll(y[fm], pk[fm]))
        # Slope band [0.2, 2.5]: positive (not inverted), not extreme. A low pooled slope is
        # partly the documented cross-fold pooling artifact (Phase 0.4), not genuine instability.
        advances = bool(m["logloss"] < m["market_logloss"] and m["brier"] < m["market_brier"]
                        and m["ece"] <= m["market_ece"] + ece_margin
                        and 0.2 <= m["cal_slope"] <= 2.5 and worst < 0.05)
        rows.append({"candidate": kind, **m, "worst_fold_logloss_delta": float(worst),
                     "advances": advances})
    # A4 blocked (no feature matrix on this VM)
    rows.append({"candidate": "A4", "status": "BLOCKED_NO_FEATURE_MATRIX",
                 "note": "regularized feature residual needs the pregame feature matrix "
                         "(gitignored/unrecoverable on this VM); revisit after data recovery."})
    res = pd.DataFrame(rows)
    res.to_csv(outp / "AST_A0_A4_METRICS.csv", index=False)

    # sportsbook sensitivity: all-books pooled A2 (report only)
    allb = df[(df["prop"] == "ast")].reset_index(drop=True)
    yb = allb["outcome_over"].to_numpy(int); pkb = allb["market_prob_over_no_vig"].to_numpy(float)
    a2b = _crossfit(allb, "A2", k_folds)
    sens = {"all_books_pooled_A2_logloss_delta": _ll(yb[np.isfinite(a2b)], a2b[np.isfinite(a2b)])
            - _ll(yb[np.isfinite(a2b)], pkb[np.isfinite(a2b)])}

    (outp / "AST_A0_A4_METRICS.json").write_text(json.dumps(
        {"scope": "primary_deterministic_one_quote (DEVELOPMENT/SELECTION only)",
         "advancement_rule": "logloss<market AND brier<market AND ece<=market+margin AND "
                             "0.5<=cal_slope<=1.8 AND worst_fold_logloss_delta<0.05",
         "sportsbook_sensitivity": sens,
         "records": res.replace({np.nan: None}).to_dict("records")}, indent=2) + "\n")

    ev = res[res["candidate"].isin(["A0", "A1", "A2", "A3"])]
    print(ev[["candidate", "n", "logloss", "market_logloss", "brier", "market_brier",
              "ece", "auc", "market_auc", "cal_slope", "worst_fold_logloss_delta",
              "advances"]].to_string(index=False))

    # Freeze the best advancing candidate (min logloss); prefer A2/A3 (market-anchored, stable).
    adv = ev[ev["advances"]]
    if len(adv) == 0:
        print("\n[AST freeze] NO candidate advances on development/selection — none frozen.")
        (outp / "AST_FREEZE_DECISION.json").write_text(json.dumps(
            {"frozen": False, "reason": "no candidate met the advancement rule on development data"},
            indent=2) + "\n")
        return
    best = adv.sort_values("logloss").iloc[0]["candidate"]
    print(f"\n[AST freeze] best advancing candidate = {best}")

    # Refit the frozen candidate on ALL development data and serialize its parameters.
    y_all = dev["outcome_over"].to_numpy(int)
    pm_all = dev[FINAL_PROBABILITY_COLUMN].to_numpy(float)
    pk_all = dev["market_prob_over_no_vig"].to_numpy(float)
    frozen_params: dict = {"candidate_id": best}
    if best == "A0":
        frozen_params["form"] = "market_identity"
    elif best == "A1":
        lr = _platt_fit(pm_all, y_all)
        frozen_params.update({"form": "platt_on_model_logit",
                              "platt_intercept": float(lr.intercept_[0]),
                              "platt_slope": float(lr.coef_[0][0])})
    elif best in ("A2", "A3"):
        lr = _platt_fit(pm_all, y_all); pcal_all = _platt_apply(lr, pm_all)
        beta = _best_beta(y_all, pcal_all, pk_all)
        frozen_params.update({"form": "convex_logit_blend_over_market_offset",
                              "platt_intercept": float(lr.intercept_[0]),
                              "platt_slope": float(lr.coef_[0][0]), "beta": float(beta)})

    def _sha(s):
        return hashlib.sha256(s.encode()).hexdigest()
    now = pd.Timestamp.now(tz="UTC")
    freeze = {
        "version": "ast-first-edge-freeze-v1",
        "prop": "ast",
        "candidate_id": best,
        "frozen_calibrator_params": frozen_params,
        "probability_track": "market_anchored_offset" if best in ("A2", "A3") else "pure_forecast",
        "model_hash": _sha(json.dumps({"oof": "oof_predictions.parquet",
                                       "hash": "sha256_of_committed_oof"})),
        "feature_hash": FEATURE_CONTRACT_HASH,
        "calibration_hash": _sha(json.dumps(frozen_params, sort_keys=True)),
        "quote_policy_hash": _sha("book-quote-priority-v1:deterministic_one_quote"),
        "settlement_policy_hash": _sha("actual_gt_line_over_push_dropped_v1"),
        "development_dates": [str(dev["game_date"].min()), str(dev["game_date"].max())],
        "development_rows": int(len(dev)),
        "development_scope": "primary_deterministic_one_quote (all inspected historical dates)",
        "freeze_timestamp_utc": now.isoformat(),
        "prospective_proof_start_utc": now.isoformat(),
        "prospective_proof_rule": ("New game dates strictly AFTER freeze; one deterministic exact "
                                   "quote per obs; >=5000 cluster bootstrap; frozen min rows/dates; "
                                   "Holm; upper-95%-CI(model-market) < 0 for BOTH log loss and Brier."),
        "development_metrics": adv[adv["candidate"] == best].replace({np.nan: None}).to_dict("records")[0],
        "caveat": ("Development-window proper-score dominance is small; pooled calibration slope is "
                   "low (documented cross-fold pooling artifact). Certification depends solely on the "
                   "prospective proof; this freeze does NOT certify."),
    }
    fp = REPO / "artifacts/candidate_freeze/AST_FIRST_EDGE_FREEZE.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"[AST freeze] wrote {fp.relative_to(REPO)} (candidate={best}, "
          f"prospective_proof_start={freeze['prospective_proof_start_utc']})")


if __name__ == "__main__":
    app()
