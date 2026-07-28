#!/usr/bin/env python3
"""Phase 8A/8B - pure PMF-shape/calibration sprint on the atomic replay universe (OUTCOMES ONLY).

For a given prop, take the base OOF PMFs (artifacts/models/calibration/oof_predictions.parquet), fit a
family of pure, PMF-coherent, outcome-only transforms with EXPANDING rolling-origin folds (each row's
transform params come only from STRICTLY EARLIER folds -> still out-of-fold), then evaluate P(over line)
on the migrated atomic decision-snapshot pairs (canonical scored universe) and compare to the exact
same-book same-line no-vig market probability.

Candidates (no market info, no line-specific calibrators, no outer-fold tuning):
  A0 base                       - current OOF PMF
  A1 temperature/dispersion     - p(k)^tau renormalised (tau fit by training log-lik)
  A2 zero-mass hurdle           - scale P(0) by m, rescale positive tail (m fit by training log-lik)
  A3 exponential tilt           - p(k)*exp(a k + b k^2) renormalised (a,b fit by training log-lik)
  A4 monotone-CDF calibration   - isotonic map of model CDF -> empirical, PMF by CDF differences
  A5 nonnegative ensemble       - simplex weights over {A0,A1,A2,A3,A4} by training log-lik

Screen: candidate must improve A0 on LL, Brier, ECE, |cal intercept|, |cal slope-1|, CRPS and full-PMF
log score, with AUC not materially worse. Winners get the full frozen market-superiority contract with a
10k date-cluster bootstrap + Holm-adjusted one-sided p-values for LL and Brier.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
OOF = REPO / "artifacts/models/calibration/oof_predictions.parquet"
PAIRS = REPO / "data/processed/atomic_quotes/atomic_pairs.parquet"
BOOK_PRIORITY = ["draftkings", "fanduel", "betonlineag", "williamhill_us", "betrivers"]
EPS = 1e-12


# ---------------------------------------------------------------- PMF utilities
def _parse_pmf(js: str, K: int) -> np.ndarray:
    d = json.loads(js)
    v = np.zeros(K + 1)
    for k, p in d.items():
        ik = int(k)
        if 0 <= ik <= K:
            v[ik] = float(p)
    s = v.sum()
    return v / s if s > 0 else v


def _p_over(pmf: np.ndarray, line: float) -> float:
    thr = int(np.floor(line)) + 1          # over line => Y >= thr
    return float(pmf[thr:].sum()) if thr <= len(pmf) - 1 else 0.0


def _cdf(pmf: np.ndarray) -> np.ndarray:
    return np.clip(np.cumsum(pmf), 0.0, 1.0)


def _novig(over_odds: float, under_odds: float) -> float:
    def imp(o):
        return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)
    po, pu = imp(over_odds), imp(under_odds)
    return po / (po + pu)


# ------------------------------------------------------------- transform fitters
def _fit_temperature(P, y):
    def nll(t):
        lg = t * np.log(P + EPS)
        lg -= logsumexp(lg, axis=1, keepdims=True)
        return -np.mean(lg[np.arange(len(y)), y])
    r = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
    return float(r.x)


def _apply_temperature(P, t):
    lg = t * np.log(P + EPS)
    lg -= logsumexp(lg, axis=1, keepdims=True)
    return np.exp(lg)


def _fit_zeromass(P, y):
    def nll(m):
        Q = P.copy()
        p0 = np.clip(P[:, 0] * m, 1e-9, 1 - 1e-9)
        pos = P[:, 1:].sum(axis=1, keepdims=True)
        scale = np.where(pos.ravel() > 0, (1 - p0) / pos.ravel(), 0.0)
        Q[:, 1:] = P[:, 1:] * scale[:, None]
        Q[:, 0] = p0
        Q = Q / Q.sum(axis=1, keepdims=True)
        return -np.mean(np.log(Q[np.arange(len(y)), y] + EPS))
    r = minimize_scalar(nll, bounds=(0.2, 3.0), method="bounded")
    return float(r.x)


def _apply_zeromass(P, m):
    Q = P.copy()
    p0 = np.clip(P[:, 0] * m, 1e-9, 1 - 1e-9)
    pos = P[:, 1:].sum(axis=1, keepdims=True)
    scale = np.where(pos.ravel() > 0, (1 - p0) / pos.ravel(), 0.0)
    Q[:, 1:] = P[:, 1:] * scale[:, None]
    Q[:, 0] = p0
    return Q / Q.sum(axis=1, keepdims=True)


def _fit_tilt(P, y):
    K = P.shape[1] - 1
    k = np.arange(K + 1)
    logP = np.log(P + EPS)

    def nll(ab):
        a, b = ab
        lg = logP + a * k + b * (k ** 2)
        lg -= logsumexp(lg, axis=1, keepdims=True)
        return -np.mean(lg[np.arange(len(y)), y])
    r = minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead",
                 options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6})
    return float(r.x[0]), float(r.x[1])


def _apply_tilt(P, a, b):
    K = P.shape[1] - 1
    k = np.arange(K + 1)
    lg = np.log(P + EPS) + a * k + b * (k ** 2)
    lg -= logsumexp(lg, axis=1, keepdims=True)
    return np.exp(lg)


def _fit_isotonic(P, y):
    K = P.shape[1] - 1
    F = np.cumsum(P, axis=1)
    thr = np.arange(K + 1)
    X = F.ravel()
    Y = (y[:, None] <= thr[None, :]).astype(float).ravel()
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(X, Y)
    return iso


def _apply_isotonic(P, iso):
    F = np.cumsum(P, axis=1)
    Fc = iso.predict(F.ravel()).reshape(F.shape)
    Fc = np.maximum.accumulate(Fc, axis=1)          # monotone nondecreasing in threshold
    Fc[:, -1] = 1.0
    Q = np.diff(Fc, axis=1, prepend=0.0)
    Q = np.clip(Q, 0.0, None)
    s = Q.sum(axis=1, keepdims=True)
    return np.where(s > 0, Q / s, P)


def _fit_ensemble(cand_P, y):
    # nonnegative simplex weights maximising training log-lik of the mixture
    idx = np.arange(len(y))
    cols = [Pc[idx, y] for Pc in cand_P]              # each (n,)
    M = np.vstack(cols).T                             # (n, n_cand)

    def nll(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + EPS)
        return -np.mean(np.log(M @ w + EPS))
    nc = len(cand_P)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bnds = [(0.0, 1.0)] * nc
    r = minimize(nll, x0=np.full(nc, 1.0 / nc), method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 200})
    w = np.clip(r.x, 0, None)
    return w / (w.sum() + EPS)


# ------------------------------------------------------------------- metrics
def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _cal_slope_intercept(p, y):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _crps(P, y):
    F = np.cumsum(P, axis=1)
    thr = np.arange(P.shape[1])
    ind = (y[:, None] <= thr[None, :]).astype(float)
    return float(np.mean(((F - ind) ** 2).sum(axis=1)))


def _full_pmf_logscore(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + EPS)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prop = args.prop
    out = Path(args.out) if args.out else REPO / f"artifacts/pure_model_completion/SPRINT_{prop.upper()}_PMF_SHAPE.json"

    oof = pd.read_parquet(OOF)
    d = oof[oof.stat == prop].copy()
    K = int(d.pmf_support_max.max())
    d["pmf_vec"] = d.pmf_json.map(lambda j: _parse_pmf(j, K))
    d = d.sort_values(["fold_id", "game_date"]).reset_index(drop=True)
    P = np.vstack(d.pmf_vec.values)
    y = d.actual_outcome.astype(int).clip(0, K).values
    folds = d.fold_id.values

    # ---- expanding rolling-origin: each fold's rows get params fit on strictly earlier folds ----
    uniq = sorted(np.unique(folds))
    cand_names = ["A0", "A1_temp", "A2_zeromass", "A3_tilt", "A4_isocdf", "A5_ensemble"]
    out_P = {c: P.copy() for c in cand_names}          # A0 stays base
    for f in uniq:
        tr = folds < f
        va = folds == f
        if tr.sum() < 200:                              # too little history -> identity (== A0)
            continue
        Ptr, ytr = P[tr], y[tr]
        t = _fit_temperature(Ptr, ytr); out_P["A1_temp"][va] = _apply_temperature(P[va], t)
        m = _fit_zeromass(Ptr, ytr); out_P["A2_zeromass"][va] = _apply_zeromass(P[va], m)
        a, b = _fit_tilt(Ptr, ytr); out_P["A3_tilt"][va] = _apply_tilt(P[va], a, b)
        iso = _fit_isotonic(Ptr, ytr); out_P["A4_isocdf"][va] = _apply_isotonic(P[va], iso)
        base_cands = [_apply_temperature(Ptr, t), _apply_zeromass(Ptr, m),
                      _apply_tilt(Ptr, a, b), _apply_isotonic(Ptr, iso), Ptr]
        w = _fit_ensemble(base_cands, ytr)
        va_cands = [out_P["A1_temp"][va], out_P["A2_zeromass"][va], out_P["A3_tilt"][va],
                    out_P["A4_isocdf"][va], P[va]]
        out_P["A5_ensemble"][va] = sum(wi * Pc for wi, Pc in zip(w, va_cands))

    # ---- canonical atomic universe: one row per (game,player,line), frozen book priority ----
    pairs = pd.read_parquet(PAIRS)
    a = pairs[(pairs.prop == prop) & (pairs.snapshot_label == "decision")
              & (pairs.binary_settled_eligible)].copy()
    a["book_rank"] = a.sportsbook.map({b: i for i, b in enumerate(BOOK_PRIORITY)}).fillna(99)
    a = a.sort_values("book_rank").groupby(["game_id", "player_id"], as_index=False).first()
    a["market_prob_over_no_vig"] = [_novig(o, u) for o, u in zip(a.over_odds, a.under_odds)]
    a["y"] = (a.outcome == "over").astype(int)
    for c in ("game_id", "player_id"):
        a[c] = a[c].astype(str); d[c] = d[c].astype(str)
    d = d.reset_index(drop=True)
    row_idx = {(g, p): i for i, (g, p) in enumerate(zip(d.game_id, d.player_id))}
    a["ri"] = [row_idx.get((g, p), -1) for g, p in zip(a.game_id, a.player_id)]
    a = a[a.ri >= 0].reset_index(drop=True)
    ri = a.ri.values
    line = a.line.values.astype(float)
    yb = a.y.values
    gdate = pd.to_datetime(a.game_date).dt.date.astype(str).values

    # ---- score every candidate on the canonical universe ----
    def _scores(Pc):
        Psub = Pc[ri]
        pov = np.clip(np.array([_p_over(Psub[i], line[i]) for i in range(len(ri))]), 1e-6, 1 - 1e-6)
        ysel = y[ri]
        sl, ic = _cal_slope_intercept(pov, yb)
        return {
            "LL": float(log_loss(yb, pov, labels=[0, 1])),
            "Brier": float(brier_score_loss(yb, pov)),
            "AUC": float(roc_auc_score(yb, pov)) if len(np.unique(yb)) > 1 else None,
            "ECE": _ece(pov, yb), "cal_slope": sl, "cal_intercept": ic,
            "CRPS": _crps(Psub, ysel), "full_pmf_logscore": _full_pmf_logscore(Psub, ysel),
        }, pov

    market = np.clip(a.market_prob_over_no_vig.values, 1e-6, 1 - 1e-6)
    mkt = {"LL": float(log_loss(yb, market, labels=[0, 1])), "Brier": float(brier_score_loss(yb, market)),
           "AUC": float(roc_auc_score(yb, market)) if len(np.unique(yb)) > 1 else None}

    results, pov_by = {}, {}
    for c in cand_names:
        results[c], pov_by[c] = _scores(out_P[c])

    a0 = results["A0"]
    def _screen(r):
        return (r["LL"] < a0["LL"] and r["Brier"] < a0["Brier"] and r["ECE"] < a0["ECE"]
                and abs(r["cal_intercept"]) < abs(a0["cal_intercept"]) + 1e-9
                and abs(r["cal_slope"] - 1) < abs(a0["cal_slope"] - 1) + 1e-9
                and r["CRPS"] < a0["CRPS"] + 1e-9 and r["full_pmf_logscore"] < a0["full_pmf_logscore"]
                and (r["AUC"] is None or r["AUC"] >= a0["AUC"] - 0.01))
    screened = {c: _screen(results[c]) for c in cand_names if c != "A0"}

    # ---- full contract for the best screened winner (10k date-cluster bootstrap + Holm) ----
    def _bootstrap_contract(pov):
        dates = np.unique(gdate)
        rng = np.random.default_rng(20260728)
        by = {dt: np.where(gdate == dt)[0] for dt in dates}
        dll = np.empty(10000); dbr = np.empty(10000)
        for i in range(10000):
            samp = np.concatenate([by[dt] for dt in rng.choice(dates, len(dates), replace=True)])
            ys, ps, ms = yb[samp], pov[samp], market[samp]
            dll[i] = log_loss(ys, ps, labels=[0, 1]) - log_loss(ys, ms, labels=[0, 1])
            dbr[i] = brier_score_loss(ys, ps) - brier_score_loss(ys, ms)
        # model better => delta<0. one-sided p = P(delta>=0)
        p_ll = float(np.mean(dll >= 0)); p_br = float(np.mean(dbr >= 0))
        # Holm across the 2-test family
        ps_sorted = sorted([("LL", p_ll), ("Brier", p_br)], key=lambda x: x[1])
        holm = {}
        for rank, (name, pv) in enumerate(ps_sorted):
            holm[name] = min(1.0, pv * (2 - rank))
        return {
            "delta_LL_mean": float(dll.mean()), "delta_LL_ci95_upper": float(np.percentile(dll, 97.5)),
            "delta_Brier_mean": float(dbr.mean()), "delta_Brier_ci95_upper": float(np.percentile(dbr, 97.5)),
            "holm_p_LL": holm["LL"], "holm_p_Brier": holm["Brier"],
        }

    winners = [c for c, ok in screened.items() if ok]
    contract = {}
    for c in winners:
        bc = _bootstrap_contract(pov_by[c])
        passes = (results[c]["LL"] < mkt["LL"] and results[c]["Brier"] < mkt["Brier"]
                  and bc["delta_LL_ci95_upper"] < 0 and bc["delta_Brier_ci95_upper"] < 0
                  and bc["holm_p_LL"] <= 0.05 and bc["holm_p_Brier"] <= 0.05)
        contract[c] = {**bc, "passes_full_contract": bool(passes)}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "prop": prop,
        "canonical_universe": {"rows": int(len(a)), "dates": int(len(np.unique(gdate))),
                               "book_policy": "one row per (game,player,line); priority " + ">".join(BOOK_PRIORITY),
                               "meets_min_rows_dates": bool(len(a) >= 300 and len(np.unique(gdate)) >= 30)},
        "market": mkt, "candidate_metrics": results, "screen_pass": screened,
        "full_contract": contract,
        "any_prop_passes": any(v.get("passes_full_contract") for v in contract.values()),
        "verdict": ("PASS" if any(v.get("passes_full_contract") for v in contract.values())
                    else "NO_PASS"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=str)
    print(json.dumps({"prop": prop, "n": int(len(a)), "market_LL": round(mkt["LL"], 4),
                      "A0_LL": round(a0["LL"], 4),
                      "best": {c: round(results[c]["LL"], 4) for c in cand_names},
                      "screened_winners": winners,
                      "contract": {c: contract[c]["passes_full_contract"] for c in winners},
                      "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
