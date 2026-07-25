"""Low-cost probability-correction candidates C0-C6 (owner critical-path step 5).

Evaluates cheap corrections that do NOT touch the core predictive model:

  C0  existing model probability (identity)
  C1  Platt scaling                       (logistic on logit(p_model))
  C2  Beta calibration                    (logistic on [log p, log(1-p)])
  C3  isotonic regression                 (only when selection support is sufficient)
  C4  convex model-market blend           (single alpha per prop)
  C5  prop-role blend                     (alpha per role bucket)
  C6  regularized market-residual stack   (L2 logistic on logit(p_model)+logit(p_market))

Development discipline (owner): rolling-origin DEVELOPMENT/SELECTION DATA ONLY. Selection
rows are split into K contiguous date-grouped folds; every candidate is CROSS-FITTED
(fit on the other folds, predict the held-out fold) so the reported selection metric is an
honest out-of-fold estimate — the untouched test/proof window is never read here.

Advancement rule (both must hold, plus acceptable ECE and no severe subgroup instability):
    candidate log loss < EXACT market log loss  AND  candidate Brier < EXACT market Brier
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
from wnba_props_model.models.probability_contract import FINAL_PROBABILITY_COLUMN  # noqa: E402

app = typer.Typer(add_completion=False)
EPS = 1e-6
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
ISO_MIN_SUPPORT = 200


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


def _ece(y, p, n_bins=10):
    y = np.asarray(y, int); p = _clip(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


# ----- candidate fit/apply: each returns a predictor callable fit on a train frame -----
def _fit_candidate(name: str, tr: pd.DataFrame):
    y = tr["outcome_over"].to_numpy(int)
    pm = tr[FINAL_PROBABILITY_COLUMN].to_numpy(float)
    pk = tr["market_prob_over_no_vig"].to_numpy(float)

    if name == "C0_identity":
        return lambda d: _clip(d[FINAL_PROBABILITY_COLUMN].to_numpy(float))

    if name == "C1_platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(pm).reshape(-1, 1), y)
        return lambda d: lr.predict_proba(_logit(d[FINAL_PROBABILITY_COLUMN]).reshape(-1, 1))[:, 1]

    if name == "C2_beta":
        X = np.column_stack([np.log(_clip(pm)), np.log(_clip(1 - pm))])
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(X, y)
        return lambda d: lr.predict_proba(
            np.column_stack([np.log(_clip(d[FINAL_PROBABILITY_COLUMN])),
                             np.log(_clip(1 - d[FINAL_PROBABILITY_COLUMN]))]))[:, 1]

    if name == "C3_isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pm, y)
        return lambda d: iso.predict(_clip(d[FINAL_PROBABILITY_COLUMN].to_numpy(float)))

    if name == "C4_blend":
        alphas = np.linspace(0, 1, 41)
        best = min(alphas, key=lambda a: _ll(y, a * pm + (1 - a) * pk))
        return lambda d: (best * d[FINAL_PROBABILITY_COLUMN].to_numpy(float)
                          + (1 - best) * d["market_prob_over_no_vig"].to_numpy(float))

    if name == "C5_role_blend":
        alphas = np.linspace(0, 1, 41)
        by_role = {}
        for role, g in tr.groupby("role_bucket"):
            yy = g["outcome_over"].to_numpy(int)
            a = min(alphas, key=lambda a: _ll(yy, a * g[FINAL_PROBABILITY_COLUMN].to_numpy(float)
                                              + (1 - a) * g["market_prob_over_no_vig"].to_numpy(float)))
            by_role[role] = a
        global_a = min(alphas, key=lambda a: _ll(y, a * pm + (1 - a) * pk))

        def _apply(d):
            a = d["role_bucket"].map(lambda r: by_role.get(r, global_a)).to_numpy(float)
            return a * d[FINAL_PROBABILITY_COLUMN].to_numpy(float) + \
                (1 - a) * d["market_prob_over_no_vig"].to_numpy(float)
        return _apply

    if name == "C6_market_residual":
        X = np.column_stack([_logit(pm), _logit(pk)])
        lr = LogisticRegression(C=1.0, solver="lbfgs")  # L2-regularized stack
        lr.fit(X, y)
        return lambda d: lr.predict_proba(np.column_stack([
            _logit(d[FINAL_PROBABILITY_COLUMN]), _logit(d["market_prob_over_no_vig"])]))[:, 1]

    raise ValueError(name)


CANDIDATES = ["C0_identity", "C1_platt", "C2_beta", "C3_isotonic",
              "C4_blend", "C5_role_blend", "C6_market_residual"]


def _date_folds(dates: np.ndarray, k: int) -> list:
    uniq = np.sort(np.unique(dates))
    return [set(c.tolist()) for c in np.array_split(uniq, k)]


def _crossfit(prop_df: pd.DataFrame, name: str, k: int) -> "np.ndarray | None":
    folds = _date_folds(prop_df["game_date"].to_numpy(), k)
    pred = np.full(len(prop_df), np.nan)
    gd = prop_df["game_date"].to_numpy()
    for fold in folds:
        te = np.array([d in fold for d in gd])
        tr = prop_df[~te]
        if len(tr) < 30 or te.sum() == 0:
            continue
        if name == "C3_isotonic" and len(tr) < ISO_MIN_SUPPORT:
            return None
        fn = _fit_candidate(name, tr)
        pred[te] = _clip(fn(prop_df[te]))
    return pred if not np.isnan(pred).all() else None


@app.command()
def main(
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/scored_candidates_g0v2.parquet", "--scored"),
    out_dir: str = typer.Option("artifacts/market_feature_proof/G0_v2", "--out-dir"),
    primary_book: str = typer.Option("primary", "--primary-book",
                                     help="'primary' = frozen deterministic one-quote per obs "
                                          "(is_primary); 'all' = all-books pooled SENSITIVITY; or a "
                                          "specific book name."),
    k_folds: int = typer.Option(5, "--k-folds"),
    ece_margin: float = typer.Option(0.03, "--ece-margin",
                                     help="Candidate ECE must be <= market ECE + margin."),
) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(scored)
    sel = df[df["split"] == "selection"].copy()
    if primary_book == "primary":
        sel = sel[sel["is_primary"]].copy()
    elif primary_book != "all":
        sel = sel[sel["book"] == primary_book].copy()

    rows = []
    for prop in DIRECT_PROPS:
        pdf = sel[sel["prop"] == prop].reset_index(drop=True)
        if len(pdf) == 0:
            rows.append({"prop": prop, "candidate": "-", "n": 0, "status": "NO_EXACT_QUOTES"})
            continue
        y = pdf["outcome_over"].to_numpy(int)
        pk = pdf["market_prob_over_no_vig"].to_numpy(float)
        mkt = {"market_logloss": _ll(y, pk), "market_brier": _brier(y, pk),
               "market_ece": _ece(y, pk), "market_auc": _auc(y, pk)}
        for name in CANDIDATES:
            pred = _crossfit(pdf, name, k_folds)
            if pred is None:
                rows.append({"prop": prop, "candidate": name, "n": int(len(pdf)),
                             "status": "SKIPPED_INSUFFICIENT_SUPPORT", **mkt})
                continue
            m = np.isfinite(pred)
            cll, cbr = _ll(y[m], pred[m]), _brier(y[m], pred[m])
            cece, cauc = _ece(y[m], pred[m]), _auc(y[m], pred[m])
            advances = bool(cll < mkt["market_logloss"] and cbr < mkt["market_brier"]
                            and cece <= mkt["market_ece"] + ece_margin)
            rows.append({
                "prop": prop, "candidate": name, "n": int(m.sum()),
                "n_dates": int(pdf.loc[m, "game_date"].nunique()),
                "cand_logloss": cll, "cand_brier": cbr, "cand_ece": cece, "cand_auc": cauc,
                **mkt,
                "logloss_delta_vs_market": cll - mkt["market_logloss"],
                "brier_delta_vs_market": cbr - mkt["market_brier"],
                "advances": advances,
                "status": "EVALUATED",
            })
    res = pd.DataFrame(rows)
    res.to_csv(outp / "LOWCOST_CANDIDATE_METRICS.csv", index=False)
    (outp / "LOWCOST_CANDIDATE_METRICS.json").write_text(
        json.dumps({"primary_book": primary_book, "k_folds": k_folds,
                    "advancement_rule": "cand_logloss<market_logloss AND cand_brier<market_brier AND cand_ece<=market_ece+margin",
                    "ece_margin": ece_margin,
                    "records": res.replace({np.nan: None}).to_dict("records")}, indent=2) + "\n")

    ev = res[res["status"] == "EVALUATED"].copy()
    print(ev[["prop", "candidate", "n", "cand_logloss", "market_logloss",
              "logloss_delta_vs_market", "cand_brier", "brier_delta_vs_market",
              "cand_ece", "advances"]].to_string(index=False))
    # best advancing candidate per prop + distance-from-passing ranking (selection).
    summary = []
    for prop in sorted(ev["prop"].unique()):
        g = ev[ev["prop"] == prop]
        adv = g[g["advances"]]
        best = (adv.sort_values("cand_logloss").iloc[0] if len(adv)
                else g.sort_values("logloss_delta_vs_market").iloc[0])
        summary.append({
            "prop": prop, "has_advancing_candidate": bool(len(adv) > 0),
            "best_candidate": best["candidate"],
            "best_logloss_delta_vs_market": float(best["logloss_delta_vs_market"]),
            "best_brier_delta_vs_market": float(best["brier_delta_vs_market"]),
            "distance_from_passing": float(max(best["logloss_delta_vs_market"], 0)
                                           + max(best["brier_delta_vs_market"], 0)),
        })
    summ = pd.DataFrame(summary).sort_values("distance_from_passing")
    summ.to_csv(outp / "CLOSEST_PROP_RANKING.csv", index=False)
    (outp / "CLOSEST_PROP_RANKING.json").write_text(json.dumps(
        {"ranking_metric": "distance_from_passing = max(dLL,0)+max(dBrier,0) on cross-fit selection",
         "records": summ.to_dict("records")}, indent=2) + "\n")
    print("\n[closest-prop ranking on selection]")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    app()
