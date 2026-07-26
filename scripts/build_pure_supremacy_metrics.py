"""PHASE 2 (P1) -- pure PMF mean/variance + probability recalibration; new OOF metrics.

Pure model only (market_probability_weight=0.0; the line is used only as a post-PMF query
threshold). Two model-only levers, both asserted market-free by the PHASE 0 contract:

  * de-DNP (mean repair): the delivered probability is the DNP-blended PMF P(over line); every
    over line is >= 0.5 so the over region is entirely in k>0, which the blend scaled by
    (1 - p_dnp). Dividing by (1 - p_dnp) recovers the pure conditional-on-play P(over) and
    removes the systematic under-projection the PHASE 1 decomposition attributes to availability.
  * pure Platt (binary recalibration): a monotone 1-parameter logistic map fit out-of-fold on
    (logit(P_over_deDNP), outcome_over) only -- no market, cannot manufacture discrimination.

Evaluation: grouped expanding-window nested rolling-origin CV (leakage-safe; max(train)<min(val)).
BEFORE = P0 current pure delivered probability; AFTER = P1 (de-DNP + pure Platt). Reports per
prop n / LL / market LL / dLL / Brier / market Brier / dBrier / AUC / market AUC / ECE / market
ECE / worst-fold dLL / monotone / genuine_pure_win against the exact no-vig market. stl/blk/
turnover carry no exact quotes and are reported NO_EXACT_QUOTES (development blocked on odds).

DEVELOPMENT selection evidence only; does not touch prospective-proof / promotion / edge gates.
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
        "genuine_pure_win": bool(_ll(y, pr) < _ll(y, pk) and _brier(y, pr) < _brier(y, pk)
                                 and worst < WORST_FOLD_LL_MAX
                                 and _ece(y, pr) <= _ece(y, pk) + ECE_MARGIN and mono),
    }


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
        per_prop[prop] = {"before_P0": before, "after_P1_deDNP_platt": after,
                          "genuine_pure_win": bool(after and after["genuine_pure_win"])}
        for tag, r in [("BEFORE_P0", before), ("AFTER_P1", after)]:
            if r:
                csv_rows.append({"prop": prop, "stage": tag, **{k: r[k] for k in (
                    "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                    "model_brier", "market_brier", "brier_delta", "model_auc", "market_auc",
                    "model_ece", "market_ece", "worst_fold_logloss_delta", "monotone",
                    "genuine_pure_win")}})

    winners = [p for p in QUOTE_COVERED if per_prop[p].get("genuine_pure_win")]
    (outp / "PURE_SUPREMACY_OOF_METRICS.json").write_text(json.dumps({
        "version": "pure-supremacy-oof-metrics-v1",
        "zero_market_blending": True,
        "pure_prediction_inputs_allowed": sorted(PURE_ALLOWED),
        "before_pipeline": "P0 current pure delivered probability",
        "after_pipeline": "P1 = de-DNP (P/(1-p_dnp)) -> out-of-fold pure Platt (model-vs-outcome only)",
        "cv": "grouped expanding-window nested rolling-origin (max(train)<min(val))",
        "win_rule": ("dLL<0 AND dBrier<0 AND worst_fold_dLL<%.2f AND model_ece<=market_ece+%.2f "
                     "AND monotone" % (WORST_FOLD_LL_MAX, ECE_MARGIN)),
        "evidence_class": "DEVELOPMENT_SELECTION_EVIDENCE (NOT prospective proof; gates unchanged)",
        "fold_manifest": manifests, "per_prop": per_prop, "genuine_pure_wins": winners,
    }, indent=2) + "\n")
    pd.DataFrame(csv_rows).to_csv(outp / "PURE_SUPREMACY_OOF_METRICS.csv", index=False)

    print("\n=== PHASE 2 P1: pure BEFORE vs AFTER vs exact no-vig market (nested CV) ===")
    print(f"{'prop':5s} {'stage':10s} {'n':>4s} {'mLL':>7s} {'kLL':>7s} {'dLL':>8s} {'dBr':>8s} "
          f"{'mAUC':>6s} {'kAUC':>6s} {'mECE':>6s} {'win':>5s}")
    for prop in QUOTE_COVERED:
        for tag, key in [("BEFORE", "before_P0"), ("AFTER", "after_P1_deDNP_platt")]:
            r = per_prop[prop][key]
            if r:
                print(f"{prop:5s} {tag:10s} {r['n']:>4d} {r['model_logloss']:>7.4f} {r['market_logloss']:>7.4f} "
                      f"{r['logloss_delta']:>+8.4f} {r['brier_delta']:>+8.4f} {r['model_auc']:>6.3f} "
                      f"{r['market_auc']:>6.3f} {r['model_ece']:>6.3f} {str(r['genuine_pure_win'])[:5]:>5s}")
    print(f"\nGENUINE PURE WINS (LL & Brier, robust): {winners or 'none'}")
    print("stl/blk/turnover: NO_EXACT_QUOTES (market comparison blocked on odds collection)")


if __name__ == "__main__":
    app()
