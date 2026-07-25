"""Phase 3 A4: targeted regularized feature-residual repair for PTS / FG3M.

Previously BLOCKED_NO_FEATURE_MATRIX. With the pregame feature matrix recovered/rebuilt
(``data/processed/wnba_player_game_features_wide.parquet``, model_feature_columns, leakage
guard PASS), this fits the A4 learner the low-cost recalibration candidates cannot: a
regularized model that may add *discrimination* (ordering) on top of the existing model and
the market, using ONLY point-in-time pregame features.

Evaluation is the SAME leakage-safe nested expanding-window rolling-origin used for C0-C6.
For every outer fold: standardize + median-impute features on the outer-train dates only,
nested-select the L2 strength C on inner expanding folds, refit on the full outer-train
block, and score once on the untouched outer-validation block. Aggregate = concatenated
outer-validation predictions (never trained on future).

Reports both gates transparently:
    proper_score : aggregate logloss<market AND brier<market AND ece ok AND worst fold ok
    strict_auc   : additionally aggregate AUC > market AUC  (the PTS/FG3M deficit is
                   discrimination, so this is the gate that matters for a genuine repair)

No freeze is written unless a prop is proper_score_selection_eligible; the exact deficit is
recorded otherwise. This introduces no new model architecture — it is the sanctioned A4
residual over existing pregame features.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lowcost_candidates import (  # noqa: E402
    MIN_TRAIN_DATES,
    VAL_BLOCK_DATES,
    _auc,
    _brier,
    _cal_slope,
    _clip,
    _ece,
    _ll,
    _logit,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.evaluation.rolling_origin import (  # noqa: E402
    expanding_window_folds,
    nested_select,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"
ECE_MARGIN = 0.03
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
FINAL = "model_prob_over_final"
MKT = "market_prob_over_no_vig"


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _design(frame: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Feature-residual design: standardized pregame features + model & market logits."""
    X = frame[feat_cols].to_numpy(float)
    base = _logit(frame[FINAL].to_numpy(float)).reshape(-1, 1)
    mkt = _logit(frame[MKT].to_numpy(float)).reshape(-1, 1)
    return np.hstack([X, base, mkt])


def _fit_predict(tr: pd.DataFrame, va: pd.DataFrame, feat_cols: list[str], C: float):
    """Median-impute + standardize on TRAIN only, ridge-logit, predict on VA."""
    Xtr = _design(tr, feat_cols)
    Xva = _design(va, feat_cols)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xva = np.where(np.isfinite(Xva), Xva, med)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    y = tr["outcome_over"].to_numpy(int)
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    lr.fit(Xtr, y)
    return _clip(lr.predict_proba(Xva)[:, 1])


def _nested_a4(pdf: pd.DataFrame, feat_cols: list[str]) -> dict | None:
    dates = pdf["game_date"].to_numpy()
    folds = expanding_window_folds(dates, min_train_dates=MIN_TRAIN_DATES, val_block_dates=VAL_BLOCK_DATES)
    if not folds:
        return None
    pred = np.full(len(pdf), np.nan)
    d = pdf["game_date"].to_numpy()
    per_fold = []
    for f in folds:
        tr = pdf[np.isin(d, list(f.train_dates))]
        va = pdf[np.isin(d, list(f.val_dates))]
        if len(tr) < 40 or len(va) == 0:
            continue

        def _score(C, itr, iva, _tr=tr):
            t = _tr[np.isin(_tr["game_date"].to_numpy(), list(itr))]
            v = _tr[np.isin(_tr["game_date"].to_numpy(), list(iva))]
            if len(t) < 30 or len(v) == 0:
                return np.inf
            p = _fit_predict(t, v, feat_cols, C)
            return _ll(v["outcome_over"].to_numpy(int), p)

        C = nested_select(list(f.train_dates), param_grid=C_GRID, score_fn=_score,
                          min_train_dates=max(8, MIN_TRAIN_DATES // 2), val_block_dates=VAL_BLOCK_DATES)
        pv = _fit_predict(tr, va, feat_cols, C)
        idx = np.where(np.isin(d, list(f.val_dates)))[0]
        pred[idx] = pv
        yv = va["outcome_over"].to_numpy(int)
        pkv = va[MKT].to_numpy(float)
        per_fold.append({"fold_id": f.fold_id, "n_val": int(len(va)), "C": float(C),
                         "logloss_delta": _ll(yv, pv) - _ll(yv, pkv),
                         "brier_delta": _brier(yv, pv) - _brier(yv, pkv)})
    m = np.isfinite(pred)
    if m.sum() == 0:
        return None
    y = pdf["outcome_over"].to_numpy(int)[m]
    pk = pdf[MKT].to_numpy(float)[m]
    pb = pdf[FINAL].to_numpy(float)[m]
    pr = pred[m]
    worst_ll = max((r["logloss_delta"] for r in per_fold), default=np.nan)
    proper = bool(_ll(y, pr) < _ll(y, pk) and _brier(y, pr) < _brier(y, pk)
                  and _ece(y, pr) <= _ece(y, pk) + ECE_MARGIN and worst_ll < 0.05)
    strict_auc = bool(proper and _auc(y, pr) > _auc(y, pk))
    return {
        "n": int(m.sum()), "n_folds": len(per_fold),
        "cand_logloss": _ll(y, pr), "market_logloss": _ll(y, pk), "base_model_logloss": _ll(y, pb),
        "cand_brier": _brier(y, pr), "market_brier": _brier(y, pk),
        "cand_ece": _ece(y, pr), "market_ece": _ece(y, pk),
        "cand_auc": _auc(y, pr), "market_auc": _auc(y, pk), "base_model_auc": _auc(y, pb),
        "cal_slope": _cal_slope(y, pr),
        "logloss_delta_vs_market": _ll(y, pr) - _ll(y, pk),
        "brier_delta_vs_market": _brier(y, pr) - _brier(y, pk),
        "auc_delta_vs_market": _auc(y, pr) - _auc(y, pk),
        "auc_gain_vs_base_model": _auc(y, pr) - _auc(y, pb),
        "worst_fold_logloss_delta": float(worst_ll),
        "selected_C_by_fold": [r["C"] for r in per_fold],
        "proper_score_selection_eligible": proper,
        "strict_auc_selection_eligible": strict_auc,
    }


@app.command()
def main(
    scored: str = typer.Option("artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet", "--scored"),
    features: str = typer.Option("data/processed/wnba_player_game_features_wide.parquet", "--features"),
    manifest: str = typer.Option("data/processed/feature_schema_manifest.json", "--manifest"),
    props: str = typer.Option("pts,fg3m", "--props"),
) -> None:
    G0.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(scored)
    feat = pd.read_parquet(features)
    man = json.loads(Path(manifest).read_text())
    feat_cols = [c for c in man["model_feature_columns"] if c in feat.columns]

    fk = feat.copy()
    fk["_gid"] = fk["game_id"].astype(str)
    fk["_pid"] = fk["player_id"].astype(str)
    fk = fk[["_gid", "_pid", *feat_cols]].drop_duplicates(["_gid", "_pid"])

    out = {
        "version": "phase3-a4-feature-residual-v1",
        "cv": "corrected nested expanding-window rolling-origin (leakage-safe)",
        "a4_status": "UNBLOCKED_FEATURE_MATRIX_RECOVERED",
        "learner": ("L2 logistic; standardized median-imputed pregame features + model logit + "
                    "market logit; C nested-selected per outer fold"),
        "feature_matrix": features,
        "feature_matrix_sha256": _sha_file(features),
        "n_model_features": len(feat_cols),
        "props": {},
    }
    for prop in [p.strip() for p in props.split(",") if p.strip()]:
        pdf = sc[sc["prop"] == prop].copy()
        pdf["_gid"] = pdf["game_id"].astype(str)
        pdf["_pid"] = pdf["player_id"].astype(str)
        j = pdf.merge(fk, on=["_gid", "_pid"], how="inner").sort_values("game_date").reset_index(drop=True)
        joined = len(j)
        if joined == 0:
            out["props"][prop] = {"status": "BLOCKED_NO_FEATURE_JOIN"}
            continue
        r = _nested_a4(j, feat_cols)
        if r is None:
            out["props"][prop] = {"status": "NO_FOLDS", "joined_rows": joined}
            continue
        r["status"] = "EVALUATED"
        r["joined_rows"] = joined
        r["repair_beats_market"] = r["proper_score_selection_eligible"]
        r["adds_discrimination_vs_market"] = bool(r["auc_delta_vs_market"] > 0)
        out["props"][prop] = r

    (G0 / "PHASE3_A4_FEATURE_RESIDUAL.json").write_text(json.dumps(out, indent=2) + "\n")
    for prop, r in out["props"].items():
        if r.get("status") != "EVALUATED":
            print(f"{prop:5s} {r.get('status')}")
            continue
        print(f"{prop:5s} n={r['n']} folds={r['n_folds']} "
              f"dLL={r['logloss_delta_vs_market']:+.4f} dBrier={r['brier_delta_vs_market']:+.4f} "
              f"dAUC_mkt={r['auc_delta_vs_market']:+.4f} AUC(cand/base/mkt)="
              f"{r['cand_auc']:.3f}/{r['base_model_auc']:.3f}/{r['market_auc']:.3f} "
              f"proper={r['proper_score_selection_eligible']} strict_auc={r['strict_auc_selection_eligible']}")


if __name__ == "__main__":
    app()
