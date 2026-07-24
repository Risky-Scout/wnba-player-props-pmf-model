"""W0.5 - fit binary P(over) calibrators with rolling-origin, complete-date CV.

Supersedes the earlier KFold approach (which trained each validation block on later
dates - temporally leaky). For each stat, we compare identity / Platt / Beta / isotonic
using grouped rolling-origin folds (train dates STRICTLY BEFORE validation dates; whole
game-dates kept together) and ship a calibrator ONLY if it improves BOTH mean CV log loss
AND mean CV Brier vs identity. Every one of the seven props gets an explicit policy entry
(identity when nothing advances). Fitting uses ONLY the selection window; integer pushes
and non-eligible OOF rows are excluded.

Usage:
    python3 scripts/fit_binary_prob_calibrators.py \\
        --oof data/oof/oof_player_stat_pmfs.parquet \\
        --closing artifacts/p1/p1_closing_consensus.parquet \\
        --split-date 2026-07-10 \\
        --out-dir artifacts/models/calibration \\
        --policy-out config/binary_calibration_policy_v1.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.models.binary_calibrators import CALIBRATOR_FAMILIES  # noqa: E402
from wnba_props_model.models.market import (  # noqa: E402
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)
KEYS = ["game_id", "player_id", "stat"]
ALL_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
_EPS = 1e-6


def _logloss(y, p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def _brier(y, p):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def _settled_p_over(pmf_json, line):
    try:
        return settled_probabilities_from_pmf(json_to_pmf(pmf_json), float(line)).p_over_settled
    except UndefinedSettledProbabilityError:
        return None


def _rolling_origin_folds(dates_sorted, n_folds):
    """Return list of (train_dates, val_dates) with train STRICTLY before val (expanding)."""
    uniq = list(dates_sorted)
    if len(uniq) < n_folds + 1:
        return []
    blocks = np.array_split(np.arange(len(uniq)), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.concatenate(blocks[:k])
        val_idx = blocks[k]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append(([uniq[i] for i in train_idx], [uniq[i] for i in val_idx]))
    return folds


def _cv_scores(g: pd.DataFrame, family_cls, n_folds):
    """Mean rolling-origin CV (logloss, brier) for a calibrator family on one stat."""
    dates = np.sort(g["_d"].dropna().unique())
    folds = _rolling_origin_folds(dates, n_folds)
    if not folds:
        return None
    lls, brs = [], []
    for train_dates, val_dates in folds:
        tr = g[g["_d"].isin(train_dates)]
        va = g[g["_d"].isin(val_dates)]
        if len(tr) < 30 or len(va) == 0 or tr["over_outcome"].nunique() < 2:
            continue
        cal = family_cls().fit(tr["p_over_raw"].to_numpy(), tr["over_outcome"].to_numpy())
        pv = cal.predict(va["p_over_raw"].to_numpy().reshape(-1, 1))
        yv = va["over_outcome"].to_numpy()
        lls.append(_logloss(yv, pv)); brs.append(_brier(yv, pv))
    if not lls:
        return None
    return float(np.mean(lls)), float(np.mean(brs))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.command()
def fit(
    oof: str = typer.Option("data/oof/oof_player_stat_pmfs.parquet", "--oof"),
    closing: str = typer.Option("artifacts/p1/p1_closing_consensus.parquet", "--closing"),
    split_date: str = typer.Option(..., "--split-date"),
    out_dir: str = typer.Option("artifacts/models/calibration", "--out-dir"),
    policy_out: str = typer.Option("config/binary_calibration_policy_v1.json", "--policy-out"),
    selection_out: str = typer.Option("artifacts/models/calibration/binary_calibrator_selection.json",
                                       "--selection-out"),
    cv_folds: int = typer.Option(4, "--cv-folds"),
    min_rows: int = typer.Option(120, "--min-rows"),
    min_improvement: float = typer.Option(1e-4, "--min-improvement"),
) -> None:
    import joblib
    oof_p, cl_p = Path(oof), Path(closing)
    if not oof_p.exists() or not cl_p.exists():
        typer.echo(f"[FATAL] missing inputs oof={oof_p.exists()} closing={cl_p.exists()}", err=True)
        raise typer.Exit(1)
    oofd = pd.read_parquet(oof_p).dropna(subset=["pmf_json", "actual_outcome"]).copy()
    # W0.5 eligibility filters: model_oof only, calibration-eligible, no prior_only/failed.
    if "oof_prediction_type" in oofd.columns:
        oofd = oofd[oofd["oof_prediction_type"] == "model_oof"]
    if "calibration_eligible" in oofd.columns:
        oofd = oofd[oofd["calibration_eligible"] == True]  # noqa: E712
    if "fit_status" in oofd.columns:
        oofd = oofd[~oofd["fit_status"].isin(["prior_only", "failed_model_fit"])]
    cc = pd.read_parquet(cl_p)
    for k in KEYS:
        oofd[k] = oofd[k].astype("string"); cc[k] = cc[k].astype("string")
    gd = "game_date" if "game_date" in oofd.columns else None
    df = cc.merge(oofd[KEYS + ["pmf_json", "actual_outcome"] + ([gd] if gd else [])],
                  on=KEYS, how="inner")
    if df.empty:
        typer.echo("[FATAL] no OOF<->closing overlap.", err=True); raise typer.Exit(1)
    if gd is None:
        df["game_date"] = pd.to_datetime(df["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["p_over_raw"] = [_settled_p_over(pj, ln) for pj, ln in zip(df["pmf_json"], df["line"])]
    df = df[df["p_over_raw"].notna()].copy()
    df = df[df["actual_outcome"].astype(float) != df["line"].astype(float)].copy()  # exclude pushes
    df["over_outcome"] = (df["actual_outcome"].astype(float) > df["line"].astype(float)).astype(int)
    df["_d"] = pd.to_datetime(df["game_date"], errors="coerce")
    selection = df[df["_d"] < pd.Timestamp(split_date)].sort_values("_d")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    artifacts, policy_props, report = {}, {}, {}
    for stat in ALL_PROPS:
        g = selection[selection["stat"] == stat]
        rec = {"n_selection": int(len(g)), "positives": int(g["over_outcome"].sum()) if len(g) else 0}
        if len(g) < min_rows or g["over_outcome"].nunique() < 2:
            policy_props[stat] = {"method": "identity"}
            rec["decision"] = "identity_insufficient_data"; report[stat] = rec; continue
        base = _cv_scores(g, CALIBRATOR_FAMILIES["identity"], cv_folds)
        rec["cv_identity"] = base
        best_family, best_ll = None, None
        cand_scores = {}
        for fam in ("platt", "beta", "isotonic"):
            sc = _cv_scores(g, CALIBRATOR_FAMILIES[fam], cv_folds)
            cand_scores[fam] = sc
            if sc is None or base is None:
                continue
            d_ll, d_br = sc[0] - base[0], sc[1] - base[1]
            # Advance only if BOTH log loss AND Brier strictly improve vs identity.
            if d_ll < -min_improvement and d_br < -min_improvement:
                if best_ll is None or sc[0] < best_ll:
                    best_family, best_ll = fam, sc[0]
        rec["cv_candidates"] = cand_scores
        if best_family is None:
            policy_props[stat] = {"method": "identity"}
            rec["decision"] = "identity_no_improvement"; report[stat] = rec; continue
        # Fit final on ALL selection rows for this stat, ship the artifact.
        model = CALIBRATOR_FAMILIES[best_family]().fit(
            g["p_over_raw"].to_numpy(), g["over_outcome"].to_numpy())
        art = out / f"binary_{best_family}_{stat}.pkl"
        joblib.dump(model, art)
        sha = _sha256(art)
        policy_props[stat] = {"method": best_family, "path": str(art).replace("\\", "/"), "sha256": sha}
        artifacts[stat] = policy_props[stat]
        rec["decision"] = f"ship_{best_family}"; report[stat] = rec

    policy = {
        "version": "binary-cal-v1",
        "enabled": bool(artifacts),
        "cv": "rolling_origin_complete_date",
        "advance_rule": "delta_logloss<0 AND delta_brier<0",
        "fit_split_date": split_date,
        "props": policy_props,           # explicit entry for ALL seven props
        "artifacts": {s: {"path": a["path"], "sha256": a["sha256"]}
                      for s, a in artifacts.items()},
    }
    Path(policy_out).parent.mkdir(parents=True, exist_ok=True)
    Path(policy_out).write_text(json.dumps(policy, indent=2) + "\n")
    Path(selection_out).parent.mkdir(parents=True, exist_ok=True)
    Path(selection_out).write_text(json.dumps(report, indent=2, default=str) + "\n")
    typer.echo(f"[binary-cal] selection rows={len(selection):,} split={split_date} folds={cv_folds}")
    for stat in ALL_PROPS:
        typer.echo(f"  {stat:9s}: {report[stat]['decision']}")
    typer.echo(f"[binary-cal] enabled={policy['enabled']} shipped={list(artifacts)} -> {policy_out}")


if __name__ == "__main__":
    app()
