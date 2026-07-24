"""Phase A.1 — fit binary P(over) calibrators against the scored betting objective.

The delivery lineage's push-safe settled P(over) is calibrated to over/under log loss
here (NOT PMF-shape calibration). Fitting uses ONLY the selection window so the proof's
forward-test window stays pristine; the per-stat winner is chosen by chronological
cross-validated log loss and a calibrator is shipped ONLY if it strictly beats identity
(else that stat stays identity). Integer-line pushes are excluded (they neither win nor
lose an over/under bet).

Artifacts are sklearn IsotonicRegression models (out_of_bounds='clip'), whose
`.predict([[p]])` contract matches BinaryCalibrationRegistry. A versioned policy file and
a selection report are written so the SAME registry loads in both delivery and the proof.

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
from wnba_props_model.models.market import (  # noqa: E402
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)
KEYS = ["game_id", "player_id", "stat"]
_EPS = 1e-6


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def _settled_p_over(pmf_json, line: float) -> float | None:
    try:
        return settled_probabilities_from_pmf(json_to_pmf(pmf_json), float(line)).p_over_settled
    except UndefinedSettledProbabilityError:
        return None


def _cv_logloss_isotonic(p: np.ndarray, y: np.ndarray, folds: int) -> tuple[float, float]:
    """Chronological (no-shuffle) CV: mean val log loss for identity vs isotonic.
    Rows are assumed pre-sorted by date. Returns (identity_ll, isotonic_ll)."""
    from sklearn.isotonic import IsotonicRegression
    n = len(p)
    if n < folds * 2:
        return float("nan"), float("nan")
    idx = np.array_split(np.arange(n), folds)
    id_lls, iso_lls = [], []
    for k in range(folds):
        val = idx[k]
        train = np.concatenate([idx[j] for j in range(folds) if j != k])
        if len(np.unique(y[train])) < 2 or len(val) == 0:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[train], y[train])
        id_lls.append(_logloss(y[val], p[val]))
        iso_lls.append(_logloss(y[val], iso.predict(p[val])))
    if not iso_lls:
        return float("nan"), float("nan")
    return float(np.mean(id_lls)), float(np.mean(iso_lls))


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
    split_date: str = typer.Option(..., "--split-date",
                                   help="Selection = games strictly before this date (fit here only)."),
    out_dir: str = typer.Option("artifacts/models/calibration", "--out-dir"),
    policy_out: str = typer.Option("config/binary_calibration_policy_v1.json", "--policy-out"),
    selection_out: str = typer.Option("artifacts/models/calibration/binary_calibrator_selection.json",
                                       "--selection-out"),
    cv_folds: int = typer.Option(4, "--cv-folds"),
    min_rows: int = typer.Option(80, "--min-rows", help="Min selection rows/stat to fit a calibrator."),
    min_improvement: float = typer.Option(1e-4, "--min-improvement",
                                          help="Min CV log-loss reduction to ship a calibrator."),
) -> None:
    from sklearn.isotonic import IsotonicRegression
    import joblib

    oof_p, cl_p = Path(oof), Path(closing)
    if not oof_p.exists() or not cl_p.exists():
        typer.echo(f"[FATAL] missing inputs: oof={oof_p.exists()} closing={cl_p.exists()}", err=True)
        raise typer.Exit(1)

    oofd = pd.read_parquet(oof_p).dropna(subset=["pmf_json", "actual_outcome"]).copy()
    cc = pd.read_parquet(cl_p)
    for k in KEYS:
        oofd[k] = oofd[k].astype("string")
        cc[k] = cc[k].astype("string")
    opt = [c for c in ("role_bucket",) if c in oofd.columns]
    gd = "game_date" if "game_date" in oofd.columns else None
    df = cc.merge(oofd[KEYS + ["pmf_json", "actual_outcome"] + ([gd] if gd else []) + opt],
                  on=KEYS, how="inner")
    if df.empty:
        typer.echo("[FATAL] no OOF<->closing overlap.", err=True)
        raise typer.Exit(1)
    if gd is None:
        df["game_date"] = pd.to_datetime(df["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")

    df["p_over_raw"] = [_settled_p_over(pj, ln) for pj, ln in zip(df["pmf_json"], df["line"])]
    df = df[df["p_over_raw"].notna()].copy()
    # Push-exclude: only integer lines can push (actual == line).
    df = df[df["actual_outcome"].astype(float) != df["line"].astype(float)].copy()
    df["over_outcome"] = (df["actual_outcome"].astype(float) > df["line"].astype(float)).astype(int)
    df["_d"] = pd.to_datetime(df["game_date"], errors="coerce")
    selection = df[df["_d"] < pd.Timestamp(split_date)].sort_values("_d")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    report: dict[str, dict] = {}
    for stat, g in selection.groupby("stat"):
        p = g["p_over_raw"].to_numpy(float)
        y = g["over_outcome"].to_numpy(int)
        rec = {"n_selection": int(len(g)), "positives": int(y.sum())}
        if len(g) < min_rows or len(np.unique(y)) < 2:
            rec["decision"] = "identity_insufficient_data"
            report[str(stat)] = rec
            continue
        id_ll, iso_ll = _cv_logloss_isotonic(p, y, cv_folds)
        rec.update({"cv_logloss_identity": id_ll, "cv_logloss_isotonic": iso_ll})
        if not np.isfinite(iso_ll) or (id_ll - iso_ll) < min_improvement:
            rec["decision"] = "identity_no_improvement"
            report[str(stat)] = rec
            continue
        # Ship: fit final isotonic on ALL selection rows for this stat.
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        art = out / f"binary_iso_{stat}.pkl"
        joblib.dump(model, art)
        artifacts[str(stat)] = {"path": str(art).replace("\\", "/"), "sha256": _sha256(art)}
        rec["decision"] = "ship_isotonic"
        rec["artifact"] = str(art)
        report[str(stat)] = rec

    policy = {
        "version": "binary-cal-v1",
        "enabled": bool(artifacts),
        "allow_role_fallback_to_prop": True,
        "fit_split_date": split_date,
        "fit_window": "selection (< split_date), chronological CV",
        "artifacts": artifacts,
    }
    Path(policy_out).parent.mkdir(parents=True, exist_ok=True)
    Path(policy_out).write_text(json.dumps(policy, indent=2) + "\n")
    Path(selection_out).parent.mkdir(parents=True, exist_ok=True)
    Path(selection_out).write_text(json.dumps(report, indent=2) + "\n")

    typer.echo(f"[binary-cal] selection rows={len(selection):,} split={split_date}")
    for stat, rec in sorted(report.items()):
        typer.echo(f"  {stat:5s}: {rec['decision']}"
                   + (f" (cv LL {rec['cv_logloss_identity']:.4f}->{rec['cv_logloss_isotonic']:.4f})"
                      if 'cv_logloss_isotonic' in rec else ""))
    typer.echo(f"[binary-cal] enabled={policy['enabled']} artifacts={list(artifacts)} -> {policy_out}")


if __name__ == "__main__":
    app()
