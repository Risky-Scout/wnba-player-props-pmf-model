"""P3 — mint the multi-stat champion package for the markets validated prequentially.

Per-market production method (identical transforms/functions used in validation → parity):
  * turnover -> location recalibration (per-role delta, scale=1);
  * pts, ast, stl -> hierarchical empirical residual PMF (role x minutes-bucket cells);
  * stocks -> correlated combo of (stl, blk) calibrated components;
  * pts_ast -> correlated combo of (pts, ast) calibrated components.
blk's hierarchical cells are stored ONLY as a stocks component (blk direct is not certified).
Writes config/certified_forecast_calibration.json (methods+artifacts), config/stat_registry.json,
config/champion_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation.pmf_recalibration import _fit_factors  # noqa: E402
from wnba_props_model.evaluation import distribution_calibration as dc  # noqa: E402
from wnba_props_model.models.simulation import estimate_oof_correlations  # noqa: E402

app = typer.Typer(add_completion=False)
HIER = ["pts", "ast", "stl", "blk"]          # blk = stocks component only
COMBOS = {"stocks": ["stl", "blk"], "pts_ast": ["pts", "ast"]}
CERTIFIED = ["turnover", "pts", "ast", "stl", "stocks", "pts_ast"]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cells_to_json(cells: dict) -> dict:
    return {k: {"vals": [int(x) for x in v[0]], "wts": [float(x) for x in v[1]],
                "shift": float(v[2])} for k, v in cells.items()}


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         result: str = typer.Option("artifacts/p3/p3_full_recovery_report.json"),
         cal_out: str = typer.Option("config/certified_forecast_calibration.json"),
         registry_out: str = typer.Option("config/stat_registry.json"),
         manifest_out: str = typer.Option("config/champion_manifest.json"),
         policy_path: str = typer.Option("config/recommendation_policy.yaml")) -> None:
    df = pd.read_parquet(oof)
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    df["_point"] = df["pmf_mean"].astype(float)
    mm = df["minutes_mean"].astype(float) if "minutes_mean" in df else pd.Series(0.0, index=df.index)
    df["_minbucket"] = pd.cut(mm, bins=[-1, 10, 20, 28, 100], labels=["m0", "m1", "m2", "m3"]).astype(str)

    calib = {"version": 2, "markets": {}}
    # turnover: location
    s = df[df["stat"] == "turnover"]
    d, sc = _fit_factors(s, dispersion=False)
    by_role = {}
    for role, g in s.groupby("role_bucket"):
        f = _fit_factors(g, dispersion=False)
        by_role[str(role)] = [round(f[0], 6), 1.0]
    calib["markets"]["turnover"] = {"method": "location", "pooled": [round(d, 6), 1.0],
                                    "by_role": by_role}
    # hierarchical stats (pts, ast, stl, blk-as-component)
    for stat in HIER:
        cells = dc.fit_residual_hist(df[df["stat"] == stat])
        calib["markets"][stat] = {"method": "hierarchical", "cells": _cells_to_json(cells),
                                  "certified": stat in CERTIFIED}
    # combos
    corr = estimate_oof_correlations(df)
    for combo, parts in COMBOS.items():
        calib["markets"][combo] = {"method": "combo", "parts": parts,
                                   "correlations": {k: float(v) for k, v in corr.items()}}
    Path(cal_out).write_text(json.dumps(calib, indent=2))
    cal_hash = _sha(calib)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    feat = "schema_v2"
    result_data = json.loads(Path(result).read_text()) if Path(result).exists() else {}
    all_markets = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                   "pts_reb", "pts_ast", "pts_reb_ast", "stocks"]
    registry = {}
    for m in all_markets:
        allowed = m in CERTIFIED
        method = (calib["markets"].get(m, {}).get("method")
                  or ("location" if m == "turnover" else None)) if allowed else None
        rr = result_data.get(m, {})
        registry[m] = {
            "forecast_allowed": bool(allowed),
            "forecast_method": method if allowed else None,
            "market_comparison_allowed": False,
            "betting_recommendation_allowed": False,
            "crps": rr.get("crps"), "crps_vs_baseline": rr.get("crps_vs_baseline"),
            "calibration_hash": cal_hash if allowed else None,
            "feature_schema": feat, "code_commit": commit,
            "validation_window": "2026 full-season OOF; strictly-prequential 5x5-date holdout",
            "suppression_reason": "" if allowed else (rr.get("suppression_reason")
                                                      or "not validated (pending structural retrain)"),
        }
    Path(registry_out).write_text(json.dumps(registry, indent=2))
    registry_hash = _sha(registry)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()[:16]

    ledger_hash = _sha(result_data) if result_data else _sha({"certified": CERTIFIED, "cal": cal_hash})
    manifest = {
        "champion": "schema_v2 multi-stat (location + hierarchical-empirical + correlated-combo)",
        "certified_markets": CERTIFIED,
        "certified_stats": CERTIFIED,
        "github_sha": os.environ.get("GITHUB_SHA", "") or commit,
        "code_commit": commit, "feature_schema": feat, "feature_hash": _sha({"schema": feat}),
        "model_hash": os.environ.get("MODEL_HASH", "schema_v2_structural"),
        "calibration_artifact": cal_out, "calibration_hash": cal_hash,
        "registry_hash": registry_hash, "policy_hash": policy_hash,
        "ledger_hash": ledger_hash,
        "status": "LIVE_VALIDATED_FORECAST_ONLY",
    }
    Path(manifest_out).write_text(json.dumps(manifest, indent=2))
    typer.echo(f"[P3] certified={CERTIFIED} cal_hash={cal_hash} registry_hash={registry_hash} commit={commit}")


if __name__ == "__main__":
    app()
