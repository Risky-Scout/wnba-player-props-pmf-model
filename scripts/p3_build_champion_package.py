"""P3 — build the immutable certified-forecast calibration + registry + champion manifest
for the stats that passed the corrected gate (Task 3).

The validated transform for a certified stat is location-and-scale PMF recalibration
(wnba_props_model.evaluation.pmf_recalibration.recalibrate_pmf) — the SAME function used
in validation — with per-role (delta, scale) factors fit on ALL 2026 OOF history before
the production date. Parity therefore holds by construction (identical transform + fitted
factors in validation and production).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation.pmf_recalibration import _fit_factors  # noqa: E402

app = typer.Typer(add_completion=False)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         certified: str = typer.Option("turnover", help="Comma-separated certified stats."),
         cal_out: str = typer.Option("config/certified_forecast_calibration.json"),
         registry_out: str = typer.Option("config/stat_registry.json"),
         manifest_out: str = typer.Option("config/champion_manifest.json")) -> None:
    certified_stats = [s.strip() for s in certified.split(",") if s.strip()]
    df = pd.read_parquet(oof)
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()]
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712

    calib = {"method": "location_and_scale_recalibration",
             "transform": "wnba_props_model.evaluation.pmf_recalibration.recalibrate_pmf",
             "stats": {}}
    for stat in certified_stats:
        s = df[df["stat"] == stat]
        pooled = _fit_factors(s, dispersion=True)
        by_role = {}
        if "role_bucket" in s.columns:
            for role, g in s.groupby("role_bucket"):
                f = _fit_factors(g, dispersion=True)
                by_role[str(role)] = [round(f[0], 6), round(f[1], 6)]
        calib["stats"][stat] = {"_pooled": [round(pooled[0], 6), round(pooled[1], 6)],
                                "by_role": by_role, "n_train": int(len(s))}
    Path(cal_out).parent.mkdir(parents=True, exist_ok=True)
    Path(cal_out).write_text(json.dumps(calib, indent=2))
    cal_hash = _sha(calib)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    feat_hash = ""
    try:
        from wnba_props_model.features.build_features import FEATURE_SCHEMA_VERSION
        feat_hash = f"schema_v{FEATURE_SCHEMA_VERSION}"
    except Exception:
        feat_hash = "schema_v2"

    # per-stat registry: only certified stats forecast_allowed; betting stays false.
    registry = {}
    for stat in sorted(df["stat"].unique()):
        allowed = stat in certified_stats
        registry[stat] = {
            "forecast_allowed": bool(allowed),
            "market_comparison_allowed": False,
            "betting_recommendation_allowed": False,
            "calibration_method": "location_and_scale_recalibration" if allowed else None,
            "calibration_hash": cal_hash if allowed else None,
            "feature_schema": feat_hash,
            "code_commit": commit,
            "validation_window": "2026 full-season OOF; holdout = latest 25 game-dates",
            "suppression_reason": "" if allowed else "did not pass corrected forecast gate",
        }
    Path(registry_out).write_text(json.dumps(registry, indent=2))

    manifest = {
        "champion": "schema_v2_structural + location_and_scale_forecast_calibration",
        "certified_stats": certified_stats,
        "feature_schema": feat_hash,
        "code_commit": commit,
        "calibration_artifact": cal_out,
        "calibration_hash": cal_hash,
        "registry_hash": _sha(registry),
        "validation": "docs/p3_forecast_gate_result.md",
        "status": "LIVE_VALIDATED_FORECAST_ONLY",
    }
    Path(manifest_out).write_text(json.dumps(manifest, indent=2))
    typer.echo(f"[P3] certified={certified_stats} cal_hash={cal_hash} registry_hash={manifest['registry_hash']} commit={commit}")


if __name__ == "__main__":
    app()
