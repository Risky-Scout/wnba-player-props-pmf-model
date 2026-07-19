"""P3 - promote validated pts_reb and pts_reb_ast combos into the immutable champion package.

Surgical: preserves the seven already-validated markets byte-for-byte in
config/certified_forecast_calibration.json; adds only the two combo-residual markets
(hierarchical empirical combo-residual PMF centered on the sum of the validated component
expectations + frozen prequential dispersion scale). Refreshes shared calibration/registry/
policy/manifest hashes consistently. Components are never retrained or recalibrated.
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
from wnba_props_model.evaluation import distribution_calibration as dc  # noqa: E402
from scripts.p3_combo_repair import _load_components, _minbucket, COMBO_PARTS  # noqa: E402

app = typer.Typer(add_completion=False)
# Frozen prequential dispersion scales (final blocks; see artifacts/p3/p3_combo_repair_result.json)
COMBO_SCALE = {"pts_reb": 0.97, "pts_reb_ast": 0.95}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cells_to_json(cells: dict) -> dict:
    return {k: {"vals": [int(x) for x in v[0]], "wts": [float(x) for x in v[1]],
                "shift": float(v[2])} for k, v in cells.items()}


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         result: str = typer.Option("artifacts/p3/p3_combo_repair_result.json"),
         cal_out: str = typer.Option("config/certified_forecast_calibration.json"),
         registry_out: str = typer.Option("config/stat_registry.json"),
         manifest_out: str = typer.Option("config/champion_manifest.json"),
         policy_path: str = typer.Option("config/recommendation_policy.yaml")) -> None:
    _df, rows = _load_components(oof, cal_out)
    calib = json.loads(Path(cal_out).read_text())
    res = json.loads(Path(result).read_text())

    for combo in ("pts_reb", "pts_reb_ast"):
        assert combo not in calib["markets"], f"{combo} already present"
        parts = COMBO_PARTS[combo]
        recs = [{"role_bucket": d["role_bucket"], "_minbucket": _minbucket(d["minutes_mean"]),
                 "_point": float(sum(d["mean"][p] for p in parts)),
                 "actual_outcome": int(round(sum(d["actual"][p] for p in parts)))}
                for d in rows.values() if all(p in d["pmf"] for p in parts)]
        cells = dc.fit_residual_hist(pd.DataFrame(recs))
        calib["markets"][combo] = {"method": "combo_residual", "parts": parts,
                                   "cells": _cells_to_json(cells), "scale": COMBO_SCALE[combo],
                                   "certified": True}
    Path(cal_out).write_text(json.dumps(calib, indent=2))
    cal_hash = _sha(calib)

    registry = json.loads(Path(registry_out).read_text())
    for combo in ("pts_reb", "pts_reb_ast"):
        r = res[combo]
        registry[combo].update({
            "forecast_allowed": True, "forecast_method": "combo_residual",
            "crps": r["crps"], "crps_vs_baseline": r["crps_vs_baseline"],
            "calibration_hash": cal_hash, "suppression_reason": "",
        })
    for m, e in registry.items():
        if e.get("forecast_allowed"):
            e["calibration_hash"] = cal_hash
    Path(registry_out).write_text(json.dumps(registry, indent=2))
    registry_hash = _sha(registry)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()[:16]
    manifest = json.loads(Path(manifest_out).read_text())
    cert = sorted([m for m, e in registry.items() if e.get("forecast_allowed")])
    manifest.update({
        "certified_markets": cert, "certified_stats": cert,
        "calibration_hash": cal_hash, "registry_hash": registry_hash,
        "policy_hash": policy_hash, "code_commit": commit, "github_sha": commit,
        "combo_ledger_hashes": {"pts_reb": res["pts_reb"]["ledger_hash"],
                                "pts_reb_ast": res["pts_reb_ast"]["ledger_hash"]},
    })
    Path(manifest_out).write_text(json.dumps(manifest, indent=2))
    typer.echo(f"[combos] added cal_hash={cal_hash} registry_hash={registry_hash} certified={cert}")


if __name__ == "__main__":
    app()
