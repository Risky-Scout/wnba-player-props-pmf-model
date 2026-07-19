"""P3 — promote validated Rebounds (Candidate D) into the immutable champion package.

Surgical: preserves the six already-validated markets byte-for-byte in
config/certified_forecast_calibration.json; adds only `reb` (hierarchical empirical residual
PMF + frozen prequential dispersion scale). Refreshes the shared calibration hash and updates
config/stat_registry.json + config/champion_manifest.json consistently.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import distribution_calibration as dc  # noqa: E402

app = typer.Typer(add_completion=False)
REB_SCALE = 0.9   # frozen prequential dispersion scale (final blocks selected 0.9)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cells_to_json(cells: dict) -> dict:
    return {k: {"vals": [int(x) for x in v[0]], "wts": [float(x) for x in v[1]],
                "shift": float(v[2])} for k, v in cells.items()}


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         result: str = typer.Option("artifacts/p3/p3_reb_repair_result.json"),
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

    calib = json.loads(Path(cal_out).read_text())
    assert "reb" not in calib["markets"], "reb already present"
    reb_cells = dc.fit_residual_hist(df[df["stat"] == "reb"])
    calib["markets"]["reb"] = {"method": "hierarchical", "cells": _cells_to_json(reb_cells),
                               "scale": REB_SCALE, "certified": True}
    Path(cal_out).write_text(json.dumps(calib, indent=2))
    cal_hash = _sha(calib)

    res = json.loads(Path(result).read_text())
    registry = json.loads(Path(registry_out).read_text())
    certified = [m for m, e in registry.items() if e.get("forecast_allowed")] + ["reb"]
    registry["reb"].update({
        "forecast_allowed": True, "forecast_method": "hierarchical",
        "crps": res["crps"], "crps_vs_baseline": res["crps_vs_baseline"],
        "calibration_hash": cal_hash, "suppression_reason": "",
    })
    # Refresh the shared calibration hash for every certified market (artifact changed; PMFs
    # for the six existing markets are byte-identical — only reb was added).
    for m, e in registry.items():
        if e.get("forecast_allowed"):
            e["calibration_hash"] = cal_hash
    Path(registry_out).write_text(json.dumps(registry, indent=2))
    registry_hash = _sha(registry)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()[:16]
    manifest = json.loads(Path(manifest_out).read_text())
    cert_stats = sorted(set(certified))
    manifest.update({
        "certified_markets": cert_stats, "certified_stats": cert_stats,
        "calibration_hash": cal_hash, "registry_hash": registry_hash,
        "policy_hash": policy_hash, "ledger_hash": res["ledger_hash"],
        "code_commit": commit, "github_sha": commit,
    })
    Path(manifest_out).write_text(json.dumps(manifest, indent=2))
    typer.echo(f"[reb] added scale={REB_SCALE} cal_hash={cal_hash} registry_hash={registry_hash} "
               f"certified={cert_stats}")


if __name__ == "__main__":
    app()
