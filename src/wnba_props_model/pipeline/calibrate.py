from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.constants import SUPPORTED_STATS
from wnba_props_model.models.calibration import RoleAwarePMFCalibrator, fit_role_aware_calibrator
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json


def fit_calibrators(
    oof_pmfs_path: str | Path,
    out_dir: str | Path = "artifacts/models/calibration",
) -> dict[str, Path]:
    """Fit per-stat role-aware isotonic calibrators from OOF PMFs."""
    oof = pd.read_parquet(oof_pmfs_path).copy()

    # Normalize column names: OOF parquet uses actual_outcome; calibrator expects outcome
    if "outcome" not in oof.columns and "actual_outcome" in oof.columns:
        oof["outcome"] = oof["actual_outcome"]

    # role_bucket may not be in OOF parquet — default to "all" (global-only calibration)
    if "role_bucket" not in oof.columns:
        oof["role_bucket"] = "all"

    # Build numpy PMF arrays from JSON
    if "pmf" not in oof.columns and "pmf_json" in oof.columns:
        oof["pmf"] = oof["pmf_json"].map(json_to_pmf)

    # Only calibrate on eligible rows (excludes prior_only, low-minute games)
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stat in sorted(set(oof["stat"]) & set(SUPPORTED_STATS)):
        cal = fit_role_aware_calibrator(oof, stat)
        path = out / f"pmf_cal_role_{stat}.pkl"
        cal.save(str(path))
        paths[stat] = path
    return paths


def apply_calibrators(
    pmfs_df: pd.DataFrame,
    cal_dir: str | Path = "artifacts/models/calibration",
) -> pd.DataFrame:
    """Apply per-stat role-aware isotonic calibrators to a PMF DataFrame.

    Input columns required: stat, role_bucket, pmf_json.
    Adds: pmf_json (overwritten with calibrated values), is_calibrated=True,
    cal_source='role_aware_isotonic'.

    Stats without a fitted calibrator are passed through unchanged with
    is_calibrated=False and cal_source='no_calibrator'.
    """
    cal_dir = Path(cal_dir)
    out = pmfs_df.copy()
    calibrators: dict[str, RoleAwarePMFCalibrator] = {}

    for stat in out["stat"].unique():
        cal_path = cal_dir / f"pmf_cal_role_{stat}.pkl"
        if cal_path.exists():
            calibrators[stat] = RoleAwarePMFCalibrator.load(str(cal_path))

    new_pmf_jsons = []
    is_calibrated_flags = []
    cal_sources = []

    for _, row in out.iterrows():
        stat = row["stat"]
        role = str(row.get("role_bucket", "unknown"))
        raw_pmf = json_to_pmf(row["pmf_json"])

        if stat in calibrators:
            try:
                cal_pmf = calibrators[stat].apply(normalize_pmf(raw_pmf), role)
                new_pmf_jsons.append(pmf_to_json(cal_pmf))
                is_calibrated_flags.append(True)
                cal_sources.append("role_aware_isotonic")
            except Exception:
                new_pmf_jsons.append(row["pmf_json"])
                is_calibrated_flags.append(False)
                cal_sources.append("calibration_error")
        else:
            new_pmf_jsons.append(row["pmf_json"])
            is_calibrated_flags.append(False)
            cal_sources.append("no_calibrator")

    out["pmf_json"] = new_pmf_jsons
    out["is_calibrated"] = is_calibrated_flags
    out["cal_source"] = cal_sources

    # Recompute summary stats from calibrated PMFs
    def _pmf_stats(pmf_json: str) -> dict:
        pmf = normalize_pmf(json_to_pmf(pmf_json))
        ks = np.arange(len(pmf))
        mean = float(np.dot(ks, pmf))
        return {
            "mean": mean,
            "median": int(np.searchsorted(np.cumsum(pmf), 0.5)),
            "mode": int(np.argmax(pmf)),
            "p0": float(pmf[0]),
        }

    stats_rows = [_pmf_stats(j) for j in out["pmf_json"]]
    for key in ("mean", "median", "mode", "p0"):
        out[key] = [r[key] for r in stats_rows]

    return out
