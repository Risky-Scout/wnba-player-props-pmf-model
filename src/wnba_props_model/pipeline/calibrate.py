from __future__ import annotations

from pathlib import Path

import pandas as pd

from wnba_props_model.constants import SUPPORTED_STATS
from wnba_props_model.models.calibration import fit_role_aware_calibrator
from wnba_props_model.models.simulation import json_to_pmf


def fit_calibrators(oof_pmfs_path: str | Path, out_dir: str | Path = "artifacts/models/calibration") -> dict[str, Path]:
    oof = pd.read_parquet(oof_pmfs_path)
    if "pmf" not in oof.columns and "pmf_json" in oof.columns:
        oof = oof.copy()
        oof["pmf"] = oof["pmf_json"].map(json_to_pmf)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stat in sorted(set(oof["stat"]) & set(SUPPORTED_STATS)):
        cal = fit_role_aware_calibrator(oof, stat)
        path = out / f"pmf_cal_role_{stat}.pkl"
        cal.save(str(path))
        paths[stat] = path
    return paths
