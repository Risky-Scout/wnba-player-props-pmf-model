"""Phase 0: evaluated == deployed parity.

The market-superiority proof MUST score the exact probability production ships. This
verifies build_market_superiority_input.py emits, for every row, the identical
`model_prob_over_final` that `build_probability_lineage` (the single delivery creator)
produces for the same (pmf, line, prop, role) — not a separate isotonic path.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.models.binary_probability_calibration import BinaryCalibrationRegistry
from wnba_props_model.models.probability_lineage import build_probability_lineage
from wnba_props_model.models.simulation import json_to_pmf

ROOT = Path(__file__).resolve().parent.parent


def _pmf_list(mean: float, n: int = 40) -> list[float]:
    from scipy.stats import poisson
    xs = np.arange(0, n)
    p = poisson.pmf(xs, mean)
    return (p / p.sum()).round(8).tolist()


def _make_inputs(tmp: Path):
    """One row per distinct line so (prop, line) uniquely identifies its PMF."""
    cc_rows, oof_rows, pmf_by_line = [], [], {}
    base = pd.Timestamp("2026-06-01")
    lines = [9.5, 10.0, 14.5, 18.0, 22.5, 7.0]  # mix of half-integer and integer
    for i, line in enumerate(lines):
        gd = (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        pmf = _pmf_list(line)  # mean near line -> not all-push, binary eligible
        pmf_by_line[line] = pmf
        cc_rows.append({"game_id": str(3000 + i), "player_id": str(i), "stat": "pts",
                        "line": float(line), "market_prob_over_no_vig": 0.5,
                        "commence_time": f"{gd}T23:00:00Z"})
        oof_rows.append({"game_id": str(3000 + i), "player_id": str(i), "stat": "pts",
                         "actual_outcome": float(int(line) + 1), "pmf_json": json.dumps(pmf),
                         "game_date": gd, "role_bucket": "all", "model_version": "test_v1"})
    cc_p, oof_p = tmp / "cc.parquet", tmp / "oof.parquet"
    pd.DataFrame(cc_rows).to_parquet(cc_p)
    pd.DataFrame(oof_rows).to_parquet(oof_p)
    return cc_p, oof_p, pmf_by_line


def test_assembler_probability_equals_delivery_lineage(tmp_path):
    cc_p, oof_p, pmf_by_line = _make_inputs(tmp_path)
    out_p = tmp_path / "msi.parquet"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_market_superiority_input.py"),
         "--closing", str(cc_p), "--oof", str(oof_p), "--out", str(out_p),
         "--split-date", "2026-06-03"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    got = pd.read_parquet(out_p)

    # Identity registry (no policy file) == current production default.
    registry = BinaryCalibrationRegistry.from_policy(None)
    for _, row in got.iterrows():
        line = float(row["line"])
        expected = build_probability_lineage(
            final_pmf=json_to_pmf(json.dumps(pmf_by_line[line])),
            line=line, prop="pts", role="all",
            binary_calibration_registry=registry,
            probability_track="pure_forecast",
        ).model_prob_over_final
        assert expected is not None
        assert row["model_prob_over_final"] == expected, (
            f"line {line}: proof shipped {row['model_prob_over_final']} != "
            f"delivery lineage {expected}")


def test_identity_registry_matches_current_production_default():
    """from_policy(absent) yields an identity-disabled registry == deliver.py default."""
    assert BinaryCalibrationRegistry.from_policy(None).status == "identity_disabled"
    assert BinaryCalibrationRegistry.from_policy(
        "config/does_not_exist.json").status == "identity_disabled"
