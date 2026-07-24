"""Tests for the Phase A binary P(over) calibrator fitter.

Verifies the tooling logic on well-powered synthetic data (independent of whether the
current small real OOF benefits): a genuinely miscalibrated-but-monotonic signal yields a
shipped isotonic calibrator that loads through BinaryCalibrationRegistry; a well-calibrated
signal stays identity.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.models.binary_probability_calibration import BinaryCalibrationRegistry

ROOT = Path(__file__).resolve().parent.parent


def _write_inputs(tmp: Path, true_fn, n=600, seed=0):
    """One row per (game,player); pmf=[1-p,p] at line 0.5 so settled P(over)=p exactly."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-01")
    cc, oof = [], []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        y = int(rng.uniform(0, 1) < true_fn(p))
        gd = (base + pd.Timedelta(days=int(i % 40))).strftime("%Y-%m-%d")
        cc.append({"game_id": str(i), "player_id": str(i), "stat": "reb",
                   "line": 0.5, "market_prob_over_no_vig": 0.5,
                   "commence_time": f"{gd}T23:00:00Z"})
        oof.append({"game_id": str(i), "player_id": str(i), "stat": "reb",
                    "pmf_json": json.dumps([1.0 - p, p]), "actual_outcome": float(y),
                    "game_date": gd, "role_bucket": "all"})
    cc_p, oof_p = tmp / "cc.parquet", tmp / "oof.parquet"
    pd.DataFrame(cc).to_parquet(cc_p)
    pd.DataFrame(oof).to_parquet(oof_p)
    return cc_p, oof_p


def _run(tmp: Path, cc_p, oof_p):
    policy = tmp / "policy.json"
    sel = tmp / "sel.json"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fit_binary_prob_calibrators.py"),
         "--oof", str(oof_p), "--closing", str(cc_p), "--split-date", "2026-07-01",
         "--out-dir", str(tmp / "cal"), "--policy-out", str(policy),
         "--selection-out", str(sel), "--min-rows", "50"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return json.loads(policy.read_text()), json.loads(sel.read_text())


def test_miscalibrated_signal_ships_isotonic_and_loads(tmp_path):
    # Overconfident-monotonic: observed frequency = p**2 (isotonic recovers it).
    policy, sel = _run(tmp_path, *_write_inputs(tmp_path, lambda p: p ** 2))
    assert sel["reb"]["decision"] == "ship_isotonic", sel
    assert policy["enabled"] is True and "reb" in policy["artifacts"]
    # Loads through the fail-closed registry and returns a calibrated (non-identity) value.
    reg = BinaryCalibrationRegistry.from_policy(str(tmp_path / "policy.json"))
    assert reg.status == "enabled"
    out = reg.apply("reb", "all", 0.8)
    assert out.calibration_status == "calibrated"
    assert 0.0 <= out.p_calibrated <= 1.0


def test_well_calibrated_signal_stays_identity(tmp_path):
    # Observed frequency == p: isotonic cannot beat identity.
    policy, sel = _run(tmp_path, *_write_inputs(tmp_path, lambda p: p))
    assert sel["reb"]["decision"] in ("identity_no_improvement", "identity_insufficient_data"), sel
    assert policy["enabled"] is False
