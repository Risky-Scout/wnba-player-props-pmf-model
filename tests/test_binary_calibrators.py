"""W0.5: binary calibrator families + rolling-origin, complete-date CV fitter."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.models.binary_calibrators import (
    BetaCalibrator,
    CALIBRATOR_FAMILIES,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)

ROOT = Path(__file__).resolve().parent.parent


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_identity_is_passthrough():
    p = np.array([0.1, 0.5, 0.9])
    out = IdentityCalibrator().fit(p, [0, 1, 1]).predict(p.reshape(-1, 1))
    assert np.allclose(out, p)


def test_predict_accepts_2d_and_returns_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(0, 1, 400) < p ** 2).astype(int)   # miscalibrated (overconfident)
    for fam in (PlattCalibrator, BetaCalibrator, IsotonicCalibrator):
        cal = fam().fit(p, y)
        out = cal.predict(np.array([[0.3], [0.7]]))       # 2D [[p]] contract
        assert out.shape == (2,)
        assert np.all((out > 0) & (out < 1))


def test_calibrators_beat_identity_on_miscalibrated_signal():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 3000)
    y = (rng.uniform(0, 1, 3000) < p ** 2).astype(int)
    base = _ll(y, p)
    for fam in (PlattCalibrator, BetaCalibrator, IsotonicCalibrator):
        cal = fam().fit(p, y)
        assert _ll(y, cal.predict(p.reshape(-1, 1))) < base


def test_rolling_origin_folds_have_train_before_val():
    spec = importlib.util.spec_from_file_location(
        "fbc", ROOT / "scripts" / "fit_binary_prob_calibrators.py")
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    dates = pd.to_datetime([f"2026-05-{d:02d}" for d in range(1, 21)]).to_numpy()
    folds = fbc._rolling_origin_folds(np.sort(np.unique(dates)), 4)
    assert len(folds) == 4
    for train_dates, val_dates in folds:
        assert max(train_dates) < min(val_dates)   # strict temporal separation


def _synth_inputs(tmp: Path, true_fn, n=900, seed=0):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-01")
    cc, oof = [], []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        y = int(rng.uniform(0, 1) < true_fn(p))
        gd = (base + pd.Timedelta(days=int(i % 60))).strftime("%Y-%m-%d")
        cc.append({"game_id": str(i), "player_id": str(i), "stat": "reb", "line": 0.5,
                   "market_prob_over_no_vig": 0.5, "commence_time": f"{gd}T23:00:00Z"})
        oof.append({"game_id": str(i), "player_id": str(i), "stat": "reb",
                    "pmf_json": json.dumps([1.0 - p, p]), "actual_outcome": float(y),
                    "game_date": gd, "oof_prediction_type": "model_oof",
                    "calibration_eligible": True})
    cc_p, oof_p = tmp / "cc.parquet", tmp / "oof.parquet"
    pd.DataFrame(cc).to_parquet(cc_p); pd.DataFrame(oof).to_parquet(oof_p)
    return cc_p, oof_p


def _run_fitter(tmp, cc_p, oof_p):
    tmp.mkdir(parents=True, exist_ok=True)
    policy, sel = tmp / "policy.json", tmp / "sel.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fit_binary_prob_calibrators.py"),
         "--oof", str(oof_p), "--closing", str(cc_p), "--split-date", "2026-08-01",
         "--out-dir", str(tmp / "cal"), "--policy-out", str(policy),
         "--selection-out", str(sel), "--min-rows", "60"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(policy.read_text()), json.loads(sel.read_text())


def test_fitter_policy_declares_all_seven_props(tmp_path):
    policy, _ = _run_fitter(tmp_path, *_synth_inputs(tmp_path, lambda p: p ** 2))
    for prop in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
        assert prop in policy["props"]
        assert policy["props"][prop]["method"] in ("identity", "platt", "beta", "isotonic")


def test_fitter_ships_calibrator_on_miscalibrated_and_identity_on_clean(tmp_path):
    pol_bad, sel_bad = _run_fitter(tmp_path / "a", *_synth_inputs(tmp_path / "a", lambda p: p ** 2))
    assert sel_bad["reb"]["decision"].startswith("ship_")
    assert pol_bad["enabled"] is True
    pol_ok, sel_ok = _run_fitter(tmp_path / "b", *_synth_inputs(tmp_path / "b", lambda p: p, seed=2))
    assert sel_ok["reb"]["decision"] in ("identity_no_improvement", "identity_insufficient_data")
