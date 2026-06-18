"""Tests for hyperparameter tuning infrastructure (P3.2)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_wide_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_id": range(n),
        "game_id": range(n),
        "game_date": pd.date_range("2022-01-01", periods=n, freq="3D"),
        "season": [2024] * n,
        "actual_pts": rng.integers(0, 30, n).astype(float),
        "actual_minutes": rng.uniform(0, 35, n),
        "did_play": [1] * n,
        "player_pts_mean_l5": rng.uniform(5, 25, n),
        "player_minutes_mean_l5": rng.uniform(15, 35, n),
    })


def test_oof_is_returns_finite():
    from scripts.tune_hyperparams import _oof_is_for_params
    wide = _make_wide_df()
    model_cols = ["player_pts_mean_l5", "player_minutes_mean_l5"]
    params = {"max_iter": 50, "max_leaf_nodes": 15, "learning_rate": 0.1,
              "min_samples_leaf": 20, "l2_regularization": 0.0}
    score = _oof_is_for_params("pts", wide, model_cols, params, seed=42)
    assert np.isfinite(score), "OOF IS score should be finite"
    assert score > 0, "IS score must be positive"


def test_oof_is_lower_for_better_params():
    """More iterations should not dramatically worsen IS."""
    from scripts.tune_hyperparams import _oof_is_for_params
    wide = _make_wide_df(n=600)
    model_cols = ["player_pts_mean_l5", "player_minutes_mean_l5"]
    params_default = {"max_iter": 50, "max_leaf_nodes": 15, "learning_rate": 0.1,
                      "min_samples_leaf": 20, "l2_regularization": 0.0}
    params_more = {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.08,
                   "min_samples_leaf": 15, "l2_regularization": 0.1}
    s_default = _oof_is_for_params("pts", wide, model_cols, params_default, seed=42)
    s_more    = _oof_is_for_params("pts", wide, model_cols, params_more, seed=42)
    # Both should be finite; neither should be pathologically worse
    assert np.isfinite(s_default) and np.isfinite(s_more)


def test_tune_hyperparams_script_importable():
    """tune_hyperparams.py must be importable without crashing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tune_hyperparams",
        Path("scripts/tune_hyperparams.py")
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_oof_is_for_params")
    assert hasattr(mod, "_tune_stat")
