"""Session-scoped fixtures for the real e2e injury PMF rebuild tests.

These fixtures create minimal, deterministic sklearn model artifacts in a
temporary directory so that tests/test_injury_pmf_rebuild.py can invoke the
actual production pipeline code (predict_player_pmfs, rebuild_affected_pmfs,
rebuild_combos_for_affected) without needing the full trained production
artifacts.

Fixture layout (all paths relative to the tmp session dir):
  model/
    minutes_model.joblib         — MinutesModel trained on 80 synthetic rows
    stat_rate_models.joblib      — {pts: StatRateModel, reb: StatRateModel}
    hurdle_models.joblib         — {} (empty)
    feature_manifest.json        — minimal feature column list
  config.yaml                    — minimal stage4_baseline config
  cal/                           — empty (calibration disabled in tests)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import yaml

# ────────────────────────────────────────────────────────────────────────────────
# Constants for fixture generation
# ────────────────────────────────────────────────────────────────────────────────

_FEATURE_COLS = [
    "player_minutes_mean_l5",
    "player_pts_mean_l5",
    "player_reb_mean_l5",
]

_N_TRAIN = 80
_SEED = 42


def _make_training_data(n: int = _N_TRAIN, seed: int = _SEED) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build synthetic training data for the minimal fixture models.

    Returns (X, y_minutes, y_pts, ctx_df) where ctx_df contains
    actual_minutes, projected_minutes_bucket, role_uncertainty_bucket.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "player_minutes_mean_l5": rng.uniform(10, 35, n),
        "player_pts_mean_l5": rng.uniform(5, 25, n),
        "player_reb_mean_l5": rng.uniform(2, 10, n),
    })
    # Targets: minutes ~ l5 + noise, pts ~ pts_l5 + noise, reb ~ reb_l5 + noise
    y_minutes = pd.Series(X["player_minutes_mean_l5"].values + rng.normal(0, 2, n))
    y_pts = pd.Series(X["player_pts_mean_l5"].values + rng.normal(0, 1.5, n))
    y_reb = pd.Series(X["player_reb_mean_l5"].values + rng.normal(0, 0.8, n))
    ctx = pd.DataFrame({
        "actual_minutes": y_minutes.clip(lower=1.0),
        "projected_minutes_bucket": rng.choice(["low", "medium", "high"], n),
        "role_uncertainty_bucket": rng.choice(["low", "uncertain"], n),
        "did_play": np.ones(n, dtype=int),
    })
    return X, y_minutes.clip(lower=0.0), y_pts.clip(lower=0.0), y_reb.clip(lower=0.0), ctx


@pytest.fixture(scope="session")
def injury_e2e_artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and return a temporary directory with minimal model artifacts.

    The directory is created once per test session and shared across all
    tests that need it.  Artifacts are written deterministically so that
    repeated runs produce identical results.
    """
    try:
        import joblib
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import StatRateModel
    except ImportError as exc:
        pytest.skip(f"wnba_props_model not importable: {exc}")

    base = tmp_path_factory.mktemp("injury_e2e")
    model_dir = base / "model"
    model_dir.mkdir(parents=True)
    cal_dir = base / "cal"
    cal_dir.mkdir(parents=True)

    X, y_min, y_pts, y_reb, ctx = _make_training_data()

    # ── MinutesModel ──────────────────────────────────────────────────────────
    min_cfg = {
        "random_seed": _SEED,
        "min_minutes_sigma": 3.0,
        "minutes_clip_min": 0.0,
        "minutes_clip_max": 40.0,
        "hgb_regressor": {"max_iter": 30},
    }
    min_model = MinutesModel(min_cfg)
    min_model.fit(X, y_min, metadata_df=ctx)
    min_model.save(str(model_dir / "minutes_model.joblib"))

    # ── StatRateModel — pts ───────────────────────────────────────────────────
    pts_cfg = {
        "random_seed": _SEED,
        "hgb_regressor": {"max_iter": 30},
        "use_minutes_offset": False,
        # Use squared_error loss so the stat mean is directly predicted
        # (not quantile=0.5, which would skip dispersion slope fitting)
    }
    pts_model = StatRateModel("pts_fixture", pts_cfg)
    pts_model.fit(X, y_pts, context_df=ctx)

    # ── StatRateModel — reb ───────────────────────────────────────────────────
    reb_cfg = {
        "random_seed": _SEED,
        "hgb_regressor": {"max_iter": 30},
        "use_minutes_offset": False,
    }
    reb_model = StatRateModel("reb_fixture", reb_cfg)
    reb_model.fit(X, y_reb, context_df=ctx)

    # Register stat names so pmf_engine can look them up
    pts_model.stat = "pts"
    reb_model.stat = "reb"

    rate_models = {"pts": pts_model, "reb": reb_model}
    hurdle_models: dict = {}

    joblib.dump(rate_models, str(model_dir / "stat_rate_models.joblib"))
    joblib.dump(hurdle_models, str(model_dir / "hurdle_models.joblib"))

    # ── Feature manifest ──────────────────────────────────────────────────────
    manifest = {"model_feature_columns": _FEATURE_COLS}
    (model_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Minimal config YAML ───────────────────────────────────────────────────
    # Use only pts and reb to keep the test fast.
    # use_minutes_marginalization=True so that OUT-player DNP blending fires
    # and produces {"0": 1.0} PMFs, which is the expected canonical behaviour.
    config = {
        "random_seed": _SEED,
        "stats": ["pts", "reb"],
        "sparse_stats": [],
        "use_minutes_marginalization": True,
        "minutes_marginalization_weights": [0.10, 0.15, 0.50, 0.15, 0.10],
        "pmf_support_caps": {"pts": 40, "reb": 20},
        "pmf_source": "e2e_fixture",
    }
    config_path = base / "config.yaml"
    config_path.write_text(yaml.dump(config))

    return base
