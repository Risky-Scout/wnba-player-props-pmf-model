"""Session-scoped fixtures for the real e2e injury PMF rebuild tests.

These fixtures create minimal, deterministic sklearn model artifacts in a
temporary directory so that tests/test_injury_pmf_rebuild.py can invoke the
actual production pipeline code (predict_player_pmfs, rebuild_affected_pmfs,
rebuild_combos_for_affected) without needing the full trained production
artifacts.

Fixture layout (all paths relative to the tmp session dir):
  model/
    minutes_model.joblib         — MinutesModel trained on 80 synthetic rows
    stat_rate_models.joblib      — {pts, reb, ast: StatRateModel}
    hurdle_models.joblib         — {stl: ZINBStatModel, blk: ZINBStatModel}
    feature_manifest.json        — minimal feature column list
  config.yaml                    — minimal stage4_baseline config
  cal/                           — empty (calibration disabled in tests)

All five combo stats (pts_reb, pts_ast, reb_ast, pts_reb_ast, stocks) are
supported by including ast (rate model) and stl/blk (hurdle models).
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


def _make_training_data(n: int = _N_TRAIN, seed: int = _SEED):
    """Build synthetic training data for the minimal fixture models.

    Returns (X, y_minutes, y_pts, y_reb, y_ast, y_stl, y_blk, ctx_df).
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "player_minutes_mean_l5": rng.uniform(10, 35, n),
        "player_pts_mean_l5": rng.uniform(5, 25, n),
        "player_reb_mean_l5": rng.uniform(2, 10, n),
    })
    y_minutes = pd.Series(X["player_minutes_mean_l5"].values + rng.normal(0, 2, n))
    y_pts = pd.Series(X["player_pts_mean_l5"].values + rng.normal(0, 1.5, n))
    y_reb = pd.Series(X["player_reb_mean_l5"].values + rng.normal(0, 0.8, n))
    y_ast = pd.Series(X["player_minutes_mean_l5"].values * 0.15 + rng.normal(0, 0.5, n))
    # Sparse stats: mostly 0 with occasional 1-3 for realism
    y_stl = pd.Series(
        np.where(rng.random(n) < 0.45, rng.integers(1, 4, n).astype(float), 0.0)
    )
    y_blk = pd.Series(
        np.where(rng.random(n) < 0.30, rng.integers(1, 3, n).astype(float), 0.0)
    )
    ctx = pd.DataFrame({
        "actual_minutes": y_minutes.clip(lower=1.0),
        "projected_minutes_bucket": rng.choice(["low", "medium", "high"], n),
        "role_uncertainty_bucket": rng.choice(["low", "uncertain"], n),
        "did_play": np.ones(n, dtype=int),
    })
    return (
        X,
        y_minutes.clip(lower=0.0),
        y_pts.clip(lower=0.0),
        y_reb.clip(lower=0.0),
        y_ast.clip(lower=0.0),
        y_stl.clip(lower=0.0),
        y_blk.clip(lower=0.0),
        ctx,
    )


@pytest.fixture(scope="session")
def injury_e2e_artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and return a temporary directory with minimal model artifacts.

    The directory is created once per test session and shared across all
    tests that need it.  Artifacts are written deterministically so that
    repeated runs produce identical results.

    Includes ast (rate model) and stl/blk (ZINB hurdle models) so that all
    five combo stats can be built: pts_reb, pts_ast, reb_ast, pts_reb_ast,
    stocks.  The hurdle models for stl/blk also allow Fix 1 hurdle ratio
    tests to run through the actual pmf_engine.
    """
    try:
        import joblib
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import StatRateModel
        from wnba_props_model.models.hurdle import ZINBStatModel
    except ImportError as exc:
        pytest.skip(f"wnba_props_model not importable: {exc}")

    base = tmp_path_factory.mktemp("injury_e2e")
    model_dir = base / "model"
    model_dir.mkdir(parents=True)
    cal_dir = base / "cal"
    cal_dir.mkdir(parents=True)

    X, y_min, y_pts, y_reb, y_ast, y_stl, y_blk, ctx = _make_training_data()

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

    _stat_cfg = {
        "random_seed": _SEED,
        "hgb_regressor": {"max_iter": 30},
        "use_minutes_offset": False,
    }

    # ── StatRateModel — pts ───────────────────────────────────────────────────
    pts_model = StatRateModel("pts_fixture", _stat_cfg)
    pts_model.fit(X, y_pts, context_df=ctx)
    pts_model.stat = "pts"

    # ── StatRateModel — reb ───────────────────────────────────────────────────
    reb_model = StatRateModel("reb_fixture", _stat_cfg)
    reb_model.fit(X, y_reb, context_df=ctx)
    reb_model.stat = "reb"

    # ── StatRateModel — ast ───────────────────────────────────────────────────
    # ast uses the same 3 features; a simple linear model is sufficient for tests.
    ast_model = StatRateModel("ast_fixture", _stat_cfg)
    ast_model.fit(X, y_ast, context_df=ctx)
    ast_model.stat = "ast"

    rate_models = {"pts": pts_model, "reb": reb_model, "ast": ast_model}

    # ── ZINBStatModel — stl and blk (hurdle models for sparse stats) ──────────
    _zinb_cfg = {
        "random_seed": _SEED,
        "hgb_regressor": {"max_iter": 30},
    }
    stl_model = ZINBStatModel("stl", _zinb_cfg)
    stl_model.fit(X, y_stl, actual_minutes=ctx["actual_minutes"].values)

    blk_model = ZINBStatModel("blk", _zinb_cfg)
    blk_model.fit(X, y_blk, actual_minutes=ctx["actual_minutes"].values)

    hurdle_models = {"stl": stl_model, "blk": blk_model}

    joblib.dump(rate_models, str(model_dir / "stat_rate_models.joblib"))
    joblib.dump(hurdle_models, str(model_dir / "hurdle_models.joblib"))

    # ── Feature manifest ──────────────────────────────────────────────────────
    manifest = {"model_feature_columns": _FEATURE_COLS}
    (model_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Minimal config YAML ───────────────────────────────────────────────────
    # Include pts, reb, ast, stl, blk to support all five combo stats.
    # sparse_stats: [stl, blk] routes stl/blk through the ZINB hurdle model path,
    # enabling Fix 1 hurdle ratio tests to run through the actual pmf_engine.
    # use_minutes_marginalization=True so that OUT-player DNP blending fires
    # and produces {"0": 1.0} PMFs, which is the expected canonical behaviour.
    config = {
        "random_seed": _SEED,
        "stats": ["pts", "reb", "ast", "stl", "blk"],
        "sparse_stats": ["stl", "blk"],
        "use_minutes_marginalization": True,
        "minutes_marginalization_weights": [0.10, 0.15, 0.50, 0.15, 0.10],
        "pmf_support_caps": {"pts": 40, "reb": 20, "ast": 15, "stl": 10, "blk": 10},
        "pmf_source": "e2e_fixture",
    }
    config_path = base / "config.yaml"
    config_path.write_text(yaml.dump(config))

    return base
