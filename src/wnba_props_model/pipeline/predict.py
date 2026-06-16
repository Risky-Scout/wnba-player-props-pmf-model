"""Stage 6+ production inference pipeline.

Uses the Stage 4 HGB engine (pmf_engine.build_all_pmfs) to generate PMFs,
then optionally applies role-aware isotonic calibrators.

Legacy quantile path (pipeline/train.py, models/base.py, models/simulation.py)
is preserved for audit purposes but is no longer invoked.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from wnba_props_model.models.pmf_engine import (
    STATS,
    build_all_pmfs,
    prepare_feature_matrix,
)
from wnba_props_model.models.minutes_model import MinutesModel
from wnba_props_model.models.rate_model import HurdleModel, StatRateModel
from wnba_props_model.models.shrinkage import apply_bayesian_shrinkage
from wnba_props_model.pipeline.calibrate import apply_calibrators

logger = logging.getLogger(__name__)


def _load_stage4_models(model_dir: str | Path) -> dict:
    """Load Stage 4 HGB artifacts from disk.

    Supports both file layout conventions:
    - Bundled: minutes_model.joblib + stat_rate_models.joblib + hurdle_models.joblib
      (produced by train_baseline_pmfs.py)
    - Per-stat: minutes_model.pkl + rate_{stat}.pkl + hurdle_{stat}.pkl
      (legacy layout)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Stage 4 model directory not found: {model_dir}\n"
            "Run `python scripts/train_baseline_pmfs.py` first."
        )

    # --- Minutes model (try both naming conventions) ---
    for minutes_name in ("minutes_model.joblib", "minutes_model.pkl"):
        minutes_path = model_dir / minutes_name
        if minutes_path.exists():
            minutes = MinutesModel.load(str(minutes_path))
            break
    else:
        raise FileNotFoundError(
            f"minutes_model not found in {model_dir} — "
            "run `python scripts/train_baseline_pmfs.py` first."
        )

    # --- Pos encoder ---
    pos_encoder = None
    for enc_name in ("pos_encoder.pkl", "pos_encoder.joblib"):
        enc_path = model_dir / enc_name
        if enc_path.exists():
            pos_encoder = joblib.load(enc_path)
            break

    # --- Stat models ---
    rate_models: dict[str, StatRateModel] = {}
    hurdle_models: dict[str, HurdleModel] = {}

    # Bundled format (train_baseline_pmfs.py output)
    bundled_rate = model_dir / "stat_rate_models.joblib"
    bundled_hurdle = model_dir / "hurdle_models.joblib"
    if bundled_rate.exists():
        rate_models = joblib.load(bundled_rate)
    if bundled_hurdle.exists():
        hurdle_models = joblib.load(bundled_hurdle)

    # Per-stat format (legacy)
    if not rate_models and not hurdle_models:
        for stat in STATS:
            hurdle_path = model_dir / f"hurdle_{stat}.pkl"
            rate_path = model_dir / f"rate_{stat}.pkl"
            if hurdle_path.exists():
                hurdle_models[stat] = HurdleModel.load(str(hurdle_path))
            elif rate_path.exists():
                rate_models[stat] = StatRateModel.load(str(rate_path))

    # --- Feature manifest ---
    manifest_path = model_dir / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_feature_cols = manifest.get("model_feature_columns", [])

    return {
        "minutes": minutes,
        "pos_encoder": pos_encoder,
        "rate_models": rate_models,
        "hurdle_models": hurdle_models,
        "model_feature_cols": model_feature_cols,
    }


def predict_player_pmfs(
    feature_df: pd.DataFrame,
    model_dir: str | Path = "artifacts/models/stage4_baseline",
    config_path: str | Path | None = "config/model/stage4_baseline.yaml",
    cal_dir: str | Path | None = "artifacts/models/calibration",
    apply_calibration: bool = True,
    apply_shrinkage: bool = True,
    shrinkage_k: float = 15.0,
) -> pd.DataFrame:
    """Generate calibrated PMFs for all players in feature_df.

    Uses the Stage 4 HGB engine. If calibrators are available and
    apply_calibration=True, applies role-aware isotonic calibration.

    Parameters
    ----------
    feature_df : wide feature DataFrame from build_features.py
    model_dir  : Stage 4 artifact directory
    config_path: stage4_baseline.yaml path (for PMF caps / source tag)
    cal_dir    : Stage 6 calibrator directory; None to skip calibration
    apply_calibration: set False to return uncalibrated PMFs

    Returns
    -------
    Long PMF DataFrame with columns:
      player_id, game_id, game_date, stat, pmf_json, mean, median, mode, p0,
      is_calibrated, cal_source, role_bucket, pmf_source, model_version
    """
    cfg: dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    artifacts = _load_stage4_models(model_dir)
    model_dir = Path(model_dir)

    # Build a synthetic long_df for inference: one row per (player, game, stat).
    # The engine uses long_df to define which (player_id, game_id) pairs exist
    # for each stat; we replicate every wide row for each target stat.
    stats_to_predict = cfg.get("stats", STATS)
    long_rows = []
    for stat in stats_to_predict:
        stat_slice = feature_df.copy()
        stat_slice["stat"] = stat
        stat_slice[f"actual_{stat}"] = np.nan  # unknown at inference time
        long_rows.append(stat_slice)
    long_df_infer = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    pmfs_long = build_all_pmfs(
        wide_df=feature_df,
        long_df=long_df_infer,
        model_feature_cols=artifacts["model_feature_cols"],
        minutes_model=artifacts["minutes"],
        stat_models=artifacts["rate_models"],
        hurdle_models=artifacts["hurdle_models"],
        cfg=cfg,
    )

    pmfs_long["model_version"] = "wnba_pmf_v1.0_hgb_calibrated"
    pmfs_long["is_calibrated"] = False
    pmfs_long["cal_source"] = "uncalibrated"

    # Apply PenaltyBlog-style Bayesian shrinkage for small-sample players
    if apply_shrinkage:
        pmfs_long = apply_bayesian_shrinkage(
            pmfs_long,
            features=feature_df,
            k=shrinkage_k,
        )

    if apply_calibration and cal_dir is not None:
        cal_dir = Path(cal_dir)
        if cal_dir.exists() and any(cal_dir.glob("pmf_cal_role_*.pkl")):
            logger.info("Applying role-aware isotonic calibrators from %s", cal_dir)
            pmfs_long = apply_calibrators(pmfs_long, cal_dir=cal_dir)
        else:
            logger.warning(
                "Calibration requested but no calibrators found in %s; "
                "run `python scripts/fit_calibrators.py` first.",
                cal_dir,
            )

    return pmfs_long


def build_features_for_prediction(player_stats: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build wide feature table for inference.

    Thin wrapper kept for backward compatibility with predict_today.py.
    """
    from wnba_props_model.features.build_features import build_player_training_table
    return build_player_training_table(player_stats, games)
