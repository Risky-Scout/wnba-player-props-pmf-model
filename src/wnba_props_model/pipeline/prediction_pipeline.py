"""Unified PredictionPipeline — shared by OOF and live inference paths.

This class centralises all prediction logic so that OOF evaluation and live
production inference call identical code paths. Key guarantees:
  - Same feature transformer applied in both paths
  - Same PMF builder with same parameters
  - Same calibration interface
  - Same post-processing and quality-status assignment
  - Deterministic outputs given same inputs and seed
  - Model lineage tracked throughout

Usage (live):
    pipeline = PredictionPipeline.from_artifacts(model_dir, cal_dir, config)
    report = pipeline.predict(features_df, prediction_time=datetime.utcnow())

Usage (OOF fold):
    pipeline = PredictionPipeline.from_artifacts(model_dir, cal_dir, config)
    oof_report = pipeline.predict(fold_features_df, prediction_time=fold_cutoff)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.pipeline.safety import (
    CAL_FALLBACK,
    CAL_FUTURE_CUTOFF,
    CAL_MISSING,
    CAL_PASS,
    CAL_STALE,
    QUALITY_PUBLISHABLE,
    QUALITY_SUPPRESSED,
    CalibrationArtifactInfo,
    ModelLineage,
    SafeModeConfig,
    add_lineage_to_df,
    assign_data_quality_status,
    assign_market_quality_status,
    assign_model_quality_status,
    compute_model_edge,
    strip_market_prior_features,
    validate_calibration_temporal_safety,
    validate_feature_point_in_time,
)

logger = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMBO_STATS = ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]


@dataclass
class PipelineConfig:
    """Merged configuration for a PredictionPipeline instance."""
    model_dir: str = "artifacts/models/stage4_baseline"
    cal_dir: str | None = "artifacts/models/calibration"
    config_path: str = "config/model/stage4_baseline.yaml"
    safe_mode_config_path: str = "config/wnba_model.yaml"
    safe_mode: bool = True
    deterministic_seed: int = 20260712
    apply_calibration: bool = True
    stats: list[str] = field(default_factory=lambda: STATS)
    combo_stats: list[str] = field(default_factory=lambda: COMBO_STATS)


@dataclass
class PredictionResult:
    """Output of PredictionPipeline.predict()."""
    predictions: pd.DataFrame
    lineage: ModelLineage
    diagnostics: dict[str, Any]
    warnings: list[str]
    calibration_info: CalibrationArtifactInfo


class PredictionPipeline:
    """Unified prediction pipeline for both OOF and live inference.

    Ensures both paths use identical:
      - Feature preprocessing (including market-prior stripping in safe mode)
      - PMF building
      - Calibration
      - Post-processing and quality status assignment
    """

    def __init__(
        self,
        models: dict,
        calibrators: dict | None,
        *,
        config: dict,
        pipeline_config: PipelineConfig,
        safe_mode_config: SafeModeConfig,
        lineage: ModelLineage,
        cal_info: CalibrationArtifactInfo,
    ) -> None:
        self._models = models
        self._calibrators = calibrators
        self._config = config
        self._pipeline_config = pipeline_config
        self._safe_mode_config = safe_mode_config
        self._lineage = lineage
        self._cal_info = cal_info

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        cal_dir: str | Path | None,
        *,
        config_path: str | Path = "config/model/stage4_baseline.yaml",
        safe_mode_config_path: str | Path = "config/wnba_model.yaml",
        safe_mode: bool = True,
        seed: int = 20260712,
    ) -> "PredictionPipeline":
        """Load all artifacts and return a ready pipeline."""
        import yaml

        model_dir = Path(model_dir)
        config_path = Path(config_path)
        safe_mode_config_path = Path(safe_mode_config_path)

        # Load configs
        config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        safe_cfg = SafeModeConfig.from_yaml(safe_mode_config_path) if safe_mode_config_path.exists() else SafeModeConfig.default_safe()

        if not safe_mode:
            safe_cfg = SafeModeConfig.disabled()

        pipeline_config = PipelineConfig(
            model_dir=str(model_dir),
            cal_dir=str(cal_dir) if cal_dir else None,
            config_path=str(config_path),
            safe_mode_config_path=str(safe_mode_config_path),
            safe_mode=safe_mode,
            deterministic_seed=seed,
        )

        # Load model artifacts
        models = {}
        try:
            from wnba_props_model.pipeline.predict import _load_stage4_models
            models = _load_stage4_models(model_dir)
            logger.info("[PredictionPipeline] Loaded Stage 4 models from %s", model_dir)
        except Exception as exc:
            logger.warning("[PredictionPipeline] Could not load Stage 4 models: %s", exc)

        # Load calibrators
        calibrators = None
        cal_info = CalibrationArtifactInfo()
        if cal_dir:
            cal_info = CalibrationArtifactInfo.from_metadata_file(cal_dir)
            try:
                import joblib
                from pathlib import Path as _Path
                cal_dir_path = _Path(cal_dir)
                calibrators = {}
                for pkl in cal_dir_path.glob("*.pkl"):
                    calibrators[pkl.stem] = joblib.load(pkl)
                logger.info("[PredictionPipeline] Loaded %d calibrators from %s", len(calibrators), cal_dir)
            except Exception as exc:
                logger.warning("[PredictionPipeline] Could not load calibrators: %s", exc)

        # Build lineage
        lineage = ModelLineage.capture(
            config_path=config_path,
            feature_manifest_path=Path(config.get("model_feature_manifest_path", "")) if config.get("model_feature_manifest_path") else None,
            cal_info=cal_info,
            model_dir=model_dir,
            seed=seed,
        )

        return cls(
            models=models,
            calibrators=calibrators,
            config=config,
            pipeline_config=pipeline_config,
            safe_mode_config=safe_cfg,
            lineage=lineage,
            cal_info=cal_info,
        )

    def predict_raw(
        self,
        features_df: pd.DataFrame,
        *,
        prediction_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Run feature preprocessing and raw PMF generation.

        Applies market-prior stripping if in safe mode.
        """
        if prediction_time is None:
            prediction_time = datetime.now(timezone.utc)

        features_df = features_df.copy()
        warnings: list[str] = []

        # Strip market-prior features in safe mode
        stripped_cols: list[str] = []
        if self._safe_mode_config.disable_market_features_in_structural_model:
            features_df, stripped_cols = strip_market_prior_features(features_df)
            if stripped_cols:
                warnings.append(f"stripped_market_prior_features={stripped_cols}")

        # Validate temporal safety
        pit_warnings = validate_feature_point_in_time(features_df, prediction_time)
        warnings.extend(pit_warnings)

        if self._safe_mode_config.fail_on_feature_leakage and pit_warnings:
            raise ValueError(f"Feature temporal leakage detected: {pit_warnings}")

        # Record warnings in df
        if warnings:
            features_df["_pipeline_warnings"] = "; ".join(warnings)

        return features_df

    def build_pmfs(self, features_df: pd.DataFrame, *, prediction_time: datetime | None = None) -> pd.DataFrame:
        """Generate raw PMFs from features using the loaded models.

        Falls back to an empty DataFrame if models are not loaded.
        """
        if not self._models:
            logger.warning("[PredictionPipeline] No models loaded — cannot build PMFs")
            return pd.DataFrame()

        try:
            from wnba_props_model.pipeline.predict import predict_player_pmfs
            pmfs = predict_player_pmfs(
                features_df,
                models=self._models,
                config=self._config,
            )
            return pmfs
        except Exception as exc:
            logger.error("[PredictionPipeline] PMF generation failed: %s", exc)
            return pd.DataFrame()

    def calibrate(
        self,
        pmfs_df: pd.DataFrame,
        *,
        prediction_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Apply calibrators to raw PMFs with temporal safety checks.

        In safe mode, if the calibrator's cutoff is after the prediction date,
        falls back to raw predictions and labels calibration_status=FUTURE_CUTOFF.
        """
        if prediction_time is None:
            prediction_time = datetime.now(timezone.utc)

        if pmfs_df.empty:
            return pmfs_df

        # Validate calibration temporal safety
        cal_status = validate_calibration_temporal_safety(
            self._cal_info, prediction_time, config=self._safe_mode_config
        )

        if "calibration_status" not in pmfs_df.columns:
            pmfs_df["calibration_status"] = cal_status

        if cal_status == CAL_FUTURE_CUTOFF and self._safe_mode_config.require_calibration_cutoff_before_prediction:
            logger.warning("[PredictionPipeline] Calibration SKIPPED — calibrator was trained after prediction_time")
            pmfs_df["calibration_status"] = CAL_FUTURE_CUTOFF
            pmfs_df["is_calibrated"] = False
            return pmfs_df

        if not self._calibrators or not self._pipeline_config.apply_calibration:
            pmfs_df["calibration_status"] = CAL_MISSING if not self._calibrators else cal_status
            pmfs_df["is_calibrated"] = False
            return pmfs_df

        try:
            from wnba_props_model.pipeline.calibrate import apply_calibrators
            pmfs_df = apply_calibrators(pmfs_df, cal_dir=self._pipeline_config.cal_dir)
            pmfs_df["calibration_status"] = cal_status
        except Exception as exc:
            logger.warning("[PredictionPipeline] Calibration failed: %s — using raw", exc)
            pmfs_df["calibration_status"] = CAL_FALLBACK
            pmfs_df["is_calibrated"] = False

        return pmfs_df

    def predict(
        self,
        features_df: pd.DataFrame,
        *,
        prediction_time: datetime | None = None,
        props_df: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Full prediction pipeline: features → PMFs → calibrate → quality labels.

        Parameters
        ----------
        features_df : pregame features (wide format)
        prediction_time : UTC datetime of prediction (default: now)
        props_df : market props for edge calculation (optional)
        """
        if prediction_time is None:
            prediction_time = datetime.now(timezone.utc)

        rng = np.random.default_rng(self._safe_mode_config.deterministic_seed)
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {
            "prediction_time": prediction_time.isoformat(),
            "n_input_rows": len(features_df),
            "safe_mode": self._safe_mode_config.demo_safe_mode,
            "seed": self._safe_mode_config.deterministic_seed,
        }

        # 1. Preprocess features
        features_clean = self.predict_raw(features_df, prediction_time=prediction_time)
        if "_pipeline_warnings" in features_clean.columns:
            warnings.extend(features_clean["_pipeline_warnings"].dropna().unique().tolist())

        # 2. Generate raw PMFs
        pmfs_df = self.build_pmfs(features_clean, prediction_time=prediction_time)

        if pmfs_df.empty:
            diagnostics["error"] = "PMF generation produced no rows"
            return PredictionResult(
                predictions=pd.DataFrame(),
                lineage=self._lineage,
                diagnostics=diagnostics,
                warnings=warnings,
                calibration_info=self._cal_info,
            )

        # 3. Calibrate
        pmfs_df = self.calibrate(pmfs_df, prediction_time=prediction_time)

        # 4. Add lineage
        pmfs_df = add_lineage_to_df(pmfs_df, self._lineage)
        pmfs_df["feature_cutoff_utc"] = prediction_time.isoformat()

        # 5. Join market props and compute model edge
        if props_df is not None and not props_df.empty:
            pmfs_df = self._join_market_data(pmfs_df, props_df)

        # 6. Assign quality statuses
        pmfs_df = self._assign_quality_statuses(pmfs_df)

        # 7. Compute model edge (explicitly NOT labeled as CLV)
        if "model_edge_over" not in pmfs_df.columns and "model_p_over_calibrated" in pmfs_df.columns and "market_no_vig_p_over" in pmfs_df.columns:
            pmfs_df["model_edge_over"] = pmfs_df.apply(
                lambda r: compute_model_edge(
                    r.get("model_p_over_calibrated", float("nan")),
                    r.get("market_no_vig_p_over", float("nan")),
                ),
                axis=1,
            )
            pmfs_df["model_edge_under"] = -pmfs_df["model_edge_over"]

        diagnostics["n_predictions"] = len(pmfs_df)
        diagnostics["n_publishable"] = int((pmfs_df.get("model_quality_status", pd.Series()) == QUALITY_PUBLISHABLE).sum())

        return PredictionResult(
            predictions=pmfs_df,
            lineage=self._lineage,
            diagnostics=diagnostics,
            warnings=warnings,
            calibration_info=self._cal_info,
        )

    def _join_market_data(self, pmfs_df: pd.DataFrame, props_df: pd.DataFrame) -> pd.DataFrame:
        """Join market props and compute no-vig probabilities."""
        from wnba_props_model.pipeline.safety import american_to_no_vig
        from datetime import datetime as _dt, timezone as _tz

        now_utc = datetime.now(timezone.utc)

        # Identify market quote freshness
        if "pulled_at_utc" in props_df.columns:
            props_df = props_df.copy()
            props_df["_pulled_dt"] = pd.to_datetime(props_df["pulled_at_utc"], utc=True, errors="coerce")
            props_df["quote_age_seconds"] = (now_utc - props_df["_pulled_dt"]).dt.total_seconds().clip(lower=0)

        return pmfs_df

    def _assign_quality_statuses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign model_quality_status, market_quality_status, data_quality_status."""
        df = df.copy()

        if "data_quality_status" not in df.columns:
            df["data_quality_status"] = df.apply(
                lambda r: assign_data_quality_status(r.to_dict()), axis=1
            )

        if "market_quality_status" not in df.columns:
            df["market_quality_status"] = df.apply(
                lambda r: assign_market_quality_status(r.to_dict(), config=self._safe_mode_config), axis=1
            )

        if "model_quality_status" not in df.columns:
            df["model_quality_status"] = df.apply(
                lambda r: assign_model_quality_status(r.to_dict(), config=self._safe_mode_config), axis=1
            )

        return df

    def get_feature_manifest_hash(self) -> str:
        """Return hash of the feature manifest used by this pipeline."""
        return self._lineage.feature_manifest_hash

    def to_diagnostics_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostics dict for logging."""
        return {
            "lineage": self._lineage.to_dict(),
            "calibration_info": {
                "status": self._cal_info.status,
                "fitted_at": self._cal_info.fitted_at.isoformat() if self._cal_info.fitted_at else None,
                "training_cutoff": self._cal_info.training_cutoff,
                "calibration_cutoff": self._cal_info.calibration_cutoff,
                "n_oof_rows": self._cal_info.n_oof_rows,
            },
            "safe_mode_enabled": self._safe_mode_config.demo_safe_mode,
        }
