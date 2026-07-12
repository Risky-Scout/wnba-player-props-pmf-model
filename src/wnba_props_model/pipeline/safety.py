"""Production-safe mode enforcement and model lineage utilities.

This module centralises all integrity checks required for safe prediction:
  - demo_safe_mode configuration loading
  - Feature leakage prevention
  - Calibration cutoff enforcement
  - Model lineage tracking
  - Quality status assignment
  - Quote freshness validation
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality status constants
# ---------------------------------------------------------------------------
QUALITY_PUBLISHABLE = "PUBLISHABLE"
QUALITY_WATCHLIST = "WATCHLIST"
QUALITY_EXPERIMENTAL = "EXPERIMENTAL"
QUALITY_SUPPRESSED = "SUPPRESSED"
QUALITY_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

QUALITY_RANK: dict[str, int] = {
    QUALITY_PUBLISHABLE: 0,
    QUALITY_WATCHLIST: 1,
    QUALITY_EXPERIMENTAL: 2,
    QUALITY_INSUFFICIENT_DATA: 3,
    QUALITY_SUPPRESSED: 4,
}

# ---------------------------------------------------------------------------
# Calibration status constants
# ---------------------------------------------------------------------------
CAL_PASS = "PASS"
CAL_FAIL = "FAIL"
CAL_FALLBACK = "FALLBACK_IDENTITY"
CAL_INSUFFICIENT = "INSUFFICIENT_DATA"
CAL_STALE = "STALE_ARTIFACT"
CAL_FUTURE_CUTOFF = "CUTOFF_AFTER_PREDICTION"
CAL_MISSING = "MISSING_ARTIFACT"

# ---------------------------------------------------------------------------
# Safe-mode configuration
# ---------------------------------------------------------------------------
_MARKET_PRIOR_FEATURES = frozenset({
    "player_market_p_over_prev",
    "player_market_line_prev",
    "player_line_movement_prev",
})


@dataclass
class SafeModeConfig:
    demo_safe_mode: bool = True
    fail_on_feature_leakage: bool = True
    fail_on_artifact_date_violation: bool = True
    require_model_lineage: bool = True
    disable_unvalidated_advanced_features: bool = True
    disable_market_features_in_structural_model: bool = True
    require_calibration_cutoff_before_prediction: bool = True
    allow_identity_calibration_fallback: bool = True
    suppress_failed_categories: bool = True
    mark_insufficient_categories: bool = True
    deterministic_seed: int = 20260712
    strip_market_prior_features_in_safe_mode: bool = True
    min_calibration_obs: int = 150
    quote_stale_seconds: int = 7200
    min_publishable_edge: float = 0.04

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SafeModeConfig":
        """Load safe-mode config from the main wnba_model.yaml."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config not found at %s — using defaults", path)
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        prod_block = raw.get("production", {})
        return cls(**{k: v for k, v in prod_block.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default_safe(cls) -> "SafeModeConfig":
        return cls()

    @classmethod
    def disabled(cls) -> "SafeModeConfig":
        """All checks disabled — for backward-compatible legacy paths only."""
        return cls(
            demo_safe_mode=False,
            fail_on_feature_leakage=False,
            fail_on_artifact_date_violation=False,
            require_model_lineage=False,
            disable_unvalidated_advanced_features=False,
            disable_market_features_in_structural_model=False,
            require_calibration_cutoff_before_prediction=False,
            allow_identity_calibration_fallback=True,
            suppress_failed_categories=False,
            mark_insufficient_categories=False,
            strip_market_prior_features_in_safe_mode=False,
        )


# ---------------------------------------------------------------------------
# Feature safety
# ---------------------------------------------------------------------------

def strip_market_prior_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove market-prior features from the feature matrix.

    Returns (cleaned_df, list_of_removed_columns).
    """
    to_remove = [c for c in df.columns if c in _MARKET_PRIOR_FEATURES]
    if to_remove:
        logger.info("[safe_mode] Stripping market-prior features from structural model: %s", to_remove)
        df = df.drop(columns=to_remove)
    return df, to_remove


def validate_feature_point_in_time(
    df: pd.DataFrame,
    prediction_time: datetime,
    *,
    source_effective_col: str | None = "feature_cutoff_utc",
    game_start_col: str | None = "scheduled_start_utc",
) -> list[str]:
    """Validate that features are temporally safe.

    Returns list of warning strings (empty = all clear).
    """
    warnings: list[str] = []
    now_utc = prediction_time.replace(tzinfo=timezone.utc) if prediction_time.tzinfo is None else prediction_time

    if source_effective_col and source_effective_col in df.columns:
        bad = df[pd.to_datetime(df[source_effective_col], utc=True, errors="coerce") > now_utc]
        if not bad.empty:
            warnings.append(
                f"feature_cutoff_utc > prediction_time for {len(bad)} rows — potential leakage"
            )

    if game_start_col and game_start_col in df.columns:
        game_starts = pd.to_datetime(df[game_start_col], utc=True, errors="coerce")
        if source_effective_col and source_effective_col in df.columns:
            cutoffs = pd.to_datetime(df[source_effective_col], utc=True, errors="coerce")
            bad_after_start = df[cutoffs >= game_starts]
            if not bad_after_start.empty:
                warnings.append(
                    f"feature_cutoff >= game_start for {len(bad_after_start)} rows — features include game-day data"
                )

    return warnings


# ---------------------------------------------------------------------------
# Calibration artifact validation
# ---------------------------------------------------------------------------

@dataclass
class CalibrationArtifactInfo:
    fitted_at: datetime | None = None
    training_cutoff: str | None = None
    calibration_cutoff: str | None = None
    n_oof_rows: int = 0
    stats: list[str] = field(default_factory=list)
    status: str = CAL_MISSING

    @classmethod
    def from_metadata_file(cls, cal_dir: str | Path) -> "CalibrationArtifactInfo":
        meta_path = Path(cal_dir) / "calibration_metadata.json"
        if not meta_path.exists():
            return cls(status=CAL_MISSING)
        try:
            meta = json.loads(meta_path.read_text())
            fitted_at = None
            if "fitted_at" in meta:
                fitted_at = datetime.fromisoformat(meta["fitted_at"])
                if fitted_at.tzinfo is None:
                    fitted_at = fitted_at.replace(tzinfo=timezone.utc)
            return cls(
                fitted_at=fitted_at,
                training_cutoff=meta.get("training_cutoff"),
                calibration_cutoff=meta.get("calibration_cutoff"),
                n_oof_rows=meta.get("n_oof_rows", 0),
                stats=meta.get("stats", []),
                status=CAL_PASS,
            )
        except Exception as exc:
            logger.warning("Could not parse calibration_metadata.json: %s", exc)
            return cls(status=CAL_MISSING)


def validate_calibration_temporal_safety(
    cal_info: CalibrationArtifactInfo,
    prediction_date: datetime,
    *,
    config: SafeModeConfig | None = None,
) -> str:
    """Validate that the calibration artifact is safe to apply to prediction_date.

    Returns one of: CAL_PASS, CAL_STALE, CAL_FUTURE_CUTOFF, CAL_MISSING.
    """
    if config is None:
        config = SafeModeConfig.default_safe()

    if cal_info.status == CAL_MISSING:
        return CAL_MISSING

    pred_utc = prediction_date.replace(tzinfo=timezone.utc) if prediction_date.tzinfo is None else prediction_date

    if cal_info.fitted_at is not None:
        fitted_utc = cal_info.fitted_at
        if fitted_utc > pred_utc:
            logger.warning(
                "[safe_mode] Calibrator fitted_at %s is AFTER prediction_date %s — temporal violation",
                fitted_utc.isoformat(), pred_utc.isoformat(),
            )
            return CAL_FUTURE_CUTOFF

        age_days = (pred_utc - fitted_utc).days
        if age_days > 90:
            logger.warning(
                "[safe_mode] Calibrator is %d days old — may be stale", age_days
            )
            return CAL_STALE

    return CAL_PASS


# ---------------------------------------------------------------------------
# Model lineage
# ---------------------------------------------------------------------------

@dataclass
class ModelLineage:
    git_commit: str = ""
    config_hash: str = ""
    feature_manifest_hash: str = ""
    training_cutoff: str = ""
    calibration_cutoff: str = ""
    prediction_timestamp_utc: str = ""
    market_data_timestamp: str = ""
    model_artifact_version: str = ""
    data_source_version: str = ""
    model_version: str = "0.1.0"
    deterministic_seed: int = 20260712

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def capture(
        cls,
        *,
        config_path: str | Path | None = None,
        feature_manifest_path: str | Path | None = None,
        cal_info: CalibrationArtifactInfo | None = None,
        model_dir: str | Path | None = None,
        seed: int = 20260712,
    ) -> "ModelLineage":
        git_commit = _get_git_commit()
        config_hash = _hash_file(config_path) if config_path else ""
        feature_manifest_hash = _hash_file(feature_manifest_path) if feature_manifest_path else ""

        training_cutoff = ""
        calibration_cutoff = ""
        model_artifact_version = ""
        if cal_info:
            training_cutoff = cal_info.training_cutoff or ""
            calibration_cutoff = cal_info.calibration_cutoff or ""

        if model_dir and Path(model_dir).exists():
            model_artifact_version = _hash_dir(model_dir)

        return cls(
            git_commit=git_commit,
            config_hash=config_hash,
            feature_manifest_hash=feature_manifest_hash,
            training_cutoff=training_cutoff,
            calibration_cutoff=calibration_cutoff,
            prediction_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_artifact_version=model_artifact_version,
            model_version="0.1.0",
            deterministic_seed=seed,
        )


def _get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _hash_dir(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    for f in sorted(p.rglob("*.pkl")) + sorted(p.rglob("*.json")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Quality status assignment
# ---------------------------------------------------------------------------

def assign_model_quality_status(
    row: dict[str, Any],
    *,
    config: SafeModeConfig | None = None,
) -> str:
    """Assign a model_quality_status to a prediction row.

    Interpretation:
      PUBLISHABLE  — point-in-time safe, valid calibration, fresh market, valid artifact
      WATCHLIST    — model valid but calibration/role certainty weaker
      EXPERIMENTAL — sparse market, unvalidated combo, or unsupported stat history
      SUPPRESSED   — stale/invalid input, probable inactive, broken artifact, leakage
      INSUFFICIENT_DATA — not enough observations to assess category
    """
    if config is None:
        config = SafeModeConfig.default_safe()

    warnings = row.get("warnings", "")
    if isinstance(warnings, list):
        warnings = "; ".join(str(w) for w in warnings)
    warnings_lower = warnings.lower() if warnings else ""

    cal_status = row.get("calibration_status", "")
    data_quality = row.get("data_quality_status", "")
    market_quality = row.get("market_quality_status", "")
    stat = row.get("stat", "")
    availability_prob = row.get("availability_probability", 1.0)
    if availability_prob is None or (isinstance(availability_prob, float) and np.isnan(availability_prob)):
        availability_prob = 1.0

    # Suppression triggers
    if cal_status == CAL_FUTURE_CUTOFF:
        return QUALITY_SUPPRESSED
    if "leakage" in warnings_lower or "temporal_violation" in warnings_lower:
        return QUALITY_SUPPRESSED
    if availability_prob < 0.3:
        return QUALITY_SUPPRESSED
    if data_quality == "STALE" or data_quality == "MISSING":
        return QUALITY_SUPPRESSED

    # Insufficient data
    cal_sample = row.get("calibration_sample_size", 0) or 0
    if cal_status == CAL_INSUFFICIENT or cal_sample < config.min_calibration_obs:
        return QUALITY_INSUFFICIENT_DATA

    # Combo stats — always experimental until joint simulation is validated
    combo_stats = {"pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "stocks", "pa", "pr", "ra", "pra"}
    if stat in combo_stats:
        return QUALITY_EXPERIMENTAL

    # Market quality
    if market_quality == "STALE" or market_quality == "MISSING_SIDE":
        return QUALITY_WATCHLIST

    # Calibration status
    if cal_status == CAL_STALE or cal_status == CAL_FALLBACK:
        return QUALITY_WATCHLIST

    # Check edge magnitude
    edge_over = row.get("model_edge_over", 0.0) or 0.0
    edge_under = row.get("model_edge_under", 0.0) or 0.0
    max_edge = max(abs(edge_over), abs(edge_under))

    if cal_status in (CAL_PASS, "") and max_edge >= config.min_publishable_edge:
        if availability_prob >= 0.8:
            return QUALITY_PUBLISHABLE
        return QUALITY_WATCHLIST

    return QUALITY_WATCHLIST


def assign_market_quality_status(
    row: dict[str, Any],
    *,
    config: SafeModeConfig | None = None,
) -> str:
    """Determine freshness/completeness of the market quote."""
    if config is None:
        config = SafeModeConfig.default_safe()

    over_odds = row.get("over_odds")
    under_odds = row.get("under_odds")
    quote_age = row.get("quote_age_seconds") or 0

    if over_odds is None or under_odds is None:
        return "MISSING_SIDE"
    if quote_age > config.quote_stale_seconds:
        return "STALE"
    if quote_age > config.quote_stale_seconds // 2:
        return "AGING"
    return "FRESH"


def assign_data_quality_status(
    row: dict[str, Any],
) -> str:
    """Validate input data freshness."""
    feature_cutoff = row.get("feature_cutoff_utc")
    if feature_cutoff is None:
        return "MISSING"
    try:
        cutoff_dt = pd.to_datetime(feature_cutoff, utc=True)
        age_days = (datetime.now(timezone.utc) - cutoff_dt).days
        if age_days > 3:
            return "STALE"
        return "FRESH"
    except Exception:
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# No-vig helpers (corrected per spec §4.5)
# ---------------------------------------------------------------------------

def american_to_implied(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def no_vig_normalize(p_over_raw: float, p_under_raw: float) -> tuple[float, float]:
    """Remove vig: normalize over/under implied probabilities to sum to 1."""
    total = p_over_raw + p_under_raw
    if total <= 0:
        return float("nan"), float("nan")
    return p_over_raw / total, p_under_raw / total


def american_to_no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Convert American odds to no-vig probabilities."""
    p_over_raw = american_to_implied(over_odds)
    p_under_raw = american_to_implied(under_odds)
    return no_vig_normalize(p_over_raw, p_under_raw)


def fair_odds_american(prob: float) -> float:
    """Convert fair probability to American odds."""
    if prob <= 0 or prob >= 1:
        return float("nan")
    if prob >= 0.5:
        return -(prob / (1.0 - prob)) * 100.0
    return ((1.0 - prob) / prob) * 100.0


# ---------------------------------------------------------------------------
# Model edge (not CLV)
# ---------------------------------------------------------------------------

def compute_model_edge(
    model_p_over: float,
    market_no_vig_p_over: float,
) -> float:
    """Compute model edge vs the no-vig market probability.

    This is model_edge_over = model_p_over - market_no_vig_p_over.
    This is NOT closing-line value (CLV). CLV requires a closing quote.
    Positive = model is more bullish on over than market.
    """
    if np.isnan(model_p_over) or np.isnan(market_no_vig_p_over):
        return float("nan")
    return model_p_over - market_no_vig_p_over


# ---------------------------------------------------------------------------
# Lineage fields for prediction rows
# ---------------------------------------------------------------------------

REQUIRED_LINEAGE_COLUMNS = [
    "prediction_timestamp_utc",
    "feature_cutoff_utc",
    "training_cutoff_utc",
    "calibration_cutoff_utc",
    "model_version",
    "config_hash",
    "feature_manifest_hash",
]


def add_lineage_to_df(df: pd.DataFrame, lineage: ModelLineage) -> pd.DataFrame:
    """Add lineage columns to a prediction DataFrame."""
    df = df.copy()
    now = lineage.prediction_timestamp_utc or datetime.now(timezone.utc).isoformat()
    if "prediction_timestamp_utc" not in df.columns:
        df["prediction_timestamp_utc"] = now
    if "model_version" not in df.columns:
        df["model_version"] = lineage.model_version
    if "config_hash" not in df.columns:
        df["config_hash"] = lineage.config_hash
    if "feature_manifest_hash" not in df.columns:
        df["feature_manifest_hash"] = lineage.feature_manifest_hash
    if "training_cutoff_utc" not in df.columns:
        df["training_cutoff_utc"] = lineage.training_cutoff
    if "calibration_cutoff_utc" not in df.columns:
        df["calibration_cutoff_utc"] = lineage.calibration_cutoff
    return df
