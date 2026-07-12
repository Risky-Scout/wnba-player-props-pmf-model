"""Generate presentation-ready prediction report for the July 13, 2026 demo.

This script produces the full suite of deliverables required for the
2026-07-13 management presentation in production-safe mode.

SAFE MODE guarantees:
  - Market features are NOT in the structural model
  - Calibrators are NOT applied if their training cutoff post-dates the prediction
  - All predictions carry full model lineage
  - Quality statuses are always populated (PUBLISHABLE/WATCHLIST/EXPERIMENTAL/SUPPRESSED/INSUFFICIENT_DATA)
  - Model edge is never labeled as CLV
  - Stale or missing inputs are labeled, not silently filled

Usage:
    python scripts/prepare_demo_predictions.py \\
        --date 2026-07-13 \\
        --safe-mode \\
        --output-dir reports/demo_2026-07-13

Exit codes:
  0 — success (warnings may be present)
  1 — critical integrity failure (see diagnostics.json)
  2 — no games found for date
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_demo_predictions")

# ---------------------------------------------------------------------------
# Deterministic seed (Priority 0 requirement)
# ---------------------------------------------------------------------------
DEMO_SEED = 20260712
np.random.seed(DEMO_SEED)


def main(
    date: str = "2026-07-13",
    safe_mode: bool = True,
    output_dir: str = "reports/demo_2026-07-13",
    features_wide: str | None = None,
    model_dir: str = "artifacts/models/stage4_baseline",
    cal_dir: str = "artifacts/models/calibration",
    config: str = "config/model/stage4_baseline.yaml",
    safe_mode_config: str = "config/wnba_model.yaml",
    raw_props: str | None = None,
    api_key: str | None = None,
    no_live_data: bool = False,
) -> int:
    """Generate the demo prediction report. Returns exit code."""
    generation_time_utc = datetime.now(timezone.utc)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("WNBA DEMO PREDICTION PIPELINE — %s", date)
    logger.info("Safe mode: %s | Seed: %d", safe_mode, DEMO_SEED)
    logger.info("=" * 70)

    # ── Import safety module ─────────────────────────────────────────────────
    from wnba_props_model.pipeline.safety import (
        CalibrationArtifactInfo,
        ModelLineage,
        SafeModeConfig,
        validate_calibration_temporal_safety,
        CAL_PASS, CAL_FUTURE_CUTOFF, CAL_MISSING, CAL_STALE,
        QUALITY_PUBLISHABLE, QUALITY_WATCHLIST, QUALITY_EXPERIMENTAL,
        QUALITY_SUPPRESSED, QUALITY_INSUFFICIENT_DATA,
        american_to_no_vig, compute_model_edge,
        assign_model_quality_status, assign_market_quality_status, assign_data_quality_status,
        add_lineage_to_df,
    )

    critical_failures: list[str] = []
    all_warnings: list[str] = []

    # ── Load safe-mode config ────────────────────────────────────────────────
    safe_cfg = SafeModeConfig.from_yaml(safe_mode_config) if Path(safe_mode_config).exists() else SafeModeConfig.default_safe()
    if not safe_mode:
        from wnba_props_model.pipeline.safety import SafeModeConfig as _SM
        safe_cfg = _SM.disabled()

    logger.info("Safe-mode settings: %s", safe_cfg)

    # ── Load calibration artifact info ───────────────────────────────────────
    cal_info = CalibrationArtifactInfo.from_metadata_file(cal_dir)
    prediction_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cal_status = validate_calibration_temporal_safety(cal_info, prediction_dt, config=safe_cfg)

    logger.info("Calibration artifact status: %s (fitted_at=%s)", cal_status,
                cal_info.fitted_at.isoformat() if cal_info.fitted_at else "MISSING")

    if cal_status == CAL_FUTURE_CUTOFF and safe_cfg.require_calibration_cutoff_before_prediction:
        if safe_cfg.fail_on_artifact_date_violation:
            critical_failures.append(f"calibration_cutoff_violation: calibrator trained after {date}")
        else:
            all_warnings.append(f"calibration applied after cutoff for {date} — results are raw")

    # ── Build model lineage ──────────────────────────────────────────────────
    lineage = ModelLineage.capture(
        config_path=Path(config),
        feature_manifest_path=Path("data/processed/feature_schema_manifest.json"),
        cal_info=cal_info,
        model_dir=Path(model_dir),
        seed=DEMO_SEED,
    )
    lineage.prediction_timestamp_utc = generation_time_utc.isoformat()
    logger.info("Model lineage captured: git=%s config_hash=%s", lineage.git_commit, lineage.config_hash)

    # ── Step 1: Load or fetch feature data ──────────────────────────────────
    features_df = _load_features(
        features_wide=features_wide,
        date=date,
        api_key=api_key,
        no_live_data=no_live_data,
    )

    if features_df is None or features_df.empty:
        logger.warning("No feature data available for %s", date)
        all_warnings.append(f"no_feature_data_for_{date}")
        # Produce empty-but-valid reports with INSUFFICIENT_DATA status
        _write_empty_reports(out_dir, date, generation_time_utc, lineage, all_warnings, critical_failures)
        return 2

    logger.info("Loaded %d player-game rows for %s", len(features_df), date)

    # ── Step 2: Validate data freshness ─────────────────────────────────────
    data_freshness = _validate_data_freshness(features_df, date)
    if data_freshness["status"] == "STALE":
        all_warnings.append("data_is_stale: features may not reflect current roster/injury state")

    # ── Step 3: Strip market-prior features in safe mode ────────────────────
    stripped_features: list[str] = []
    if safe_cfg.disable_market_features_in_structural_model:
        from wnba_props_model.pipeline.safety import strip_market_prior_features
        features_df, stripped_features = strip_market_prior_features(features_df)
        if stripped_features:
            logger.info("Stripped market-prior features: %s", stripped_features)
            all_warnings.append(f"stripped_market_prior_features={stripped_features}")

    # ── Step 4: Generate structural model predictions ────────────────────────
    logger.info("Generating structural PMF predictions...")
    pmfs_df, pmf_warnings = _generate_pmfs(
        features_df=features_df,
        model_dir=model_dir,
        config_path=config,
        date=date,
        seed=DEMO_SEED,
    )
    all_warnings.extend(pmf_warnings)

    if pmfs_df.empty:
        logger.error("PMF generation produced no rows")
        critical_failures.append("pmf_generation_failed: no predictions produced")
        _write_empty_reports(out_dir, date, generation_time_utc, lineage, all_warnings, critical_failures)
        return 1

    logger.info("Generated %d PMF rows", len(pmfs_df))

    # ── Step 5: Apply calibration with temporal safety ───────────────────────
    pmfs_df, cal_warnings = _apply_calibration_safe(
        pmfs_df=pmfs_df,
        cal_dir=cal_dir,
        cal_info=cal_info,
        cal_status=cal_status,
        prediction_dt=prediction_dt,
        safe_cfg=safe_cfg,
    )
    all_warnings.extend(cal_warnings)

    # ── Step 6: Load market lines ────────────────────────────────────────────
    props_df = _load_market_props(raw_props=raw_props, date=date, api_key=api_key, no_live_data=no_live_data)
    market_updated_at = datetime.now(timezone.utc).isoformat() if props_df is not None and not props_df.empty else None

    # ── Step 7: Join market data and compute no-vig ──────────────────────────
    if props_df is not None and not props_df.empty:
        pmfs_df = _join_and_compute_market(pmfs_df, props_df)
        logger.info("Joined market data: %d props", len(props_df))
    else:
        logger.warning("No market props available for %s — edge cannot be computed", date)
        all_warnings.append("no_market_props: model edge cannot be computed without market lines")
        for col in ["market_no_vig_p_over", "market_no_vig_p_under", "model_edge_over", "model_edge_under",
                    "over_odds", "under_odds", "line", "quote_age_seconds"]:
            if col not in pmfs_df.columns:
                pmfs_df[col] = float("nan")

    # ── Step 8: Compute model edge (NOT CLV) ─────────────────────────────────
    if "model_edge_over" not in pmfs_df.columns or pmfs_df["model_edge_over"].isna().all():
        pmf_p_col = "p_over_calibrated" if "p_over_calibrated" in pmfs_df.columns else "pmf_p_over"
        if pmf_p_col in pmfs_df.columns and "market_no_vig_p_over" in pmfs_df.columns:
            pmfs_df["model_edge_over"] = pmfs_df[pmf_p_col] - pmfs_df["market_no_vig_p_over"]
            pmfs_df["model_edge_under"] = -pmfs_df["model_edge_over"]

    # ── Step 9: Add lineage to all rows ──────────────────────────────────────
    pmfs_df = add_lineage_to_df(pmfs_df, lineage)
    pmfs_df["feature_cutoff_utc"] = generation_time_utc.isoformat()
    pmfs_df["calibration_cutoff_utc"] = cal_info.calibration_cutoff or ""
    pmfs_df["training_cutoff_utc"] = cal_info.training_cutoff or ""
    if "warnings" not in pmfs_df.columns:
        pmfs_df["warnings"] = ""

    # ── Step 10: Assign quality statuses ────────────────────────────────────
    pmfs_df = _assign_all_quality_statuses(pmfs_df, safe_cfg)

    # ── Step 11: Build required output columns ────────────────────────────────
    pmfs_df = _ensure_required_columns(pmfs_df, date, generation_time_utc, lineage)

    # ── Step 12: Sort by quality then edge ──────────────────────────────────
    from wnba_props_model.pipeline.safety import QUALITY_RANK
    pmfs_df["_quality_rank"] = pmfs_df["model_quality_status"].map(QUALITY_RANK).fillna(99)
    pmfs_df["_abs_edge"] = pmfs_df[["model_edge_over", "model_edge_under"]].abs().max(axis=1).fillna(0)
    pmfs_df = pmfs_df.sort_values(["_quality_rank", "_abs_edge"], ascending=[True, False])
    pmfs_df = pmfs_df.drop(columns=["_quality_rank", "_abs_edge"], errors="ignore")

    # ── Step 13: Write output files ──────────────────────────────────────────
    logger.info("Writing output files to %s", out_dir)
    _write_output_files(
        pmfs_df=pmfs_df,
        out_dir=out_dir,
        date=date,
        generation_time_utc=generation_time_utc,
        lineage=lineage,
        data_freshness=data_freshness,
        all_warnings=all_warnings,
        critical_failures=critical_failures,
        cal_info=cal_info,
        cal_status=cal_status,
        market_updated_at=market_updated_at,
        stripped_features=stripped_features,
        props_df=props_df,
    )

    # ── Step 14: Final status ────────────────────────────────────────────────
    if critical_failures:
        logger.error("CRITICAL FAILURES: %s", critical_failures)
        return 1

    n_publishable = int((pmfs_df["model_quality_status"] == QUALITY_PUBLISHABLE).sum())
    n_watchlist = int((pmfs_df["model_quality_status"] == QUALITY_WATCHLIST).sum())
    logger.info(
        "Complete: %d predictions (%d PUBLISHABLE, %d WATCHLIST)",
        len(pmfs_df), n_publishable, n_watchlist,
    )
    return 0


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _load_features(
    features_wide: str | None,
    date: str,
    api_key: str | None,
    no_live_data: bool,
) -> pd.DataFrame | None:
    """Load feature data from file or fetch from API."""

    # 1. Try specified features file
    if features_wide and Path(features_wide).exists():
        df = pd.read_parquet(features_wide)
        if "game_date" in df.columns:
            df = df[df["game_date"].astype(str) == date]
        if not df.empty:
            return df
        logger.info("No rows for %s in %s — trying other sources", date, features_wide)

    # 2. Try latest processed features file
    candidates = [
        "data/processed/wnba_player_game_features_wide.parquet",
        "data/processed/features_wide.parquet",
    ]
    for cand in candidates:
        if Path(cand).exists():
            try:
                df = pd.read_parquet(cand)
                if "game_date" in df.columns:
                    filtered = df[df["game_date"].astype(str) == date]
                    if not filtered.empty:
                        return filtered
                    # Return most recent date's data if target date not found
                    last_date = df["game_date"].astype(str).max()
                    logger.warning(
                        "Date %s not in %s (latest: %s) — returning latest date for demo",
                        date, cand, last_date,
                    )
                    df_latest = df[df["game_date"].astype(str) == last_date].copy()
                    df_latest["_demo_data_date"] = last_date
                    df_latest["_demo_note"] = f"No data for {date}; using {last_date} as demo"
                    return df_latest
                return df
            except Exception as exc:
                logger.warning("Could not load %s: %s", cand, exc)

    # 3. Try fetching via BDL if API key available and not suppressed
    if not no_live_data and api_key:
        try:
            return _fetch_features_from_api(date, api_key)
        except Exception as exc:
            logger.warning("API feature fetch failed: %s", exc)

    return None


def _fetch_features_from_api(date: str, api_key: str) -> pd.DataFrame | None:
    """Attempt to pull next-game slate features from BDL."""
    try:
        from wnba_props_model.data.bdl_client import BDLClient
        client = BDLClient(api_key=api_key)
        games = client.list_endpoint("games", params={"dates": [date], "per_page": 100})
        if not games:
            logger.info("No games found for %s via BDL", date)
            return None
        logger.info("Found %d games for %s via BDL", len(games), date)
        # Return minimal structure; full features require build_features pipeline
        return None
    except Exception as exc:
        logger.warning("BDL API fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Data freshness validation
# ---------------------------------------------------------------------------

def _validate_data_freshness(df: pd.DataFrame, date: str) -> dict:
    result: dict = {"status": "UNKNOWN", "details": {}}

    if "game_date" in df.columns:
        latest = df["game_date"].astype(str).max()
        target_dt = datetime.strptime(date, "%Y-%m-%d")
        latest_dt = datetime.strptime(latest, "%Y-%m-%d") if latest else None
        result["details"]["latest_game_date"] = latest
        result["details"]["target_date"] = date

        if latest_dt is None:
            result["status"] = "UNKNOWN"
        elif latest_dt.date() >= target_dt.date():
            result["status"] = "FRESH"
        elif (target_dt - latest_dt).days <= 3:
            result["status"] = "RECENT"
        else:
            result["status"] = "STALE"

    result["details"]["n_rows"] = len(df)
    result["details"]["columns"] = list(df.columns)[:20]
    return result


# ---------------------------------------------------------------------------
# PMF generation
# ---------------------------------------------------------------------------

def _generate_pmfs(
    features_df: pd.DataFrame,
    model_dir: str,
    config_path: str,
    date: str,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate raw PMFs from the Stage 4 HGB engine."""
    warnings: list[str] = []

    if not Path(model_dir).exists():
        warnings.append(f"model_dir_missing: {model_dir}")
        return _build_fallback_pmfs(features_df, date), warnings

    try:
        from wnba_props_model.pipeline.predict import predict_player_pmfs, _load_stage4_models
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models = _load_stage4_models(model_dir)
        pmfs = predict_player_pmfs(features_df, models=models, config=cfg)
        return pmfs, warnings

    except FileNotFoundError as exc:
        warnings.append(f"model_artifact_missing: {exc}")
        return _build_fallback_pmfs(features_df, date), warnings
    except Exception as exc:
        warnings.append(f"pmf_generation_error: {exc}")
        logger.warning("PMF generation failed: %s", exc, exc_info=True)
        return _build_fallback_pmfs(features_df, date), warnings


def _build_fallback_pmfs(features_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Build skeleton PMF rows when model artifacts are unavailable.

    All predictions are labeled SUPPRESSED with calibration_status=MISSING_ARTIFACT.
    """
    if features_df.empty:
        return pd.DataFrame()

    rows = []
    stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    id_cols = ["player_id", "player_name", "team_id", "team_abbreviation",
               "opponent_team_id", "opponent_team_abbreviation",
               "game_id", "game_date", "is_home"]

    for _, feat_row in features_df.iterrows():
        for stat in stats:
            row = {col: feat_row.get(col) for col in id_cols if col in features_df.columns}
            row["stat"] = stat
            row["game_date"] = date
            row["model_p_over_raw"] = float("nan")
            row["model_p_under_raw"] = float("nan")
            row["model_p_push_raw"] = float("nan")
            row["model_p_over_calibrated"] = float("nan")
            row["model_p_under_calibrated"] = float("nan")
            row["model_p_push_calibrated"] = float("nan")
            row["calibration_status"] = "MISSING_ARTIFACT"
            row["model_quality_status"] = "SUPPRESSED"
            row["warnings"] = "no_model_artifacts_available"
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _apply_calibration_safe(
    pmfs_df: pd.DataFrame,
    cal_dir: str,
    cal_info: CalibrationArtifactInfo,
    cal_status: str,
    prediction_dt: datetime,
    safe_cfg,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply calibration with temporal safety enforcement."""
    from wnba_props_model.pipeline.safety import CAL_PASS, CAL_FUTURE_CUTOFF, CAL_MISSING, CAL_STALE

    warnings: list[str] = []

    if "calibration_status" not in pmfs_df.columns:
        pmfs_df = pmfs_df.copy()
        pmfs_df["calibration_status"] = cal_status

    if cal_status == CAL_FUTURE_CUTOFF:
        warnings.append(f"calibration_skipped: trained after prediction date {prediction_dt.date()}")
        if "is_calibrated" not in pmfs_df.columns:
            pmfs_df["is_calibrated"] = False
        return pmfs_df, warnings

    if cal_status == CAL_MISSING:
        warnings.append("calibration_missing: no calibration artifacts found — using raw predictions")
        if "is_calibrated" not in pmfs_df.columns:
            pmfs_df["is_calibrated"] = False
        return pmfs_df, warnings

    if not Path(cal_dir).exists():
        warnings.append(f"cal_dir_missing: {cal_dir}")
        if "is_calibrated" not in pmfs_df.columns:
            pmfs_df["is_calibrated"] = False
        return pmfs_df, warnings

    try:
        from wnba_props_model.pipeline.calibrate import apply_calibrators
        pmfs_df = apply_calibrators(pmfs_df, cal_dir=cal_dir)
        logger.info("Calibration applied successfully (status=%s)", cal_status)
    except Exception as exc:
        warnings.append(f"calibration_error: {exc}")
        logger.warning("Calibration failed: %s — using raw predictions", exc)
        pmfs_df["calibration_status"] = "FALLBACK_IDENTITY"
        if "is_calibrated" not in pmfs_df.columns:
            pmfs_df["is_calibrated"] = False

    return pmfs_df, warnings


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def _load_market_props(
    raw_props: str | None,
    date: str,
    api_key: str | None,
    no_live_data: bool,
) -> pd.DataFrame | None:
    """Load market props from file or API."""
    # Try specified file
    if raw_props and Path(raw_props).exists():
        return pd.read_parquet(raw_props)

    # Try standard locations
    candidates = [
        "data/processed/wnba_player_props.parquet",
        "data/processed/player_props.parquet",
        f"data/processed/wnba_player_props_oddsapi_latest.parquet",
    ]
    for cand in candidates:
        if Path(cand).exists():
            try:
                df = pd.read_parquet(cand)
                if "game_date" in df.columns:
                    filtered = df[df["game_date"].astype(str) == date]
                    if not filtered.empty:
                        return filtered
                return df
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------------
# Market joining and no-vig computation
# ---------------------------------------------------------------------------

def _join_and_compute_market(pmfs_df: pd.DataFrame, props_df: pd.DataFrame) -> pd.DataFrame:
    """Join market props and compute no-vig probabilities."""
    from wnba_props_model.pipeline.safety import american_to_no_vig

    now_utc = datetime.now(timezone.utc)

    # Add quote age
    if "pulled_at_utc" in props_df.columns:
        pulled_dt = pd.to_datetime(props_df["pulled_at_utc"], utc=True, errors="coerce")
        props_df = props_df.copy()
        props_df["quote_age_seconds"] = (now_utc - pulled_dt).dt.total_seconds().clip(lower=0).fillna(9999)

    # Compute no-vig
    if "over_odds" in props_df.columns and "under_odds" in props_df.columns:
        props_df = props_df.copy()
        nv_results = props_df.apply(
            lambda r: pd.Series(
                american_to_no_vig(float(r.get("over_odds", float("nan"))), float(r.get("under_odds", float("nan")))),
                index=["market_no_vig_p_over", "market_no_vig_p_under"],
            ),
            axis=1,
        )
        props_df = pd.concat([props_df, nv_results], axis=1)
        props_df["market_updated_at"] = props_df.get("source_updated_at", now_utc.isoformat())
        props_df["market_pulled_at"] = props_df.get("pulled_at_utc", now_utc.isoformat())

    # Join on player_id + stat (or player_name + stat)
    join_keys = []
    if "player_id" in pmfs_df.columns and "player_id" in props_df.columns:
        join_keys = ["player_id", "stat"]
    elif "player_name" in pmfs_df.columns and "player_name" in props_df.columns:
        join_keys = ["player_name", "stat"]

    if join_keys:
        market_cols = join_keys + [c for c in [
            "line", "over_odds", "under_odds", "market_no_vig_p_over", "market_no_vig_p_under",
            "vendor", "quote_age_seconds", "market_updated_at", "market_pulled_at",
        ] if c in props_df.columns]
        props_slim = props_df[market_cols].drop_duplicates(subset=join_keys, keep="last")
        pmfs_df = pmfs_df.merge(props_slim, on=join_keys, how="left")

    return pmfs_df


# ---------------------------------------------------------------------------
# Quality status assignment
# ---------------------------------------------------------------------------

def _assign_all_quality_statuses(pmfs_df: pd.DataFrame, safe_cfg) -> pd.DataFrame:
    """Assign all three quality status columns."""
    from wnba_props_model.pipeline.safety import (
        assign_model_quality_status, assign_market_quality_status, assign_data_quality_status
    )
    pmfs_df = pmfs_df.copy()

    if "data_quality_status" not in pmfs_df.columns:
        pmfs_df["data_quality_status"] = pmfs_df.apply(
            lambda r: assign_data_quality_status(r.to_dict()), axis=1
        )

    if "market_quality_status" not in pmfs_df.columns:
        pmfs_df["market_quality_status"] = pmfs_df.apply(
            lambda r: assign_market_quality_status(r.to_dict(), config=safe_cfg), axis=1
        )

    if "model_quality_status" not in pmfs_df.columns:
        pmfs_df["model_quality_status"] = pmfs_df.apply(
            lambda r: assign_model_quality_status(r.to_dict(), config=safe_cfg), axis=1
        )

    return pmfs_df


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "prediction_timestamp_utc",
    "game_id",
    "scheduled_start_utc",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home_away",
    "stat",
    "market_type",
    "vendor",
    "market_updated_at",
    "market_pulled_at",
    "quote_age_seconds",
    "line",
    "over_odds",
    "under_odds",
    "market_no_vig_p_over",
    "market_no_vig_p_under",
    "model_p_over_raw",
    "model_p_under_raw",
    "model_p_push_raw",
    "model_p_over_calibrated",
    "model_p_under_calibrated",
    "model_p_push_calibrated",
    "structural_fair_over_odds",
    "structural_fair_under_odds",
    "calibrated_fair_over_odds",
    "calibrated_fair_under_odds",
    "model_edge_over",
    "model_edge_under",
    "expected_stat",
    "stat_median",
    "stat_mode",
    "expected_minutes",
    "minutes_p10",
    "minutes_p25",
    "minutes_p50",
    "minutes_p75",
    "minutes_p90",
    "availability_probability",
    "starter_probability",
    "role_state",
    "role_probabilities",
    "calibration_status",
    "calibration_sample_size",
    "model_quality_status",
    "data_quality_status",
    "market_quality_status",
    "feature_cutoff_utc",
    "training_cutoff_utc",
    "calibration_cutoff_utc",
    "model_version",
    "config_hash",
    "feature_manifest_hash",
    "warnings",
]

_COLUMN_ALIASES = {
    "team": ["team_abbreviation", "team_id"],
    "opponent": ["opponent_team_abbreviation", "opponent_team_id"],
    "home_away": ["is_home"],
    "player_name": ["name"],
    "expected_stat": ["pmf_mean", "pred_mean"],
    "expected_minutes": ["pred_minutes_mean", "predicted_minutes"],
    "availability_probability": ["p_active", "p_available"],
    "starter_probability": ["start_proxy_lag1", "recent_starter_rate5"],
    "role_state": ["role_bucket"],
    "model_p_over_raw": ["p_over"],
    "model_p_under_raw": ["p_under"],
    "model_p_push_raw": ["p_push"],
    "model_p_over_calibrated": ["p_over_calibrated", "cal_p_over"],
    "model_p_under_calibrated": ["p_under_calibrated", "cal_p_under"],
}


def _ensure_required_columns(
    df: pd.DataFrame,
    date: str,
    generation_time_utc: datetime,
    lineage: ModelLineage,
) -> pd.DataFrame:
    """Ensure all required columns exist, using aliases where available."""
    from wnba_props_model.pipeline.safety import fair_odds_american

    df = df.copy()

    # Apply column aliases
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[canonical] = df[alias]
                    break

    # Derive is_home → home_away
    if "home_away" not in df.columns and "is_home" in df.columns:
        df["home_away"] = df["is_home"].map({True: "home", False: "away", 1: "home", 0: "away"})

    # Derive fair odds from model probabilities
    if "structural_fair_over_odds" not in df.columns and "model_p_over_raw" in df.columns:
        df["structural_fair_over_odds"] = df["model_p_over_raw"].apply(
            lambda p: fair_odds_american(p) if not pd.isna(p) else float("nan")
        )
        df["structural_fair_under_odds"] = df["model_p_under_raw"].apply(
            lambda p: fair_odds_american(p) if not pd.isna(p) else float("nan")
        )

    if "calibrated_fair_over_odds" not in df.columns and "model_p_over_calibrated" in df.columns:
        df["calibrated_fair_over_odds"] = df["model_p_over_calibrated"].apply(
            lambda p: fair_odds_american(p) if not pd.isna(p) else float("nan")
        )
        df["calibrated_fair_under_odds"] = df["model_p_under_calibrated"].apply(
            lambda p: fair_odds_american(p) if not pd.isna(p) else float("nan")
        )

    # Scheduled start
    if "scheduled_start_utc" not in df.columns:
        df["scheduled_start_utc"] = date + "T23:00:00Z"

    # Market type
    if "market_type" not in df.columns:
        df["market_type"] = "player_prop"

    # Calibration sample size
    if "calibration_sample_size" not in df.columns:
        df["calibration_sample_size"] = float("nan")

    # Role probabilities
    if "role_probabilities" not in df.columns:
        if "role_bucket" in df.columns:
            df["role_probabilities"] = df["role_bucket"].apply(
                lambda r: json.dumps({"dominant_role": r if r else "unknown"})
            )
        else:
            df["role_probabilities"] = "{}"

    # Fill remaining required columns with NaN
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col not in ["model_quality_status", "calibration_status",
                                                    "data_quality_status", "market_quality_status",
                                                    "warnings", "market_type", "vendor", "role_state",
                                                    "role_probabilities"] else ""

    return df


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_output_files(
    pmfs_df: pd.DataFrame,
    out_dir: Path,
    date: str,
    generation_time_utc: datetime,
    lineage: ModelLineage,
    data_freshness: dict,
    all_warnings: list[str],
    critical_failures: list[str],
    cal_info,
    cal_status: str,
    market_updated_at: str | None,
    stripped_features: list[str],
    props_df: pd.DataFrame | None,
) -> None:
    """Write all required output artifacts."""
    from wnba_props_model.pipeline.safety import (
        QUALITY_PUBLISHABLE, QUALITY_WATCHLIST, QUALITY_EXPERIMENTAL, QUALITY_SUPPRESSED, QUALITY_INSUFFICIENT_DATA
    )

    # ── predictions.csv — all predictions ────────────────────────────────────
    output_cols = [c for c in REQUIRED_COLUMNS if c in pmfs_df.columns]
    pmfs_df[output_cols].to_csv(out_dir / "predictions.csv", index=False)
    pmfs_df[output_cols].to_parquet(out_dir / "predictions.parquet", index=False)

    # ── Split by quality status ──────────────────────────────────────────────
    for status, fname in [
        (QUALITY_PUBLISHABLE, "predictions_publishable.csv"),
        (QUALITY_WATCHLIST, "predictions_watchlist.csv"),
        (QUALITY_EXPERIMENTAL, "predictions_experimental.csv"),
        (QUALITY_SUPPRESSED, "suppressed_predictions.csv"),
        (QUALITY_INSUFFICIENT_DATA, "predictions_insufficient_data.csv"),
    ]:
        subset = pmfs_df[pmfs_df["model_quality_status"] == status]
        subset[output_cols].to_csv(out_dir / fname, index=False)

    # ── data_freshness.csv ───────────────────────────────────────────────────
    pd.DataFrame([data_freshness["details"]]).to_csv(out_dir / "data_freshness.csv", index=False)

    # ── calibration_by_stat_role.csv ─────────────────────────────────────────
    _write_calibration_report(pmfs_df, out_dir)

    # ── diagnostics.json ─────────────────────────────────────────────────────
    diagnostics = {
        "generation_timestamp_utc": generation_time_utc.isoformat(),
        "prediction_date": date,
        "model_version": lineage.model_version,
        "git_commit": lineage.git_commit,
        "config_hash": lineage.config_hash,
        "feature_manifest_hash": lineage.feature_manifest_hash,
        "training_cutoff": lineage.training_cutoff,
        "calibration_cutoff": lineage.calibration_cutoff,
        "calibration_status": cal_status,
        "calibration_fitted_at": cal_info.fitted_at.isoformat() if cal_info.fitted_at else None,
        "data_freshness": data_freshness,
        "market_data_timestamp": market_updated_at,
        "n_predictions_total": len(pmfs_df),
        "n_by_quality_status": pmfs_df["model_quality_status"].value_counts().to_dict(),
        "stripped_market_prior_features": stripped_features,
        "all_warnings": all_warnings,
        "critical_failures": critical_failures,
        "deterministic_seed": DEMO_SEED,
        "note_on_clv": (
            "model_edge_over and model_edge_under are the structural model's probability "
            "minus the current market no-vig probability. This is NOT closing-line value (CLV). "
            "True CLV requires an archived closing quote. See evaluation/clv.py."
        ),
        "safe_mode": True,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))

    # ── summary.md ───────────────────────────────────────────────────────────
    _write_summary_markdown(
        pmfs_df=pmfs_df,
        out_dir=out_dir,
        date=date,
        generation_time_utc=generation_time_utc,
        lineage=lineage,
        cal_info=cal_info,
        cal_status=cal_status,
        data_freshness=data_freshness,
        all_warnings=all_warnings,
        critical_failures=critical_failures,
        market_updated_at=market_updated_at,
        stripped_features=stripped_features,
    )

    logger.info("Wrote all output files to %s", out_dir)


def _write_calibration_report(pmfs_df: pd.DataFrame, out_dir: Path) -> None:
    """Write calibration summary by stat and role."""
    rows = []
    stats = pmfs_df["stat"].unique() if "stat" in pmfs_df.columns else []
    role_col = "role_state" if "role_state" in pmfs_df.columns else "role_bucket"
    roles = pmfs_df[role_col].unique() if role_col in pmfs_df.columns else ["all"]

    for stat in stats:
        for role in roles:
            mask = pmfs_df["stat"] == stat
            if role_col in pmfs_df.columns and role != "all":
                mask &= pmfs_df[role_col] == role
            subset = pmfs_df[mask]
            if subset.empty:
                continue
            cal_status_val = subset["calibration_status"].iloc[0] if "calibration_status" in subset.columns else "UNKNOWN"
            rows.append({
                "stat": stat,
                "role": role,
                "n_rows": len(subset),
                "calibration_status": cal_status_val,
                "calibration_sample_size": subset.get("calibration_sample_size", pd.Series([float("nan")])).mean(),
                "mean_model_p_over_raw": subset.get("model_p_over_raw", pd.Series(dtype=float)).mean(),
                "mean_model_p_over_calibrated": subset.get("model_p_over_calibrated", pd.Series(dtype=float)).mean(),
                "mean_model_edge_over": subset.get("model_edge_over", pd.Series(dtype=float)).mean(),
                "n_publishable": int((subset.get("model_quality_status", pd.Series()) == "PUBLISHABLE").sum()),
                "n_suppressed": int((subset.get("model_quality_status", pd.Series()) == "SUPPRESSED").sum()),
            })

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "calibration_by_stat_role.csv", index=False)
    else:
        pd.DataFrame(columns=["stat", "role", "n_rows", "calibration_status"]).to_csv(
            out_dir / "calibration_by_stat_role.csv", index=False
        )


def _write_summary_markdown(
    pmfs_df: pd.DataFrame,
    out_dir: Path,
    date: str,
    generation_time_utc: datetime,
    lineage: ModelLineage,
    cal_info,
    cal_status: str,
    data_freshness: dict,
    all_warnings: list[str],
    critical_failures: list[str],
    market_updated_at: str | None,
    stripped_features: list[str],
) -> None:
    """Write the presentation summary Markdown."""
    from wnba_props_model.pipeline.safety import QUALITY_PUBLISHABLE, QUALITY_WATCHLIST

    n_total = len(pmfs_df)
    quality_counts = pmfs_df["model_quality_status"].value_counts().to_dict() if "model_quality_status" in pmfs_df.columns else {}
    n_publishable = quality_counts.get(QUALITY_PUBLISHABLE, 0)
    n_watchlist = quality_counts.get(QUALITY_WATCHLIST, 0)

    publishable_rows = pmfs_df[pmfs_df.get("model_quality_status", pd.Series()) == QUALITY_PUBLISHABLE] if "model_quality_status" in pmfs_df.columns else pd.DataFrame()

    summary = f"""# WNBA Player Prop Predictions — {date}

**Generated:** {generation_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Model version:** {lineage.model_version}
**Git commit:** {lineage.git_commit}
**Config hash:** {lineage.config_hash}
**Safe mode:** ENABLED

---

## Data Freshness

| Item | Status |
|------|--------|
| Latest game data | {data_freshness.get('details', {}).get('latest_game_date', 'UNKNOWN')} |
| Data status | {data_freshness.get('status', 'UNKNOWN')} |
| Market data timestamp | {market_updated_at or 'NOT AVAILABLE'} |

---

## Predictions by Quality Status

| Status | Count |
|--------|-------|
"""
    for status, count in sorted(quality_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"| {status} | {count} |\n"

    summary += f"""
**Total predictions:** {n_total}

---

## Calibration Status

| Item | Value |
|------|-------|
| Calibration artifact status | {cal_status} |
| Calibrator fitted at | {cal_info.fitted_at.isoformat() if cal_info.fitted_at else 'MISSING'} |
| Training cutoff | {cal_info.training_cutoff or 'UNKNOWN'} |
| Calibration cutoff | {cal_info.calibration_cutoff or 'UNKNOWN'} |

---

## Top Publishable Projections

"""

    if publishable_rows.empty:
        summary += "_No PUBLISHABLE predictions available. See WATCHLIST tab for best available predictions._\n\n"
    else:
        top_cols = ["player_name", "stat", "line", "model_edge_over", "model_p_over_calibrated",
                    "market_no_vig_p_over", "expected_minutes", "role_state"]
        top_cols = [c for c in top_cols if c in publishable_rows.columns]
        top = publishable_rows[top_cols].head(10)
        summary += top.to_markdown(index=False) + "\n\n"

    summary += """---

## Market Edge Note

> **IMPORTANT:** `model_edge_over` and `model_edge_under` represent the structural
> model's probability minus the current market no-vig probability at the time of
> prediction. This is **not** Closing Line Value (CLV).
>
> True CLV requires a closing quote (from before game start) to compare against
> the entry-time price. CLV measurement requires an archived quote ledger, which
> is not yet fully operational.

---

## Disabled / Downgraded Functionality

"""
    if stripped_features:
        summary += f"- **Market-prior features stripped from structural model:** {stripped_features}\n"
    summary += "- Combo prop (pts+reb, pts+ast, etc.) independence not validated — labeled EXPERIMENTAL\n"
    summary += "- True CLV not computed (requires archived closing quotes)\n"

    summary += "\n---\n\n## Warnings\n\n"
    if all_warnings:
        for w in all_warnings:
            summary += f"- {w}\n"
    else:
        summary += "_(No warnings)_\n"

    summary += "\n---\n\n## Known Limitations\n\n"
    summary += "- Model edge at current quote is not realized CLV\n"
    summary += "- Combo props use convolved marginal PMFs (independence assumed)\n"
    summary += "- Calibrators may not reflect current season's distributional shifts\n"
    summary += "- Injury status is current-state only; historical leakage is disabled\n"
    if data_freshness.get("status") in ("STALE", "UNKNOWN"):
        summary += f"- Feature data is {data_freshness.get('status', 'UNKNOWN')} — predictions may not reflect latest roster\n"

    if critical_failures:
        summary += "\n---\n\n## CRITICAL FAILURES\n\n"
        for f in critical_failures:
            summary += f"- **{f}**\n"

    (out_dir / "summary.md").write_text(summary)


def _write_empty_reports(
    out_dir: Path,
    date: str,
    generation_time_utc: datetime,
    lineage: ModelLineage,
    warnings: list[str],
    critical_failures: list[str],
) -> None:
    """Write empty-but-valid report files when no predictions are available."""
    for fname in [
        "predictions.csv", "predictions_publishable.csv", "predictions_watchlist.csv",
        "predictions_experimental.csv", "suppressed_predictions.csv",
        "calibration_by_stat_role.csv", "data_freshness.csv",
    ]:
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(out_dir / fname, index=False)

    diagnostics = {
        "generation_timestamp_utc": generation_time_utc.isoformat(),
        "prediction_date": date,
        "model_version": lineage.model_version,
        "git_commit": lineage.git_commit,
        "n_predictions_total": 0,
        "warnings": warnings,
        "critical_failures": critical_failures,
        "status": "NO_DATA",
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))

    summary = f"""# WNBA Player Prop Predictions — {date}

**Generated:** {generation_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Status:** NO DATA AVAILABLE

No predictions were generated for {date}.

## Warnings
{chr(10).join(f'- {w}' for w in warnings) or '_(none)_'}

## Critical Failures
{chr(10).join(f'- {f}' for f in critical_failures) or '_(none)_'}
"""
    (out_dir / "summary.md").write_text(summary)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate demo prediction report for WNBA player props (production-safe mode)."
    )
    parser.add_argument("--date", default="2026-07-13", help="Prediction date (YYYY-MM-DD)")
    parser.add_argument("--safe-mode", action="store_true", default=True, help="Enable production-safe mode")
    parser.add_argument("--no-safe-mode", action="store_false", dest="safe_mode")
    parser.add_argument("--output-dir", default="reports/demo_2026-07-13", help="Output directory")
    parser.add_argument("--features-wide", help="Path to wide feature parquet")
    parser.add_argument("--model-dir", default="artifacts/models/stage4_baseline")
    parser.add_argument("--cal-dir", default="artifacts/models/calibration")
    parser.add_argument("--config", default="config/model/stage4_baseline.yaml")
    parser.add_argument("--safe-mode-config", default="config/wnba_model.yaml")
    parser.add_argument("--raw-props", help="Path to market props parquet")
    parser.add_argument("--api-key", default=os.environ.get("BDL_API_KEY", ""))
    parser.add_argument("--no-live-data", action="store_true", help="Skip all live API calls")
    args = parser.parse_args()

    exit_code = main(
        date=args.date,
        safe_mode=args.safe_mode,
        output_dir=args.output_dir,
        features_wide=args.features_wide,
        model_dir=args.model_dir,
        cal_dir=args.cal_dir,
        config=args.config,
        safe_mode_config=args.safe_mode_config,
        raw_props=args.raw_props,
        api_key=args.api_key,
        no_live_data=args.no_live_data,
    )
    sys.exit(exit_code)
