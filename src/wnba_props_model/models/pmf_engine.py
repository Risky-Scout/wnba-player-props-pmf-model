"""Stage 4 PMF engine.

Converts model predictions (minutes_mean, stat_mean / p_nonzero + pos_mean)
into full discrete atom PMFs over non-negative integer support.

Key invariants:
- All PMFs sum to 1 within 1e-6
- All probabilities non-negative and finite
- Support starts at 0
- is_calibrated = False (Stage 6 will calibrate)
- pmf_source = "stage4_baseline_uncalibrated_model_only"
- No market data used anywhere in this pipeline
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES
from wnba_props_model.models.pmf_utils import (
    hurdle_pmf_batch,
    negbinom_pmf_batch,
    pmf_matrix_to_json_list,
    pmf_mean_var,
    pmf_pge,
    poisson_pmf_batch,
    validate_pmf_matrix,
)

PMF_SOURCE = "stage4_baseline_uncalibrated_model_only"
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_feature_matrix(
    df: pd.DataFrame,
    model_feature_cols: list[str],
    pos_encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder | None]:
    """Convert DataFrame to numeric feature matrix.

    Encodes 'position' column (categorical) using OrdinalEncoder.
    All other columns must already be numeric; NaN is handled by HGBC natively.

    Returns (X, encoder).  encoder is None if 'position' not in model_feature_cols.
    """
    # Verify no forbidden columns
    bad = [c for c in model_feature_cols if c in FORBIDDEN_MODEL_FEATURES]
    if bad:
        raise ValueError(f"Forbidden columns in model_feature_cols: {bad}")

    # Filter to available columns
    available = [c for c in model_feature_cols if c in df.columns]
    X = df[available].copy()

    # Encode categorical position column
    if "position" in X.columns:
        pos_series = X[["position"]].fillna("unknown").astype(str)
        if fit_encoder:
            pos_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            pos_encoder.fit(pos_series)
        if pos_encoder is not None:
            X["position"] = pos_encoder.transform(pos_series).ravel()
        else:
            X["position"] = -1.0

    # Convert booleans to float
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(float)

    # Replace inf
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, pos_encoder


# ---------------------------------------------------------------------------
# Main PMF builder
# ---------------------------------------------------------------------------

def build_all_pmfs(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    model_feature_cols: list[str],
    minutes_model: Any,
    stat_models: dict[str, Any],
    hurdle_models: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build full PMF table (one row per player_id × game_id × stat).

    Returns a DataFrame with columns:
        game_id, game_date, season, player_id, player_name,
        team_id, team_abbreviation, opponent_team_id, opponent_team_abbreviation,
        stat, actual_outcome, actual_minutes, did_play,
        minutes_mean, minutes_sigma,
        stat_mean, stat_variance,
        pmf_json, pmf_support_min, pmf_support_max,
        pmf_mean, pmf_variance, p0, p_ge_1, p_ge_2, p_ge_3, p_ge_5, p_ge_10,
        model_version, pmf_source, is_calibrated
    """
    # ------------------------------------------------------------------ #
    # Prepare wide-table features (used for minutes prediction and as
    # the base features for each stat model)
    # ------------------------------------------------------------------ #
    X_wide, pos_encoder = prepare_feature_matrix(
        wide_df, model_feature_cols, fit_encoder=False
    )
    # pos_encoder was already fitted at training time; load it from the model
    # For inference, just re-use whatever preprocessing was done.
    # (The train script fits the encoder and passes it in the model artifacts.)
    # If the encoder is attached to the minutes_model, use it:
    if hasattr(minutes_model, "_pos_encoder") and minutes_model._pos_encoder is not None:
        X_wide, _ = prepare_feature_matrix(
            wide_df, model_feature_cols,
            pos_encoder=minutes_model._pos_encoder, fit_encoder=False
        )

    # ------------------------------------------------------------------ #
    # Minutes predictions
    # ------------------------------------------------------------------ #
    min_means, min_sigmas = minutes_model.predict(X_wide, wide_df)
    wide_with_min = wide_df.assign(
        minutes_mean=min_means,
        minutes_sigma=min_sigmas,
    )

    # ------------------------------------------------------------------ #
    # Per-stat PMF generation
    # ------------------------------------------------------------------ #
    support_caps = cfg.get("pmf_support_caps", {})
    sparse_stats = set(cfg.get("sparse_stats", ["stl", "blk"]))
    pmf_source = cfg.get("pmf_source", PMF_SOURCE)

    all_frames: list[pd.DataFrame] = []

    for stat in cfg.get("stats", STATS):
        cap = support_caps.get(stat, 20)
        target_col = f"actual_{stat}"

        # Merge long-table rows for this stat with wide-table min predictions
        stat_rows = long_df[long_df["stat"] == stat].copy()
        stat_rows = stat_rows.merge(
            wide_with_min[["player_id", "game_id", "minutes_mean", "minutes_sigma"]],
            on=["player_id", "game_id"], how="left"
        )

        if len(stat_rows) == 0:
            continue

        # Feature matrix for this stat (use wide features, align to stat_rows)
        X_stat = wide_df.set_index(["player_id", "game_id"]).reindex(
            pd.MultiIndex.from_frame(stat_rows[["player_id", "game_id"]])
        ).reset_index(drop=True)
        # Rebuild feature matrix with correct row alignment
        X_stat_df, _ = prepare_feature_matrix(
            X_stat, model_feature_cols,
            pos_encoder=getattr(minutes_model, "_pos_encoder", None),
            fit_encoder=False,
        )

        # ---- Predict stat ------------------------------------------------
        if stat in hurdle_models:
            model = hurdle_models[stat]
            p_nz, pos_mus = model.predict(X_stat_df)
            stat_means = p_nz * pos_mus  # E[Y] = P(Y>0) * E[Y|Y>0]
        else:
            model = stat_models[stat]
            stat_means = model.predict_mean(X_stat_df)
            p_nz = None
            pos_mus = None

        # ---- Build PMF matrix ---------------------------------------------
        pmf_mat = _build_pmf_matrix(
            stat, stat_means, p_nz, pos_mus,
            stat_models, hurdle_models, cap
        )

        validate_pmf_matrix(pmf_mat)

        # ---- Extract summary statistics -----------------------------------
        pmf_means, pmf_vars = pmf_mean_var(pmf_mat)
        p0_arr = pmf_mat[:, 0]
        p_ge_1_arr = pmf_pge(pmf_mat, 1)
        p_ge_2_arr = pmf_pge(pmf_mat, 2)
        p_ge_3_arr = pmf_pge(pmf_mat, 3)
        p_ge_5_arr = pmf_pge(pmf_mat, 5) if cap >= 5 else np.zeros(len(pmf_mat))
        p_ge_10_arr = pmf_pge(pmf_mat, 10) if cap >= 10 else np.zeros(len(pmf_mat))

        # ---- Build PMF JSON strings ----------------------------------------
        pmf_jsons = pmf_matrix_to_json_list(pmf_mat)

        # ---- Assemble output frame ----------------------------------------
        model_version = getattr(model, "VERSION", "stage4_baseline_v1")
        stat_var_arr = np.full(len(stat_rows), float(
            getattr(model, "_global_var",
                    getattr(model, "_pos_var", np.nan))
        ))

        frame = pd.DataFrame({
            "game_id":                  stat_rows["game_id"].values,
            "game_date":                stat_rows["game_date"].values,
            "season":                   stat_rows["season"].values,
            "player_id":                stat_rows["player_id"].values,
            "player_name":              stat_rows["player_name"].values,
            "team_id":                  stat_rows["team_id"].values,
            "team_abbreviation":        stat_rows["team_abbreviation"].values,
            "opponent_team_id":         stat_rows["opponent_team_id"].values,
            "opponent_team_abbreviation": stat_rows["opponent_team_abbreviation"].values
                                        if "opponent_team_abbreviation" in stat_rows.columns
                                        else None,
            "stat":                     stat,
            "actual_outcome":           stat_rows["actual_outcome"].values,
            "actual_minutes":           stat_rows["actual_minutes"].values,
            "did_play":                 stat_rows["did_play"].values
                                        if "did_play" in stat_rows.columns else None,
            "minutes_mean":             stat_rows["minutes_mean"].values,
            "minutes_sigma":            stat_rows["minutes_sigma"].values,
            "stat_mean":                stat_means,
            "stat_variance":            stat_var_arr,
            "pmf_json":                 pmf_jsons,
            "pmf_support_min":          0,
            "pmf_support_max":          cap,
            "pmf_mean":                 pmf_means,
            "pmf_variance":             pmf_vars,
            "p0":                       p0_arr,
            "p_ge_1":                   p_ge_1_arr,
            "p_ge_2":                   p_ge_2_arr,
            "p_ge_3":                   p_ge_3_arr,
            "p_ge_5":                   p_ge_5_arr,
            "p_ge_10":                  p_ge_10_arr,
            "model_version":            model_version,
            "pmf_source":               pmf_source,
            "is_calibrated":            False,
        })
        all_frames.append(frame)

    if not all_frames:
        raise ValueError("No PMF frames built — check stats list and long table")
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal PMF matrix construction
# ---------------------------------------------------------------------------

def _build_pmf_matrix(
    stat: str,
    stat_means: np.ndarray,
    p_nz: np.ndarray | None,
    pos_mus: np.ndarray | None,
    stat_models: dict,
    hurdle_models: dict,
    cap: int,
) -> np.ndarray:
    """Build PMF matrix (n × cap+1) for a stat."""
    if stat in hurdle_models:
        model = hurdle_models[stat]
        pos_r = model.pos_dispersion_r
        pmf_mat = hurdle_pmf_batch(p_nz, pos_mus, pos_r, cap)  # type: ignore[arg-type]
    else:
        model = stat_models.get(stat)
        r = getattr(model, "dispersion_r", None) if model is not None else None
        if r is not None:
            pmf_mat = negbinom_pmf_batch(stat_means, r, cap)
        else:
            pmf_mat = poisson_pmf_batch(stat_means, cap)
    return pmf_mat


# ---------------------------------------------------------------------------
# Wide PMF table (one row per player_id × game_id, stats as columns)
# ---------------------------------------------------------------------------

def build_wide_pmf_table(pmf_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long PMF table into wide format.

    One row per player_id × game_id.
    For each stat: {stat}_pmf_mean, {stat}_p0, {stat}_p_ge_1, {stat}_p_ge_5,
                   {stat}_stat_mean, {stat}_pmf_json.
    """
    id_cols = [
        "game_id", "game_date", "season", "player_id", "player_name",
        "team_id", "team_abbreviation", "opponent_team_id",
        "actual_minutes", "minutes_mean", "minutes_sigma",
    ]
    stat_metrics = ["pmf_mean", "pmf_variance", "p0", "p_ge_1", "p_ge_5",
                    "stat_mean", "pmf_json", "actual_outcome"]

    available_id = [c for c in id_cols if c in pmf_df.columns]
    # Ensure player_id and game_id are present (needed for merge keys)
    for key in ("player_id", "game_id"):
        if key not in available_id:
            available_id.append(key)
    id_df = pmf_df[available_id].drop_duplicates(subset=["player_id", "game_id"])

    for stat in STATS:
        sub = pmf_df[pmf_df["stat"] == stat]
        if sub.empty:
            continue
        metrics = {c: f"{stat}_{c}" for c in stat_metrics if c in sub.columns}
        sub_pivot = sub[["player_id", "game_id"] + list(metrics.keys())].rename(
            columns=metrics
        )
        id_df = id_df.merge(sub_pivot, on=["player_id", "game_id"], how="left")

    return id_df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
