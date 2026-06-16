"""OOF fold generation and prior-PMF utilities for Stage 5.

Public API:
  generate_oof_folds(game_dates, window_days=14)
    → list of fold dicts

  make_prior_only_pmfs(val_wide, val_long, fold_meta, cfg)
    → pd.DataFrame  (long PMF table with prior_only prediction type)
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.models.pmf_utils import (
    dispersion_from_moments,
    hurdle_pmf_batch,
    negbinom_pmf_batch,
    pmf_matrix_to_json_list,
    pmf_mean_var,
    pmf_pge,
    poisson_pmf_batch,
    validate_pmf_matrix,
)


def generate_oof_folds(
    game_dates: list[date],
    window_days: int = 14,
) -> list[dict[str, Any]]:
    """Generate non-overlapping chronological OOF fold definitions.

    Each fold's validation window covers exactly `window_days` calendar days.
    Training for fold i = all dates strictly before fold_i.val_start_date.

    CRITICAL: train_end_date < val_start_date for every fold.

    Args:
        game_dates: All unique game dates (as datetime.date objects), sorted.
        window_days: Calendar days per validation window.

    Returns:
        List of fold dicts, one per non-empty validation window.
    """
    game_dates = sorted(set(game_dates))
    if not game_dates:
        return []

    min_date = game_dates[0]
    max_date = game_dates[-1]

    folds: list[dict[str, Any]] = []
    fold_id = 0
    current_start = min_date

    while current_start <= max_date:
        val_end = current_start + timedelta(days=window_days - 1)
        val_dates = [d for d in game_dates if current_start <= d <= val_end]

        if val_dates:
            train_end = current_start - timedelta(days=1)
            train_dates = [d for d in game_dates if d < current_start]
            folds.append({
                "fold_id": fold_id,
                "train_start_date": train_dates[0] if train_dates else None,
                "train_end_date": train_end,
                "val_start_date": current_start,
                "val_end_date": val_end,
                "val_dates": val_dates,
                "train_games": len(train_dates),
                "val_games": len(val_dates),
            })
            fold_id += 1

        current_start = val_end + timedelta(days=1)

    return folds


def make_prior_only_pmfs(
    val_wide: pd.DataFrame,
    val_long: pd.DataFrame,
    fold_meta: dict[str, Any],
    cfg: dict[str, Any],
    error_msg: str | None = None,
) -> pd.DataFrame:
    """Generate prior-only PMFs for folds without sufficient training data.

    Uses league-average priors from config. All rows are marked:
        oof_prediction_type = "prior_only"
        calibration_eligible = False

    Args:
        val_wide: Validation wide features (for identity columns).
        val_long: Validation long features.
        fold_meta: Fold metadata dict.
        cfg: Stage 5 config dict.
        error_msg: If set, marks oof_prediction_type = "failed_model_fit".

    Returns:
        Long PMF DataFrame for this fold.
    """
    stats = cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"])
    support_caps = cfg.get("pmf_support_caps", {})
    priors = cfg.get("league_priors", {})
    pmf_source = cfg.get("pmf_source", "stage5_walk_forward_oof_uncalibrated_model_only")
    oof_type = "failed_model_fit" if error_msg else "prior_only"

    all_frames: list[pd.DataFrame] = []

    for stat in stats:
        cap = support_caps.get(stat, 20)
        stat_rows = val_long[val_long["stat"] == stat].copy()
        if stat_rows.empty:
            continue

        prior = priors.get(stat, {})
        mu = max(float(prior.get("mean", 1.0)), 0.01)
        var = float(prior.get("var", mu))

        r = dispersion_from_moments(mu, var)
        mus = np.full(len(stat_rows), mu)
        pmf_mat = (negbinom_pmf_batch(mus, r, cap)
                   if r is not None else poisson_pmf_batch(mus, cap))

        validate_pmf_matrix(pmf_mat)
        pmf_means, pmf_vars = pmf_mean_var(pmf_mat)
        pmf_jsons = pmf_matrix_to_json_list(pmf_mat)

        frame = pd.DataFrame({
            "game_id":                      stat_rows["game_id"].values,
            "game_date":                    stat_rows["game_date"].values,
            "season":                       stat_rows["season"].values,
            "player_id":                    stat_rows["player_id"].values,
            "player_name":                  stat_rows["player_name"].values,
            "team_id":                      stat_rows["team_id"].values,
            "team_abbreviation":            stat_rows["team_abbreviation"].values,
            "opponent_team_id":             stat_rows["opponent_team_id"].values,
            "opponent_team_abbreviation":   stat_rows.get(
                "opponent_team_abbreviation", pd.Series([None]*len(stat_rows))).values,
            "is_home":                      stat_rows.get(
                "is_home", pd.Series([None]*len(stat_rows))).values,
            "home_away":                    stat_rows.get(
                "home_away", pd.Series([None]*len(stat_rows))).values,
            "stat":                         stat,
            "actual_outcome":               stat_rows["actual_outcome"].values,
            "actual_minutes":               stat_rows["actual_minutes"].values,
            "did_play":                     stat_rows["did_play"].values
                                            if "did_play" in stat_rows.columns else None,
            "fold_id":                      fold_meta["fold_id"],
            "fold_train_start_date":        fold_meta.get("train_start_date"),
            "fold_train_end_date":          fold_meta.get("train_end_date"),
            "fold_validation_start_date":   fold_meta.get("val_start_date"),
            "fold_validation_end_date":     fold_meta.get("val_end_date"),
            "fold_train_rows":              fold_meta.get("train_wide_rows", 0),
            "fold_train_rows_stat":         0,
            "fold_train_games":             fold_meta.get("train_games", 0),
            "fold_validation_rows":         len(stat_rows),
            "oof_prediction_type":          oof_type,
            "calibration_eligible":         False,
            "minutes_mean":                 np.nan,
            "minutes_sigma":                np.nan,
            "minutes_prediction_type":      "prior",
            "stat_mean":                    mus,
            "stat_variance":                np.full(len(stat_rows), var),
            "stat_model_type":              "prior",
            "pmf_json":                     pmf_jsons,
            "pmf_support_min":              0,
            "pmf_support_max":              cap,
            "pmf_mean":                     pmf_means,
            "pmf_variance":                 pmf_vars,
            "p0":                           pmf_mat[:, 0],
            "p_ge_1":                       pmf_pge(pmf_mat, 1),
            "p_ge_2":                       pmf_pge(pmf_mat, 2),
            "p_ge_3":                       pmf_pge(pmf_mat, 3),
            "p_ge_5":                       pmf_pge(pmf_mat, 5) if cap >= 5 else np.zeros(len(stat_rows)),
            "p_ge_10":                      pmf_pge(pmf_mat, 10) if cap >= 10 else np.zeros(len(stat_rows)),
            "low_minutes_adjustment_applied": False,
            "low_minutes_adjustment_count": 0,
            "model_version":                "stage5_oof_prior_v1",
            "pmf_source":                   pmf_source,
            "is_calibrated":                False,
            "error_message":                error_msg or "",
        })
        all_frames.append(frame)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
