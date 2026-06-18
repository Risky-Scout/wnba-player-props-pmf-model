"""Refactored model training and prediction for Stage 5+ reuse.

Provides clean train_fold() / generate_fold_pmfs() functions that both
the full-data Stage 4 trainer and the Stage 5 OOF loop can call with
explicit train/validation DataFrames.

Public API:
  train_fold(train_wide, train_long, model_feature_cols, cfg)
    → FoldModel

  generate_fold_pmfs(fold_model, val_wide, val_long, fold_meta, cfg)
    → pd.DataFrame   (long PMF table for this fold)

  apply_low_minutes_adjustment(pmf_mat, minutes_means, threshold)
    → (pmf_mat, n_adjusted)

  encode_features(df, model_cols, pos_encoder, fit_encoder)
    → (X_numeric, pos_encoder)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES
from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket
from wnba_props_model.models.log_linear_stat_model import LogLinearStatModel
from wnba_props_model.models.minutes_model import MinutesModel
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
from wnba_props_model.models.rate_model import HurdleModel, StatRateModel

OOF_PMF_SOURCE_S5 = "stage5_walk_forward_oof_uncalibrated_model_only"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldModel:
    """All trained artifacts for a single OOF fold."""

    minutes_model: MinutesModel
    stat_models: dict[str, StatRateModel]
    hurdle_models: dict[str, HurdleModel]
    pos_encoder: OrdinalEncoder | None
    feature_cols: list[str]
    summaries: dict[str, Any]
    train_wide_rows: int
    train_long_rows: int
    train_stat_rows: dict[str, int] = field(default_factory=dict)
    bb_models: dict = field(default_factory=dict)  # BetaBinomialStatModel keyed by stat


# ---------------------------------------------------------------------------
# Feature encoding (shared with pmf_engine.py pattern)
# ---------------------------------------------------------------------------

def encode_features(
    df: pd.DataFrame,
    model_feature_cols: list[str],
    pos_encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder | None]:
    """Prepare numeric feature matrix from a DataFrame.

    Encodes 'position' with OrdinalEncoder.  All other columns must be
    numeric; NaN is handled natively by HistGradientBoosting.

    Args:
        df: Source DataFrame.
        model_feature_cols: Authoritative allow-list from manifest.
        pos_encoder: Pre-fitted encoder (None → fit fresh if fit_encoder=True).
        fit_encoder: If True, fit a new encoder on df['position'].

    Returns:
        (X_numeric, encoder)
    """
    bad = [c for c in model_feature_cols if c in FORBIDDEN_MODEL_FEATURES]
    if bad:
        raise ValueError(f"Forbidden columns in model_feature_cols: {bad}")

    available = [c for c in model_feature_cols if c in df.columns]
    X = df[available].copy()

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

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, pos_encoder


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def train_fold(
    train_wide: pd.DataFrame,
    train_long: pd.DataFrame,
    model_feature_cols: list[str],
    cfg: dict[str, Any],
) -> FoldModel:
    """Train minutes + per-stat models on a training fold.

    Args:
        train_wide: Wide feature table (all rows in training window).
        train_long: Long feature table (all stat rows in training window).
        model_feature_cols: From manifest — no forbidden cols.
        cfg: Stage 5 (or Stage 4) YAML config dict.

    Returns:
        FoldModel containing all fitted models and metadata.
    """
    # ---- Feature encoding ------------------------------------------------
    X_train, pos_encoder = encode_features(train_wide, model_feature_cols, fit_encoder=True)

    # Drop all-NaN columns: sklearn's BinMapper raises
    # "window shape cannot be larger than input array shape" when a column is
    # entirely NaN (common in early OOF folds before rolling history exists).
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_nan_cols:
        X_train = X_train.drop(columns=all_nan_cols)

    if X_train.empty or len(X_train.columns) == 0:
        raise ValueError(
            f"All {len(model_feature_cols)} feature columns are entirely NaN — "
            "insufficient game history for this fold."
        )

    usable_feature_cols = list(X_train.columns)

    # ---- Temporal sample weights (exponential decay) ---------------------
    # Exponential decay: games from 6 months ago have weight ~0.5; 1-year-old
    # games have weight ~0.25. Reduces systematic bias from stale season trends.
    sample_weight: np.ndarray | None = None
    halflife = cfg.get("sample_weight_halflife_days", None)
    if halflife and "game_date" in train_wide.columns:
        cutoff = pd.to_datetime(train_wide["game_date"]).max()
        days_ago = (cutoff - pd.to_datetime(train_wide["game_date"])).dt.days.fillna(0)
        sw = np.exp(-np.log(2) / halflife * days_ago.values)
        sw = sw / sw.mean()  # normalize so total effective sample size is preserved
        sample_weight = sw.astype(np.float64)

    # ---- Minutes model ---------------------------------------------------
    y_min = train_wide["actual_minutes"].fillna(0.0)
    min_model = MinutesModel(cfg)
    min_model.fit(X_train, y_min, train_wide, sample_weight=sample_weight)
    # Remember which columns were usable so inference aligns to the same set
    min_model._feature_cols = usable_feature_cols
    min_model._pos_encoder = pos_encoder  # store for inference
    min_summary = min_model.get_training_summary()

    # ---- Stat models -----------------------------------------------------
    sparse_stats = set(cfg.get("sparse_stats", ["stl", "blk"]))
    stats = cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"])

    played_mask = (train_wide["did_play"].astype(bool)
                   if "did_play" in train_wide.columns
                   else (train_wide["actual_minutes"] > 0))
    X_played = X_train[played_mask].reset_index(drop=True)

    # Filter sample_weight to played rows before passing to stat models.
    # The temporal decay weight is computed over all train_wide rows; stat models
    # train only on did_play=True rows (X_played), so the weight must be subsetted.
    sample_weight_played = sample_weight[played_mask] if sample_weight is not None else None

    stat_models: dict[str, StatRateModel] = {}
    hurdle_models: dict[str, HurdleModel] = {}
    bb_models: dict = {}
    summaries: dict[str, Any] = {"minutes": min_summary}
    train_stat_rows: dict[str, int] = {}

    # Check which special distributions are enabled
    stat_overrides_cfg = cfg.get("stat_overrides", {})
    use_beta_binomial_fg3m = (
        stat_overrides_cfg.get("fg3m", {}).get("use_beta_binomial", False)
        and "actual_fg3a" in train_wide.columns  # fg3a must be available
    )

    for stat in stats:
        target_col = f"actual_{stat}"
        if target_col not in train_wide.columns:
            continue

        y_stat = train_wide.loc[played_mask, target_col].reset_index(drop=True)
        n_rows = len(y_stat)
        train_stat_rows[stat] = n_rows

        min_stat = cfg.get("min_train_stat_rows", 250)
        if n_rows < min_stat:
            continue

        if stat in sparse_stats:
            min_pos = cfg.get("min_sparse_positive_rows", 50)
            if int((y_stat > 0).sum()) < min_pos:
                continue
            use_zinb = cfg.get("use_zinb_for_sparse_stats", False)
            if use_zinb:
                from wnba_props_model.models.hurdle import ZINBStatModel  # noqa: PLC0415
                m = ZINBStatModel(stat, cfg)
                m.fit(X_played, y_stat, sample_weight=sample_weight_played)
            else:
                m = HurdleModel(stat, cfg)
                m.fit(X_played, y_stat, sample_weight=sample_weight_played)
            hurdle_models[stat] = m
            summaries[stat] = m.get_training_summary()
        elif stat == "fg3m" and use_beta_binomial_fg3m:
            # Beta-Binomial model for fg3m (uses fg3a as attempt count)
            from wnba_props_model.models.beta_binomial import BetaBinomialStatModel  # noqa: PLC0415
            played_ctx = train_wide[played_mask].reset_index(drop=True)
            fg3a_col = "actual_fg3a" if "actual_fg3a" in played_ctx.columns else None
            y_attempts = played_ctx[fg3a_col] if fg3a_col else None
            stat_cfg = {**cfg, **stat_overrides_cfg.get(stat, {})}
            bb_m = BetaBinomialStatModel(stat_cfg)
            bb_m.fit(X_played, y_stat, y_attempts, sample_weight=sample_weight_played)
            bb_models["fg3m"] = bb_m
            summaries["fg3m_bb"] = {
                "stat": "fg3m", "model_type": "BetaBinomial",
                "alpha": bb_m.alpha_, "beta": bb_m.beta_,
            }
            # Also fit a standard stat model as fallback stored in stat_models
            stat_cfg2 = {**cfg, **stat_overrides_cfg.get(stat, {})}
            m = StatRateModel(stat, stat_cfg2)
            m.fit(X_played, y_stat, context_df=played_ctx, sample_weight=sample_weight_played)
            stat_models[stat] = m
        else:
            # Pass context_df so StatRateModel can compute per-role dispersion.
            played_ctx = train_wide[played_mask].reset_index(drop=True)
            # Merge global config with per-stat overrides (stat_overrides.{stat})
            stat_cfg = {**cfg, **stat_overrides_cfg.get(stat, {})}
            if cfg.get("use_log_linear", False):
                m = LogLinearStatModel(stat, stat_cfg)
                m.fit(X_played, y_stat, context_df=played_ctx, sample_weight=sample_weight_played)
            else:
                m = StatRateModel(stat, stat_cfg)
                m.fit(X_played, y_stat, context_df=played_ctx, sample_weight=sample_weight_played)
            stat_models[stat] = m
            summaries[stat] = {"stat": stat, "model_type": type(m).__name__}

    return FoldModel(
        minutes_model=min_model,
        stat_models=stat_models,
        hurdle_models=hurdle_models,
        bb_models=bb_models,
        pos_encoder=pos_encoder,
        # Use only columns that had non-NaN values in training (all-NaN cols were
        # dropped before fitting; inference must align to the same column set).
        feature_cols=usable_feature_cols,
        summaries=summaries,
        train_wide_rows=len(train_wide),
        train_long_rows=len(train_long),
        train_stat_rows=train_stat_rows,
    )


# ---------------------------------------------------------------------------
# Fold PMF generation
# ---------------------------------------------------------------------------

def generate_fold_pmfs(
    fold_model: FoldModel,
    val_wide: pd.DataFrame,
    val_long: pd.DataFrame,
    fold_meta: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Generate PMFs for one validation fold.

    Args:
        fold_model: Trained models from train_fold().
        val_wide: Validation wide feature table.
        val_long: Validation long feature table.
        fold_meta: Dict with fold_id, dates, row counts, etc.
        cfg: Config dict (stage5_oof.yaml).

    Returns:
        Long PMF DataFrame with all OOF output columns.
    """
    X_val, _ = encode_features(
        val_wide, fold_model.feature_cols,
        pos_encoder=fold_model.pos_encoder, fit_encoder=False
    )

    # Minutes predictions (aligned with val_wide)
    min_means, min_sigmas, p_dnp_arr = fold_model.minutes_model.predict(X_val, val_wide)

    # Build lookup: (player_id, game_id) → (minutes_mean, minutes_sigma, p_dnp)
    min_lookup = {
        (row.player_id, row.game_id): (min_means[i], min_sigmas[i], p_dnp_arr[i])
        for i, row in enumerate(val_wide.itertuples(index=False))
    }

    support_caps = cfg.get("pmf_support_caps", {})
    sparse_stats = set(cfg.get("sparse_stats", ["stl", "blk"]))
    stats = cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"])
    pmf_source = cfg.get("pmf_source", OOF_PMF_SOURCE_S5)
    dnp_threshold = cfg.get("dnp_minutes_threshold", 1.0)
    apply_dnp = cfg.get("low_minutes_zero_inflation_enabled", True)
    cal_eligible_types = set(cfg.get("calibration_eligible_prediction_types", ["model_oof"]))

    all_frames: list[pd.DataFrame] = []

    for stat in stats:
        cap = support_caps.get(stat, 20)
        stat_rows = val_long[val_long["stat"] == stat].copy()
        if stat_rows.empty:
            continue

        # Attach minutes predictions
        _default_min = (0.0, cfg.get("min_minutes_sigma", 3.0), 0.0)
        stat_rows["minutes_mean"]  = [min_lookup.get((r.player_id, r.game_id), _default_min)[0] for r in stat_rows.itertuples(index=False)]
        stat_rows["minutes_sigma"] = [min_lookup.get((r.player_id, r.game_id), _default_min)[1] for r in stat_rows.itertuples(index=False)]
        stat_rows["p_dnp"]         = [min_lookup.get((r.player_id, r.game_id), _default_min)[2] for r in stat_rows.itertuples(index=False)]

        # Feature matrix aligned to stat_rows (use val_wide indexed by player+game)
        wide_idx = val_wide.set_index(["player_id", "game_id"])
        aligned_wide = wide_idx.reindex(
            pd.MultiIndex.from_arrays(
                [stat_rows["player_id"].values, stat_rows["game_id"].values]
            )
        ).reset_index(drop=True)

        X_stat, _ = encode_features(
            aligned_wide, fold_model.feature_cols,
            pos_encoder=fold_model.pos_encoder, fit_encoder=False
        )

        # ---- Predict stat ------------------------------------------------
        stat_model_type = "missing"
        stat_means_out = np.full(len(stat_rows), cfg.get("min_stat_mean", 0.01))
        stat_var_out = np.full(len(stat_rows), np.nan)
        p_nz_out = None
        pos_mus_out = None

        if stat in fold_model.hurdle_models:
            model = fold_model.hurdle_models[stat]
            p_nz, pos_mus = model.predict(X_stat)
            stat_means_out = p_nz * pos_mus
            stat_var_out = np.full(len(stat_rows), float(model._pos_var))
            stat_model_type = "hurdle"
            p_nz_out = p_nz
            pos_mus_out = pos_mus
        elif stat in fold_model.stat_models:
            model = fold_model.stat_models[stat]
            stat_means_out = model.predict_mean(X_stat)
            stat_var_out = np.full(len(stat_rows), float(model._global_var))
            stat_model_type = "rate"

        # ---- Build PMF matrix --------------------------------------------
        roles = stat_rows["role_bucket"].values if "role_bucket" in stat_rows.columns else None
        use_marginalization = cfg.get("use_minutes_marginalization", False)
        bb_models_fold = getattr(fold_model, "bb_models", {})

        if (use_marginalization
                and hasattr(fold_model.minutes_model, "_quantile_models")
                and fold_model.minutes_model._quantile_models):
            # Get quantile predictions for all stat_rows
            wide_idx2 = val_wide.set_index(["player_id", "game_id"])
            aligned_wide2 = wide_idx2.reindex(
                pd.MultiIndex.from_arrays(
                    [stat_rows["player_id"].values, stat_rows["game_id"].values]
                )
            ).reset_index(drop=True)
            X_for_quant, _ = encode_features(
                aligned_wide2, fold_model.feature_cols,
                pos_encoder=fold_model.pos_encoder, fit_encoder=False
            )
            quant_mat = fold_model.minutes_model.predict_quantiles(X_for_quant, aligned_wide2)
            quad_weights = np.array(cfg.get(
                "minutes_marginalization_weights", [0.10, 0.15, 0.50, 0.15, 0.10]
            ))
            from wnba_props_model.models.pmf_engine import (  # noqa: PLC0415
                _build_marginalized_pmf_matrix, _blend_with_dnp,
            )
            pmf_mat = _build_marginalized_pmf_matrix(
                stat, quant_mat, quad_weights, p_nz_out, pos_mus_out,
                fold_model.stat_models, fold_model.hurdle_models, cap, roles=roles
            )
        elif stat == "fg3m" and bb_models_fold.get("fg3m") is not None:
            pmf_mat = bb_models_fold["fg3m"].predict_pmf_matrix(X_stat, cap=cap)
        elif stat in fold_model.hurdle_models:
            model = fold_model.hurdle_models[stat]
            pmf_mat = hurdle_pmf_batch(p_nz_out, pos_mus_out, model.pos_dispersion_r, cap)
        elif stat in fold_model.stat_models:
            model = fold_model.stat_models[stat]
            # Role-aware NegBinom: batch by role so stars get fatter tails
            if roles is not None and getattr(model, "_role_dispersion", None):
                n = len(stat_means_out)
                pmf_mat = np.zeros((n, cap + 1))
                for role in np.unique(roles):
                    mask = roles == role
                    r_role = model.get_dispersion(str(role))
                    mu_role = stat_means_out[mask]
                    if r_role is not None:
                        pmf_mat[mask] = negbinom_pmf_batch(mu_role, r_role, cap)
                    else:
                        pmf_mat[mask] = poisson_pmf_batch(mu_role, cap)
            else:
                r = model.dispersion_r
                pmf_mat = (negbinom_pmf_batch(stat_means_out, r, cap)
                           if r is not None else poisson_pmf_batch(stat_means_out, cap))
        else:
            # Stat not trained — fall back to Poisson with global prior
            prior = cfg.get("league_priors", {}).get(stat, {})
            mu_prior = prior.get("mean", 1.0)
            pmf_mat = poisson_pmf_batch(np.full(len(stat_rows), mu_prior), cap)
            stat_model_type = "prior_fallback"

        # ---- DNP / zero-inflation -----------------------------------------
        low_min_applied = False
        low_min_count = 0
        if use_marginalization:
            # Use DNP model output for zero inflation
            p_dnp_row = stat_rows["p_dnp"].fillna(0.0).values.astype(float)
            if np.any(p_dnp_row > 0.0):
                from wnba_props_model.models.pmf_engine import _blend_with_dnp  # noqa: PLC0415
                pmf_mat = _blend_with_dnp(pmf_mat, p_dnp_row)
                low_min_applied = True
                low_min_count = int(np.sum(p_dnp_row > 0.01))
        elif apply_dnp:
            pmf_mat, low_min_count = apply_low_minutes_adjustment(
                pmf_mat, stat_rows["minutes_mean"].values, dnp_threshold
            )
            low_min_applied = low_min_count > 0

        validate_pmf_matrix(pmf_mat)

        # ---- Summary stats -----------------------------------------------
        pmf_means, pmf_vars = pmf_mean_var(pmf_mat)
        p0_arr      = pmf_mat[:, 0]
        p_ge_1_arr  = pmf_pge(pmf_mat, 1)
        p_ge_2_arr  = pmf_pge(pmf_mat, 2)
        p_ge_3_arr  = pmf_pge(pmf_mat, 3)
        p_ge_5_arr  = pmf_pge(pmf_mat, 5) if cap >= 5 else np.zeros(len(pmf_mat))
        p_ge_10_arr = pmf_pge(pmf_mat, 10) if cap >= 10 else np.zeros(len(pmf_mat))
        pmf_jsons   = pmf_matrix_to_json_list(pmf_mat)

        oof_type = fold_meta.get("oof_prediction_type", "model_oof")
        cal_elig = oof_type in cal_eligible_types

        frame = pd.DataFrame({
            "game_id":                      stat_rows["game_id"].values,
            "game_date":                    stat_rows["game_date"].values,
            "season":                       stat_rows["season"].values,
            "player_id":                    stat_rows["player_id"].values,
            "player_name":                  stat_rows["player_name"].values,
            "team_id":                      stat_rows["team_id"].values,
            "team_abbreviation":            stat_rows["team_abbreviation"].values,
            "opponent_team_id":             stat_rows["opponent_team_id"].values,
            "opponent_team_abbreviation":   stat_rows.get("opponent_team_abbreviation",
                                                pd.Series([None] * len(stat_rows))).values,
            "is_home":                      stat_rows.get("is_home",
                                                pd.Series([None] * len(stat_rows))).values,
            "home_away":                    stat_rows.get("home_away",
                                                pd.Series([None] * len(stat_rows))).values,
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
            "fold_train_rows_stat":         fold_meta.get("train_stat_rows", {}).get(stat, 0),
            "fold_train_games":             fold_meta.get("train_games", 0),
            "fold_validation_rows":         len(stat_rows),
            "oof_prediction_type":          oof_type,
            "calibration_eligible":         cal_elig,
            "minutes_mean":                 stat_rows["minutes_mean"].values,
            "minutes_sigma":                stat_rows["minutes_sigma"].values,
            "p_dnp":                        stat_rows["p_dnp"].values if "p_dnp" in stat_rows.columns else np.zeros(len(stat_rows)),
            "minutes_prediction_type":      "model" if oof_type == "model_oof" else "prior",
            "stat_mean":                    stat_means_out,
            "stat_variance":                stat_var_out,
            "stat_model_type":              stat_model_type,
            "pmf_json":                     pmf_jsons,
            "pmf_support_min":              0,
            "pmf_support_max":              cap,
            "pmf_mean":                     pmf_means,
            "pmf_variance":                 pmf_vars,
            "p0":                           p0_arr,
            "p_ge_1":                       p_ge_1_arr,
            "p_ge_2":                       p_ge_2_arr,
            "p_ge_3":                       p_ge_3_arr,
            "p_ge_5":                       p_ge_5_arr,
            "p_ge_10":                      p_ge_10_arr,
            "low_minutes_adjustment_applied": low_min_applied,
            "low_minutes_adjustment_count": low_min_count,
            "model_version":                "stage5_oof_v1",
            "pmf_source":                   pmf_source,
            "is_calibrated":                False,
        })
        all_frames.append(frame)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)

    # Attach ex-ante role_bucket (required for per-role calibrator fitting).
    # Without this, fit_calibrators() defaults every OOF row to "all" and the
    # role-aware calibrators are never trained — only a global calibrator ships.
    if "minutes_mean" in result.columns and "role_bucket" not in result.columns:
        unique_pg = result[["player_id", "game_id", "minutes_mean"]].drop_duplicates()
        unique_pg = add_ex_ante_role_bucket(unique_pg, minutes_col="minutes_mean")
        rb_map = unique_pg.set_index(["player_id", "game_id"])["role_bucket"]
        result["role_bucket"] = result.set_index(["player_id", "game_id"]).index.map(rb_map).values
        result["role_bucket"] = result["role_bucket"].fillna("all")

    return result


# ---------------------------------------------------------------------------
# Low-minutes zero-inflation adjustment
# ---------------------------------------------------------------------------

def apply_low_minutes_adjustment(
    pmf_mat: np.ndarray,
    minutes_means: np.ndarray,
    threshold: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Zero-inflate PMF rows where minutes_mean < threshold.

    For each such row:
        alpha = 1 - minutes_mean / threshold   (0 when at threshold, 1 when at 0)
        pmf[k] = (1 - alpha) * pmf[k]  for k > 0
        pmf[0] += alpha

    The row still sums to 1:
        (1-alpha) * 1 + alpha = 1 ✓

    Does NOT use actual_minutes — only predicted minutes_mean.

    Returns:
        (adjusted pmf_mat, number of rows adjusted)
    """
    pmf_mat = pmf_mat.copy()
    adj_mask = (minutes_means < threshold) & (minutes_means >= 0)
    n_adjusted = int(adj_mask.sum())

    if n_adjusted == 0:
        return pmf_mat, 0

    for i in np.where(adj_mask)[0]:
        mu = float(minutes_means[i])
        alpha = max(0.0, 1.0 - mu / threshold) if threshold > 0 else 1.0
        alpha = min(alpha, 0.99)  # never fully degenerate
        pmf_mat[i] = (1.0 - alpha) * pmf_mat[i]
        pmf_mat[i, 0] += alpha

    return pmf_mat, n_adjusted
