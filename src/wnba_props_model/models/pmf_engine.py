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
    sanitize_pmf_matrix,
    validate_pmf_matrix,
    zinb_pmf_batch,
)

PMF_SOURCE = "stage4_baseline_uncalibrated_model_only"
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

# ---------------------------------------------------------------------------
# Per-role, per-position REB dispersion (Phase G)
# ---------------------------------------------------------------------------

_REB_ROLE_DISPERSION: dict[str, float] = {
    "bench_low":      8.0,
    "bench_rotation": 6.0,
    "rotation":       4.5,
    "starter":        3.5,
    "workhorse":      3.0,
    "core":           4.0,
    "fringe":         9.0,
}

_REB_POSITION_MODIFIER: dict[str, float] = {"G": 1.5, "F": 1.0, "C": 0.7}


def compute_reb_effective_r(role: str, position: str) -> float:
    """Return position + role adjusted NegBinom dispersion r for REB.

    Higher r = tighter PMF (less overdispersion).
    - Centers have lower r (more variance relative to mean, spread around role)
    - Bench/fringe have higher r (tight around zero)
    """
    base_r = _REB_ROLE_DISPERSION.get(str(role), 4.0)
    pos_mod = _REB_POSITION_MODIFIER.get(str(position)[:1].upper(), 1.0)
    return base_r * pos_mod


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
    # Verify no forbidden columns.
    # During training (fit_encoder=True): hard error so we catch schema drift early.
    # During inference (fit_encoder=False): gracefully drop — forbidden features carry
    # no signal by definition, so their absence does not degrade predictions. This
    # prevents crashes when loading model artifacts trained before a feature was banned.
    bad = [c for c in model_feature_cols if c in FORBIDDEN_MODEL_FEATURES]
    if bad:
        if fit_encoder:
            raise ValueError(f"Forbidden columns in model_feature_cols: {bad}")
        import warnings as _warnings
        _warnings.warn(
            f"Dropping {len(bad)} forbidden columns from inference feature set: {bad}",
            stacklevel=2,
        )
        model_feature_cols = [c for c in model_feature_cols if c not in FORBIDDEN_MODEL_FEATURES]

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
    bb_models: dict[str, Any] | None = None,
    minutes_correction: Any | None = None,
    pts_hurdle_model: Any | None = None,
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
    min_means, min_sigmas, p_dnp = minutes_model.predict(X_wide, wide_df)

    # Apply role-aware minutes correction if available (fitted during weekly_calibration).
    if minutes_correction is not None and getattr(minutes_correction, "fitted", False):
        try:
            _role_col = wide_df["role_bucket"].values if "role_bucket" in wide_df.columns else np.full(len(wide_df), "rotation")
            _corr = minutes_correction.correct(min_means, min_sigmas, p_dnp, _role_col)
            min_means  = _corr["minutes_mean"]
            min_sigmas = _corr["minutes_sigma"]
            p_dnp      = _corr["p_dnp"]
        except Exception as _mc_exc:
            import warnings as _w
            _w.warn(f"minutes_correction.correct() failed: {_mc_exc}; using raw predictions", stacklevel=2)

    # Defect 3: save baseline minutes BEFORE injury override, so we can compute
    # the scenario minutes ratio and apply it to stat PMFs later.
    _has_inj_mult = "_injury_minutes_multiplier" in wide_df.columns
    baseline_minutes_mean = min_means.copy()

    # Injury minutes override: if ``_injury_minutes_multiplier`` column is
    # present in wide_df, apply it AFTER model prediction and corrections.
    # This ensures PMFs for injured players are built from the correct
    # expected minutes rather than just scaling post-hoc means.
    # Multiplier = 0.0 → OUT player (p_dnp forced to 1.0, minutes forced to 0).
    # Multiplier in (0, 1) → limited status (reduced conditional minutes).
    # Multiplier > 1.0 → teammate receiving redistributed minutes.
    # NOTE (Blocker 1): multiplier = cond_mult ONLY; availability_probability is
    # carried separately and is NOT multiplied into conditional minutes.
    if _has_inj_mult:
        _mult = wide_df["_injury_minutes_multiplier"].fillna(1.0).values.astype(float)
        _changed = _mult != 1.0
        if _changed.any():
            min_means  = min_means * _mult
            min_sigmas = min_sigmas * np.clip(_mult, 0.0, None)
            # OUT players (mult==0): force p_dnp=1.0 so DNP blending zeroes the PMF
            _is_out = _mult <= 0.0
            p_dnp = np.where(_is_out, 1.0, p_dnp)
            import logging as _log_inj  # noqa: PLC0415
            _log_inj.getLogger(__name__).info(
                "[pmf_engine] _injury_minutes_multiplier applied to %d / %d rows "
                "(%d OUT, %d partial)",
                int(_changed.sum()), len(_mult),
                int(_is_out.sum()),
                int((_changed & ~_is_out).sum()),
            )

    # Blocker 2: apply conditional_minutes_cap to the complete minutes distribution.
    # The cap is enforced AFTER multiplier application, ensuring the full
    # distribution (mean, sigma) respects the cap.
    if "conditional_minutes_cap" in wide_df.columns:
        _cap_raw = wide_df["conditional_minutes_cap"].values
        try:
            _cap = _cap_raw.astype(float)
        except Exception:
            _cap = np.full(len(_cap_raw), np.nan, dtype=float)
        _has_cap = np.isfinite(_cap) & (_cap > 0)
        if _has_cap.any():
            _needs_clamp = _has_cap & (min_means > _cap)
            if _needs_clamp.any():
                # Compute ratio to proportionally reduce sigma with the cap
                _cap_ratio = np.where(
                    _needs_clamp,
                    _cap / np.maximum(min_means, 1e-9),
                    1.0,
                )
                min_means  = np.where(_needs_clamp, _cap, min_means)
                min_sigmas = np.where(_needs_clamp, min_sigmas * _cap_ratio, min_sigmas)
                import logging as _log_cap  # noqa: PLC0415
                _log_cap.getLogger(__name__).info(
                    "[pmf_engine] conditional_minutes_cap applied to %d / %d rows",
                    int(_needs_clamp.sum()), len(_cap),
                )

    # Defect 3: compute injury minutes ratio for stat PMF scaling.
    # ratio = final_minutes_mean / max(baseline_minutes_mean, epsilon).
    # For normal (non-injury) players: ratio == 1.0 → no stat PMF change.
    # For OUT (mult=0): ratio == 0 → stat_means scaled to ~0 (floor applied).
    # For limited (mult=0.65, cap=20, baseline=35): ratio = 20/35 ≈ 0.571.
    # For teammate boost (baseline=20, final=28): ratio = 28/20 = 1.4.
    if _has_inj_mult:
        _injury_minutes_ratio = min_means / np.maximum(baseline_minutes_mean, 1e-9)
    else:
        _injury_minutes_ratio = np.ones(len(min_means))

    wide_with_min = wide_df.assign(
        minutes_mean=min_means,
        minutes_sigma=min_sigmas,
        p_dnp=p_dnp,
    )

    # Build lookup dicts for injury ratio/multiplier/cap (indexed by player_id + game_id)
    # to efficiently align to per-stat stat_rows in the loop below.
    if _has_inj_mult:
        _wide_index = pd.MultiIndex.from_frame(wide_df[["player_id", "game_id"]])
        _inj_ratio_series = pd.Series(_injury_minutes_ratio, index=_wide_index)
        _inj_mult_series = pd.Series(
            wide_df["_injury_minutes_multiplier"].fillna(1.0).values.astype(float),
            index=_wide_index,
        )
        _inj_cap_series = (
            pd.Series(wide_df["conditional_minutes_cap"].values, index=_wide_index)
            if "conditional_minutes_cap" in wide_df.columns
            else None
        )
    else:
        _inj_ratio_series = None
        _inj_mult_series = None
        _inj_cap_series = None

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
            wide_with_min[["player_id", "game_id", "minutes_mean", "minutes_sigma", "p_dnp"]],
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
            _role_series_hurdle = stat_rows["role_bucket"] if "role_bucket" in stat_rows.columns else None
            p_nz, pos_mus = model.predict(X_stat_df, role_series=_role_series_hurdle)
            stat_means = p_nz * pos_mus  # E[Y] = P(Y>0) * E[Y|Y>0]
        else:
            model = stat_models[stat]
            # P3.3: use Bayesian shrinkage ensemble when enabled
            use_ensemble = cfg.get("use_model_ensemble", False)
            _role_series_pmf = stat_rows["role_bucket"] if "role_bucket" in stat_rows.columns else None
            if use_ensemble and hasattr(model, "predict_with_shrinkage"):
                stat_means = model.predict_with_shrinkage(X_stat_df, X_stat, role_series=_role_series_pmf)
            else:
                stat_means = model.predict_mean(X_stat_df, role_series=_role_series_pmf)
            # Guard: replace any NaN/inf stat_means with minimum floor before PMF generation
            if not np.isfinite(stat_means).all():
                import warnings
                _n_bad = (~np.isfinite(stat_means)).sum()
                warnings.warn(f"[pmf_engine] {stat}: {_n_bad} non-finite stat_means replaced with floor 0.1", stacklevel=2)
                stat_means = np.where(np.isfinite(stat_means), stat_means, 0.1)
            p_nz = None
            pos_mus = None

        # Part 4: Market prior blending — shrink model mean toward market-implied mean.
        # mean_final = (1 - λ) * mean_model + λ * mean_market
        # The sportsbook line is the market's best estimate of the median; using it
        # as an anchor reduces extreme premiums (40-74% combo OVERs) toward realistic levels.
        _mpl = cfg.get("market_prior_lambda", 0.0)
        if _mpl > 0 and "market_line" in stat_rows.columns:
            _market_lines = stat_rows["market_line"].values
            _valid_market = np.isfinite(_market_lines) & (_market_lines > 0)
            if _valid_market.any():
                stat_means = np.where(
                    _valid_market,
                    (1.0 - _mpl) * stat_means + _mpl * _market_lines,
                    stat_means,
                )
                stat_means = np.clip(stat_means, 0.01, None)

        # Part 6: CLV head signal — soft-nudge stat_mean by ≤5% based on
        # whether the model's direction beats the closing line historically.
        if stat in stat_models:
            _clv_model = stat_models[stat]
            _clv_head = getattr(_clv_model, "clv_head", None)
            if _clv_head is not None:
                try:
                    _clv_p = _clv_head.predict_proba(X_stat_df)[:, 1]  # P(beat closing)
                    # ±5% nudge centred at 0.5: positive edge → increase mean
                    _clv_adj = (_clv_p - 0.5) * 0.10
                    stat_means = stat_means * (1.0 + _clv_adj)
                    stat_means = np.clip(stat_means, 0.01, None)
                except Exception:
                    pass

        # ---- Defect 3: Apply injury minutes ratio to conditional stat opportunity ----
        # The stat model predictions use historical features (player_pts_mean_l5, etc.)
        # which do not reflect the injury-adjusted minutes.  Scale the stat means by
        # the scenario minutes ratio so that limited/teammate-boost scenarios change
        # the full PMF, not just the minutes distribution.
        #
        # minutes_ratio = final_minutes_mean / max(baseline_minutes_mean, epsilon)
        # For questionable (cond_mult=1.0): ratio=1.0 → conditional PMF UNCHANGED
        # For limited (cond_mult=0.65, cap=20, baseline=35): ratio≈0.571 → PMF reduced
        # For teammate boost (baseline=20, final=28): ratio=1.4 → PMF increased
        #
        # Do NOT apply availability_probability to the conditional performance PMF.
        # Flags and saved pre-Fix1 parameters used later for Newton PMF-mean correction.
        _fix1_applied = False
        _p_nz_base_fix1: "np.ndarray | None" = None
        _pos_mus_base_fix1: "np.ndarray | None" = None
        _inj_ratio_fix1: "np.ndarray | None" = None

        if _inj_ratio_series is not None:
            _stat_idx = pd.MultiIndex.from_frame(stat_rows[["player_id", "game_id"]])
            _inj_ratio = _inj_ratio_series.reindex(_stat_idx).fillna(1.0).values.astype(float)
            if not np.allclose(_inj_ratio, 1.0):
                baseline_stat_mean = stat_means.copy()
                if p_nz is not None and pos_mus is not None:
                    # Correct hurdle/ZINB EV scaling (Fix 1):
                    # 1. Compute target E[Y] = baseline_E[Y] * ratio  (linear EV scaling)
                    # 2. Update P(nonzero) via exponential-hazard model
                    # 3. Solve conditional positive mean: pos_mu = target / p_nz_adj
                    # Do NOT also multiply pos_mus by ratio — that over-adjusts.
                    #
                    # Save baseline p_nz, pos_mus, and the injury ratio for the Newton
                    # PMF-mean correction applied below (after effective_quant is known).
                    _p_nz_base_fix1  = p_nz.copy()
                    _pos_mus_base_fix1 = pos_mus.copy()
                    _inj_ratio_fix1   = _inj_ratio.copy()
                    _fix1_applied = True

                    _epsilon = 1e-9
                    _target_ev = baseline_stat_mean * _inj_ratio  # (n,) element-wise
                    _p_nz_clipped = np.clip(p_nz, _epsilon, 1.0 - _epsilon)
                    p_nz = np.clip(
                        1.0 - np.power(1.0 - _p_nz_clipped, _inj_ratio),
                        _epsilon, 1.0 - _epsilon,
                    )
                    # Solve conditional positive mean from target
                    pos_mus = _target_ev / np.maximum(p_nz, _epsilon)
                    # For zero-truncated positive-count distributions: if pos_mus_adj
                    # would be below minimum feasible, reduce p_nz instead so that
                    # p_nz * pos_mus == target_ev exactly.
                    _min_pos_mu = _epsilon
                    _too_small = pos_mus < _min_pos_mu
                    if _too_small.any():
                        pos_mus = np.where(_too_small, _min_pos_mu, pos_mus)
                        p_nz = np.clip(
                            np.where(_too_small, _target_ev / _min_pos_mu, p_nz),
                            _epsilon, 1.0 - _epsilon,
                        )
                    stat_means = p_nz * pos_mus
                else:
                    stat_means = np.maximum(stat_means * _inj_ratio, 1e-9)
            else:
                baseline_stat_mean = stat_means
        else:
            baseline_stat_mean = stat_means

        # ---- Build PMF matrix ---------------------------------------------
        # If role_bucket is missing from stat_rows but dispersion_r_by_role config
        # is present, derive role_bucket from predicted minutes_mean so the role-aware
        # dispersion path in _build_pmf_matrix can fire.  Without this, roles=None
        # and the global r is used — causing calibrators to over-correct by ~40%.
        if "role_bucket" not in stat_rows.columns and "minutes_mean" in stat_rows.columns:
            _disp_cfg_local = cfg.get("dispersion_r_by_role", {})
            if _disp_cfg_local:
                try:
                    from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket as _add_rb  # noqa: PLC0415
                    stat_rows = _add_rb(stat_rows, minutes_col="minutes_mean")
                except Exception:
                    pass
        roles = stat_rows["role_bucket"].values if "role_bucket" in stat_rows.columns else None

        # Enhancement 19: use rotation model for bimodal minutes if enabled
        use_rotation_model = cfg.get("use_rotation_model", False)
        if use_rotation_model and "projected_minutes" in stat_rows.columns:
            try:
                from wnba_props_model.models.rotation_model import build_rotation_minutes_samples  # noqa: PLC0415
                _rotation_samples_list = []
                for _, _pr in stat_rows.iterrows():
                    _feats = {
                        "projected_minutes":     float(_pr.get("projected_minutes", 20.0)),
                        "pregame_win_probability": float(_pr.get("pregame_win_probability", 0.50)),
                        "blowout_probability":    float(_pr.get("blowout_probability", 0.15)),
                    }
                    _rotation_samples_list.append(build_rotation_minutes_samples(_feats, n_samples=1000))
                # Summarise into 5-quantile matrix matching _build_marginalized_pmf_matrix expectations
                import numpy as _np_r  # noqa: PLC0415
                quant_mat_rotation = _np_r.array([
                    [_np_r.percentile(s, q) for q in [10, 25, 50, 75, 90]]
                    for s in _rotation_samples_list
                ])
                # Temporarily override quant_mat below
                _use_rotation_quants = True
            except Exception as _rme:
                _use_rotation_quants = False
        else:
            _use_rotation_quants = False

        use_marginalization = cfg.get("use_minutes_marginalization", False)
        # Defaults for the Newton correction tracker (set to True/False inside each branch).
        _pmf_built_marginalized = False
        _pmf_built_quant_mat    = None
        if use_marginalization and hasattr(minutes_model, "_quantile_models") and minutes_model._quantile_models:
            # Retrieve per-player quantile minutes for quadrature
            X_for_quant = wide_df.set_index(["player_id", "game_id"]).reindex(
                pd.MultiIndex.from_frame(stat_rows[["player_id", "game_id"]])
            ).reset_index(drop=True)
            X_for_quant_aligned, _ = prepare_feature_matrix(
                X_for_quant, model_feature_cols,
                pos_encoder=getattr(minutes_model, "_pos_encoder", None),
                fit_encoder=False,
            )
            quant_mat = minutes_model.predict_quantiles(X_for_quant_aligned, X_for_quant)
            quad_weights = np.array(cfg.get(
                "minutes_marginalization_weights", [0.10, 0.15, 0.50, 0.15, 0.10]
            ))
            # Use rotation model bimodal quantiles if available (E19)
            effective_quant = quant_mat_rotation if _use_rotation_quants else quant_mat

            # Defect 4: Apply injury multiplier and cap to ALL minutes quantiles.
            # Same transformation as for min_means: multiply each quantile by
            # conditional_minutes_multiplier, then cap at conditional_minutes_cap.
            # Also applies to rotation-model quantiles (same transformation).
            # Enforces monotonicity (q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90) after transformation.
            if _inj_mult_series is not None:
                _stat_idx_q = pd.MultiIndex.from_frame(stat_rows[["player_id", "game_id"]])
                _q_mult = _inj_mult_series.reindex(_stat_idx_q).fillna(1.0).values.astype(float)
                if not np.allclose(_q_mult, 1.0):
                    effective_quant = effective_quant.copy()
                    effective_quant = effective_quant * _q_mult[:, np.newaxis]
                    # Apply cap to each quantile
                    if _inj_cap_series is not None:
                        _q_cap_raw = _inj_cap_series.reindex(_stat_idx_q).values
                        _q_cap_f = pd.to_numeric(
                            pd.Series(_q_cap_raw), errors="coerce"
                        ).values.astype(float)
                        _has_cap_q = np.isfinite(_q_cap_f) & (_q_cap_f > 0)
                        if _has_cap_q.any():
                            effective_quant[_has_cap_q] = np.minimum(
                                effective_quant[_has_cap_q],
                                _q_cap_f[_has_cap_q, np.newaxis],
                            )
                    # Enforce monotonicity: q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
                    for _qi in range(1, effective_quant.shape[1]):
                        effective_quant[:, _qi] = np.maximum(
                            effective_quant[:, _qi], effective_quant[:, _qi - 1]
                        )

            pmf_mat = _build_marginalized_pmf_matrix(
                stat, effective_quant, quad_weights, p_nz, pos_mus,
                stat_models, hurdle_models, cap, roles=roles,
                stat_means=stat_means,
            )
            _pmf_built_marginalized = True
            _pmf_built_quant_mat = quant_mat  # pre-injury quantile matrix (for Newton baseline)
        else:
            # Extract per-player position for REB dispersion
            _positions = None
            if "position" in stat_rows.columns:
                _positions = stat_rows["position"].fillna("F").values
            elif "position" in wide_df.columns:
                _pos_lu = wide_df.set_index(["player_id", "game_id"])["position"]
                _positions = stat_rows.set_index(["player_id", "game_id"]).index.map(_pos_lu).fillna("F").values
            pmf_mat = _build_pmf_matrix(
                stat, stat_means, p_nz, pos_mus,
                stat_models, hurdle_models, cap, roles=roles,
                bb_models=bb_models, X_stat_df=X_stat_df,
                pts_hurdle_model=pts_hurdle_model,
                stat_rows=stat_rows,
                positions=_positions,
            )
            _pmf_built_marginalized = False
            _pmf_built_quant_mat = None

        # ---- Fix 1 Newton PMF-mean correction (ZINB hurdle stats only) ------
        # The initial Fix 1 formula targets p_nz * pos_mus = target_ev, but the
        # actual PMF mean after cap truncation and renormalization may differ (by
        # up to ~P_tail per element).  One Newton step corrects pos_mus so the
        # built PMF mean equals baseline_pmf_mean * ratio within ~1e-14.
        if _fix1_applied and _p_nz_base_fix1 is not None and _pos_mus_base_fix1 is not None:
            from wnba_props_model.models.hurdle import ZINBStatModel as _ZINBSMnc  # noqa: PLC0415
            if stat in hurdle_models and isinstance(hurdle_models[stat], _ZINBSMnc):
                _k_nc = np.arange(pmf_mat.shape[1], dtype=float)
                _trial_pmf_mean_nc = (pmf_mat * _k_nc[np.newaxis, :]).sum(axis=1)

                # Build baseline PMF (no injury) to get baseline_pmf_mean.
                _pi_base_nc     = np.clip(1.0 - _p_nz_base_fix1, 0.0, 1.0)
                _smbase_nc      = _p_nz_base_fix1 * _pos_mus_base_fix1
                _r_nc           = hurdle_models[stat]._r
                if _pmf_built_marginalized and _pmf_built_quant_mat is not None:
                    _base_pmf_nc = _build_marginalized_pmf_matrix(
                        stat, _pmf_built_quant_mat, quad_weights,
                        _p_nz_base_fix1, _pos_mus_base_fix1,
                        stat_models, hurdle_models, cap, roles=roles,
                        stat_means=_smbase_nc,
                    )
                else:
                    _base_pmf_nc = zinb_pmf_batch(_pi_base_nc, _pos_mus_base_fix1, _r_nc, cap)
                _kbase_nc        = np.arange(_base_pmf_nc.shape[1], dtype=float)
                _base_pmf_mean_nc = (_base_pmf_nc * _kbase_nc[np.newaxis, :]).sum(axis=1)
                _target_pmf_mean_nc = _base_pmf_mean_nc * _inj_ratio_fix1

                # Iterative Newton correction: scale pos_mus until PMF mean == target.
                # One step reduces the residual by ~10-100x; 8 iterations reach 1e-14.
                _pi_adj_nc  = np.clip(1.0 - p_nz, 0.0, 1.0)
                _pos_mus_nc = pos_mus.copy()
                _cur_mean_nc = _trial_pmf_mean_nc.copy()
                for _nc_iter in range(8):
                    _safe_cur  = np.maximum(_cur_mean_nc, 1e-15)
                    _cf        = _target_pmf_mean_nc / _safe_cur
                    _needs_nc  = np.abs(_cf - 1.0) > 1e-12
                    if not _needs_nc.any():
                        break
                    _pos_mus_nc = np.where(_needs_nc, _pos_mus_nc * _cf, _pos_mus_nc)
                    _pos_mus_nc = np.maximum(_pos_mus_nc, 1e-9)
                    if _pmf_built_marginalized and _pmf_built_quant_mat is not None:
                        _cur_pmf_nc = _build_marginalized_pmf_matrix(
                            stat, effective_quant, quad_weights,
                            p_nz, _pos_mus_nc,
                            stat_models, hurdle_models, cap, roles=roles,
                            stat_means=p_nz * _pos_mus_nc,
                        )
                    else:
                        _cur_pmf_nc = zinb_pmf_batch(_pi_adj_nc, _pos_mus_nc, _r_nc, cap)
                    _kc_nc      = np.arange(_cur_pmf_nc.shape[1], dtype=float)
                    _cur_mean_nc = (_cur_pmf_nc * _kc_nc[np.newaxis, :]).sum(axis=1)

                # Assign the converged PMF and update pos_mus / stat_means.
                if _pmf_built_marginalized and _pmf_built_quant_mat is not None:
                    pmf_mat = _build_marginalized_pmf_matrix(
                        stat, effective_quant, quad_weights,
                        p_nz, _pos_mus_nc,
                        stat_models, hurdle_models, cap, roles=roles,
                        stat_means=p_nz * _pos_mus_nc,
                    )
                else:
                    pmf_mat = zinb_pmf_batch(_pi_adj_nc, _pos_mus_nc, _r_nc, cap)
                pos_mus    = _pos_mus_nc
                stat_means = p_nz * pos_mus

        # ---- Apply DNP blending -------------------------------------------
        p_dnp_arr = stat_rows["p_dnp"].fillna(0.0).values.astype(float)
        if use_marginalization and np.any(p_dnp_arr > 0.0):
            pmf_mat = _blend_with_dnp(pmf_mat, p_dnp_arr)

        # Sanitize before validation — prevents single-row numerical issues aborting pipeline
        pmf_mat, _n_fixed = sanitize_pmf_matrix(pmf_mat)
        if _n_fixed > 0:
            import warnings as _w_san
            _w_san.warn(
                f"[pmf_engine] {stat}: {_n_fixed} PMF row(s) sanitized (non-finite/zero-sum) before validation",
                stacklevel=2,
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
            "season":                   stat_rows["season"].values
                                        if "season" in stat_rows.columns else None,
            "player_id":                stat_rows["player_id"].values,
            "player_name":              stat_rows["player_name"].values
                                        if "player_name" in stat_rows.columns else None,
            "team_id":                  stat_rows["team_id"].values
                                        if "team_id" in stat_rows.columns else None,
            "team_abbreviation":        stat_rows["team_abbreviation"].values
                                        if "team_abbreviation" in stat_rows.columns else None,
            "opponent_team_id":         stat_rows["opponent_team_id"].values
                                        if "opponent_team_id" in stat_rows.columns
                                        else None,
            "opponent_team_abbreviation": stat_rows["opponent_team_abbreviation"].values
                                        if "opponent_team_abbreviation" in stat_rows.columns
                                        else None,
            "stat":                     stat,
            "actual_outcome":           stat_rows["actual_outcome"].values
                                        if "actual_outcome" in stat_rows.columns
                                        else None,
            "actual_minutes":           stat_rows["actual_minutes"].values
                                        if "actual_minutes" in stat_rows.columns
                                        else None,
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
            "pmf_mean_engine":          pmf_means,  # engine float64 mean, not JSON-recomputed
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
    roles: np.ndarray | None = None,
    bb_models: dict | None = None,
    X_stat_df: "pd.DataFrame | None" = None,
    pts_hurdle_model: "Any | None" = None,
    stat_rows: "pd.DataFrame | None" = None,
    positions: "np.ndarray | None" = None,
) -> np.ndarray:
    """Build PMF matrix (n × cap+1) for a stat.

    When ``roles`` is provided and the model has per-role dispersion, PMFs are
    batched by role_bucket so each group gets its own NegBinom r parameter.
    Typically 4-6 role groups — this is fast.

    When ``bb_models`` contains a BetaBinomialStatModel for fg3m and X_stat_df
    is provided, uses the Beta-Binomial PMF instead of NegBinom.

    When ``pts_hurdle_model`` is provided and stat == 'pts', uses the minutes-
    conditional hurdle model to compute per-player P(nonzero) then builds a
    truncated-zero NegBinom PMF for each player.
    """
    # ---- PTS hurdle model (minutes-conditional zero-inflation) ---------------
    if (stat == "pts" and pts_hurdle_model is not None and stat_rows is not None):
        try:
            n = len(stat_means)
            pmf_mat = np.zeros((n, cap + 1))
            _p_dnp = stat_rows["p_dnp"].fillna(0.0).values.astype(float) if "p_dnp" in stat_rows.columns else np.zeros(n)
            _min_mean = stat_rows["minutes_mean"].fillna(20.0).values.astype(float) if "minutes_mean" in stat_rows.columns else np.full(n, 20.0)
            _min_sigma = stat_rows["minutes_sigma"].fillna(4.0).values.astype(float) if "minutes_sigma" in stat_rows.columns else np.full(n, 4.0)
            _roles_arr = stat_rows["role_bucket"].fillna("rotation").values if "role_bucket" in stat_rows.columns else np.full(n, "rotation")
            p_nz_hurdle = pts_hurdle_model.predict_p_nonzero(_p_dnp, _roles_arr, _min_mean, _min_sigma)
            for i in range(n):
                role_i = str(_roles_arr[i]) if not pd.isna(_roles_arr[i]) else "rotation"
                r_i = pts_hurdle_model.role_dispersion.get(role_i, 3.5)
                pos_mu_i = float(stat_means[i]) / max(float(p_nz_hurdle[i]), 0.01)
                pos_mu_i = max(pos_mu_i, 0.1)
                pmf_mat[i] = pts_hurdle_model.build_pmf(float(p_nz_hurdle[i]), pos_mu_i, r_i, cap)
            return pmf_mat
        except Exception as _phe:
            import warnings as _w
            _w.warn(f"PtsHurdleModel failed, falling back to global NegBinom: {_phe}", stacklevel=3)

    # ---- Beta-Binomial for fg3m -----------------------------------------
    if (stat == "fg3m" and bb_models is not None and "fg3m" in bb_models
            and X_stat_df is not None):
        return bb_models["fg3m"].predict_pmf_matrix(X_stat_df, cap=cap)

    if stat in hurdle_models:
        model = hurdle_models[stat]
        # PositionStratifiedSparseModel returns per-player r values via predict()
        from wnba_props_model.models.sparse_stats_v2 import PositionStratifiedSparseModel as _PSSM  # noqa: PLC0415
        if isinstance(model, _PSSM) and X_stat_df is not None:
            try:
                preds = model.predict(X_stat_df)
                _p_nz   = preds["p_nz"]
                _pos_mu = preds["pos_mus"]
                _r_arr  = preds["role_rs"]
                n = len(_p_nz)
                pmf_mat = np.zeros((n, cap + 1))
                for i in range(n):
                    pmf_mat[i] = model.build_pmf(float(_p_nz[i]), float(_pos_mu[i]), float(_r_arr[i]), cap)
                return pmf_mat
            except Exception as _pssm_exc:
                import warnings as _w
                _w.warn(f"PositionStratifiedSparseModel failed, falling back: {_pssm_exc}", stacklevel=3)
        # ZINB: use the conditional mean E[Y|Y>0] = stat_means / p_nz for the NegBinom
        # component rather than the raw unconditional mean (pos_mus from predict()).
        # stat_means = E[Y] = p_nz * mu_negbinom; dividing recovers E[Y|Y>0].
        from wnba_props_model.models.hurdle import ZINBStatModel as _ZINBStatModel  # noqa: PLC0415
        if isinstance(model, _ZINBStatModel):
            _pi_arr = np.clip(1.0 - p_nz, 0.0, 1.0)  # type: ignore[arg-type]
            _p_nz_zinb = np.clip(p_nz, 1e-6, 1.0)  # type: ignore[arg-type]
            pos_mus = stat_means / _p_nz_zinb  # conditional mean E[Y|Y>0]
            return zinb_pmf_batch(_pi_arr, pos_mus, model._r, cap)  # type: ignore[arg-type]
        pos_r = model.pos_dispersion_r
        return hurdle_pmf_batch(p_nz, pos_mus, pos_r, cap)  # type: ignore[arg-type]

    model = stat_models.get(stat)
    if model is None:
        return poisson_pmf_batch(stat_means, cap)

    # Part 3: Feature-based learned dispersion model (highest priority).
    # Uses the fitted dispersion_model to predict log(r) per player from features,
    # giving player-specific, context-specific NegBinom tails.
    if (X_stat_df is not None
            and getattr(model, "dispersion_model", None) is not None):
        try:
            _usable = getattr(model, "_usable_cols", None)
            _X_disp = X_stat_df.reindex(columns=_usable) if _usable else X_stat_df
            _log_r_vec = model.dispersion_model.predict(_X_disp)
            _r_vec = np.exp(_log_r_vec).clip(0.3, 15.0)
            n = len(stat_means)
            pmf_mat = np.zeros((n, cap + 1))
            for i in range(n):
                pmf_mat[i] = negbinom_pmf_batch(stat_means[i:i+1], float(_r_vec[i]), cap)[0]
            return pmf_mat
        except Exception:
            pass  # fall through to existing dispersion logic

    # P3.1: mean-dependent dispersion — per-row r(mu_i)
    use_mean_dep = getattr(model, "cfg", {}).get("use_mean_dependent_dispersion", False)
    if (use_mean_dep and getattr(model, "_dispersion_slope", None) is not None):
        n = len(stat_means)
        pmf_mat = np.zeros((n, cap + 1))
        for i in range(n):
            r_i = model.get_dispersion("", mu=float(stat_means[i]))
            if r_i is not None:
                pmf_mat[i] = negbinom_pmf_batch(stat_means[i:i+1], r_i, cap)[0]
            else:
                pmf_mat[i] = poisson_pmf_batch(stat_means[i:i+1], cap)[0]
        return pmf_mat

    # Per-role, per-position REB dispersion (Phase G): position adjusts the role-
    # level base r so centers (low r, wide distribution) and guards (high r, tight)
    # get appropriately shaped PMFs regardless of whether the model has _role_dispersion.
    if stat == "reb" and roles is not None and positions is not None:
        n = len(stat_means)
        pmf_mat = np.zeros((n, cap + 1))
        for i in range(n):
            r_i = compute_reb_effective_r(str(roles[i]), str(positions[i]))
            pmf_mat[i] = negbinom_pmf_batch(stat_means[i:i+1], r_i, cap)[0]
        return pmf_mat

    # AST: role-aware ZINB zero-inflation to correct systematic UNDER bias.
    # Bench/fringe players get 0 assists in ~30-40% of games; starters ~10-15%.
    # Only fires when ast is NOT handled by an explicit hurdle/ZINB model above.
    if stat == "ast" and roles is not None:
        ROLE_ZERO_INFLATION: dict[str, float] = {
            "starter": 0.10,
            "core": 0.12,
            "rotation": 0.20,
            "bench": 0.32,
            "fringe": 0.40,
            "inactive_risk": 0.50,
        }
        n = len(stat_means)
        pmf_mat = np.zeros((n, cap + 1))
        for i in range(n):
            role = str(roles[i])
            pi_zero = ROLE_ZERO_INFLATION.get(role, 0.20)
            # E[Y] = (1 - pi_zero) * mu_nb  →  mu_nb = stat_means[i] / (1 - pi_zero)
            mu_nb = float(stat_means[i]) / max(1.0 - pi_zero, 0.1)
            r_ast = 2.0
            if model is not None and hasattr(model, "get_dispersion"):
                r_val = model.get_dispersion(role)
                if r_val:
                    r_ast = float(r_val)
            nb_pmf = negbinom_pmf_batch(np.array([mu_nb]), r_ast, cap)[0]
            # ZINB: P(Y=0) = pi_zero + (1-pi_zero)*P_NB(0); P(Y=k>0) = (1-pi_zero)*P_NB(k)
            zinb = np.zeros(cap + 1)
            zinb[0] = pi_zero + (1.0 - pi_zero) * nb_pmf[0]
            zinb[1:] = (1.0 - pi_zero) * nb_pmf[1:]
            zinb /= zinb.sum()
            pmf_mat[i] = zinb
        return pmf_mat

    # Role-aware NegBinom batching: star players have fatter tails than bench.
    if roles is not None and getattr(model, "_role_dispersion", None):
        n = len(stat_means)
        pmf_mat = np.zeros((n, cap + 1))
        for role in np.unique(roles):
            mask = roles == role
            r_role = model.get_dispersion(str(role))
            mu_role = stat_means[mask]
            if r_role is not None:
                pmf_mat[mask] = negbinom_pmf_batch(mu_role, r_role, cap)
            else:
                pmf_mat[mask] = poisson_pmf_batch(mu_role, cap)
        return pmf_mat

    # Global dispersion fallback
    r = getattr(model, "dispersion_r", None)
    if r is not None:
        return negbinom_pmf_batch(stat_means, r, cap)
    return poisson_pmf_batch(stat_means, cap)


# ---------------------------------------------------------------------------
# Minutes-marginalized PMF construction (F1)
# ---------------------------------------------------------------------------

def _build_marginalized_pmf_matrix(
    stat: str,
    quant_mat: np.ndarray,
    quad_weights: np.ndarray,
    p_nz: np.ndarray | None,
    pos_mus: np.ndarray | None,
    stat_models: dict,
    hurdle_models: dict,
    cap: int,
    roles: np.ndarray | None = None,
    stat_means: np.ndarray | None = None,
) -> np.ndarray:
    """Build minutes-marginalized PMF matrix using Gauss-style quadrature.

    For each of the 5 quantile minute points (q10..q90):
      mu_i = stat_mean * (q_i / q50)  (scale the stat-model predicted count)
      PMF_i = PMF at mu_i
    Final PMF = sum(weight_i * PMF_i)

    For hurdle models the p_nz component is held fixed (non-playing probability
    does not change with minute variance); only the positive tail is blended.

    Args:
        stat_means: Predicted stat counts from the stat model (e.g. 3.5 reb).
            Must be provided for non-hurdle stats so the scaling uses the actual
            model output rather than the raw minute quantiles.  When None, falls
            back to the median-minutes column of quant_mat — this is incorrect
            for most stats but kept as a safety net.
    """
    n = quant_mat.shape[0]
    n_q = quant_mat.shape[1]  # 5 quantile points
    if len(quad_weights) != n_q:
        quad_weights = np.full(n_q, 1.0 / n_q)
    quad_weights = quad_weights / quad_weights.sum()

    pmf_acc = np.zeros((n, cap + 1), dtype=float)

    # Median projected minutes — used only as the denominator for the scale factor
    # so that scale(q50) == 1 and the stat mean is unmodified at the median.
    q50_mins = quant_mat[:, 2].clip(0.001)

    for qi in range(n_q):
        q_mins = quant_mat[:, qi].clip(0.0)
        # Scale factor: q_i / q50  (relative deviation from median minutes)
        scale = np.where(q50_mins > 0, q_mins / q50_mins, 1.0)

        if stat in hurdle_models:
            model = hurdle_models[stat]
            # Scale pos_mus/mu by the minute ratio; p_nz / π unchanged
            scaled_pos_mus = np.clip(pos_mus * scale, 1e-9, None)  # type: ignore[operator]
            from wnba_props_model.models.hurdle import ZINBStatModel as _ZINBStatModel  # noqa: PLC0415
            if isinstance(model, _ZINBStatModel):
                _pi_arr = np.clip(1.0 - p_nz, 0.0, 1.0)  # type: ignore[arg-type]
                pmf_i = zinb_pmf_batch(_pi_arr, scaled_pos_mus, model._r, cap)
            else:
                pmf_i = hurdle_pmf_batch(p_nz, scaled_pos_mus, model.pos_dispersion_r, cap)
        else:
            model = stat_models.get(stat)
            # Scale the actual stat-model predictions by the minute ratio.
            # stat_means is the predicted stat count (e.g. 3.5 reb) — NOT minutes.
            # Using q50_mins here was the bug: it put minutes in the PMF mean slot,
            # inflating reb/ast/fg3m/turnover predictions by 3–6×.
            baseline = stat_means if stat_means is not None else q50_mins
            scaled_means = (baseline * scale).clip(1e-9)

            # ast: apply role-stratified zero-inflation (ZINB) to handle the large
            # spike of 0-assist games for bench/fringe players. Without this, the NB
            # is fitted to the zero-inflated population, dragging the effective mean
            # down and generating persistent UNDER bias on positive assist lines.
            if stat == "ast" and roles is not None:
                _AST_ZERO_INFLATION = {
                    "starter": 0.10, "core": 0.12, "rotation": 0.20,
                    "bench": 0.32, "fringe": 0.40, "inactive_risk": 0.50,
                }
                pmf_i = np.zeros((n, cap + 1))
                r_ast_default = 2.0
                if model is not None:
                    r_ast_default = getattr(model, "dispersion_r", None) or (
                        model.get_dispersion(str(roles[0])) if hasattr(model, "get_dispersion") else 2.0
                    ) or 2.0
                for i in range(n):
                    role = str(roles[i])
                    pi_zero = _AST_ZERO_INFLATION.get(role, 0.20)
                    # Rescale NB mean upward so unconditional E[Y] matches scaled_means[i]
                    mu_nb = float(scaled_means[i]) / max(1.0 - pi_zero, 0.1)
                    r_i = r_ast_default
                    if model is not None and hasattr(model, "get_dispersion"):
                        r_val = model.get_dispersion(role)
                        if r_val is not None:
                            r_i = r_val
                    nb_pmf = negbinom_pmf_batch(np.array([mu_nb]), float(r_i), cap)[0]
                    zinb = np.zeros(cap + 1)
                    zinb[0] = pi_zero + (1.0 - pi_zero) * nb_pmf[0]
                    zinb[1:] = (1.0 - pi_zero) * nb_pmf[1:]
                    total = zinb.sum()
                    pmf_i[i] = zinb / total if total > 0 else zinb

            elif model is None:
                pmf_i = poisson_pmf_batch(scaled_means, cap)
            elif roles is not None and getattr(model, "_role_dispersion", None):
                pmf_i = np.zeros((n, cap + 1))
                for role in np.unique(roles):
                    mask = roles == role
                    r_role = model.get_dispersion(str(role))
                    if r_role is not None:
                        pmf_i[mask] = negbinom_pmf_batch(scaled_means[mask], r_role, cap)
                    else:
                        pmf_i[mask] = poisson_pmf_batch(scaled_means[mask], cap)
            else:
                r = getattr(model, "dispersion_r", None)
                if r is not None:
                    pmf_i = negbinom_pmf_batch(scaled_means, r, cap)
                else:
                    pmf_i = poisson_pmf_batch(scaled_means, cap)

        pmf_acc += quad_weights[qi] * pmf_i

    # Renormalize (weights sum to 1 but floating-point may drift)
    row_sums = pmf_acc.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return pmf_acc / row_sums


def _blend_with_dnp(pmf_mat: np.ndarray, p_dnp: np.ndarray) -> np.ndarray:
    """Blend PMF with degenerate-at-zero using DNP probability.

    final_pmf = p_dnp * [1, 0, 0, ...] + (1 - p_dnp) * pmf

    Replaces the crude apply_low_minutes_adjustment for rows where DNP model is available.
    """
    p_dnp = np.clip(p_dnp, 0.0, 0.99)  # never fully degenerate
    result = pmf_mat.copy()
    result[:, 0] = p_dnp + (1.0 - p_dnp) * pmf_mat[:, 0]
    result[:, 1:] = (1.0 - p_dnp[:, np.newaxis]) * pmf_mat[:, 1:]
    return result


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
