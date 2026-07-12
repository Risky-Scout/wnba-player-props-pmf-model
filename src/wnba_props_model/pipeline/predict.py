"""Stage 6+ production inference pipeline.

Uses the Stage 4 HGB engine (pmf_engine.build_all_pmfs) to generate PMFs,
then optionally applies role-aware isotonic calibrators.

Legacy quantile path (pipeline/train.py, models/base.py, models/simulation.py)
is preserved for audit purposes but is no longer invoked.

Shadow mode
-----------
If ``WNBA_USE_QUANTILE_MODEL`` is set in the environment (any non-empty value),
``predict_player_pmfs`` will also run the new ``WNBAPlayerPropPipeline`` from
``models/quantile_model.py`` in parallel.

- The shadow pipeline trains on historical data from
  ``data/processed/wnba_player_game_features_wide.parquet`` (when available)
  and predicts on the current ``feature_df``.
- Its output is always written to
  ``deliveries/next_game/quantile_edge_board.parquet`` for logging/comparison.
- Summary statistics (OVER count, UNDER count, top-10 edges) are logged at
  INFO level for both the legacy pipeline and the quantile pipeline.
- When ``WNBA_USE_QUANTILE_MODEL=1`` the quantile edge board is ALSO written
  as the primary output alongside the existing ``publishable_edges.parquet``.
  The legacy PMF output is unchanged — shadow mode does not replace it.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from wnba_props_model.constants import DOMAIN_MAX
from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket
from wnba_props_model.models.bivariate_pmf import (
    _DEFAULT_CORRELATIONS,
    _DEFAULT_CORRELATIONS_BY_POS,
    adjust_combo_pmf_for_correlation,
)
from wnba_props_model.models.pmf_engine import (
    STATS,
    build_all_pmfs,
    negbinom_pmf_batch,
    prepare_feature_matrix,
)
from wnba_props_model.models.minutes_model import MinutesModel
from wnba_props_model.models.rate_model import HurdleModel, StatRateModel
from wnba_props_model.models.shrinkage import apply_bayesian_shrinkage
from wnba_props_model.models.simulation import build_combo_pmfs, json_to_pmf, pmf_to_json
from wnba_props_model.pipeline.calibrate import apply_calibrators, apply_role_stratified_corrections

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
    # Priority:
    #   1. model_dir/feature_manifest.json  (written by train_baseline_pmfs.py >= this fix)
    #   2. data/processed/feature_schema_manifest.json  (fallback for pre-fix artifacts)
    #   3. Empty list → causes constant-prediction bug; warn loudly
    manifest: dict = {}
    manifest_path = model_dir / "feature_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        # Fallback: look for the schema manifest produced by build_features.py
        _schema_candidates = [
            Path("data/processed/feature_schema_manifest.json"),
            Path("data/processed/schema_manifest.json"),
            model_dir.parent.parent / "data" / "processed" / "feature_schema_manifest.json",
        ]
        for _cand in _schema_candidates:
            if _cand.exists():
                manifest = json.loads(_cand.read_text())
                logger.info(
                    "feature_manifest.json missing in %s — using fallback: %s",
                    model_dir, _cand,
                )
                # Write a copy into the model dir so future runs are self-contained
                try:
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    logger.info("Wrote feature_manifest.json to %s", manifest_path)
                except Exception as _we:
                    logger.warning("Could not write feature_manifest.json: %s", _we)
                break

    model_feature_cols = manifest.get("model_feature_columns", [])
    if not model_feature_cols:
        logger.warning(
            "No model_feature_columns found — all players will receive identical "
            "predictions (global mean). Run train_baseline_pmfs.py to rebuild artifacts."
        )

    # --- Beta-Binomial models (fg3m) ---
    bb_models: dict = {}
    bb_path = model_dir / "bb_models.joblib"
    if bb_path.exists():
        try:
            bb_models = joblib.load(bb_path)
            logger.info("Loaded bb_models from %s (stats: %s)", bb_path, list(bb_models.keys()))
        except Exception as _bb_exc:
            logger.warning("Could not load bb_models.joblib: %s", _bb_exc)

    return {
        "minutes": minutes,
        "pos_encoder": pos_encoder,
        "rate_models": rate_models,
        "hurdle_models": hurdle_models,
        "model_feature_cols": model_feature_cols,
        "bb_models": bb_models,
    }


def _attach_role_bucket(pmfs_long: pd.DataFrame, feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach ex-ante role_bucket to PMF rows based on predicted minutes_mean.

    role_bucket drives per-role calibration (isotonic calibrators are fitted
    per stat × role).  Without this wiring, all predictions use the global
    calibrator, losing the per-role precision.

    Uses max(minutes_mean_l5, minutes_mean_season * 0.85) so that players with
    recent DNP/injury games (which depress their L5 average) are not wrongly
    demoted to bench/fringe role — which would collapse their predictions.
    """
    if "minutes_mean" not in pmfs_long.columns:
        pmfs_long["role_bucket"] = "all"
        return pmfs_long

    unique_pg = pmfs_long[["player_id", "game_id", "minutes_mean"]].drop_duplicates().copy()

    # Stabilise role bucket: use max(L5 minutes, 85% of season average).
    # This prevents a few DNP/injury games from wrongly demoting starters to bench.
    if feature_df is not None and not feature_df.empty:
        _season_col = next(
            (c for c in ["player_minutes_mean_season", "player_minutes_mean_l20", "player_minutes_mean_l15"]
             if c in feature_df.columns),
            None,
        )
        if _season_col is not None:
            try:
                _feat_idx = feature_df.set_index("player_id")[_season_col]
                _season_mins = unique_pg["player_id"].map(_feat_idx)
                # Take the higher of L5 vs 85% of season average to prevent injury-driven demotion
                unique_pg["minutes_mean"] = np.maximum(
                    unique_pg["minutes_mean"].fillna(0),
                    (_season_mins.fillna(0) * 0.85),
                )
            except Exception:
                pass

    unique_pg = add_ex_ante_role_bucket(unique_pg, minutes_col="minutes_mean")
    rb_map = unique_pg.set_index(["player_id", "game_id"])["role_bucket"]

    pmfs_long["role_bucket"] = pmfs_long.set_index(["player_id", "game_id"]).index.map(rb_map).values
    pmfs_long["role_bucket"] = pmfs_long["role_bucket"].fillna("all")
    return pmfs_long


def _build_combo_pmf_rows(
    pmfs_long: pd.DataFrame,
    corr_map: dict[str, float] | None = None,
    corr_map_by_pos: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Convolve per-stat PMFs into combo-prop PMFs (stocks, pts_ast, etc.).

    For correlated pairs (pts+ast, pts+reb, reb+ast, stl+blk), applies a
    Gaussian copula correction using empirically estimated Pearson correlations.
    When ``corr_map_by_pos`` is provided (position-stratified from P3.5), the
    player's primary position is used to select the appropriate correlation map;
    falls back to the flat ``corr_map`` for unknown positions.

    Canonical stat key mapping:
        stocks   = stl + blk
        pts_ast  = pts + ast   (BDL prop: "points_assists")
        pts_reb  = pts + reb   (BDL prop: "points_rebounds")
        reb_ast  = reb + ast   (BDL prop: "rebounds_assists")
        pts_reb_ast = pts+reb+ast (BDL prop: "points_rebounds_assists")
    """
    # Map build_combo_pmfs output keys to canonical stat names stored in delivery
    _COMBO_KEY_TO_STAT = {
        "stocks": "stocks",
        "pa":     "pts_ast",
        "pr":     "pts_reb",
        "ra":     "reb_ast",
        "pra":    "pts_reb_ast",
    }
    # Component pairs for bivariate copula adjustment (two-component combos only)
    _COMBO_KEY_PAIRS: dict[str, tuple[str, str]] = {
        "stocks": ("stl", "blk"),
        "pa":     ("pts", "ast"),
        "pr":     ("pts", "reb"),
        "ra":     ("reb", "ast"),
    }
    if corr_map is None:
        corr_map = _DEFAULT_CORRELATIONS

    # Load variance compression factors for combo stats from calibration artifact
    _vc_path = Path("artifacts/models/calibration/variance_compress.json")
    _var_compress: dict[str, float] = {}
    if _vc_path.exists():
        try:
            _var_compress = json.loads(_vc_path.read_text())
        except Exception:
            pass

    # Build position lookup from pmfs_long if available (P3.5 position-stratified copula)
    _has_position = "position" in pmfs_long.columns
    _pos_map: dict[tuple, str] = {}
    if _has_position and corr_map_by_pos:
        _pos_map = (
            pmfs_long[["player_id", "game_id", "position"]]
            .drop_duplicates(subset=["player_id", "game_id"])
            .set_index(["player_id", "game_id"])["position"]
            .to_dict()
        )

    combo_rows: list[dict] = []
    # Track mean deviation for logging
    _combo_mean_devs_before: list[float] = []
    _combo_mean_devs_after: list[float] = []

    # Component stats for each combo key (used for expected-mean computation)
    _COMBO_COMPONENTS: dict[str, tuple[str, ...]] = {
        "stocks": ("stl", "blk"),
        "pa":     ("pts", "ast"),
        "pr":     ("pts", "reb"),
        "ra":     ("reb", "ast"),
        "pra":    ("pts", "reb", "ast"),
    }

    for (player_id, game_id), grp in pmfs_long.groupby(["player_id", "game_id"], sort=False):
        # Collect component PMF arrays indexed by stat key
        component_pmfs: dict[str, np.ndarray] = {}
        for _, row in grp.iterrows():
            stat = row["stat"]
            if stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"):
                try:
                    component_pmfs[stat] = json_to_pmf(row["pmf_json"])
                except Exception:
                    pass

        if not component_pmfs:
            continue

        combos = build_combo_pmfs(component_pmfs)
        if not combos:
            continue

        # Use the first stat row as a metadata template
        tmpl = grp.iloc[0].to_dict()

        for combo_key, pmf_arr in combos.items():
            canonical_stat = _COMBO_KEY_TO_STAT.get(combo_key, combo_key)
            cap = DOMAIN_MAX.get(combo_key, DOMAIN_MAX.get(canonical_stat, 105))

            # Apply bivariate copula correction for two-component combos (P3.5)
            if combo_key in _COMBO_KEY_PAIRS:
                s1, s2 = _COMBO_KEY_PAIRS[combo_key]
                if s1 in component_pmfs and s2 in component_pmfs:
                    try:
                        # Resolve position-stratified corr map if available
                        _pos = _pos_map.get((player_id, game_id)) if _pos_map else None
                        _active_corr = corr_map
                        if corr_map_by_pos and _pos:
                            _pos_key = _pos[0].upper() if _pos else None  # "Guard" → "G"
                            if _pos_key and _pos_key in corr_map_by_pos:
                                _active_corr = corr_map_by_pos[_pos_key]
                            elif "all" in corr_map_by_pos:
                                _active_corr = corr_map_by_pos["all"]
                        pmf_arr = adjust_combo_pmf_for_correlation(
                            component_pmfs[s1], component_pmfs[s2],
                            s1, s2, corr_map=_active_corr,
                        )
                    except Exception as exc:
                        logger.debug("[combo:%s] Copula adjustment failed: %s; using convolution", combo_key, exc)

            # Truncate to domain cap and renormalize
            pmf_arr = pmf_arr[: cap + 1]
            if pmf_arr.sum() > 1e-9:
                pmf_arr = pmf_arr / pmf_arr.sum()

            # Apply variance compression from variance_compress.json (combo stats have no
            # .pkl calibrators, so this must be applied directly at inference time).
            # Fits a tighter NegBinomial with the same mean but reduced variance.
            vc_factor = float(_var_compress.get(canonical_stat, 1.0))
            if vc_factor > 1.05 and pmf_arr.sum() > 1e-9:
                ks_vc = np.arange(len(pmf_arr))
                mu_vc = float(ks_vc @ pmf_arr)
                var_vc = float((ks_vc ** 2) @ pmf_arr - mu_vc ** 2)
                var_target = max(var_vc / vc_factor, mu_vc * 1.01)  # floor: slightly super-Poisson
                if var_target > mu_vc and mu_vc > 0.5:
                    r_new = mu_vc ** 2 / (var_target - mu_vc)
                    compressed = negbinom_pmf_batch(np.array([mu_vc]), float(r_new), cap)[0]
                    if compressed.sum() > 1e-9:
                        pmf_arr = compressed / compressed.sum()

            # Enforce marginal mean integrity: E[combo] must equal sum of component means.
            # Copula reweighting and truncation can shift the marginal mean; correct it here.
            _combo_comp_stats = _COMBO_COMPONENTS.get(combo_key)
            if _combo_comp_stats and pmf_arr.sum() > 1e-9:
                _comp_means = []
                for _cs in _combo_comp_stats:
                    if _cs in component_pmfs:
                        _cs_arr = component_pmfs[_cs]
                        if _cs_arr.sum() > 1e-9:
                            _cs_ks = np.arange(len(_cs_arr))
                            _comp_means.append(float(_cs_ks @ _cs_arr / _cs_arr.sum()))
                if _comp_means:
                    _expected_mean = float(np.sum(_comp_means))
                    _actual_mean_pre = float(np.arange(len(pmf_arr)) @ pmf_arr / pmf_arr.sum())
                    _mean_dev_pre = abs(_actual_mean_pre - _expected_mean)
                    _combo_mean_devs_before.append(_mean_dev_pre)
                    if _expected_mean > 0.01 and _mean_dev_pre > 1e-4:
                        from wnba_props_model.pipeline.calibrate import (  # noqa: PLC0415
                            _apply_mean_bias_correction as _combo_abc,
                        )
                        _mean_factor = _expected_mean / max(_actual_mean_pre, 1e-9)
                        pmf_arr = _combo_abc(pmf_arr, float(_mean_factor))
                        if pmf_arr.sum() > 1e-9:
                            pmf_arr = pmf_arr / pmf_arr.sum()
                        _actual_mean_post = float(np.arange(len(pmf_arr)) @ pmf_arr / max(pmf_arr.sum(), 1e-9))
                        _combo_mean_devs_after.append(abs(_actual_mean_post - _expected_mean))
                    else:
                        _combo_mean_devs_after.append(_mean_dev_pre)

            ks = np.arange(len(pmf_arr))
            pmf_mean = float(ks @ pmf_arr)
            pmf_var  = float((ks ** 2) @ pmf_arr - pmf_mean ** 2)
            p0       = float(pmf_arr[0]) if len(pmf_arr) > 0 else 0.0

            # Phantom floor: only suppress truly degenerate near-zero predictions
            # (e.g. pmf_mean=0.016). Legitimate rotation/bench projections (e.g.
            # pts_reb=6.5) are NOT suppressed — OVER/UNDER signals at those levels
            # are real. Suppression is enforced in deliver.py only for UNDER edges.
            _COMBO_PHANTOM_FLOOR: dict[str, float] = {
                "pts_reb":     1.0,
                "pts_ast":     1.0,
                "pts_reb_ast": 2.0,
                "reb_ast":     0.8,
                "stocks":      0.3,
                "blk_stl":     0.3,
            }
            _suppressed = pmf_mean < _COMBO_PHANTOM_FLOOR.get(canonical_stat, 0.0)

            row_dict = {
                k: v for k, v in tmpl.items()
                if k not in ("stat", "pmf_json", "mean", "pmf_mean", "pmf_variance",
                             "stat_mean", "stat_variance", "p0", "actual_outcome",
                             "actual_minutes", "did_play", "pmf_support_max")
            }
            row_dict.update({
                "stat":           canonical_stat,
                "pmf_json":       pmf_to_json(pmf_arr),
                "mean":           round(pmf_mean, 4),
                "pmf_mean":       round(pmf_mean, 4),
                "pmf_variance":   round(pmf_var, 4),
                "stat_mean":      round(pmf_mean, 4),
                "stat_variance":  round(pmf_var, 4),
                "p0":             round(p0, 6),
                "pmf_support_max": cap,
                "pmf_source":     "combo_convolution",
                "actual_outcome": np.nan,
                "combo_suppressed": _suppressed,
            })
            combo_rows.append(row_dict)

    if not combo_rows:
        return pd.DataFrame()
    if _combo_mean_devs_before:
        logger.info(
            "[combo] Mean integrity correction: max_dev_before=%.6f max_dev_after=%.6f "
            "(n=%d combos; must be < 0.02 after)",
            float(np.max(_combo_mean_devs_before)),
            float(np.max(_combo_mean_devs_after)) if _combo_mean_devs_after else 0.0,
            len(_combo_mean_devs_before),
        )
    return pd.DataFrame(combo_rows)


def predict_player_pmfs(
    feature_df: pd.DataFrame,
    model_dir: str | Path = "artifacts/models/stage4_baseline",
    config_path: str | Path | None = "config/model/stage4_baseline.yaml",
    cal_dir: str | Path | None = "artifacts/models/calibration",
    apply_calibration: bool = True,
    apply_shrinkage: bool = True,
    shrinkage_k: float | None = None,
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
    # WNBA_USE_QUANTILE_MODEL=1: shadow mode will run below and promote quantile
    # board as the primary publishable_edges.parquet output. The PMF pipeline
    # still runs to completion — the quantile board is copied on top at the end.

    cfg: dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    artifacts = _load_stage4_models(model_dir)
    model_dir = Path(model_dir)

    # Inject OOF-calibrated role-aware dispersion r into rate models when the config
    # supplies `dispersion_r_by_role`.  The stage4 model is trained with a single
    # global r (≈1.63) but the calibrators were fitted on OOF PMFs that reflected
    # higher r for starters/core (r≈2.84/2.36).  Without this patch, the stage4
    # model produces higher-variance PMFs that the calibrators over-correct by ~40%,
    # causing 98% UNDER on the betting sheet.  The patch is non-destructive: it only
    # sets `_role_dispersion` when the model doesn't already have one.
    _disp_cfg: dict = cfg.get("dispersion_r_by_role", {})
    if _disp_cfg:
        for _stat_name, _disp_by_role in _disp_cfg.items():
            _model = artifacts.get("rate_models", {}).get(_stat_name)
            if _model is not None:
                _base = getattr(_model, "_base_model", _model)
                try:
                    _base._role_dispersion = _disp_by_role
                    logger.info(
                        "[predict] Applied OOF dispersion override for %s: %s",
                        _stat_name, _disp_by_role,
                    )
                except AttributeError:
                    pass

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
        bb_models=artifacts.get("bb_models") or None,
    )

    pmfs_long["model_version"] = "wnba_pmf_v1.0_hgb_calibrated"
    pmfs_long["is_calibrated"] = False
    pmfs_long["cal_source"] = "uncalibrated"

    # Fix minutes-offset stats: use MinutesModel prediction instead of lagging feature
    # The StatRateModel fits on per-minute rate and re-multiplies by player_minutes_mean_l5
    # (a lagging feature). Re-multiply by MinutesModel prediction instead.
    _MINUTES_OFFSET_STATS = ["turnover", "ast"]
    for _stat in _MINUTES_OFFSET_STATS:
        _mask = pmfs_long["stat"] == _stat
        if _mask.any() and "minutes_mean" in pmfs_long.columns:
            if "player_minutes_mean_l5" in feature_df.columns:
                try:
                    _feat_mins = feature_df.set_index(["player_id", "game_id"])["player_minutes_mean_l5"]
                    _pg = pmfs_long[_mask].set_index(["player_id", "game_id"])
                    _feat_min_vals = _feat_mins.reindex(_pg.index).values
                    _safe_feat = np.where(_feat_min_vals > 1.0, _feat_min_vals, 1.0)
                    _model_mins = pmfs_long.loc[_mask, "minutes_mean"].values
                    _old_means = pmfs_long.loc[_mask, "stat_mean"].values.copy()
                    _rate_per_min = _old_means / _safe_feat
                    pmfs_long.loc[_mask, "stat_mean"] = _rate_per_min * np.clip(_model_mins, 0, 45)
                    logger.info(
                        "Minutes-offset fix %s: %d rows, old_mean=%.2f → new_mean=%.2f",
                        _stat, int(_mask.sum()),
                        float(np.nanmean(_old_means)),
                        float(np.nanmean(pmfs_long.loc[_mask, "stat_mean"].values)),
                    )
                except Exception as _mof_exc:
                    logger.warning("Minutes-offset fix failed for %s: %s", _stat, _mof_exc)

    # Pre-role-bucket: apply minutes_mean_override and role_bucket_override from
    # player_form_corrections_2026.json. These fix players whose minutes model
    # misfires (e.g. due to zero_minute_flag in a recent DNP game row), which
    # would otherwise cause a wrong role_bucket assignment and catastrophic
    # calibrator over-correction. Must be applied BEFORE _attach_role_bucket.
    if cal_dir is not None:
        _pfc_early_path = Path(cal_dir) / "player_form_corrections_2026.json"
        if _pfc_early_path.exists():
            try:
                import json as _pfc_early_json  # noqa: PLC0415
                _pfc_early = _pfc_early_json.loads(_pfc_early_path.read_text())
                _pfc_early_players = _pfc_early.get("players", {})
                _pfc_early_n = 0
                for _pfc_pid_str, _pfc_pdata in _pfc_early_players.items():
                    try:
                        _pfc_pid_int = int(_pfc_pid_str)
                    except ValueError:
                        continue
                    _pfc_mask = pmfs_long["player_id"] == _pfc_pid_int
                    if not _pfc_mask.any():
                        continue
                    _mm_override = _pfc_pdata.get("minutes_mean_override")
                    if _mm_override is not None:
                        pmfs_long = pmfs_long.copy()
                        pmfs_long.loc[_pfc_mask, "minutes_mean"] = float(_mm_override)
                        _pfc_early_n += 1
                        logger.info("[predict] minutes_mean override for player_id=%s (%s): → %.1f",
                                    _pfc_pid_str, _pfc_pdata.get("player_name", ""), float(_mm_override))
                if _pfc_early_n > 0:
                    logger.info("[predict] Applied %d minutes_mean override(s) before role-bucket assignment",
                                _pfc_early_n)
            except Exception as _pfc_e_exc:
                logger.warning("[predict] player_form_corrections minutes override failed (non-fatal): %s", _pfc_e_exc)

    # Attach ex-ante role bucket (needed for per-role calibration & dispersion).
    # Pass feature_df so the role bucket uses max(L5, 85%×season) minutes,
    # preventing recent DNP/injury games from wrongly demoting starters to bench.
    pmfs_long = _attach_role_bucket(pmfs_long, feature_df=feature_df)

    # Post-role-bucket: apply role_bucket_override from player_form_corrections_2026.json.
    # Also collect overrides into _shrinkage_role_overrides so that apply_bayesian_shrinkage
    # uses the corrected role prior (shrinkage reads role from features["role_status"], not
    # from pmfs_long["role_bucket"], so it needs the override passed explicitly).
    _shrinkage_role_overrides: dict[int, str] = {}
    if cal_dir is not None:
        _pfc_rb_path = Path(cal_dir) / "player_form_corrections_2026.json"
        if _pfc_rb_path.exists():
            try:
                import json as _pfc_rb_json  # noqa: PLC0415
                _pfc_rb = _pfc_rb_json.loads(_pfc_rb_path.read_text())
                for _pfc_rb_pid, _pfc_rb_data in _pfc_rb.get("players", {}).items():
                    _rb_override = _pfc_rb_data.get("role_bucket_override")
                    if _rb_override:
                        try:
                            _pfc_rb_pid_int = int(_pfc_rb_pid)
                        except ValueError:
                            continue
                        _rb_mask = pmfs_long["player_id"] == _pfc_rb_pid_int
                        if _rb_mask.any():
                            pmfs_long = pmfs_long.copy()
                            pmfs_long.loc[_rb_mask, "role_bucket"] = str(_rb_override)
                            _shrinkage_role_overrides[_pfc_rb_pid_int] = str(_rb_override)
                            logger.info("[predict] role_bucket override for player_id=%s (%s): → %s",
                                        _pfc_rb_pid, _pfc_rb_data.get("player_name", ""), _rb_override)
            except Exception as _pfc_rb_exc:
                logger.warning("[predict] role_bucket_override failed (non-fatal): %s", _pfc_rb_exc)

    # P3.5: Attach player position so the copula uses position-stratified correlations.
    if "position" in feature_df.columns and "position" not in pmfs_long.columns:
        _pos_lu = (
            feature_df[["player_id", "game_id", "position"]]
            .drop_duplicates(subset=["player_id", "game_id"])
        )
        pmfs_long = pmfs_long.merge(_pos_lu, on=["player_id", "game_id"], how="left")

    # P3.5: Load position-stratified combo correlations if artifact exists;
    # fall back to theory-grounded hardcoded estimates when the file is absent.
    _corr_by_pos: dict[str, dict[str, float]] = dict(_DEFAULT_CORRELATIONS_BY_POS)
    _corr_by_pos_path = model_dir / "combo_correlations_by_pos.json"
    if _corr_by_pos_path.exists():
        try:
            import json as _cjson
            with open(_corr_by_pos_path) as _cf:
                _corr_by_pos = _cjson.load(_cf)
            logger.info("Loaded position-stratified combo correlations from %s", _corr_by_pos_path)
        except Exception as _ce:
            logger.warning("Failed to load corr_by_pos: %s — using hardcoded defaults", _ce)
    else:
        logger.info("combo_correlations_by_pos.json not found; using _DEFAULT_CORRELATIONS_BY_POS")

    # CALIBRATION: apply isotonic calibrators on raw PMFs first.
    # Per the training contract the calibrators were fitted on uncorrected PMFs,
    # so role-stratified corrections are applied afterwards as residual adjustments
    # (calibrate → role → shrinkage). Shrinkage then blends the role-corrected PMF
    # toward the league prior.
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

    # Role-stratified bias corrections: applied AFTER isotonic calibration as
    # residual role-level adjustments. Must still run BEFORE apply_bayesian_shrinkage
    # so shrinkage blends the corrected PMF toward the league prior.
    if apply_calibration and cal_dir is not None:
        _role_corr_path = Path(cal_dir) / "bias_corrections_by_role.json"
        if _role_corr_path.exists():
            logger.info("[predict] Applying role-stratified bias corrections from %s", _role_corr_path)
            pmfs_long = apply_role_stratified_corrections(pmfs_long, cal_dir=cal_dir)

    # Apply PenaltyBlog-style Bayesian shrinkage AFTER calibration so that the
    # per-player prior blend operates on the already-corrected (calibrated) PMF.
    # k=None lets apply_bayesian_shrinkage use its per-stat learned Gamma prior
    # (hierarchical Bayes); k=None is now the default so the data drives shrinkage
    # strength instead of a fixed value that ignores inter-stat variance differences.
    if apply_shrinkage:
        pmfs_long = apply_bayesian_shrinkage(
            pmfs_long,
            features=feature_df,
            k=shrinkage_k,  # None → use per-stat Gamma prior.beta
            player_role_overrides=_shrinkage_role_overrides,
        )

    # Archetype-conditioned shrinkage (Phase 4): augments the single-prior
    # Gamma-Poisson shrinkage with per-archetype priors fitted from player clusters.
    # Only applied when the archetype model artifact exists.
    if apply_shrinkage and cal_dir is not None:
        import json as _json  # noqa: PLC0415
        _arch_pkl = Path(cal_dir).parent / "archetype_shrinkage.pkl"
        if _arch_pkl.exists():
            try:
                from wnba_props_model.models.archetype_shrinkage import ArchetypeConditionedShrinkage  # noqa: PLC0415
                from wnba_props_model.models.simulation import json_to_pmf as _jtpmf, pmf_to_json as _ptmj  # noqa: PLC0415
                _arch = ArchetypeConditionedShrinkage.load(str(_arch_pkl))
                _ngames_map = {}
                if feature_df is not None and "player_id" in feature_df.columns:
                    _gc = "player_games_prior" if "player_games_prior" in feature_df.columns else None
                    if _gc:
                        _ngames_map = (feature_df.groupby("player_id")[_gc].max().to_dict())
                _shrunk_jsons = []
                for _, _arow in pmfs_long.iterrows():
                    _pid = str(_arow.get("player_id", ""))
                    _stat = str(_arow.get("stat", ""))
                    _role = str(_arow.get("role_bucket", "starter"))
                    _ng = int(_ngames_map.get(int(_pid), 0)) if _pid.isdigit() else 0
                    _pmf_arr = _jtpmf(_arow["pmf_json"])
                    _shrunk = _arch.shrink_pmf(_pmf_arr, _pid, _stat, _ng, _role)
                    _shrunk_jsons.append(_ptmj(_shrunk))
                pmfs_long = pmfs_long.copy()
                pmfs_long["pmf_json"] = _shrunk_jsons
                logger.info("[predict] Applied archetype-conditioned shrinkage from %s", _arch_pkl)
            except Exception as _ae:
                logger.warning("[predict] Archetype shrinkage failed (non-fatal): %s", _ae)

    # Part C.5: Apply per-player 2026 in-season form corrections.
    # player_form_corrections_2026.json has TWO correction sources:
    #   1. flat_corrections: {"PlayerName|stat": multiplier} — residual corrections
    #      applied POST-shrinkage so shrinkage does not dampen individual form signals.
    #      These are for players whose in-season actuals exceed role-adjusted model
    #      predictions by >15%.
    #   2. players: {"player_id": {"stats": {...}}} — legacy format (now empty of stats
    #      since role-stratified corrections handle population-level bias).
    # Applied AFTER calibration + shrinkage, BEFORE conformal intervals.
    if cal_dir is not None:
        _pfc_path = Path(cal_dir) / "player_form_corrections_2026.json"
        if _pfc_path.exists():
            try:
                import json as _pfc_json  # noqa: PLC0415
                from wnba_props_model.models.simulation import json_to_pmf as _pfc_jtpmf, pmf_to_json as _pfc_ptmj  # noqa: PLC0415
                from wnba_props_model.pipeline.calibrate import _apply_mean_bias_correction as _pfc_abc  # noqa: PLC0415
                _pfc_data = _pfc_json.loads(_pfc_path.read_text())

                # --- Source 1: flat_corrections — DISABLED ---
                # Dangerous [0.10, 25x] clips caused runaway corrections for players
                # with corrupted features. The live feature-based form correction
                # (Part C.6 below) handles recency with tighter [0.80, 1.25] clips.
                _pfc_flat: dict[str, float] = {}  # disabled

                # --- Source 2: legacy players dict (backward compat; stats may be empty) ---
                _pfc_players: dict[str, dict] = _pfc_data.get("players", {})

                _pfc_new_jsons = []
                _pfc_n_applied = 0
                for _, _pfc_row in pmfs_long.iterrows():
                    _pfc_pid = str(int(_pfc_row.get("player_id", 0)))
                    _pfc_stat = str(_pfc_row.get("stat", ""))
                    _pfc_name = str(_pfc_row.get("player_name", ""))

                    # Resolve multiplier: flat_corrections takes priority over legacy stats dict
                    _pfc_flat_key = f"{_pfc_name}|{_pfc_stat}"
                    _pfc_mult: float | None = None
                    if _pfc_flat_key in _pfc_flat:
                        _pfc_mult = float(_pfc_flat[_pfc_flat_key])
                    else:
                        _legacy_mult = _pfc_players.get(_pfc_pid, {}).get("stats", {}).get(_pfc_stat)
                        if _legacy_mult is not None:
                            _pfc_mult = float(_legacy_mult)

                    if _pfc_mult is not None and abs(_pfc_mult - 1.0) > 0.01:
                        _pfc_arr = _pfc_jtpmf(_pfc_row["pmf_json"])
                        _pfc_k = float(np.arange(len(_pfc_arr), dtype=float) @ _pfc_arr / max(_pfc_arr.sum(), 1e-9))
                        if _pfc_k > 0.01:
                            # Wide clip: flat_corrections can be 0.1–25x for players with
                            # severely broken features (e.g. Alyssa Thomas, wrong minutes).
                            # Combo stats for AT need corrections up to 25x since the raw
                            # model outputs near-zero due to corrupted feature inputs.
                            _pfc_corrected = _pfc_abc(_pfc_arr, float(np.clip(_pfc_mult, 0.10, 25.0)))
                            _pfc_new_jsons.append(_pfc_ptmj(_pfc_corrected))
                            _pfc_n_applied += 1
                        else:
                            _pfc_new_jsons.append(_pfc_row["pmf_json"])
                    else:
                        _pfc_new_jsons.append(_pfc_row["pmf_json"])

                pmfs_long = pmfs_long.copy()
                pmfs_long["pmf_json"] = _pfc_new_jsons
                # Recompute pmf_mean from the corrected PMF JSON so downstream
                # columns (mean_disagreement, model_market_ratio, display) are consistent.
                _pfc_new_means = []
                for _pfc_jstr in _pfc_new_jsons:
                    try:
                        _pfc_d = _pfc_json.loads(_pfc_jstr)
                        _pfc_kk = np.array([float(k) for k in _pfc_d.keys()])
                        _pfc_vv = np.array(list(_pfc_d.values()), dtype=float)
                        _pfc_new_means.append(float((_pfc_kk * _pfc_vv).sum() / max(_pfc_vv.sum(), 1e-9)))
                    except Exception:
                        _pfc_new_means.append(np.nan)
                pmfs_long["pmf_mean"] = _pfc_new_means
                if _pfc_n_applied > 0:
                    logger.info("[predict] Applied 2026 form corrections to %d PMF rows from %s "
                                "(%d flat, %d legacy)",
                                _pfc_n_applied, _pfc_path,
                                sum(1 for k in _pfc_flat if k),
                                sum(1 for v in _pfc_players.values() if v.get("stats")))
            except Exception as _pfc_exc:
                logger.warning("[predict] player_form_corrections_2026 failed (non-fatal): %s", _pfc_exc)

    # Part C.6: Feature-based in-season form correction (computed live from fresh BDL data).
    # feature_df is built from fresh BDL data on every pipeline run, so stat means and
    # per-minute rates reflect current July games.
    #
    # Two correction paths — stat-specific to prevent double-counting minutes:
    #
    # pts / fg3m / stl / blk / turnover — absolute-count blend:
    #   blended = 0.35*L10_count + 0.65*season_count vs pmf_mean
    #   clip [0.87, 1.35], threshold 15%.  These stats' PMF means scale linearly with
    #   the count feature, so the ratio is structurally sound.
    #
    # reb / ast — per-minute rate ratio (minutes-neutral):
    #   raw_ratio = (reb_per_min_l10) / (reb_per_min_season)
    #   Using rate vs rate cancels the minutes term embedded in the calibrated PMF mean,
    #   preventing the double-counting that caused 0% reb_ast OVER (the absolute-count
    #   path hit the 0.87 floor when players had fewer recent minutes, even though the
    #   calibrator had already accounted for those minutes).
    #   Applied via symmetric log-space correction:
    #     log_adj = 0.15 * reliability * log(clip(raw_ratio, floor, ceil))
    #     factor  = clip(exp(log_adj), 0.97, 1.03)
    #   reliability = min(L10_support,10) / (min(L10_support,10)+24)  ≈ 0.29 at full L10
    #   Max practical effect: exp(0.15*0.29*ln(1.18)) ≈ ±0.7% per player.
    #   Identity fallback when per-minute rates unavailable.
    if feature_df is not None and not feature_df.empty:
        try:
            from wnba_props_model.models.simulation import json_to_pmf as _fc6_jtpmf, pmf_to_json as _fc6_ptmj  # noqa: PLC0415
            from wnba_props_model.pipeline.calibrate import _apply_mean_bias_correction as _fc6_abc  # noqa: PLC0415
            _FEAT_STATS = {"pts":      ("player_pts_mean_l10",      "player_pts_mean_season"),
                           "reb":      ("player_reb_mean_l10",      "player_reb_mean_season"),
                           "ast":      ("player_ast_mean_l10",      "player_ast_mean_season"),
                           "fg3m":     ("player_fg3m_mean_l10",     "player_fg3m_mean_season"),
                           "stl":      ("player_stl_mean_l10",      "player_stl_mean_season"),
                           "blk":      ("player_blk_mean_l10",      "player_blk_mean_season"),
                           "turnover": ("player_turnover_mean_l10", "player_turnover_mean_season")}
            # reb and ast get minutes-neutral per-minute rate correction.
            _FC6_RATE_STATS = {"reb", "ast"}

            # Build lookup: player_id → {stat → blended_actual, and per-min rates for reb/ast}
            _fc6_lookup: dict[int, dict[str, float]] = {}
            for _fc6_pid, _fc6_grp in feature_df.groupby("player_id"):
                _fc6_row_vals: dict[str, float] = {}
                for _fc6_stat, (_fc6_l10_col, _fc6_s_col) in _FEAT_STATS.items():
                    _fc6_l10 = (float(_fc6_grp[_fc6_l10_col].dropna().iloc[-1])
                                if _fc6_l10_col in _fc6_grp.columns and not _fc6_grp[_fc6_l10_col].dropna().empty
                                else None)
                    _fc6_sea = (float(_fc6_grp[_fc6_s_col].dropna().iloc[-1])
                                if _fc6_s_col in _fc6_grp.columns and not _fc6_grp[_fc6_s_col].dropna().empty
                                else None)
                    if _fc6_l10 is not None and _fc6_sea is not None:
                        _fc6_row_vals[_fc6_stat] = 0.35 * _fc6_l10 + 0.65 * _fc6_sea
                    elif _fc6_sea is not None:
                        _fc6_row_vals[_fc6_stat] = _fc6_sea
                    elif _fc6_l10 is not None:
                        _fc6_row_vals[_fc6_stat] = _fc6_l10
                    # For reb/ast: also store pre-computed per-minute rates and support count
                    if _fc6_stat in _FC6_RATE_STATS:
                        for _sfx, _col in (("_per_min_l10",   f"player_{_fc6_stat}_per_min_l10"),
                                           ("_per_min_season", f"player_{_fc6_stat}_per_min_season"),
                                           ("_l10_support",    f"player_{_fc6_stat}_l10_support")):
                            if _col in _fc6_grp.columns:
                                _v = _fc6_grp[_col].dropna()
                                if not _v.empty:
                                    _fc6_row_vals[f"{_fc6_stat}{_sfx}"] = float(_v.iloc[-1])
                _fc6_lookup[int(_fc6_pid)] = _fc6_row_vals

            _fc6_new_jsons, _fc6_new_means, _fc6_n = [], [], 0
            # Per-minute path counters for reb/ast (to verify correction fires)
            _fc6_pm_applied: dict[str, int] = {"reb": 0, "ast": 0}
            _fc6_pm_identity: dict[str, int] = {"reb": 0, "ast": 0}
            _fc6_pm_factors: dict[str, list] = {"reb": [], "ast": []}
            for _, _fc6_pmf_row in pmfs_long.iterrows():
                _fc6_stat = str(_fc6_pmf_row.get("stat", ""))
                if _fc6_stat not in _FEAT_STATS:
                    _fc6_new_jsons.append(_fc6_pmf_row["pmf_json"])
                    _fc6_new_means.append(_fc6_pmf_row.get("pmf_mean", np.nan))
                    continue

                _fc6_pid_int = int(_fc6_pmf_row.get("player_id", 0))
                _fc6_pid_vals = _fc6_lookup.get(_fc6_pid_int, {})
                _fc6_cur_mean = float(_fc6_pmf_row.get("pmf_mean", 0) or 0)

                if _fc6_stat in _FC6_RATE_STATS:
                    # Minutes-neutral path for reb/ast
                    _pm_l10  = _fc6_pid_vals.get(f"{_fc6_stat}_per_min_l10")
                    _pm_sea  = _fc6_pid_vals.get(f"{_fc6_stat}_per_min_season")
                    _sup     = float(_fc6_pid_vals.get(f"{_fc6_stat}_l10_support", 10) or 10)
                    if _pm_l10 is None or _pm_sea is None or _pm_sea < 0.001 or _fc6_cur_mean < 0.1:
                        _fc6_new_jsons.append(_fc6_pmf_row["pmf_json"])
                        _fc6_new_means.append(_fc6_cur_mean)
                        continue
                    _raw_ratio   = float(_pm_l10) / float(_pm_sea)
                    # Reliability: Bayesian shrink to 1.0; k=24 → full weight at ~half-season
                    _reliability = min(_sup, 10.0) / (min(_sup, 10.0) + 24.0)
                    _r_floor     = 0.82 if _fc6_stat == "ast" else 0.85
                    _r_ceil      = 1.20 if _fc6_stat == "ast" else 1.18
                    _log_adj     = 0.15 * _reliability * float(np.log(np.clip(_raw_ratio, _r_floor, _r_ceil)))
                    _fc6_ratio_new = float(np.clip(np.exp(_log_adj), 0.97, 1.03))
                    # Only skip if factor is floating-point-identical to 1.0 (truly a no-op).
                    # The old 0.01 threshold caused every reb/ast row to be skipped because
                    # max possible factor is ~0.0073 (strength=0.15 × reliability≈0.294 × ln(1.18)≈0.166).
                    if abs(_fc6_ratio_new - 1.0) < 1e-6:
                        _fc6_new_jsons.append(_fc6_pmf_row["pmf_json"])
                        _fc6_new_means.append(_fc6_cur_mean)
                        if _fc6_stat in _fc6_pm_identity:
                            _fc6_pm_identity[_fc6_stat] += 1
                        continue
                    _fc6_arr       = _fc6_jtpmf(_fc6_pmf_row["pmf_json"])
                    _fc6_corrected = _fc6_abc(_fc6_arr, _fc6_ratio_new)
                    _fc6_new_jsons.append(_fc6_ptmj(_fc6_corrected))
                    _fc6_new_mean  = float(np.arange(len(_fc6_corrected)) @ _fc6_corrected / max(_fc6_corrected.sum(), 1e-9))
                    _fc6_new_means.append(_fc6_new_mean)
                    _fc6_n += 1
                    if _fc6_stat in _fc6_pm_applied:
                        _fc6_pm_applied[_fc6_stat] += 1
                        _fc6_pm_factors[_fc6_stat].append(_fc6_ratio_new)
                    continue

                # Absolute-count blend path: pts, fg3m, stl, blk, turnover
                _fc6_actual = _fc6_pid_vals.get(_fc6_stat)
                if _fc6_actual is None or _fc6_actual < 0.1 or _fc6_cur_mean < 0.1:
                    _fc6_new_jsons.append(_fc6_pmf_row["pmf_json"])
                    _fc6_new_means.append(_fc6_cur_mean)
                    continue
                _fc6_ratio = float(np.clip(_fc6_actual / _fc6_cur_mean, 0.87, 1.35))
                if abs(_fc6_ratio - 1.0) < 0.15:
                    _fc6_new_jsons.append(_fc6_pmf_row["pmf_json"])
                    _fc6_new_means.append(_fc6_cur_mean)
                    continue
                _fc6_arr       = _fc6_jtpmf(_fc6_pmf_row["pmf_json"])
                _fc6_corrected = _fc6_abc(_fc6_arr, _fc6_ratio)
                _fc6_new_jsons.append(_fc6_ptmj(_fc6_corrected))
                _fc6_new_mean  = float(np.arange(len(_fc6_corrected)) @ _fc6_corrected / max(_fc6_corrected.sum(), 1e-9))
                _fc6_new_means.append(_fc6_new_mean)
                _fc6_n += 1

            pmfs_long = pmfs_long.copy()
            pmfs_long["pmf_json"]  = _fc6_new_jsons
            pmfs_long["pmf_mean"]  = _fc6_new_means
            logger.info("[predict] Feature-based form correction applied to %d / %d PMF rows", _fc6_n, len(pmfs_long))
            for _pm_stat in ("reb", "ast"):
                _n_adj = _fc6_pm_applied[_pm_stat]
                _n_id  = _fc6_pm_identity[_pm_stat]
                _facs  = _fc6_pm_factors[_pm_stat]
                if _facs:
                    logger.info(
                        "[predict] C.6 per-min rate correction: %d %s rows adjusted (identity: %d) "
                        "factor min=%.6f median=%.6f max=%.6f",
                        _n_adj, _pm_stat, _n_id,
                        float(np.min(_facs)), float(np.median(_facs)), float(np.max(_facs)),
                    )
                else:
                    logger.info(
                        "[predict] C.6 per-min rate correction: 0 %s rows adjusted (identity: %d)",
                        _pm_stat, _n_id,
                    )
        except Exception as _fc6_exc:
            logger.warning("[predict] Feature-based form correction failed (non-fatal): %s", _fc6_exc)

    # Build combo-prop PMFs from fully-corrected base-stat PMFs.
    # Must happen AFTER all base-stat corrections (calibration, shrinkage, form corrections
    # C.5 and C.6) so that component means are final.  No pre-correction is needed because
    # the base-stat pmf_json values already reflect the correct calibrated means.

    # Combo build: assert uniqueness of base-stat rows, strip stale combo rows,
    # build combos exactly once, then assert uniqueness of final result.
    _COMBO_STATS = {"pts_reb", "pts_ast", "pts_reb_ast", "reb_ast", "stocks", "blk_stl"}

    # Hard error on duplicate base-stat rows — indicates a broken construction path.
    _base_pmfs = pmfs_long[~pmfs_long["stat"].isin(_COMBO_STATS)].copy()
    _dup_mask = _base_pmfs.duplicated(["player_id", "game_id", "stat"], keep=False)
    if _dup_mask.any():
        _dup_info = (
            _base_pmfs.loc[_dup_mask, ["player_id", "game_id", "stat"]]
            .sort_values(["player_id", "stat"])
        )
        raise ValueError(
            f"[combo_guard] {int(_dup_mask.sum())} duplicate base PMF rows detected before combo build.\n"
            "This indicates a broken construction path. Fix the upstream source.\n"
            f"{_dup_info.to_string(index=False)}"
        )

    # Strip any stale combo rows from a prior build attempt before constructing fresh ones.
    pmfs_long = pmfs_long[~pmfs_long["stat"].isin(_COMBO_STATS)].copy()

    combo_rows = _build_combo_pmf_rows(pmfs_long, corr_map_by_pos=_corr_by_pos)
    if not combo_rows.empty:
        pmfs_long = pd.concat([pmfs_long, combo_rows], ignore_index=True)
        logger.info(
            "[predict] Built %d combo PMF rows from fully-corrected base stats (%s)",
            len(combo_rows),
            sorted(combo_rows["stat"].unique().tolist()),
        )

    # Hard error if combo build introduced any duplicates.
    _all_dup = pmfs_long.duplicated(["player_id", "game_id", "stat"], keep=False)
    if _all_dup.any():
        raise ValueError(
            f"[post_combo] {int(_all_dup.sum())} duplicate PMF rows after combo build. "
            "This must not happen."
        )

    # Part D: Apply guaranteed conformal prediction intervals.
    # conformal_predictor.pkl is fitted weekly by fit_calibrators() via ConformalPropPredictor.
    # Without this, conformal_90_ci in the output is merely ±1.645σ (no coverage guarantee).
    if cal_dir is not None:
        import pickle as _pickle  # noqa: PLC0415
        _conformal_pkl = Path(cal_dir) / "conformal_predictor.pkl"
        if _conformal_pkl.exists():
            try:
                with open(_conformal_pkl, "rb") as _cpf:
                    _conformal = _pickle.load(_cpf)
                # Vectorised conformal interval: add conformal_lower / conformal_upper columns
                _lowers, _uppers = [], []
                for _row in pmfs_long.itertuples():
                    _stat_r = str(getattr(_row, "stat", ""))
                    _role_r = str(getattr(_row, "role_bucket", "all"))
                    _mean_r = float(getattr(_row, "pmf_mean", 0.0))
                    _lo, _hi = _conformal.predict_interval(_mean_r, _stat_r, _role_r)
                    _lowers.append(round(max(0.0, _lo), 1))
                    _uppers.append(round(_hi, 1))
                pmfs_long = pmfs_long.copy()
                pmfs_long["conformal_lower"] = _lowers
                pmfs_long["conformal_upper"] = _uppers
                pmfs_long["conformal_source"] = "split_conformal"
            except Exception as _ce:
                logger.warning("[predict] Conformal interval computation failed: %s", _ce)

    # Direction balance warning: flag any base stat whose predicted mean is
    # unusually concentrated low (median pmf_mean < 40% of expected floor),
    # which is a leading indicator of OVER-rate collapse on the edge board.
    try:
        _BASE_EXPECTED_FLOOR = {"pts": 4.0, "reb": 1.0, "ast": 0.5, "fg3m": 0.2,
                                 "stl": 0.1, "blk": 0.1, "turnover": 0.3}
        for _dw_stat, _dw_floor in _BASE_EXPECTED_FLOOR.items():
            _dw_mask = pmfs_long["stat"] == _dw_stat
            if not _dw_mask.any():
                continue
            _dw_median = float(pmfs_long.loc[_dw_mask, "pmf_mean"].median())
            if _dw_median < _dw_floor:
                logger.warning(
                    "[direction_guard] stat=%s median pmf_mean=%.3f is below expected "
                    "floor %.2f — possible systematic underprediction; check OVER rate",
                    _dw_stat, _dw_median, _dw_floor,
                )
    except Exception as _dw_exc:
        logger.debug("[direction_guard] warning check failed (non-fatal): %s", _dw_exc)

    # Shadow mode: run multi-quantile pipeline in parallel for comparison.
    # Always runs when historical data is available; extra output written to
    # deliveries/next_game/quantile_edge_board.parquet.
    # Set WNBA_USE_QUANTILE_MODEL=1 to promote quantile board as primary.
    if feature_df is not None and not feature_df.empty:
        try:
            _run_quantile_shadow(
                feature_df=feature_df,
                pmfs_long=pmfs_long,
                model_feature_cols=artifacts.get("model_feature_cols", []),
            )
        except Exception as _shadow_exc:
            logger.warning("[predict] Quantile shadow mode failed (non-fatal): %s", _shadow_exc)

    return pmfs_long


def _run_quantile_shadow(
    feature_df: pd.DataFrame,
    pmfs_long: pd.DataFrame,
    model_feature_cols: list[str],
    delivery_dir: str | Path = "deliveries/next_game",
    hist_features_path: str | Path = "data/processed/wnba_player_game_features_wide.parquet",
    stats: list[str] | None = None,
) -> None:
    """Run multi-quantile pipeline in shadow mode alongside the legacy pipeline.

    Trains ``WNBAPlayerPropPipeline`` on historical feature data (when available),
    generates quantile-level predictions on ``feature_df``, and writes the
    resulting edge board to ``deliveries/next_game/quantile_edge_board.parquet``.

    Logs OVER/UNDER counts and top-10 edges from both legacy and quantile boards.
    All errors are caught and logged — this function must never crash the
    production prediction pipeline.

    Args:
        feature_df: Today's wide feature DataFrame (inference rows).
        pmfs_long: Legacy pipeline's long PMF output (for comparison logging).
        model_feature_cols: Feature column names used by the Stage 4 models.
        delivery_dir: Directory for quantile edge board output.
        hist_features_path: Path to historical features parquet for training.
        stats: Stats to model. Defaults to ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'turnover'].
    """
    try:
        from wnba_props_model.models.quantile_model import QUANTILES, WNBAPlayerPropPipeline  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("[quantile_shadow] Import failed — skipping shadow mode: %s", exc)
        return

    if stats is None:
        stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    delivery_dir = Path(delivery_dir)
    delivery_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load historical data and train the quantile pipeline
    # ------------------------------------------------------------------
    hist_path = Path(hist_features_path)
    if not hist_path.exists():
        logger.warning(
            "[quantile_shadow] Historical features not found at %s — "
            "skipping quantile shadow training",
            hist_path,
        )
        return

    try:
        hist_df = pd.read_parquet(hist_path)
    except Exception as exc:
        logger.warning("[quantile_shadow] Failed to read historical features: %s", exc)
        return

    # Resolve feature columns: use model_feature_cols filtered to what exists in hist_df
    feat_cols = [c for c in model_feature_cols if c in hist_df.columns]
    if len(feat_cols) < 5:
        logger.warning(
            "[quantile_shadow] Too few usable feature cols (%d) in historical data — "
            "skipping shadow training",
            len(feat_cols),
        )
        return

    # Build y_dict (stat → actual values) for historical rows with did_play=True
    play_col = "did_play"
    if play_col in hist_df.columns:
        hist_train = hist_df[hist_df[play_col].fillna(False).astype(bool)].copy()
    else:
        hist_train = hist_df.copy()

    y_dict: dict[str, pd.Series] = {}
    for stat in stats:
        actual_col = f"actual_{stat}"
        if actual_col in hist_train.columns:
            y_dict[stat] = hist_train[actual_col].reset_index(drop=True)

    if not y_dict:
        logger.warning("[quantile_shadow] No actual_* columns found in historical data — skipping")
        return

    X_hist = hist_train[feat_cols].reset_index(drop=True)
    dates_hist = (
        pd.to_datetime(hist_train["game_date"], errors="coerce").reset_index(drop=True)
        if "game_date" in hist_train.columns
        else pd.Series(range(len(hist_train)))
    )
    pid_hist = (
        hist_train["player_id"].reset_index(drop=True)
        if "player_id" in hist_train.columns
        else pd.Series(range(len(hist_train)))
    )

    try:
        pipeline = WNBAPlayerPropPipeline(quantiles=QUANTILES, n_oof_splits=3)
        pipeline.fit(X_hist, y_dict, dates_hist, pid_hist)
        logger.info("[quantile_shadow] Pipeline trained on %d historical rows", len(hist_train))
    except Exception as exc:
        logger.warning("[quantile_shadow] Pipeline training failed: %s", exc)
        return

    # ------------------------------------------------------------------
    # 2. Generate quantile predictions on today's feature_df
    # ------------------------------------------------------------------
    feat_cols_infer = [c for c in feat_cols if c in feature_df.columns]
    if len(feat_cols_infer) < 5:
        logger.warning("[quantile_shadow] Insufficient feature cols at inference — skipping predict")
        return

    X_infer = feature_df[feat_cols_infer].reset_index(drop=True)

    # Build player_games list from pmfs_long (leverages existing PMF metadata)
    player_games: list[dict] = []
    infer_rows: list[int] = []  # row indices in feature_df for each player_game

    # Use pmfs_long to understand which (player, stat) pairs are being predicted
    pmf_stats = pmfs_long["stat"].unique().tolist() if "stat" in pmfs_long.columns else stats
    pmf_stats = [s for s in pmf_stats if s in stats]  # only base stats

    # Map player_id + game_id → feature_df row index
    _feat_index_map: dict[tuple, int] = {}
    if "player_id" in feature_df.columns and "game_id" in feature_df.columns:
        for idx, row in feature_df.iterrows():
            _feat_index_map[(row["player_id"], row["game_id"])] = int(idx)

    for _, pmf_row in pmfs_long[pmfs_long["stat"].isin(pmf_stats)].iterrows():
        pid = pmf_row.get("player_id")
        gid = pmf_row.get("game_id")
        stat = pmf_row.get("stat")
        row_idx = _feat_index_map.get((pid, gid), -1)
        if row_idx < 0:
            continue
        player_games.append({
            "player_id": pid,
            "game_id": gid,
            "stat": stat,
            "line": float(pmf_row.get("pmf_mean", 0.0)),  # use legacy pmf_mean as proxy line
            "implied_over": 0.5,   # no market lines available here; placeholder
            "implied_under": 0.5,
        })
        infer_rows.append(row_idx)

    if not player_games:
        logger.warning("[quantile_shadow] No player_games built — skipping edge board generation")
        return

    # Build X for each player_game row (with column alignment)
    try:
        X_pg = X_infer.iloc[infer_rows].reset_index(drop=True)
        q_board = pipeline.predict_edge_board(X_pg, player_games)
    except Exception as exc:
        logger.warning("[quantile_shadow] predict_edge_board failed: %s", exc)
        return

    # ------------------------------------------------------------------
    # 3. Log comparison statistics
    # ------------------------------------------------------------------
    leg_over = 0
    leg_under = 0
    if "p_over" in pmfs_long.columns and "line" in pmfs_long.columns:
        leg_over = int((pmfs_long["p_over"] > 0.5).sum())
        leg_under = int((pmfs_long["p_over"] <= 0.5).sum())
    elif "pmf_mean" in pmfs_long.columns:
        leg_over = len(pmfs_long[pmfs_long["stat"].isin(pmf_stats)])

    q_over = sum(1 for r in q_board if r["direction"] == "OVER")
    q_under = sum(1 for r in q_board if r["direction"] == "UNDER")

    logger.info(
        "[quantile_shadow] === Edge Board Comparison ===\n"
        "  Legacy pipeline : OVER=%d, UNDER=%d, total_rows=%d\n"
        "  Quantile pipeline: OVER=%d, UNDER=%d, total_edges=%d (min_edge=%.2f)",
        leg_over, leg_under, len(pmfs_long),
        q_over, q_under, len(q_board), pipeline.min_edge,
    )

    top10 = q_board[:10]
    if top10:
        logger.info("[quantile_shadow] Top-10 edges (quantile pipeline):")
        for rank, rec in enumerate(top10, 1):
            logger.info(
                "  #%d  %s %s %s  line=%.1f  edge=%.3f  p_model=%.3f  mu=%.2f",
                rank,
                rec.get("player_id", "?"),
                rec.get("stat", "?"),
                rec.get("direction", "?"),
                rec.get("line", 0.0),
                rec.get("edge", 0.0),
                rec.get("prob_model", 0.0),
                rec.get("mu", 0.0),
            )

    # ------------------------------------------------------------------
    # 4. Write quantile edge board to parquet
    # ------------------------------------------------------------------
    try:
        q_board_df = pd.DataFrame(q_board) if q_board else pd.DataFrame()
        out_path = delivery_dir / "quantile_edge_board.parquet"
        q_board_df.to_parquet(out_path, index=False)
        logger.info("[quantile_shadow] Quantile edge board written to %s (%d rows)", out_path, len(q_board_df))
    except Exception as exc:
        logger.warning("[quantile_shadow] Failed to write quantile edge board: %s", exc)

    # ------------------------------------------------------------------
    # 5. If WNBA_USE_QUANTILE_MODEL=1, promote quantile board as primary output
    # ------------------------------------------------------------------
    if os.environ.get("WNBA_USE_QUANTILE_MODEL", "").strip() == "1":
        try:
            import shutil
            q_src = delivery_dir / "quantile_edge_board.parquet"
            q_dst = delivery_dir / "publishable_edges.parquet"
            if q_src.exists() and len(q_board_df) > 0:
                shutil.copy2(str(q_src), str(q_dst))
                n_over  = int((q_board_df.get("edge_over", pd.Series(dtype=float)).fillna(0) > 0).sum())
                n_under = int((q_board_df.get("edge_under", pd.Series(dtype=float)).fillna(0) > 0).sum())
                logger.info(
                    "[quantile_shadow] QUANTILE MODEL PROMOTED as primary edge board — "
                    "%d edges (%d OVER / %d UNDER) → %s",
                    len(q_board_df), n_over, n_under, q_dst,
                )
            else:
                logger.warning(
                    "[quantile_shadow] WNBA_USE_QUANTILE_MODEL=1 but quantile board is empty "
                    "or missing — legacy PMF output retained as primary"
                )
        except Exception as _promote_exc:
            logger.warning("[quantile_shadow] Failed to promote quantile board: %s", _promote_exc)


def build_features_for_prediction(player_stats: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build wide feature table for inference.

    Thin wrapper kept for backward compatibility with predict_today.py.
    """
    from wnba_props_model.features.build_features import build_player_training_table
    return build_player_training_table(player_stats, games)
