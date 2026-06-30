from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.constants import COMBO_STATS, DIRECT_STATS, SUPPORTED_STATS
from wnba_props_model.models.calibration import RoleAwarePMFCalibrator, fit_role_aware_calibrator
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json

logger = logging.getLogger(__name__)

# Combo stats need looser ECE/PIT gates than direct stats because their PMFs are
# derived by convolution (sum of two independent Poisson-like distributions), which
# is already close to well-calibrated when base stats are calibrated.
_COMBO_STATS_SET = set(COMBO_STATS)


def _build_oof_combo_pmfs(oof_base: pd.DataFrame) -> pd.DataFrame:
    """Convolve OOF base-stat PMFs to produce OOF combo PMFs for calibration.

    Only called internally when combo stats are not already in the OOF parquet.
    Requires actual_outcome cols for the component stats to be joinable.
    """
    from wnba_props_model.pipeline.predict import _build_combo_pmf_rows  # noqa: PLC0415

    # _build_combo_pmf_rows needs 'actual_outcome' — for OOF this is the real outcome
    combo_rows = _build_combo_pmf_rows(oof_base)
    if combo_rows.empty:
        return combo_rows

    # Carry calibration_eligible from the parent rows (True only when base is model_oof)
    pg_cal = (
        oof_base[oof_base["calibration_eligible"] == True]  # noqa: E712
        [["player_id", "game_id"]]
        .drop_duplicates()
    ) if "calibration_eligible" in oof_base.columns else None

    if pg_cal is not None:
        combo_rows["calibration_eligible"] = combo_rows.set_index(
            ["player_id", "game_id"]
        ).index.isin(pg_cal.set_index(["player_id", "game_id"]).index)
    else:
        combo_rows["calibration_eligible"] = True

    # Derive actual_outcome for combos from component actuals in oof_base
    # (e.g. pts_ast = actual_pts + actual_ast)
    _COMBO_COMPONENTS: dict[str, list[str]] = {
        "stocks":      ["stl", "blk"],
        "pts_ast":     ["pts", "ast"],
        "pts_reb":     ["pts", "reb"],
        "reb_ast":     ["reb", "ast"],
        "pts_reb_ast": ["pts", "reb", "ast"],
    }
    wide_actuals = (
        oof_base[oof_base["stat"].isin(DIRECT_STATS)]
        .groupby(["player_id", "game_id", "stat"])["actual_outcome"]
        .first()
        .unstack("stat")
        .rename(columns=lambda c: f"actual_{c}")
        .reset_index()
    )

    combo_rows = combo_rows.merge(wide_actuals, on=["player_id", "game_id"], how="left")
    for combo_stat, components in _COMBO_COMPONENTS.items():
        comp_cols = [f"actual_{c}" for c in components if f"actual_{c}" in combo_rows.columns]
        if comp_cols:
            mask = combo_rows["stat"] == combo_stat
            combo_rows.loc[mask, "actual_outcome"] = combo_rows.loc[mask, comp_cols].sum(axis=1)

    # Keep only rows where actual_outcome is available (needed for calibration)
    combo_rows = combo_rows[combo_rows["actual_outcome"].notna()].copy()

    if "outcome" not in combo_rows.columns and "actual_outcome" in combo_rows.columns:
        combo_rows["outcome"] = combo_rows["actual_outcome"]

    # Combo stats calibrate globally (role_bucket="all") — not enough per-role volume
    combo_rows["role_bucket"] = "all"

    logger.info(
        "[calibrate] Built %d OOF combo PMF rows for calibration (%s stats)",
        len(combo_rows),
        sorted(combo_rows["stat"].unique().tolist()),
    )
    return combo_rows


def fit_beta_calibrators(
    oof_pmfs_path: str | Path,
    out_dir: str | Path = "artifacts/models/calibration",
    line_col: str = "line",
) -> dict[str, Path]:
    """Fit Beta calibrators for P(over) on top of per-stat isotonic calibrators.

    Saved alongside the isotonic calibrators as beta_cal_{stat}.pkl.
    The beta calibrator operates on P(over) outputs and is the preferred
    method when n_oof >= 50 (Manokhin & Grønhaug, 2026).
    """
    import joblib  # noqa: PLC0415
    from wnba_props_model.evaluation.beta_calibration import fit_best_calibrator  # noqa: PLC0415

    _BETA_MIN_CAL_MINUTES = 10  # Match fit_calibrators threshold: only prop-eligible games
    oof = pd.read_parquet(oof_pmfs_path).copy()
    if "outcome" not in oof.columns and "actual_outcome" in oof.columns:
        oof["outcome"] = oof["actual_outcome"]
    if "pmf" not in oof.columns and "pmf_json" in oof.columns:
        oof["pmf"] = oof["pmf_json"].map(json_to_pmf)
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712
    # Apply same DNP + low-minutes filter as fit_calibrators
    if "did_play" in oof.columns:
        oof = oof[oof["did_play"] == True].copy()  # noqa: E712
    if "actual_minutes" in oof.columns:
        oof = oof[oof["actual_minutes"].fillna(0) >= _BETA_MIN_CAL_MINUTES].copy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for stat in sorted(set(oof["stat"].dropna()) & set(SUPPORTED_STATS)):
        stat_rows = oof[oof["stat"] == stat]
        if line_col not in stat_rows.columns or stat_rows[line_col].isna().all():
            continue

        rows = stat_rows.dropna(subset=[line_col, "outcome"])
        if len(rows) < 20:
            continue

        # P(over) from raw PMF at each line
        def _p_over(row: pd.Series) -> float:
            pmf = row.get("pmf")
            if pmf is None:
                return float("nan")
            line = float(row[line_col])
            if isinstance(pmf, dict):
                return sum(p for k, p in pmf.items() if k > line)
            arr = np.asarray(pmf, dtype=float)
            k_int = int(line)
            return float(arr[k_int + 1:].sum()) if k_int + 1 < len(arr) else 0.0

        scores = rows.apply(_p_over, axis=1).values
        # Exclude integer push rows (actual == line) — they are neither over nor under.
        # Including them as "under" biases calibration toward over-confidence on the under.
        push_mask = (rows["outcome"].values != rows[line_col].values)
        labels = (rows["outcome"].values[push_mask] > rows[line_col].values[push_mask]).astype(float)
        scores = scores[push_mask]

        mask = ~np.isnan(scores)
        if mask.sum() < 20:
            continue

        cal = fit_best_calibrator(scores[mask], labels[mask])
        path = out / f"beta_cal_{stat}.pkl"
        joblib.dump(cal, path)
        paths[stat] = path
        logger.info("[calibrate] Beta calibrator saved for %s → %s", stat, path)

    return paths


def apply_beta_calibrators(
    p_over_series: "pd.Series",
    stat: str,
    cal_dir: "str | Path" = "artifacts/models/calibration",
) -> "pd.Series":
    """Apply a fitted Beta calibrator to a series of P(over) scalars.

    Returns recalibrated P(over) values in [0, 1]. Falls back to the
    original series if no Beta calibrator is found for the stat.
    """
    import joblib as _jl  # noqa: PLC0415

    cal_path = Path(cal_dir) / f"beta_cal_{stat}.pkl"
    if not cal_path.exists():
        return p_over_series
    try:
        cal = _jl.load(cal_path)
        vals = p_over_series.fillna(0.5).clip(1e-6, 1 - 1e-6).values.reshape(-1, 1)
        cal_probs = cal.predict(vals)
        return pd.Series(cal_probs, index=p_over_series.index)
    except Exception as exc:
        logger.warning("[calibrate] apply_beta_calibrators(%s) failed: %s", stat, exc)
        return p_over_series


def _apply_mean_bias_correction(
    pmf_arr: np.ndarray,
    alpha: float,
    cap: int | None = None,
) -> np.ndarray:
    """Scale a discrete PMF so its mean shifts by factor alpha.

    Uses NegBinom moment-matching: estimates r from current mean and variance,
    rebuilds the PMF at the target mean with the same r. Falls back to the
    original PMF if estimation fails.
    """
    from wnba_props_model.models.pmf_utils import negbinom_pmf_batch  # noqa: PLC0415

    arr = normalize_pmf(pmf_arr)
    if abs(alpha - 1.0) < 0.005:
        return arr  # no correction needed
    n = len(arr)
    k = np.arange(n, dtype=float)
    mu = float(np.dot(k, arr))
    if mu < 0.05:
        return arr
    var = float(np.dot(k ** 2, arr)) - mu ** 2
    # NegBinom: var = mu + mu²/r  →  r = mu²/(var - mu)
    excess_var = max(var - mu, 1e-4)
    r_est = float(np.clip(mu ** 2 / excess_var, 0.3, 20.0))
    target_mu = float(np.clip(mu * alpha, 0.05, None))
    support = (cap or n) - 1
    try:
        new_pmf = negbinom_pmf_batch(np.array([target_mu]), r_est, support)[0]
        return normalize_pmf(new_pmf)
    except Exception:
        return arr


def fit_calibrators(
    oof_pmfs_path: str | Path,
    out_dir: str | Path = "artifacts/models/calibration",
) -> dict[str, Path]:
    """Fit per-stat role-aware isotonic calibrators from OOF PMFs.

    For direct stats (pts, reb, ast, fg3m, stl, blk, turnover): fit with
    per-role calibrators if role_bucket is present in the OOF parquet.

    For combo stats (stocks, pts_ast, pts_reb, reb_ast, pts_reb_ast): generate
    OOF combo PMFs on the fly via convolution if not already present, then fit
    global-only (role_bucket=all) calibrators.
    """
    oof = pd.read_parquet(oof_pmfs_path).copy()

    # Normalize column names: OOF parquet uses actual_outcome; calibrator expects outcome
    if "outcome" not in oof.columns and "actual_outcome" in oof.columns:
        oof["outcome"] = oof["actual_outcome"]

    # role_bucket must exist for role-aware calibration.
    # With the t1-role-bucket-oof fix, generate_fold_pmfs() now writes it.
    # Fallback to "all" only for legacy OOF files built before that fix.
    if "role_bucket" not in oof.columns:
        oof["role_bucket"] = "all"
        logger.warning(
            "[calibrate] OOF parquet has no role_bucket column — using global-only "
            "calibration. Re-run build_oof_pmfs.py to get per-role calibrators."
        )

    # Build numpy PMF arrays from JSON
    if "pmf" not in oof.columns and "pmf_json" in oof.columns:
        oof["pmf"] = oof["pmf_json"].map(json_to_pmf)

    # Only calibrate on eligible rows (excludes prior_only rows).
    # CRITICAL: also exclude DNP games (did_play=False) and garbage-time
    # appearances (actual_minutes < 10).  Including DNPs contaminates PIT
    # distributions: a player predicted at 14 pts who DNPs scores 0, giving
    # PIT ≈ 0.05.  The isotonic calibrator interprets this as systematic
    # over-prediction and learns to compress every distribution by ~2×,
    # causing severe under-prediction vs the market in live delivery.
    _MIN_CAL_MINUTES = 10  # prop-eligibility floor: only games where player played >=10 min
                           # (filters bench/fringe games that contaminate calibration training)
    # #region agent log
    import time as _time_dbg
    _DBG_LOG = "/Users/josephshackelford/worldcup2026-model/.cursor/debug-3f8dcc.log"
    def _dbg(msg: str, data: dict, hyp: str) -> None:
        import json as _json_dbg
        try:
            with open(_DBG_LOG, "a") as _f:
                _f.write(_json_dbg.dumps({"sessionId":"3f8dcc","timestamp":int(_time_dbg.time()*1000),"location":"calibrate.py:fit_calibrators","message":msg,"data":data,"hypothesisId":hyp}) + "\n")
        except Exception:
            pass
    _dbg("FIT_CALIBRATORS_ENTRY", {"min_cal_minutes": _MIN_CAL_MINUTES, "total_oof_rows": len(oof)}, "H-B")
    # #endregion agent log
    if "calibration_eligible" in oof.columns:
        oof_eligible = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712
    else:
        oof_eligible = oof.copy()

    # Apply DNP + low-minutes filter (only for rows that have these columns)
    _pre_dnp_n = len(oof_eligible)
    _has_did_play = "did_play" in oof_eligible.columns
    _has_actual_minutes = "actual_minutes" in oof_eligible.columns
    if _has_did_play:
        oof_eligible = oof_eligible[oof_eligible["did_play"] == True].copy()  # noqa: E712
    if _has_actual_minutes:
        oof_eligible = oof_eligible[
            oof_eligible["actual_minutes"].fillna(0) >= _MIN_CAL_MINUTES
        ].copy()
    _post_dnp_n = len(oof_eligible)
    logger.info(
        "[calibrate] DNP/low-min filter: %d → %d rows "
        "(removed %d DNP/low-minute games from calibration training set)",
        _pre_dnp_n, _post_dnp_n, _pre_dnp_n - _post_dnp_n,
    )
    print(f"[calibrate] Total eligible rows before DNP filter: {_pre_dnp_n:,}")
    print(f"[calibrate] has_did_play={_has_did_play} has_actual_minutes={_has_actual_minutes}")
    print(f"[calibrate] After DNP/low-min filter: {_post_dnp_n:,} rows (removed {_pre_dnp_n - _post_dnp_n:,})")
    # Prop-eligible calibration filter: exclude bench/fringe players the model
    # projects near-zero output for. Calibrators must match the distribution they
    # are applied to in production (prop-eligible starters + core only).
    # Including bench fringe rows creates a "false balance": aggregate mean_pit ≈ 0.49
    # masks severe miscalibration for starters (mean_pit 0.58–0.74), causing isotonic
    # calibrators to compress starter means by ~46% (PTS: 8.96 → 4.86, actual 9.32).
    _PROP_ELIGIBLE_PMF_MIN: dict[str, float] = {
        "pts":      4.0,
        "reb":      1.5,
        "ast":      0.8,
        "fg3m":     0.25,
        "stl":      0.25,
        "blk":      0.15,
        "turnover": 0.5,
        "stocks":   0.4,
        "pts_ast":  5.0,
        "pts_reb":  5.5,
        "reb_ast":  2.3,
        "pts_reb_ast": 6.0,
    }
    if "pmf_mean" in oof_eligible.columns and "stat" in oof_eligible.columns:
        _pre_prop_n = len(oof_eligible)
        _prop_masks = []
        for _sname, _pfloor in _PROP_ELIGIBLE_PMF_MIN.items():
            _prop_masks.append(
                (oof_eligible["stat"] == _sname) & (oof_eligible["pmf_mean"] >= _pfloor)
            )
        _unknown_stats = ~oof_eligible["stat"].isin(_PROP_ELIGIBLE_PMF_MIN.keys())
        _combined = _unknown_stats.copy()
        for _m in _prop_masks:
            _combined = _combined | _m
        oof_eligible = oof_eligible[_combined].copy()
        _post_prop_n = len(oof_eligible)
        print(f"[calibrate] Prop-eligible filter: {_pre_prop_n:,} → {_post_prop_n:,} rows (removed {_pre_prop_n - _post_prop_n:,} bench/fringe rows)")
        # #region agent log
        import time as _tprop, json as _jprop
        _DBG_PROP = "/Users/josephshackelford/worldcup2026-model/.cursor/debug-3f8dcc.log"
        _prop_by_stat = {}
        for _s2 in ["pts","reb","ast","fg3m","stl","blk"]:
            _ss2 = oof_eligible[oof_eligible["stat"]==_s2].dropna(subset=["pmf_mean","actual_outcome"]) if "stat" in oof_eligible.columns else oof_eligible.iloc[0:0]
            if len(_ss2) > 0:
                _prop_by_stat[_s2] = {"n": len(_ss2), "pmf_mean": round(float(_ss2["pmf_mean"].mean()),3), "actual": round(float(_ss2["actual_outcome"].mean()),3)}
        try:
            with open(_DBG_PROP, "a") as _fdbg:
                _fdbg.write(_jprop.dumps({"sessionId":"3f8dcc","timestamp":int(_tprop.time()*1000),"location":"calibrate.py:prop_eligible_filter","message":"POST_PROP_FILTER","data":{"pre_n":_pre_prop_n,"post_n":_post_prop_n,"by_stat":_prop_by_stat},"hypothesisId":"H-C"}) + "\n")
        except Exception:
            pass
        # #endregion agent log
    # #region agent log
    _bias_by_stat = {}
    _pmf_floor_counts = {}
    if "stat" in oof_eligible.columns and "pmf_mean" in oof_eligible.columns and "actual_outcome" in oof_eligible.columns:
        for _s in ["pts","reb","ast","fg3m","stl","blk"]:
            _ss = oof_eligible[oof_eligible["stat"]==_s].dropna(subset=["pmf_mean","actual_outcome"])
            if len(_ss) > 0:
                _bias_by_stat[_s] = {"n": len(_ss), "pmf_mean": round(float(_ss["pmf_mean"].mean()),3), "actual": round(float(_ss["actual_outcome"].mean()),3), "bias_pct": round((float(_ss["pmf_mean"].mean())-float(_ss["actual_outcome"].mean()))/max(float(_ss["actual_outcome"].mean()),0.01)*100,1)}
                _pmf_floor_counts[_s] = {"n_above_4": int((_ss["pmf_mean"]>=4.0).sum()) if _s=="pts" else int((_ss["pmf_mean"]>=1.0).sum())}
    _dbg("POST_DNP_FILTER_BIAS", {"pre_n": _pre_dnp_n, "post_n": _post_dnp_n, "min_cal_minutes": _MIN_CAL_MINUTES, "by_stat": _bias_by_stat}, "H-A")
    _dbg("PMF_FLOOR_PREVIEW", {"counts_above_prop_floor": _pmf_floor_counts}, "H-C")
    # #endregion agent log
    if _has_actual_minutes and _post_dnp_n > 0:
        for _stat in ["pts", "reb", "ast"]:
            _sub = oof_eligible[oof_eligible["stat"] == _stat] if "stat" in oof_eligible.columns else oof_eligible
            if len(_sub) > 0 and "actual_outcome" in _sub.columns and "pmf_mean" in _sub.columns:
                _bias = _sub["pmf_mean"].mean() - _sub["actual_outcome"].mean()
                print(f"[calibrate] Post-DNP-filter stat={_stat}: model_mean={_sub['pmf_mean'].mean():.2f} actual_mean={_sub['actual_outcome'].mean():.2f} bias={_bias:+.2f} n={len(_sub):,}")

    # Append combo OOF PMFs if not already present
    existing_stats = set(oof_eligible["stat"].unique())
    missing_combos = _COMBO_STATS_SET - existing_stats
    if missing_combos:
        try:
            combo_oof = _build_oof_combo_pmfs(oof_eligible)
            if not combo_oof.empty:
                if "pmf" not in combo_oof.columns and "pmf_json" in combo_oof.columns:
                    combo_oof["pmf"] = combo_oof["pmf_json"].map(json_to_pmf)
                oof_eligible = pd.concat([oof_eligible, combo_oof], ignore_index=True)
        except Exception as exc:
            logger.warning("[calibrate] Combo OOF generation failed: %s — skipping combo calibration", exc)

    # #region agent log
    print(
        f"[calibrate] OOF eligible stats: {sorted(oof_eligible['stat'].unique().tolist())} "
        f"roles: {sorted(oof_eligible['role_bucket'].unique().tolist())} "
        f"rows: {len(oof_eligible):,}"
    )
    # #endregion agent log

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Per-stat multiplicative mean bias correction.
    # Isotonic regression corrects PMF *shape* (PIT uniformity) but is unreliable
    # for correcting global *level* bias. We separate these: first shift the mean
    # by a multiplicative factor (ratio of actual to model mean on prop-eligible OOF),
    # then let isotonic correct residual shape. This prevents the calibrator from
    # over-correcting the tails to compensate for a mean shift.
    _bias_corrections: dict[str, float] = {}
    for _bc_stat in sorted(set(oof_eligible["stat"].dropna()) & set(SUPPORTED_STATS)):
        _bc_sub = oof_eligible[oof_eligible["stat"] == _bc_stat].dropna(
            subset=["pmf_mean", "actual_outcome"]
        )
        if len(_bc_sub) < 200:
            logger.warning("[calibrate] Too few rows (%d) for bias correction stat=%s; using 1.0", len(_bc_sub), _bc_stat)
            _bias_corrections[_bc_stat] = 1.0
            continue
        _model_mean_bc = float(_bc_sub["pmf_mean"].mean())
        _actual_mean_bc = float(_bc_sub["actual_outcome"].mean())
        if _model_mean_bc > 0.01:
            _raw_ratio = _actual_mean_bc / _model_mean_bc
            # Cap correction to ±40% to prevent runaway corrections on thin stats
            _bias_corrections[_bc_stat] = float(np.clip(_raw_ratio, 0.60, 1.40))
        else:
            _bias_corrections[_bc_stat] = 1.0
    (out / "bias_corrections.json").write_text(json.dumps(_bias_corrections, indent=2))
    print(f"[calibrate] Bias corrections saved: { {k: round(v,3) for k,v in _bias_corrections.items()} }")
    # #region agent log
    try:
        import time as _tprop_bc, json as _jprop_bc
        _DBG_PROP_BC = "/Users/josephshackelford/worldcup2026-model/.cursor/debug-3f8dcc.log"
        with open(_DBG_PROP_BC, "a") as _fdbg2:
            _fdbg2.write(_jprop_bc.dumps({"sessionId":"3f8dcc","timestamp":int(_tprop_bc.time()*1000),"location":"calibrate.py:bias_corrections","message":"BIAS_CORRECTIONS_COMPUTED","data":{"corrections":{k:round(v,4) for k,v in _bias_corrections.items()}},"hypothesisId":"H-D"}) + "\n")
    except Exception:
        pass
    # #endregion agent log

    # Apply bias corrections to OOF PMFs BEFORE fitting isotonic calibrators.
    # This aligns the calibrator training distribution with inference distribution
    # (in apply_calibrators, bias correction is applied before isotonic calibration).
    # Without this, there is a training/inference mismatch: calibrators trained on
    # original PMFs would see bias-corrected PMFs at inference, producing wrong mappings.
    if any(abs(v - 1.0) > 0.005 for v in _bias_corrections.values()):
        _bc_pmf_jsons = []
        for _bcr_idx, _bcr_row in oof_eligible.iterrows():
            _bc_alpha = _bias_corrections.get(str(_bcr_row.get("stat", "")), 1.0)
            if abs(_bc_alpha - 1.0) > 0.005 and pd.notna(_bcr_row.get("pmf_json")):
                _bc_pmf = _apply_mean_bias_correction(json_to_pmf(_bcr_row["pmf_json"]), _bc_alpha)
                _bc_pmf_jsons.append(pmf_to_json(_bc_pmf))
            else:
                _bc_pmf_jsons.append(_bcr_row.get("pmf_json"))
        oof_eligible = oof_eligible.copy()
        oof_eligible["pmf_json"] = _bc_pmf_jsons
        oof_eligible["pmf"] = oof_eligible["pmf_json"].map(json_to_pmf)
        print(f"[calibrate] Applied bias corrections to {len(oof_eligible):,} OOF PMFs for calibrator training alignment")

    paths = {}
    for stat in sorted(set(oof_eligible["stat"]) & set(SUPPORTED_STATS)):
        cal = fit_role_aware_calibrator(oof_eligible, stat)
        path = out / f"pmf_cal_role_{stat}.pkl"
        cal.save(str(path))
        paths[stat] = path
        # #region agent log
        print(
            f"[calibrate] Fitted role-aware calibrator stat={stat} "
            f"bucket_count={cal.bucket_counts} "
            f"quality_tiers={list(cal.quality_tier_calibrators.keys())} "
            f"thresholds={cal.quality_tier_thresholds}"
        )
        # #endregion agent log

    # Write calibration_metadata.json for freshness guard in predict_today.py
    import datetime as _dt
    _meta = {
        "fitted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_rows": int(_post_dnp_n),
        "stats": sorted(paths.keys()),
    }
    (out / "calibration_metadata.json").write_text(json.dumps(_meta, indent=2))
    print(f"[calibrate] Wrote calibration_metadata.json: {_meta['fitted_at']} n_rows={_meta['n_rows']}")

    # Also fit Beta calibrators for P(over) (Item 5A)
    try:
        beta_paths = fit_beta_calibrators(oof_pmfs_path, out_dir)
        paths.update({f"beta_{k}": v for k, v in beta_paths.items()})
    except Exception as exc:
        logger.warning("[calibrate] Beta calibrator fitting failed: %s", exc)

    # Fit Conformal Prediction Intervals (Item 5D)
    # Uses split conformal on OOF residuals: |actual - model_mean|
    try:
        import pickle  # noqa: PLC0415
        from wnba_props_model.evaluation.conformal import ConformalPropPredictor  # noqa: PLC0415

        conformal = ConformalPropPredictor(alpha=0.10)
        def _safe_pmf_mean(raw_pmf, domain_max: int = 80) -> float:
            """Compute mean of a PMF that may be ndarray, list, JSON string, or dict."""
            if isinstance(raw_pmf, (str, dict)):
                arr = json_to_pmf(raw_pmf, domain_max=domain_max)
            elif hasattr(raw_pmf, "__len__"):
                arr = np.asarray(raw_pmf, dtype=float)
            else:
                return 0.0
            if arr.sum() == 0:
                return 0.0
            arr = arr / arr.sum()
            return float((np.arange(len(arr)) * arr).sum())

        for (stat, role), grp in oof_eligible.groupby(["stat", "role_bucket"]):
            if "pmf" not in grp.columns and "pmf_json" not in grp.columns:
                continue
            if "actual_outcome" not in grp.columns:
                continue
            pmf_col = grp["pmf"] if "pmf" in grp.columns else grp["pmf_json"].map(json_to_pmf)
            preds = np.array([_safe_pmf_mean(p) for p in pmf_col])
            actuals = grp["actual_outcome"].to_numpy(dtype=float)
            conformal.fit(preds, actuals, stat=str(stat), role=str(role))

        conformal_path = out / "conformal_predictor.pkl"
        with open(conformal_path, "wb") as f:
            pickle.dump(conformal, f, protocol=5)
        paths["conformal"] = conformal_path
        logger.info("[calibrate] Conformal predictor fitted for %d (stat, role) buckets, saved to %s",
                    len(conformal.quantiles), conformal_path)
    except Exception as exc:
        logger.warning("[calibrate] Conformal predictor fitting failed (non-fatal): %s", exc)

    return paths


def apply_calibrators(
    pmfs_df: pd.DataFrame,
    cal_dir: str | Path = "artifacts/models/calibration",
) -> pd.DataFrame:
    """Apply per-stat role-aware isotonic calibrators to a PMF DataFrame.

    Input columns required: stat, role_bucket, pmf_json.
    Adds: pmf_json (overwritten with calibrated values), is_calibrated=True,
    cal_source='role_aware_isotonic'.

    Stats without a fitted calibrator are passed through unchanged with
    is_calibrated=False and cal_source='no_calibrator'.
    """
    cal_dir = Path(cal_dir)
    # Load per-stat multiplicative mean bias corrections (computed during fit_calibrators)
    _bias_path_ac = cal_dir / "bias_corrections.json"
    _bias_corrections_ac: dict[str, float] = {}
    if _bias_path_ac.exists():
        try:
            _bias_corrections_ac = json.loads(_bias_path_ac.read_text())
        except Exception as exc_bc:
            logger.warning("[calibrate] Could not load bias_corrections.json: %s", exc_bc)
    out = pmfs_df.copy()
    calibrators: dict[str, RoleAwarePMFCalibrator] = {}
    # #region agent log
    import time as _time_dbg2, json as _json_dbg2
    _DBG_LOG2 = "/Users/josephshackelford/worldcup2026-model/.cursor/debug-3f8dcc.log"
    _bias_corrections_exist = (cal_dir / "bias_corrections.json").exists()
    try:
        with open(_DBG_LOG2, "a") as _f2:
            _f2.write(_json_dbg2.dumps({"sessionId":"3f8dcc","timestamp":int(_time_dbg2.time()*1000),"location":"calibrate.py:apply_calibrators","message":"APPLY_CALIBRATORS_ENTRY","data":{"cal_dir":str(cal_dir),"bias_corrections_json_exists":_bias_corrections_exist,"n_rows":len(pmfs_df)},"hypothesisId":"H-D"}) + "\n")
    except Exception:
        pass
    # #endregion agent log

    for stat in out["stat"].unique():
        cal_path = cal_dir / f"pmf_cal_role_{stat}.pkl"
        if cal_path.exists():
            calibrators[stat] = RoleAwarePMFCalibrator.load(str(cal_path))

    new_pmf_jsons = []
    is_calibrated_flags = []
    cal_sources = []

    for _, row in out.iterrows():
        stat = row["stat"]
        role = str(row.get("role_bucket", "unknown"))
        player_id = row.get("player_id", None)
        raw_pmf = json_to_pmf(row["pmf_json"])

        # Apply multiplicative mean bias correction before isotonic shape calibration
        _alpha_ac = _bias_corrections_ac.get(stat, 1.0)
        if abs(_alpha_ac - 1.0) > 0.005:
            raw_pmf = _apply_mean_bias_correction(raw_pmf, _alpha_ac)

        if stat in calibrators:
            try:
                cal_pmf = calibrators[stat].apply(
                    normalize_pmf(raw_pmf), role, player_id=player_id
                )
                new_pmf_jsons.append(pmf_to_json(cal_pmf))
                is_calibrated_flags.append(True)
                cal_sources.append("role_aware_isotonic")
            except Exception:
                new_pmf_jsons.append(row["pmf_json"])
                is_calibrated_flags.append(False)
                cal_sources.append("calibration_error")
        else:
            new_pmf_jsons.append(row["pmf_json"])
            is_calibrated_flags.append(False)
            cal_sources.append("no_calibrator")

    out["pmf_json"] = new_pmf_jsons
    out["is_calibrated"] = is_calibrated_flags
    out["cal_source"] = cal_sources

    # Recompute summary stats from calibrated PMFs
    def _pmf_stats(pmf_json: str) -> dict:
        pmf = normalize_pmf(json_to_pmf(pmf_json))
        ks = np.arange(len(pmf))
        mean = float(np.dot(ks, pmf))
        return {
            "mean": mean,
            "median": int(np.searchsorted(np.cumsum(pmf), 0.5)),
            "mode": int(np.argmax(pmf)),
            "p0": float(pmf[0]),
        }

    stats_rows = [_pmf_stats(j) for j in out["pmf_json"]]
    for key in ("mean", "median", "mode", "p0"):
        out[key] = [r[key] for r in stats_rows]

    return out


# ---------------------------------------------------------------------------
# Part I: Live calibrators — fitted on in-game (live) PMF data
# ---------------------------------------------------------------------------

def fit_live_calibrators(
    live_cal_path: str | Path = "data/processed/live_calibration_data.parquet",
    cal_dir: str | Path = "artifacts/models/calibration",
    min_samples: int = 30,
) -> dict[str, str]:
    """Part I: Fit separate role-aware calibrators from live (in-game) prediction data.

    Live predictions conditioned on in-game box score have different calibration
    characteristics than pre-game predictions. They need their own calibrators.

    Input parquet schema:
      player_id, game_id, stat, role_bucket, quarter, pmf_json, actual_outcome
      (produced by scripts/build_live_calibration_data.py)

    Output:
      artifacts/models/calibration/live_pmf_cal_role_{stat}.pkl
    """
    live_path = Path(live_cal_path)
    cal_dir_p = Path(cal_dir)
    cal_dir_p.mkdir(parents=True, exist_ok=True)

    if not live_path.exists():
        logger.warning(
            "[live_calibrators] No live calibration data at %s — "
            "run scripts/build_live_calibration_data.py first.",
            live_path,
        )
        return {}

    live_df = pd.read_parquet(live_path)
    required = {"stat", "role_bucket", "pmf_json", "actual_outcome"}
    if not required.issubset(live_df.columns):
        logger.warning(
            "[live_calibrators] Missing columns in %s: need %s, got %s",
            live_path, required, set(live_df.columns),
        )
        return {}

    paths: dict[str, str] = {}
    for stat in live_df["stat"].unique():
        stat_df = live_df[live_df["stat"] == stat].dropna(subset=["pmf_json", "actual_outcome"])
        if len(stat_df) < min_samples:
            logger.info("[live_calibrators] %s: too few rows (%d) — skip", stat, len(stat_df))
            continue
        try:
            cal = fit_role_aware_calibrator(stat_df, stat=stat)
            out_path = cal_dir_p / f"live_pmf_cal_role_{stat}.pkl"
            cal.save(str(out_path))
            paths[stat] = str(out_path)
            logger.info("[live_calibrators] %s: fitted calibrator → %s", stat, out_path)
        except Exception as exc:
            logger.warning("[live_calibrators] %s: calibrator fitting failed: %s", stat, exc)

    return paths
