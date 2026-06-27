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

    _BETA_MIN_CAL_MINUTES = 5
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
    _MIN_CAL_MINUTES = 5   # exclude pure garbage-time (< 5 min) while keeping fringe/bench games
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
    # #region agent log
    print(f"[calibrate] Total eligible rows before DNP filter: {_pre_dnp_n:,}")
    print(f"[calibrate] has_did_play={_has_did_play} has_actual_minutes={_has_actual_minutes}")
    print(f"[calibrate] After DNP/low-min filter: {_post_dnp_n:,} rows (removed {_pre_dnp_n - _post_dnp_n:,})")
    if _has_actual_minutes and _post_dnp_n > 0:
        for _stat in ["pts", "reb", "ast"]:
            _sub = oof_eligible[oof_eligible["stat"] == _stat] if "stat" in oof_eligible.columns else oof_eligible
            if len(_sub) > 0 and "actual_outcome" in _sub.columns and "pmf_mean" in _sub.columns:
                _bias = _sub["pmf_mean"].mean() - _sub["actual_outcome"].mean()
                print(f"[calibrate] Post-DNP-filter stat={_stat}: model_mean={_sub['pmf_mean'].mean():.2f} actual_mean={_sub['actual_outcome'].mean():.2f} bias={_bias:+.2f} n={len(_sub):,}")
    import json as _jh, time as _th
    try:
        with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
            _f.write(_jh.dumps({"sessionId": "94807e", "hypothesisId": "H2", "location": "calibrate.py:fit_calibrators", "message": "dnp_filter_result", "data": {"pre_dnp_n": _pre_dnp_n, "post_dnp_n": _post_dnp_n, "removed": _pre_dnp_n - _post_dnp_n, "has_did_play": _has_did_play, "has_actual_minutes": _has_actual_minutes}, "timestamp": int(_th.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion agent log

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
    out = pmfs_df.copy()
    calibrators: dict[str, RoleAwarePMFCalibrator] = {}

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
        raw_pmf = json_to_pmf(row["pmf_json"])

        if stat in calibrators:
            try:
                cal_pmf = calibrators[stat].apply(normalize_pmf(raw_pmf), role)
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
