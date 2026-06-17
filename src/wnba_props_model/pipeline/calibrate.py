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

    # Only calibrate on eligible rows (excludes prior_only, low-minute games)
    if "calibration_eligible" in oof.columns:
        oof_eligible = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712
    else:
        oof_eligible = oof.copy()

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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    # #region agent log — H1 verification: confirm "turnover" in calibration set
    import json as _json, time as _time
    _stats_in_oof = sorted(set(oof_eligible["stat"]))
    _stats_to_cal = sorted(set(oof_eligible["stat"]) & set(SUPPORTED_STATS))
    try:
        with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _lf:
            _lf.write(_json.dumps({"sessionId": "94807e", "runId": "post-fix-tov", "hypothesisId": "H1", "location": "calibrate.py:fit_calibrators", "message": "stats_intersection_check", "data": {"oof_stats": _stats_in_oof, "supported_stats": sorted(SUPPORTED_STATS), "intersection": _stats_to_cal, "turnover_in_oof": "turnover" in _stats_in_oof, "turnover_in_supported": "turnover" in SUPPORTED_STATS, "turnover_in_intersection": "turnover" in _stats_to_cal}, "timestamp": int(_time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    for stat in sorted(set(oof_eligible["stat"]) & set(SUPPORTED_STATS)):
        cal = fit_role_aware_calibrator(oof_eligible, stat)
        path = out / f"pmf_cal_role_{stat}.pkl"
        cal.save(str(path))
        paths[stat] = path
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
