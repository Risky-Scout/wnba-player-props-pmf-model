from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

    # Preserve role_bucket from the parent player's base rows.
    _rb_map = (
        oof_base[["player_id", "game_id", "role_bucket"]]
        .drop_duplicates(subset=["player_id", "game_id"])
        .set_index(["player_id", "game_id"])["role_bucket"]
    )
    combo_rows["role_bucket"] = (
        combo_rows.set_index(["player_id", "game_id"])
        .index.map(_rb_map)
        .fillna("all")
        .values
    )
    logger.info(
        "[calibrate] Combo stats role distribution: %s",
        combo_rows["role_bucket"].value_counts().to_dict(),
    )

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


def _apply_protected_mean_bias_correction(
    model_means: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Apply additive mean bias correction proportionally by deviation from median.

    High performers (large deviation from median) get smaller corrections so that
    workhorse underprediction is not caused by the global correction pushing their
    means down. Bench players (small predictions, near median) get the full correction.

    Parameters
    ----------
    model_means : Per-player predicted stat means (batch).
    bias        : OOF-measured additive bias: actual_mean - model_mean.
                  Positive bias → model underpredicts → shift means up.
    """
    if abs(bias) < 1e-6:
        return model_means
    deviation = np.abs(model_means - np.median(model_means))
    max_dev = max(float(deviation.max()), 1.0)
    weights = 1.0 - 0.7 * np.minimum(deviation / max_dev, 1.0)
    return model_means + bias * weights


def _apply_mean_bias_correction(
    pmf_arr: np.ndarray,
    alpha: float,
    cap: int | None = None,
    variance_compress_factor: float = 1.0,
) -> np.ndarray:
    """Scale a discrete PMF so its mean shifts by factor alpha, optionally compressing variance.

    Uses NegBinom moment-matching: estimates r from current mean and variance,
    rebuilds the PMF at the target mean. When variance_compress_factor > 1.0,
    the dispersion parameter r is multiplied by that factor (higher r = less
    overdispersion = narrower PMF), correcting over-wide distributions confirmed
    by runtime OOF evidence (fg3m ratio=1.54, ast ratio=1.43).
    Falls back to the original PMF if estimation fails.
    """
    from wnba_props_model.models.pmf_utils import negbinom_pmf_batch  # noqa: PLC0415

    arr = normalize_pmf(pmf_arr)
    needs_alpha = abs(alpha - 1.0) >= 0.005
    needs_compress = variance_compress_factor > 1.05
    if not needs_alpha and not needs_compress:
        return arr
    n = len(arr)
    k = np.arange(n, dtype=float)
    mu = float(np.dot(k, arr))
    if mu < 0.05:
        return arr
    var = float(np.dot(k ** 2, arr)) - mu ** 2
    # NegBinom: var = mu + mu²/r  →  r = mu²/(var - mu)
    excess_var = max(var - mu, 1e-4)
    r_est = float(np.clip(mu ** 2 / excess_var, 0.3, 20.0))
    # Variance compression: multiply r by the compress factor.
    # Runtime evidence: fg3m model_var/actual_var=1.54, ast=1.43.
    # Higher r tightens the NegBinom distribution toward its mean.
    if needs_compress:
        r_est = float(np.clip(r_est * variance_compress_factor, 0.5, 60.0))
    target_mu = float(np.clip(mu * alpha, 0.05, None))
    support = (cap or n) - 1
    try:
        new_pmf = negbinom_pmf_batch(np.array([target_mu]), r_est, support)[0]
        return normalize_pmf(new_pmf)
    except Exception:
        return arr


def fit_pnz_calibrators(
    oof: pd.DataFrame,
    out: Path,
) -> dict[str, Path]:
    """Fit isotonic calibrators for P(nonzero) on hurdle-model stats (stl, blk).

    Uses OOF p_nz predictions vs actual nonzero indicators.
    A well-calibrated pi model produces uniform PIT, but the structural zero
    probability p_nz = 1-pi can still be biased in the mean even with uniform PIT.
    Isotonic regression corrects this monotonically without overfitting.
    """
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    import joblib  # noqa: PLC0415

    pnz_paths: dict[str, Path] = {}
    for stat in ("stl", "blk"):
        sub = oof[(oof["stat"] == stat) & (oof["p_nz"].notna())].copy()
        if len(sub) < 100:
            logger.warning(
                "[calibrate] Insufficient OOF rows for p_nz calibration of %s (%d rows)",
                stat, len(sub),
            )
            continue
        y_actual = (sub["actual_outcome"] > 0).astype(float).values
        y_pred   = sub["p_nz"].clip(1e-6, 1 - 1e-6).values

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_pred, y_actual)

        y_cal = ir.predict(y_pred)
        cal_mean = float(y_cal.mean())
        emp_rate = float(y_actual.mean())
        logger.info(
            "[calibrate] p_nz calibrator %s: empirical_nonzero=%.3f cal_mean=%.3f (delta=%.4f) n=%d",
            stat, emp_rate, cal_mean, cal_mean - emp_rate, len(sub),
        )

        path = out / f"pnz_cal_{stat}.pkl"
        joblib.dump(ir, path)
        pnz_paths[stat] = path
        print(f"[calibrate] Saved p_nz calibrator: {path}")

    return pnz_paths


def fit_pdnp_calibrator(
    oof: pd.DataFrame,
    out: Path,
) -> Path | None:
    """Fit isotonic calibrator for P(DNP) — probability a player does not play.

    Calibrates the model's p_dnp predictions against actual did_play outcomes
    from OOF data. A well-calibrated P(DNP) prevents overconfident projections
    for players with uncertain availability.
    """
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    import joblib  # noqa: PLC0415

    if "p_dnp" not in oof.columns or "did_play" not in oof.columns:
        logger.warning("[calibrate] p_dnp or did_play column missing — skipping P(DNP) calibrator")
        return None

    sub = oof[oof["p_dnp"].notna() & oof["did_play"].notna()].copy()
    if len(sub) < 50:
        logger.warning(
            "[calibrate] Insufficient OOF rows for P(DNP) calibration (%d rows, need 50)",
            len(sub),
        )
        return None

    # Target: 1 = DNP (did NOT play), 0 = played
    y_actual = (sub["did_play"] == False).astype(float).values  # noqa: E712
    y_pred = sub["p_dnp"].clip(1e-6, 1 - 1e-6).values

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_pred, y_actual)

    y_cal = ir.predict(y_pred)
    logger.info(
        "[calibrate] P(DNP) calibrator: empirical_dnp=%.3f cal_mean=%.3f (delta=%.4f) n=%d",
        float(y_actual.mean()), float(y_cal.mean()),
        float(y_cal.mean()) - float(y_actual.mean()), len(sub),
    )

    path = out / "pdnp_cal.pkl"
    joblib.dump(ir, path)
    print(f"[calibrate] Saved P(DNP) calibrator: {path}")
    return path


def fit_calibrators(
    oof_pmfs_path: str | Path,
    out_dir: str | Path = "artifacts/models/calibration",
    props_parquet_path: str | Path | None = None,
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
        "pts":         5.0,   # raised from 4.0 — matches sportsbook offering floor
        "reb":         2.0,   # raised from 1.5
        "ast":         1.0,   # raised from 0.8
        "fg3m":        0.30,  # raised from 0.25
        "stl":         0.30,  # raised from 0.25
        "blk":         0.18,  # raised from 0.15
        "turnover":    0.6,   # raised from 0.5
        "stocks":      0.5,   # raised from 0.4
        "pts_ast":     6.0,   # raised from 5.0
        "pts_reb":     6.5,   # raised from 5.5
        "reb_ast":     2.8,   # raised from 2.3
        "pts_reb_ast": 7.0,   # raised from 6.0
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

    # Compute per-stat variance compression factors and save for inference.
    # These factors are passed to _apply_mean_bias_correction at inference to tighten
    # over-wide PMF distributions that inflate P(over) for outlier players.
    _var_compress: dict[str, float] = {}
    for _vc_stat in sorted(set(oof_eligible["stat"].dropna()) & set(SUPPORTED_STATS)):
        _vc_sub = oof_eligible[oof_eligible["stat"] == _vc_stat].dropna(subset=["pmf_mean", "actual_outcome", "pmf_json"])
        if len(_vc_sub) < 200:
            _var_compress[_vc_stat] = 1.0
            continue
        _vc_pmf_vars: list[float] = []
        for _, _vc_row in _vc_sub.iterrows():
            try:
                _vc_p = normalize_pmf(json_to_pmf(_vc_row["pmf_json"]))
                _vc_k = np.arange(len(_vc_p), dtype=float)
                _vc_m = float(np.dot(_vc_k, _vc_p))
                _vc_pmf_vars.append(float(np.dot(_vc_k ** 2, _vc_p)) - _vc_m ** 2)
            except Exception:
                pass
        _vc_model_var = float(np.mean(_vc_pmf_vars)) if _vc_pmf_vars else 0.0
        _vc_actual_var = float(_vc_sub["actual_outcome"].var())
        if _vc_actual_var > 0.01 and _vc_model_var > 0.01:
            # Clamp: compress at most 3× to avoid over-squeezing thin-tailed distributions
            _var_compress[_vc_stat] = float(np.clip(_vc_model_var / _vc_actual_var, 1.0, 3.0))
        else:
            _var_compress[_vc_stat] = 1.0
    (out / "variance_compress.json").write_text(json.dumps(_var_compress, indent=2))
    print(f"[calibrate] Variance compression factors saved: { {k: round(v,3) for k,v in _var_compress.items()} }")

    # Apply variance compression to OOF PMFs BEFORE fitting isotonic calibrators,
    # matching what will be done at inference (same alignment logic as bias corrections).
    # NOTE: At this point oof_eligible["pmf_json"] may already have bias corrections applied
    # (from the block above). Pass alpha=1.0 here so we only compress variance, not
    # double-apply the mean shift.
    _any_compress = any(v > 1.05 for v in _var_compress.values())
    if _any_compress:
        _vc_pmf_jsons = []
        for _, _vc_bcr_row in oof_eligible.iterrows():
            _vc_factor = _var_compress.get(str(_vc_bcr_row.get("stat", "")), 1.0)
            if _vc_factor > 1.05 and pd.notna(_vc_bcr_row.get("pmf_json")):
                _vc_pmf = _apply_mean_bias_correction(
                    json_to_pmf(_vc_bcr_row["pmf_json"]),
                    alpha=1.0,  # mean already shifted by bias correction loop above
                    variance_compress_factor=_vc_factor,
                )
                _vc_pmf_jsons.append(pmf_to_json(_vc_pmf))
            else:
                _vc_pmf_jsons.append(_vc_bcr_row.get("pmf_json"))
        oof_eligible = oof_eligible.copy()
        oof_eligible["pmf_json"] = _vc_pmf_jsons
        oof_eligible["pmf"] = oof_eligible["pmf_json"].map(json_to_pmf)
        print(f"[calibrate] Applied variance compression to {len(oof_eligible):,} OOF PMFs")

    # Per-player variance compression: compute player-specific VC factors from OOF.
    # Applied after global stat-level VC to capture residual player-level overdispersion.
    _MIN_PLAYER_VC_ROWS = 30
    if "player_id" in oof_eligible.columns and "pmf_variance" in oof_eligible.columns:
        _player_vc: dict[str, float] = {}
        for _pvc_pid, _pvc_grp in oof_eligible.groupby("player_id"):
            if len(_pvc_grp) < _MIN_PLAYER_VC_ROWS:
                continue
            _pvc_model_var = float(_pvc_grp["pmf_variance"].mean())
            _pvc_actual_var = float(_pvc_grp["actual_outcome"].var())
            if _pvc_actual_var > 1e-4:
                _pvc_ratio = float(np.clip(_pvc_model_var / _pvc_actual_var, 0.5, 3.0))
                _player_vc[str(_pvc_pid)] = round(_pvc_ratio, 4)
        if _player_vc:
            (out / "player_variance_compress.json").write_text(json.dumps(_player_vc, indent=2))
            print(
                f"[calibrate] Per-player variance compression: {len(_player_vc)} players, "
                f"mean_ratio={float(np.mean(list(_player_vc.values()))):.3f}"
            )

    # Fit PerRoleVarianceCompressor (Phase 5) — per-role + per-player factors.
    # Saved to artifacts/models/calibration/variance_compressor.pkl for inference.
    try:
        from wnba_props_model.calibration.venn_abers import PerRoleVarianceCompressor  # noqa: PLC0415
        _prvc_compressor = PerRoleVarianceCompressor(min_samples=50)
        for _prvc_stat in sorted(set(oof_eligible["stat"].dropna()) & set(SUPPORTED_STATS)):
            _prvc_sub = oof_eligible[oof_eligible["stat"] == _prvc_stat].dropna(
                subset=["pmf_mean", "actual_outcome"]
            )
            if len(_prvc_sub) < 50:
                continue
            _prvc_preds = _prvc_sub["pmf_mean"].values
            _prvc_actuals = _prvc_sub["actual_outcome"].values
            _prvc_roles = _prvc_sub["role_bucket"].fillna("unknown").values if "role_bucket" in _prvc_sub.columns else np.array(["unknown"] * len(_prvc_sub))
            _prvc_players = _prvc_sub["player_id"].astype(str).values if "player_id" in _prvc_sub.columns else np.array(["0"] * len(_prvc_sub))
            _prvc_compressor.fit(_prvc_preds, _prvc_actuals, _prvc_roles, _prvc_players, stat=_prvc_stat)
        _prvc_path = out / "variance_compressor.pkl"
        _prvc_compressor.save(str(_prvc_path))
        print(f"[calibrate] PerRoleVarianceCompressor saved: {_prvc_path} "
              f"({len(_prvc_compressor.role_factors)} role factors, "
              f"{len(_prvc_compressor.player_factors)} player factors)")
    except Exception as _prvc_exc:
        logger.warning("[calibrate] PerRoleVarianceCompressor fitting failed (non-fatal): %s", _prvc_exc)

    # Fit Venn-Abers calibrators per (stat, role_bucket) when OOF+lines parquet is available.
    # Saved to artifacts/models/calibration/venn_abers_{stat}_{role}.pkl
    _va_oof_path = Path(oof_pmfs_path).parent / "oof_pmfs_with_lines.parquet"
    if _va_oof_path.exists():
        try:
            import math as _math  # noqa: PLC0415
            from wnba_props_model.calibration.venn_abers import VennAbersCalibrator  # noqa: PLC0415
            _va_oof = pd.read_parquet(_va_oof_path)
            if "outcome" not in _va_oof.columns and "actual_outcome" in _va_oof.columns:
                _va_oof["outcome"] = _va_oof["actual_outcome"]
            if "pmf" not in _va_oof.columns and "pmf_json" in _va_oof.columns:
                _va_oof["pmf"] = _va_oof["pmf_json"].map(json_to_pmf)
            _va_stats = sorted(set(_va_oof["stat"].dropna()) & set(SUPPORTED_STATS)) if "stat" in _va_oof.columns else []
            _va_roles = sorted(_va_oof["role_bucket"].dropna().unique().tolist()) if "role_bucket" in _va_oof.columns else ["all"]
            for _va_stat in _va_stats:
                for _va_role in _va_roles:
                    _va_sub = _va_oof[
                        (_va_oof["stat"] == _va_stat) &
                        (_va_oof.get("role_bucket", pd.Series(["all"] * len(_va_oof))) == _va_role)
                    ].dropna(subset=["outcome", "line"]) if "line" in _va_oof.columns else pd.DataFrame()
                    if len(_va_sub) < 30:
                        continue
                    # Compute P(over) scores from PMF at line
                    _va_scores = []
                    _va_labels = []
                    for _, _va_row in _va_sub.iterrows():
                        try:
                            _va_pmf = normalize_pmf(json_to_pmf(_va_row["pmf_json"]))
                            _va_line = float(_va_row["line"])
                            _va_p = float(_va_pmf[_math.ceil(_va_line):].sum())
                            _va_label = float(_va_row["outcome"] > _va_line)
                            if _va_row["outcome"] != _va_line:  # skip pushes
                                _va_scores.append(_va_p)
                                _va_labels.append(_va_label)
                        except Exception:
                            continue
                    if len(_va_scores) < 20:
                        continue
                    _va_cal = VennAbersCalibrator()
                    _va_cal.fit(np.array(_va_scores), np.array(_va_labels))
                    _va_role_safe = str(_va_role).replace("/", "_").replace(" ", "_")
                    _va_out = out / f"venn_abers_{_va_stat}_{_va_role_safe}.pkl"
                    _va_cal.save(str(_va_out))
            print(f"[calibrate] Venn-Abers calibrators saved for {_va_stats} × {_va_roles}")
        except Exception as _va_exc:
            logger.warning("[calibrate] Venn-Abers fitting failed (non-fatal): %s", _va_exc)
    else:
        logger.info("[calibrate] Skipping Venn-Abers fit — OOF+lines parquet not found at %s", _va_oof_path)

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
        "n_rows": int(len(oof_eligible)),
        "stats": sorted(paths.keys()),
    }
    (out / "calibration_metadata.json").write_text(json.dumps(_meta, indent=2))
    print(f"[calibrate] Wrote calibration_metadata.json: {_meta['fitted_at']} n_rows={_meta['n_rows']}")

    # Fit P(DNP) calibrator if p_dnp and did_play columns are present in OOF
    if "p_dnp" in oof.columns and "did_play" in oof.columns:
        _pdnp_path = fit_pdnp_calibrator(oof, out)
        if _pdnp_path:
            paths["pdnp"] = _pdnp_path

    # Also fit Beta calibrators for P(over) (Item 5A)
    # Beta calibrators require a 'line' column to compute P(over) during training.
    # The OOF parquet intentionally forbids 'line' to prevent lookahead bias.
    # Join lines from a separate historical props parquet if provided.
    try:
        _beta_oof_path = oof_pmfs_path
        if props_parquet_path is not None:
            _props_path = Path(props_parquet_path)
            if _props_path.exists():
                _oof_for_beta = pd.read_parquet(oof_pmfs_path).copy()
                if "line" not in _oof_for_beta.columns:
                    _props_df = pd.read_parquet(_props_path)
                    # Normalize stat column name if needed
                    try:
                        from wnba_props_model.constants import BDL_PROP_TO_STAT  # noqa: PLC0415
                        if "stat" not in _props_df.columns and "prop_type" in _props_df.columns:
                            _props_df["stat"] = _props_df["prop_type"].map(BDL_PROP_TO_STAT)
                    except ImportError:
                        pass
                    _line_cols = [c for c in ["game_id", "player_id", "stat"] if c in _props_df.columns]
                    _line_val_col = next((c for c in ["line_value", "line"] if c in _props_df.columns), None)
                    if _line_cols and _line_val_col:
                        _lines = (
                            _props_df[_line_cols + [_line_val_col]]
                            .dropna(subset=[_line_val_col])
                            .rename(columns={_line_val_col: "line"})
                            .groupby(_line_cols, as_index=False)["line"]
                            .median()
                        )
                        _oof_for_beta = _oof_for_beta.merge(_lines, on=_line_cols, how="left")
                        _n_with_line = int(_oof_for_beta["line"].notna().sum())
                        print(f"[calibrate] Joined lines for beta calibration: {_n_with_line:,} OOF rows have a line")
                        import tempfile as _tf  # noqa: PLC0415
                        with _tf.NamedTemporaryFile(suffix=".parquet", delete=False) as _tmp:
                            _tmp_path = _tmp.name
                        _oof_for_beta.to_parquet(_tmp_path, index=False)
                        _beta_oof_path = _tmp_path
        beta_paths = fit_beta_calibrators(_beta_oof_path, out_dir)
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

    # Fit p_nz isotonic calibrators for hurdle-model stats (stl, blk)
    if "p_nz" in oof_eligible.columns:
        pnz_paths = fit_pnz_calibrators(oof_eligible, out)
        paths.update({f"pnz_{k}": v for k, v in pnz_paths.items()})
    else:
        logger.info(
            "[calibrate] p_nz column not in OOF — skipping p_nz calibrators "
            "(rebuild OOF with updated training.py to enable)"
        )

    return paths


def apply_role_stratified_corrections(
    pmfs_long: pd.DataFrame,
    cal_dir: str | Path = "artifacts/models/calibration",
) -> pd.DataFrame:
    """Apply ONLY role-stratified bias corrections (no player-level form corrections here).

    Replaces the single global bias correction (bias_corrections.json) with
    per-role corrections (bias_corrections_by_role.json).

    IMPORTANT: Player-level flat form corrections are applied AFTER Bayesian
    shrinkage (in predict.py) to prevent shrinkage from dampening individual
    player corrections that exceed the population prior by a large margin.

    Call AFTER apply_calibrators, BEFORE apply_bayesian_shrinkage.

    Parameters
    ----------
    pmfs_long : DataFrame with pmf_json, stat, role_bucket, player_name columns
    cal_dir   : directory containing calibration artifacts

    Returns
    -------
    Copy of pmfs_long with role-corrected pmf_json, pmf_mean, mean, median, mode, p0
    """
    cal_dir = Path(cal_dir)

    role_corr_path = cal_dir / "bias_corrections_by_role.json"
    if not role_corr_path.exists():
        logger.warning(
            "[calibrate] bias_corrections_by_role.json not found at %s — "
            "skipping role-stratified corrections",
            role_corr_path,
        )
        return pmfs_long

    role_corrections: dict[str, dict[str, float]] = json.loads(role_corr_path.read_text())

    global_corrections: dict[str, float] = {}
    global_corr_path = cal_dir / "bias_corrections.json"
    if global_corr_path.exists():
        try:
            global_corrections = json.loads(global_corr_path.read_text())
        except Exception as exc:
            logger.warning("[calibrate] Could not load bias_corrections.json: %s", exc)

    _COMBO_STATS: set[str] = {"pts_reb", "pts_ast", "pts_reb_ast", "reb_ast", "stocks", "blk_stl"}

    out = pmfs_long.copy()
    new_pmf_jsons: list[str] = []
    n_corrected = 0

    for _, row in out.iterrows():
        stat = str(row.get("stat", ""))
        if stat in _COMBO_STATS:
            new_pmf_jsons.append(row["pmf_json"])
            continue

        role = str(row.get("role_bucket", "rotation"))
        global_corr = float(global_corrections.get(stat, 1.0))
        role_corr = float(role_corrections.get(role, {}).get(stat, global_corr))

        # Net multiplier: swap global bias correction for role-specific one.
        # For starters: net_mult ≈ 1.15 (pts), 1.25 (reb) → increases calibrated mean.
        # For fringe/rotation: net_mult ≈ 1.0 (no change from global).
        net_mult = (role_corr / global_corr) if global_corr > 0.0 else 1.0

        if abs(net_mult - 1.0) < 0.01:
            new_pmf_jsons.append(row["pmf_json"])
            continue

        player_name = str(row.get("player_name", ""))
        try:
            raw_pmf = json_to_pmf(row["pmf_json"])
            corrected = _apply_mean_bias_correction(
                raw_pmf,
                float(np.clip(net_mult, 0.50, 2.50)),
            )
            new_pmf_jsons.append(pmf_to_json(corrected))
            n_corrected += 1
        except Exception as exc:
            logger.debug("[calibrate] role_stratified: failed row %s/%s: %s", player_name, stat, exc)
            new_pmf_jsons.append(row["pmf_json"])

    out["pmf_json"] = new_pmf_jsons

    # Recompute summary stats from updated PMFs
    def _pmf_stats(pmf_json: str) -> dict:
        pmf = normalize_pmf(json_to_pmf(pmf_json))
        ks = np.arange(len(pmf))
        mean = float(np.dot(ks, pmf))
        return {
            "pmf_mean": round(mean, 4),
            "mean": round(mean, 4),
            "median": int(np.searchsorted(np.cumsum(pmf), 0.5)),
            "mode": int(np.argmax(pmf)),
            "p0": float(pmf[0]),
        }

    stats_rows = [_pmf_stats(j) for j in out["pmf_json"]]
    for key in ("pmf_mean", "mean", "median", "mode", "p0"):
        out[key] = [r[key] for r in stats_rows]

    logger.info(
        "[calibrate] apply_role_stratified_corrections: corrected %d / %d PMF rows "
        "(role_corr_roles=%s)",
        n_corrected, len(out), sorted(role_corrections.keys()),
    )
    return out


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

    for stat in out["stat"].unique():
        cal_path = cal_dir / f"pmf_cal_role_{stat}.pkl"
        if cal_path.exists():
            calibrators[stat] = RoleAwarePMFCalibrator.load(str(cal_path))

    # Load per-stat variance compression factors (computed during fit_calibrators)
    _var_compress_ac: dict[str, float] = {}
    _var_compress_path = cal_dir / "variance_compress.json"
    if _var_compress_path.exists():
        try:
            _var_compress_ac = json.loads(_var_compress_path.read_text())
        except Exception as exc_vc:
            logger.warning("[calibrate] Could not load variance_compress.json: %s", exc_vc)

    # Load per-player variance compression factors (computed during fit_calibrators)
    _player_vc_map: dict[str, float] = {}
    _player_vc_path = cal_dir / "player_variance_compress.json"
    if _player_vc_path.exists():
        try:
            _player_vc_map = json.loads(_player_vc_path.read_text())
        except Exception:
            pass

    # Load p_nz calibrators for hurdle stats (stl, blk)
    import joblib as _joblib_pnz  # noqa: PLC0415
    _pnz_calibrators: dict[str, Any] = {}
    for _pnz_stat in ("stl", "blk"):
        _pnz_path = cal_dir / f"pnz_cal_{_pnz_stat}.pkl"
        if _pnz_path.exists():
            _pnz_calibrators[_pnz_stat] = _joblib_pnz.load(_pnz_path)
            logger.info("[calibrate] Loaded p_nz calibrator for %s", _pnz_stat)

    # Load P(DNP) calibrator
    _pdnp_calibrator: Any = None
    _pdnp_cal_path = cal_dir / "pdnp_cal.pkl"
    if _pdnp_cal_path.exists():
        try:
            _pdnp_calibrator = _joblib_pnz.load(_pdnp_cal_path)
            logger.info("[calibrate] Loaded P(DNP) calibrator from %s", _pdnp_cal_path)
        except Exception as _pdnp_exc:
            logger.warning("[calibrate] Could not load pdnp_cal.pkl: %s", _pdnp_exc)

    # Apply P(DNP) calibration to p_dnp column if available
    if _pdnp_calibrator is not None and "p_dnp" in out.columns:
        _raw_pdnp = out["p_dnp"].fillna(0.0).clip(1e-6, 1 - 1e-6).values
        _cal_pdnp = _pdnp_calibrator.predict(_raw_pdnp)
        out = out.copy()
        out["p_dnp"] = np.clip(_cal_pdnp, 0.0, 1.0)
        logger.info("[calibrate] Applied P(DNP) calibration: mean_raw=%.3f mean_cal=%.3f",
                    float(_raw_pdnp.mean()), float(_cal_pdnp.mean()))

    # Protected mean bias correction (Phase H): apply additive non-uniform correction
    # per-stat in batch BEFORE the per-row PMF loop.  This prevents the global
    # correction from over-correcting workhorse players (large predicted means), which
    # causes the observed -1.22 underprediction gap for elite scorers.
    # We convert the multiplicative alpha (actual/model ratio) to an additive bias
    # (actual_mean - model_mean ≈ model_mean * (alpha - 1)) and apply it via
    # _apply_protected_mean_bias_correction(), which weights corrections by proximity
    # to the median prediction (high-performers get ~30%, bench gets 100%).
    _protected_target_means: dict[str, np.ndarray] = {}
    if "pmf_mean" in out.columns:
        for _pbc_stat, _pbc_grp in out.groupby("stat"):
            _pbc_alpha = _bias_corrections_ac.get(str(_pbc_stat), 1.0)
            if abs(_pbc_alpha - 1.0) > 0.005:
                _pbc_means = _pbc_grp["pmf_mean"].fillna(0.0).values.astype(float)
                _pbc_bias = float(np.mean(_pbc_means)) * (_pbc_alpha - 1.0)
                _protected_target_means[str(_pbc_stat)] = _apply_protected_mean_bias_correction(
                    _pbc_means, _pbc_bias
                )

    # Build a row-index → corrected mean map for fast lookup in the loop below
    _row_corrected_mean: dict[int, float] = {}
    if _protected_target_means:
        for _pbc_stat, _pbc_target in _protected_target_means.items():
            _pbc_idx = out[out["stat"] == _pbc_stat].index.tolist()
            for _enum_i, _ridx in enumerate(_pbc_idx):
                if _enum_i < len(_pbc_target):
                    _row_corrected_mean[_ridx] = float(_pbc_target[_enum_i])

    new_pmf_jsons = []
    is_calibrated_flags = []
    cal_sources = []

    for _row_idx, row in out.iterrows():
        stat = row["stat"]
        role = str(row.get("role_bucket", "unknown"))
        player_id = row.get("player_id", None)
        raw_pmf = json_to_pmf(row["pmf_json"])

        # Apply protected mean bias correction: use per-player corrected target mean
        # (from _apply_protected_mean_bias_correction) when available; fall back to
        # standard multiplicative correction otherwise.
        _alpha_ac = _bias_corrections_ac.get(stat, 1.0)
        _vc_factor_ac = float(_var_compress_ac.get(stat, 1.0))
        if _row_idx in _row_corrected_mean and abs(_alpha_ac - 1.0) > 0.005:
            # Recompute alpha for this specific player from the protected target mean
            _pmf_k = np.arange(len(raw_pmf), dtype=float)
            _curr_mean = float(np.dot(_pmf_k, raw_pmf / max(raw_pmf.sum(), 1e-9)))
            _target_mean = _row_corrected_mean[_row_idx]
            _player_alpha = (_target_mean / _curr_mean) if _curr_mean > 0.01 else _alpha_ac
            _player_alpha = float(np.clip(_player_alpha, 0.60, 1.40))
            raw_pmf = _apply_mean_bias_correction(raw_pmf, _player_alpha, variance_compress_factor=_vc_factor_ac)
        elif abs(_alpha_ac - 1.0) > 0.005 or _vc_factor_ac > 1.05:
            raw_pmf = _apply_mean_bias_correction(raw_pmf, _alpha_ac, variance_compress_factor=_vc_factor_ac)

        # Per-player variance compression override (applied after global stat-level VC)
        _player_vc_factor = _player_vc_map.get(str(row.get("player_id", "")))
        if _player_vc_factor is not None and abs(_player_vc_factor - 1.0) > 0.05:
            raw_pmf = _apply_mean_bias_correction(
                raw_pmf,
                alpha=1.0,  # mean already corrected above; compress variance only
                variance_compress_factor=float(_player_vc_factor),
            )

        # P_nz calibration for hurdle stats (stl, blk): rescale zero mass
        if stat in _pnz_calibrators:
            _p_nz_raw = float(1.0 - raw_pmf[0])
            _p_nz_raw_clipped = np.clip(_p_nz_raw, 1e-6, 1 - 1e-6)
            _p_nz_cal = float(_pnz_calibrators[stat].predict([_p_nz_raw_clipped])[0])
            _p_nz_cal = np.clip(_p_nz_cal, 1e-6, 1 - 1e-6)
            if _p_nz_raw > 1e-6:
                # Rescale non-zero atoms proportionally; preserve shape of conditional distribution
                raw_pmf[1:] = raw_pmf[1:] * (_p_nz_cal / _p_nz_raw)
            raw_pmf[0] = 1.0 - _p_nz_cal
            raw_pmf = raw_pmf / raw_pmf.sum()  # renormalize

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
