#!/usr/bin/env python3
"""Stage 4 baseline PMF training script.

Trains minutes model + stat rate / hurdle models, generates full atom PMFs
for every player_id × game_id × stat row, and writes model artifacts + output tables.

Usage:
    python3 scripts/train_baseline_pmfs.py \\
      --features-wide data/processed/wnba_player_game_features_wide.parquet \\
      --features-long data/processed/wnba_player_game_features_long.parquet \\
      --manifest data/processed/feature_schema_manifest.json \\
      --config config/model/stage4_baseline.yaml \\
      --model-dir artifacts/models/stage4_baseline \\
      --out-dir data/model_outputs/stage4_baseline \\
      --audit-out artifacts/audits/stage4_training_audit.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.preprocessing import OrdinalEncoder

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.features.feature_contract import (
    FORBIDDEN_MODEL_FEATURES,
    assert_no_forbidden_features,
)
from wnba_props_model.models.minutes_model import MinutesModel
from wnba_props_model.models.pmf_engine import (
    STATS,
    build_all_pmfs,
    build_wide_pmf_table,
    prepare_feature_matrix,
)
from wnba_props_model.models.rate_model import HurdleModel, StatRateModel

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _load_and_validate(
    wide_path: Path,
    long_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    print(f"Loading wide features: {wide_path}")
    wide = pd.read_parquet(wide_path)
    print(f"  {len(wide):,} rows, {wide.shape[1]} columns")

    print(f"Loading long features: {long_path}")
    long = pd.read_parquet(long_path)
    print(f"  {len(long):,} rows")

    print(f"Loading manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_cols: list[str] = manifest["model_feature_columns"]
    print(f"  {len(model_cols)} model feature columns")

    # Leakage guard
    assert_no_forbidden_features(model_cols)
    print("  Leakage guard: PASS")

    # Verify targets not in model_cols
    targets = manifest.get("target_columns", [])
    leaked = [c for c in model_cols if c in targets]
    if leaked:
        raise ValueError(f"Target columns in model_feature_cols: {leaked}")

    return wide, long, manifest, model_cols


def _prepare_X(
    df: pd.DataFrame,
    model_cols: list[str],
    pos_encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder | None]:
    """Encode features; return (X_numeric, pos_encoder)."""
    available = [c for c in model_cols if c in df.columns]
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} manifest columns not in table (absent features)")

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

    # Safety guard: drop any remaining object-dtype columns (e.g. new string bucket cols
    # added to features but not yet added to ROLE_BUCKET_COLS exclusion list).
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print(f"  [WARN] Dropping {len(obj_cols)} non-numeric model cols from X: {obj_cols}")
        X = X.drop(columns=obj_cols)

    return X, pos_encoder


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@app.command()
def train(
    features_wide: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_wide.parquet"),
        "--features-wide",
    ),
    features_long: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_long.parquet"),
        "--features-long",
    ),
    manifest: Path = typer.Option(
        Path("data/processed/feature_schema_manifest.json"),
        "--manifest",
    ),
    config_path: Path = typer.Option(
        Path("config/model/stage4_baseline.yaml"), "--config"
    ),
    model_dir: Path = typer.Option(
        Path("artifacts/models/stage4_baseline"), "--model-dir"
    ),
    out_dir: Path = typer.Option(
        Path("data/model_outputs/stage4_baseline"), "--out-dir"
    ),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage4_training_audit.json"), "--audit-out"
    ),
    time_decay_xi: float | None = typer.Option(
        None,
        "--time-decay-xi",
        help=(
            "Dixon-Coles decay xi: sample_weight = exp(-xi * days_ago). "
            "Overrides sample_weight_halflife_days from config when provided. "
            "At xi=0.003, a game 180 days ago gets weight exp(-0.54)=0.58 vs yesterday=1.0."
        ),
    ),
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("Stage 4 — Baseline PMF Training")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Config + setup
    # ------------------------------------------------------------------
    cfg: dict = yaml.safe_load(config_path.read_text())
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    wide, long, manifest_dict, model_cols = _load_and_validate(
        features_wide, features_long, manifest
    )

    # ------------------------------------------------------------------
    # 3. Feature matrix (fit OrdinalEncoder on full wide table)
    # ------------------------------------------------------------------
    print("\nPreparing feature matrix...")
    X_all, pos_encoder = _prepare_X(wide, model_cols, fit_encoder=True)
    print(f"  Feature matrix shape: {X_all.shape}")

    # Sanity: no forbidden cols in X
    bad_in_X = [c for c in X_all.columns if c in FORBIDDEN_MODEL_FEATURES]
    if bad_in_X:
        raise ValueError(f"Forbidden columns in feature matrix: {bad_in_X}")

    # ------------------------------------------------------------------
    # 3b. Temporal sample weights (exponential decay)
    # ------------------------------------------------------------------
    sample_weight: np.ndarray | None = None
    if "game_date" in wide.columns:
        cutoff = pd.to_datetime(wide["game_date"]).max()
        days_ago = (cutoff - pd.to_datetime(wide["game_date"])).dt.days.fillna(0)

        if time_decay_xi is not None:
            # Dixon-Coles style: weight = exp(-xi * days_ago)
            sw = np.exp(-time_decay_xi * days_ago.values)
            sw = sw / sw.mean()
            sample_weight = sw.astype(np.float64)
            print(f"\nTemporal weighting (Dixon-Coles xi={time_decay_xi}): "
                  f"weight range [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")
        else:
            halflife = cfg.get("sample_weight_halflife_days", None)
            if halflife:
                sw = np.exp(-np.log(2) / halflife * days_ago.values)
                sw = sw / sw.mean()
                sample_weight = sw.astype(np.float64)
                print(f"\nTemporal weighting: halflife={halflife}d, "
                      f"weight range [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")

    # ------------------------------------------------------------------
    # 4. Train minutes model
    # ------------------------------------------------------------------
    print("\nTraining minutes model...")
    y_minutes = wide["actual_minutes"].fillna(0.0)
    minutes_mdl = MinutesModel(cfg)
    minutes_mdl.fit(X_all, y_minutes, wide, sample_weight=sample_weight)
    minutes_mdl._pos_encoder = pos_encoder  # store for inference
    minutes_path = model_dir / "minutes_model.joblib"
    minutes_mdl.save(str(minutes_path))
    print(f"  Saved: {minutes_path}")
    min_summary = minutes_mdl.get_training_summary()
    print(f"  Global sigma: {min_summary['global_sigma']:.2f} min")
    print(f"  Sigma buckets: {min_summary['n_sigma_buckets']}")

    # Minutes train predictions for audit
    y_min_pred, _, _p_dnp_audit = minutes_mdl.predict(X_all, wide)
    min_residuals = y_minutes.values - y_min_pred
    min_mae = float(np.abs(min_residuals).mean())
    min_rmse = float(np.sqrt((min_residuals ** 2).mean()))
    print(f"  Train MAE: {min_mae:.2f}  RMSE: {min_rmse:.2f}")

    # ------------------------------------------------------------------
    # 5. Train stat models
    # ------------------------------------------------------------------
    sparse_stats = set(cfg.get("sparse_stats", ["stl", "blk"]))
    stats = cfg.get("stats", STATS)
    played_mask = wide["did_play"].astype(bool) if "did_play" in wide.columns else (
        wide["actual_minutes"] > 0
    )

    stat_models: dict[str, StatRateModel] = {}
    hurdle_models: dict[str, HurdleModel] = {}
    stat_summaries: dict[str, dict] = {}

    # P3.2: Load tuned hyperparameters when use_tuned_hyperparams is set
    tuned_params: dict[str, dict] = {}
    if cfg.get("use_tuned_hyperparams", False):
        import json as _json  # noqa: PLC0415
        hp_path = Path(cfg.get("hgb_hyperparams_path", "artifacts/hyperparams/best_params_all.json"))
        if hp_path.exists():
            with open(hp_path) as _f:
                tuned_params = _json.load(_f)
            print(f"\nLoaded tuned hyperparams from {hp_path}: {list(tuned_params.keys())}")
        else:
            print(f"\n[WARN] use_tuned_hyperparams=true but {hp_path} not found — using defaults")

    X_played = X_all[played_mask].reset_index(drop=True)

    for stat in stats:
        target_col = f"actual_{stat}"
        if target_col not in wide.columns:
            print(f"\n  WARNING: {target_col} not in wide table — skipping {stat}")
            continue

        y_stat = wide.loc[played_mask, target_col].reset_index(drop=True)
        n_rows = len(y_stat)
        n_played = n_rows  # X_played is already filtered to played rows
        zero_rate = float((y_stat == 0).mean())

        print(f"\nTraining {stat} model  (n={n_played:,}, zero_rate={zero_rate:.3f})")

        # Sample weights for played rows (subset of full weight vector)
        sw_played = sample_weight[played_mask] if sample_weight is not None else None
        if sw_played is not None:
            sw_played = sw_played[: len(X_played)]  # guard against index mismatch

        # P3.2: merge tuned hyperparams for this stat into config copy
        stat_cfg = dict(cfg)
        if stat in tuned_params:
            tp = tuned_params[stat]
            hgb_r = dict(stat_cfg.get("hgb_regressor", {}))
            hgb_r.update({k: tp[k] for k in ("max_iter", "max_leaf_nodes", "learning_rate",
                                               "min_samples_leaf", "l2_regularization")
                           if k in tp})
            stat_cfg["hgb_regressor"] = hgb_r
            hgb_c = dict(stat_cfg.get("hgb_classifier", {}))
            hgb_c.update({k: tp[k] for k in ("max_iter", "max_leaf_nodes", "learning_rate",
                                               "min_samples_leaf") if k in tp})
            stat_cfg["hgb_classifier"] = hgb_c

        if stat in sparse_stats:
            model_h = HurdleModel(stat, stat_cfg)
            model_h.fit(X_played, y_stat, sample_weight=sw_played)
            hurdle_models[stat] = model_h
            s = model_h.get_training_summary()
            print(f"  HurdleModel  P(Y>0)≈{1-zero_rate:.3f}  "
                  f"pos_mean={s['pos_mean']:.3f}  pos_r={s['pos_dispersion_r']}")
        else:
            played_ctx = wide[played_mask].reset_index(drop=True)
            model_r = StatRateModel(stat, stat_cfg)
            model_r.fit(X_played, y_stat, context_df=played_ctx, sample_weight=sw_played)
            stat_models[stat] = model_r
            s = model_r.get_training_summary()
            print(f"  StatRateModel  mean={s['global_mean']:.3f}  "
                  f"var={s['global_var']:.3f}  "
                  f"type={s['pmf_type']}  r={s['dispersion_r']}  "
                  f"role_buckets={len(s.get('role_dispersion', {}))}")

        stat_summaries[stat] = s

    # P3.5: Compute and save position-stratified combo correlations
    try:
        from wnba_props_model.models.bivariate_pmf import estimate_correlations  # noqa: PLC0415
        pos_col = "position" if "position" in wide.columns else None
        corr_by_pos = estimate_correlations(wide, position_col=pos_col)
        corr_path = model_dir / "combo_correlations_by_pos.json"
        import json as _json2  # noqa: PLC0415
        with open(corr_path, "w") as _cf:
            _json2.dump(corr_by_pos, _cf, indent=2, default=str)
        print(f"\nSaved combo correlations by position: {corr_path}")
    except Exception as _exc:
        print(f"\n[WARN] Could not compute position-stratified correlations: {_exc}")

    # Part 6: CLV Secondary Head — trained from historical closing-line labels.
    # Classifies whether the model's edge direction beats the closing line.
    # Soft-nudges stat_mean at inference by up to ±5% based on CLV signal.
    _clv_labels_path = Path("data/oof/oof_clv_labels.parquet")
    if _clv_labels_path.exists():
        try:
            print("\nTraining CLV secondary heads from closing-line labels...")
            from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC  # noqa: PLC0415
            _clv_df = pd.read_parquet(_clv_labels_path)
            _clv_stats_fitted = 0
            for _stat in stats:
                _stat_clv = _clv_df[_clv_df["stat"] == _stat].copy() if "stat" in _clv_df.columns else pd.DataFrame()
                if len(_stat_clv) < 100:
                    continue
                # Join CLV labels with wide features via (player_id, game_id)
                _join_keys = [k for k in ["player_id", "game_id"] if k in _stat_clv.columns and k in wide.columns]
                if not _join_keys:
                    continue
                _clv_feat = wide.merge(_stat_clv[_join_keys + ["beat_closing"]], on=_join_keys, how="inner")
                if len(_clv_feat) < 50:
                    continue
                _X_clv, _ = _prepare_X(_clv_feat, model_cols, pos_encoder=pos_encoder)
                _y_clv = _clv_feat["beat_closing"].astype(int)
                # Part A: Sample-weight by |closing_p_over - 0.5| so that rows where
                # the market has a confident view (far from 50/50) receive higher weight.
                # This forces the model to focus on games where sharp consensus is clear
                # — disagreeing with that consensus is the CLV opportunity.
                _sw_clv = np.ones(len(_clv_feat))
                if "closing_p_over" in _clv_feat.columns:
                    _close_conf = np.abs(_clv_feat["closing_p_over"].fillna(0.5).values - 0.5).clip(0.01, 0.5)
                    _sw_clv = _close_conf * 2.0  # scale [0, 1]
                elif "clv_label" in _clv_feat.columns:
                    _sw_clv = np.abs(_clv_feat["clv_label"].fillna(0.0).values).clip(0.01, 2.0)
                _sw_clv = _sw_clv / _sw_clv.mean()
                _clv_head = _HGBC(max_iter=50, max_leaf_nodes=15, learning_rate=0.05, random_state=42)
                _clv_head.fit(_X_clv, _y_clv, sample_weight=_sw_clv)
                # Attach CLV head to the stat model
                _mdl = stat_models.get(_stat)
                if _mdl is not None:
                    _mdl.clv_head = _clv_head
                    _clv_stats_fitted += 1
            print(f"  CLV heads fitted for {_clv_stats_fitted}/{len(stats)} stats")
        except Exception as _clv_exc:
            print(f"  [WARN] CLV head training failed (non-fatal): {_clv_exc}")
    else:
        print(f"\n[INFO] No CLV labels at {_clv_labels_path} — CLV head skipped. "
              "Run scripts/build_clv_training_labels.py to enable.")

    # Save models
    stat_path = model_dir / "stat_rate_models.joblib"
    hurdle_path = model_dir / "hurdle_models.joblib"
    joblib.dump(stat_models, str(stat_path))
    joblib.dump(hurdle_models, str(hurdle_path))
    print(f"\nSaved stat models: {stat_path}")
    print(f"Saved hurdle models: {hurdle_path}")

    # Model manifest
    model_manifest = {
        "minutes_model": str(minutes_path),
        "stat_rate_models": str(stat_path),
        "hurdle_models": str(hurdle_path),
        "stats_trained": sorted(stat_models.keys()),
        "hurdle_stats_trained": sorted(hurdle_models.keys()),
        "model_feature_count": len(X_all.columns),
        "stage": "stage4_baseline",
        "pmf_source": cfg.get("pmf_source", "stage4_baseline_uncalibrated_model_only"),
        "is_calibrated": False,
        "git_commit": _git_commit(),
        "config_hash": _config_hash(cfg),
    }
    manifest_path_out = model_dir / "model_manifest.json"
    manifest_path_out.write_text(json.dumps(model_manifest, indent=2))
    print(f"Saved model manifest: {manifest_path_out}")

    # Write feature_manifest.json so predict.py can load the exact feature columns
    # used at training time without needing the original data/processed manifest.
    feature_manifest_out = model_dir / "feature_manifest.json"
    feature_manifest_out.write_text(json.dumps({"model_feature_columns": model_cols}, indent=2))
    print(f"Saved feature manifest: {feature_manifest_out} ({len(model_cols)} cols)")

    # ------------------------------------------------------------------
    # 6. Generate PMFs
    # ------------------------------------------------------------------
    print("\nGenerating PMFs...")
    pmf_df = build_all_pmfs(
        wide, long, model_cols,
        minutes_mdl, stat_models, hurdle_models, cfg
    )
    print(f"  PMF rows: {len(pmf_df):,}")
    print(f"  Rows by stat:")
    for s, cnt in pmf_df.groupby("stat").size().sort_index().items():
        print(f"    {s}: {cnt:,}")

    # Validate
    sum_errors = pmf_df.apply(
        lambda row: abs(sum(json.loads(row["pmf_json"]).values()) - 1.0),
        axis=1
    )
    max_sum_error = float(sum_errors.max())
    invalid_pmf_count = int((sum_errors > 1e-6).sum())
    print(f"  Max PMF sum error: {max_sum_error:.2e}")
    print(f"  Invalid PMFs (sum error > 1e-6): {invalid_pmf_count}")
    if invalid_pmf_count > 0:
        raise ValueError(f"{invalid_pmf_count} PMFs are invalid — aborting")

    # ------------------------------------------------------------------
    # 7. Write PMF outputs
    # ------------------------------------------------------------------
    pmf_long_path = out_dir / "player_stat_pmfs.parquet"
    pmf_df.to_parquet(pmf_long_path, index=False)
    print(f"\nSaved long PMFs: {pmf_long_path}")

    pmf_wide_path = out_dir / "player_stat_pmfs_wide.parquet"
    wide_pmf = build_wide_pmf_table(pmf_df)
    wide_pmf.to_parquet(pmf_wide_path, index=False)
    print(f"Saved wide PMFs: {pmf_wide_path}  ({len(wide_pmf):,} rows)")

    # ------------------------------------------------------------------
    # 8. Training audit
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    training_audit = {
        "stage": "stage4_baseline",
        "elapsed_seconds": round(elapsed, 1),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(cfg),
        "model_feature_count": len(model_cols),
        "forbidden_feature_check": "PASS",
        "target_leakage_check": "PASS",
        "minutes_model": {
            "train_rows": int(len(wide)),
            "target_mean": float(y_minutes.mean()),
            "target_std": float(y_minutes.std()),
            "train_mae": min_mae,
            "train_rmse": min_rmse,
            "global_sigma": min_summary["global_sigma"],
            "sigma_buckets": min_summary["n_sigma_buckets"],
        },
        "stat_models": {},
        "pmf_summary": {
            "total_pmf_rows": int(len(pmf_df)),
            "max_sum_error": max_sum_error,
            "invalid_pmf_count": invalid_pmf_count,
            "is_calibrated": False,
            "pmf_source": cfg.get("pmf_source", "stage4_baseline_uncalibrated_model_only"),
        },
    }

    for stat in stats:
        target_col = f"actual_{stat}"
        played_actuals = wide.loc[played_mask, target_col] if target_col in wide.columns else pd.Series(dtype=float)
        pmf_stat = pmf_df[pmf_df["stat"] == stat]
        training_audit["stat_models"][stat] = {
            "type": "hurdle" if stat in hurdle_models else "rate",
            "train_rows": int(played_mask.sum()),
            "target_mean": float(played_actuals.mean()) if len(played_actuals) > 0 else None,
            "target_var": float(played_actuals.var()) if len(played_actuals) > 1 else None,
            "zero_rate": float((played_actuals == 0).mean()) if len(played_actuals) > 0 else None,
            "pmf_mean_vs_actual": {
                "pmf_mean": float(pmf_stat["pmf_mean"].mean()) if len(pmf_stat) > 0 else None,
                "actual_mean": float(pmf_stat["actual_outcome"].mean()) if len(pmf_stat) > 0 else None,
            },
            "model_detail": stat_summaries.get(stat, {}),
        }

    audit_out.write_text(json.dumps(training_audit, indent=2, default=str))
    print(f"\nSaved training audit: {audit_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 4 Training Complete")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  PMF rows: {len(pmf_df):,}")
    print(f"  Max sum error: {max_sum_error:.2e}")
    print(f"  Invalid PMFs: {invalid_pmf_count}")
    print("=" * 70)


if __name__ == "__main__":
    app()
