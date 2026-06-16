#!/usr/bin/env python3
"""Stage 5 — Build strict walk-forward OOF PMFs.

Generates OOF PMFs using expanding-window chronological splits.
Each validation fold is predicted by a model trained exclusively on
game_date < fold_validation_start_date (strict temporal separation).

Usage:
    python3 scripts/build_oof_pmfs.py \\
      --features-wide data/processed/wnba_player_game_features_wide.parquet \\
      --features-long data/processed/wnba_player_game_features_long.parquet \\
      --manifest data/processed/feature_schema_manifest.json \\
      --config config/model/stage5_oof.yaml \\
      --out-dir data/oof \\
      --audit-out artifacts/audits/stage5_oof_audit.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.features.feature_contract import assert_no_forbidden_features
from wnba_props_model.models.oof_engine import generate_oof_folds, make_prior_only_pmfs
from wnba_props_model.models.training import encode_features, train_fold, generate_fold_pmfs

app = typer.Typer(add_completion=False)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


@app.command()
def build(
    features_wide: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_wide.parquet"),
        "--features-wide",
    ),
    features_long: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_long.parquet"),
        "--features-long",
    ),
    manifest_path: Path = typer.Option(
        Path("data/processed/feature_schema_manifest.json"),
        "--manifest",
    ),
    config_path: Path = typer.Option(
        Path("config/model/stage5_oof.yaml"), "--config"
    ),
    out_dir: Path = typer.Option(Path("data/oof"), "--out-dir"),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage5_oof_audit.json"), "--audit-out"
    ),
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("Stage 5 — Walk-forward OOF PMF Generation")
    print("=" * 70)

    cfg: dict = yaml.safe_load(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading: {features_wide}")
    wide = pd.read_parquet(features_wide)
    wide["game_date"] = pd.to_datetime(wide["game_date"])
    print(f"  {len(wide):,} rows")

    print(f"Loading: {features_long}")
    long = pd.read_parquet(features_long)
    long["game_date"] = pd.to_datetime(long["game_date"])
    print(f"  {len(long):,} rows")

    print(f"Loading manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_cols: list[str] = manifest["model_feature_columns"]
    print(f"  {len(model_cols)} model feature columns")

    # Leakage guard
    assert_no_forbidden_features(model_cols)
    targets = manifest.get("target_columns", [])
    leaked = [c for c in model_cols if c in targets]
    if leaked:
        raise ValueError(f"Target columns in model_feature_cols: {leaked}")
    print("  Leakage guard: PASS")

    # ------------------------------------------------------------------
    # 2. Fit global position encoder (on all data for category discovery)
    # ------------------------------------------------------------------
    # We fit the encoder once globally so each fold knows all categories.
    # The actual model fitting uses per-fold training data only.
    _, global_pos_encoder = encode_features(wide, model_cols, fit_encoder=True)
    print(f"\nGlobal pos_encoder fitted on {len(wide):,} rows")

    # ------------------------------------------------------------------
    # 3. Generate folds
    # ------------------------------------------------------------------
    game_dates_all: list[date] = sorted(
        wide["game_date"].dt.date.unique().tolist()
    )
    folds = generate_oof_folds(game_dates_all, cfg.get("validation_window_days", 14))
    print(f"\nFolds generated: {len(folds)}")
    print(f"  First val window: {folds[0]['val_start_date']} – {folds[0]['val_end_date']}")
    print(f"  Last  val window: {folds[-1]['val_start_date']} – {folds[-1]['val_end_date']}")

    min_train = cfg.get("min_train_long_rows", 2000)

    # ------------------------------------------------------------------
    # 4. OOF loop
    # ------------------------------------------------------------------
    all_pmf_frames: list[pd.DataFrame] = []
    fold_records: list[dict] = []
    stats = cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"])

    for fold in folds:
        fid = fold["fold_id"]
        val_start: date = fold["val_start_date"]

        # STRICT temporal split: train on game_date < val_start (never <=)
        train_mask_wide = wide["game_date"].dt.date < val_start
        val_mask_wide   = wide["game_date"].dt.date.isin(fold["val_dates"])
        train_mask_long = long["game_date"].dt.date < val_start
        val_mask_long   = long["game_date"].dt.date.isin(fold["val_dates"])

        train_wide_df = wide[train_mask_wide].reset_index(drop=True)
        val_wide_df   = wide[val_mask_wide].reset_index(drop=True)
        train_long_df = long[train_mask_long].reset_index(drop=True)
        val_long_df   = long[val_mask_long].reset_index(drop=True)

        n_train_long  = len(train_long_df)
        n_val_wide    = len(val_wide_df)
        n_train_games = int(train_wide_df["game_id"].nunique()) if len(train_wide_df) > 0 else 0

        fold_meta_base = {
            "fold_id":         fid,
            "train_start_date": fold.get("train_start_date"),
            "train_end_date":  fold["train_end_date"],
            "val_start_date":  val_start,
            "val_end_date":    fold["val_end_date"],
            "train_wide_rows": len(train_wide_df),
            "train_long_rows": n_train_long,
            "train_games":     n_train_games,
        }

        # Print fold header
        print(f"\n  Fold {fid:2d}: val={val_start}–{fold['val_end_date']}"
              f"  train_rows={n_train_long:,}  val_rows={n_val_wide}")

        if n_val_wide == 0:
            print("    → no validation rows, skipping")
            fold_records.append({**fold_meta_base,
                "fit_status": "skipped", "error_message": "no_val_rows",
                "validation_long_rows": 0})
            continue

        # --- Check eligibility ---
        if n_train_long < min_train:
            print(f"    → insufficient training data ({n_train_long} < {min_train}) → prior_only")
            fold_meta = {**fold_meta_base, "oof_prediction_type": "prior_only"}
            pmf_frame = make_prior_only_pmfs(val_wide_df, val_long_df, fold_meta, cfg)
            fold_records.append({**fold_meta_base,
                "fit_status": "prior_only", "error_message": "",
                "validation_long_rows": len(val_long_df)})
        else:
            try:
                # --- Train fold models ---
                fold_model = train_fold(train_wide_df, train_long_df, model_cols, cfg)
                # Override encoder with global one (has all categories)
                fold_model.pos_encoder = global_pos_encoder
                fold_model.minutes_model._pos_encoder = global_pos_encoder

                fold_meta = {
                    **fold_meta_base,
                    "oof_prediction_type": "model_oof",
                    "train_stat_rows": fold_model.train_stat_rows,
                }

                pmf_frame = generate_fold_pmfs(
                    fold_model, val_wide_df, val_long_df, fold_meta, cfg
                )
                status = "model_oof"
                errmsg = ""
                print(f"    → model_oof  PMF rows={len(pmf_frame):,}")
            except Exception as e:
                print(f"    → FAILED: {e}")
                fold_meta = {**fold_meta_base, "oof_prediction_type": "failed_model_fit"}
                pmf_frame = make_prior_only_pmfs(
                    val_wide_df, val_long_df, fold_meta, cfg, error_msg=str(e)
                )
                status = "failed_model_fit"
                errmsg = str(e)

            fold_records.append({**fold_meta_base,
                "fit_status": status, "error_message": errmsg,
                "validation_long_rows": len(val_long_df)})

        if not pmf_frame.empty:
            all_pmf_frames.append(pmf_frame)

    # ------------------------------------------------------------------
    # 5. Concatenate and validate
    # ------------------------------------------------------------------
    if not all_pmf_frames:
        raise ValueError("No OOF PMF frames generated — check data and config")

    print("\nConcatenating OOF frames...")
    oof_df = pd.concat(all_pmf_frames, ignore_index=True)
    print(f"  Total OOF rows: {len(oof_df):,}")

    # Duplicate key check
    dup_count = oof_df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
    if dup_count > 0:
        raise ValueError(f"Duplicate player × game × stat keys: {dup_count}")
    print(f"  Duplicate keys: 0 (PASS)")

    # PMF sum check
    sum_errors = oof_df.apply(
        lambda r: abs(sum(json.loads(r["pmf_json"]).values()) - 1.0), axis=1
    )
    max_err = float(sum_errors.max())
    invalid = int((sum_errors > 1e-6).sum())
    if invalid > 0:
        raise ValueError(f"{invalid} invalid OOF PMFs (sum error > 1e-6)")
    print(f"  Max PMF sum error: {max_err:.2e}  (PASS)")

    # Forbidden field check
    for col in ["line", "over_odds", "under_odds", "book", "vendor", "sportsbook"]:
        if col in oof_df.columns:
            raise ValueError(f"Forbidden column '{col}' in OOF output")
    print("  Forbidden field check: PASS")

    # is_calibrated check
    if (oof_df["is_calibrated"] != False).any():  # noqa: E712
        raise ValueError("OOF rows with is_calibrated != False")
    print("  is_calibrated = False: PASS")

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    long_out = out_dir / "oof_player_stat_pmfs.parquet"
    oof_df.to_parquet(long_out, index=False)
    print(f"\nSaved long OOF PMFs: {long_out}")

    # Wide pivot
    wide_out = _build_wide_oof_table(oof_df, stats)
    wide_out.to_parquet(out_dir / "oof_player_stat_pmfs_wide.parquet", index=False)
    print(f"Saved wide OOF PMFs: {out_dir}/oof_player_stat_pmfs_wide.parquet"
          f"  ({len(wide_out):,} rows)")

    # Fold manifest
    fold_df = pd.DataFrame(fold_records)
    fold_df["created_at_utc"] = pd.Timestamp.utcnow()
    fold_df["stats_in_fold"] = json.dumps(stats)
    fold_df["train_wide_rows"] = fold_df.get("train_wide_rows", 0)
    fold_df["validation_wide_rows"] = fold_df.get("validation_long_rows", 0)
    fold_df.to_parquet(out_dir / "oof_fold_manifest.parquet", index=False)
    print(f"Saved fold manifest: {out_dir}/oof_fold_manifest.parquet")

    # ------------------------------------------------------------------
    # 7. Audit
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    n_model_oof  = int((oof_df["oof_prediction_type"] == "model_oof").sum())
    n_prior_only = int((oof_df["oof_prediction_type"] == "prior_only").sum())
    n_failed     = int((oof_df["oof_prediction_type"] == "failed_model_fit").sum())
    n_cal_elig   = int((oof_df["calibration_eligible"] == True).sum())  # noqa: E712
    low_adj      = int(oof_df.get("low_minutes_adjustment_count",
                        pd.Series([0]*len(oof_df))).sum())

    audit = {
        "stage": "stage5_oof",
        "elapsed_seconds": round(elapsed, 1),
        "git_commit": _git_commit(),
        "n_folds": len(folds),
        "n_fold_model_oof": sum(1 for r in fold_records if r["fit_status"] == "model_oof"),
        "n_fold_prior_only": sum(1 for r in fold_records if r["fit_status"] == "prior_only"),
        "n_fold_failed": sum(1 for r in fold_records if r["fit_status"] == "failed_model_fit"),
        "n_fold_skipped": sum(1 for r in fold_records if r["fit_status"] == "skipped"),
        "first_val_date": str(folds[0]["val_start_date"]),
        "last_val_date": str(folds[-1]["val_end_date"]),
        "oof_pmf_rows_total": int(len(oof_df)),
        "oof_pmf_rows_by_stat": {
            s: int((oof_df["stat"] == s).sum()) for s in stats
        },
        "n_model_oof_rows": n_model_oof,
        "n_prior_only_rows": n_prior_only,
        "n_failed_rows": n_failed,
        "n_calibration_eligible_rows": n_cal_elig,
        "duplicate_key_count": int(dup_count),
        "invalid_pmf_count": invalid,
        "max_pmf_sum_error": max_err,
        "is_calibrated_all_false": True,
        "pmf_source_correct": bool(
            (oof_df["pmf_source"] == cfg["pmf_source"]).all()
        ),
        "forbidden_feature_check": "PASS",
        "target_leakage_check": "PASS",
        "same_day_leakage_check": "PASS",
        "low_minutes_adjustment_count": low_adj,
        "model_feature_count": len(model_cols),
    }
    audit_out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nSaved audit: {audit_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 5 OOF Generation Complete")
    print(f"  Elapsed:             {elapsed:.1f}s")
    print(f"  Folds:               {len(folds)}")
    print(f"  model_oof rows:      {n_model_oof:,}")
    print(f"  prior_only rows:     {n_prior_only:,}")
    print(f"  calibration_eligible:{n_cal_elig:,}")
    print(f"  OOF PMF rows:        {len(oof_df):,}")
    print(f"  Max sum error:       {max_err:.2e}")
    print(f"  Low-min adjustments: {low_adj:,}")
    print("=" * 70)


def _build_wide_oof_table(oof_df: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Pivot OOF long table to wide (one row per player × game)."""
    id_cols = ["game_id", "game_date", "season", "player_id", "player_name",
               "team_id", "team_abbreviation", "opponent_team_id",
               "actual_minutes", "fold_id", "fold_validation_start_date"]
    metric_cols = ["pmf_mean", "p0", "p_ge_1", "p_ge_5", "stat_mean",
                   "actual_outcome", "calibration_eligible"]

    available_id = [c for c in id_cols if c in oof_df.columns]
    # Ensure player_id and game_id are included (they're already in id_cols; avoid dups)
    for col in ("player_id", "game_id"):
        if col not in available_id:
            available_id.append(col)
    id_df = (oof_df[available_id]
             .drop_duplicates(subset=["player_id", "game_id"]))

    for stat in stats:
        sub = oof_df[oof_df["stat"] == stat]
        if sub.empty:
            continue
        metrics = {c: f"{stat}_{c}" for c in metric_cols if c in sub.columns}
        sub_piv = sub[["player_id", "game_id"] + list(metrics.keys())].rename(
            columns=metrics
        )
        id_df = id_df.merge(sub_piv, on=["player_id", "game_id"], how="left")

    return id_df.reset_index(drop=True)


if __name__ == "__main__":
    app()
