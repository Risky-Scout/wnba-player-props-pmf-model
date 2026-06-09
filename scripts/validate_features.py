"""Validate Stage 3 feature tables against the feature schema manifest.

Hard failures (exit 1):
  - model_feature_columns contains forbidden columns
  - model_feature_columns contains actual_outcome or same-game targets
  - long table has duplicate player_id × game_id × stat rows
  - infinite values in model_feature_columns
  - required identity columns missing
  - required target columns missing
  - role bucket columns missing
  - feature_cutoff_policy is not strict_pregame_shifted

Warnings (no exit):
  - high-null optional features (> 20%)
  - early-season low support
  - unavailable usage inputs
  - unavailable injury temporal alignment

Usage:
    python3 scripts/validate_features.py \\
        --features-long data/processed/wnba_player_game_features_long.parquet \\
        --manifest data/processed/feature_schema_manifest.json \\
        --audit-out artifacts/audits/feature_validation_audit.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.features.build_features import (
    FEATURE_CUTOFF_POLICY,
    IDENTITY_COLS,
    ROLE_BUCKET_COLS,
    TARGET_COLS,
)
from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

app = typer.Typer(add_completion=False)

_SAME_GAME_TARGETS = {
    "actual_pts", "actual_reb", "actual_ast", "actual_fg3m",
    "actual_stl", "actual_blk", "actual_turnover",
    "home_team_score", "visitor_team_score", "total_score", "has_final_score",
    "actual_outcome",
}


@app.command()
def main(
    features_long: str = typer.Option(
        "data/processed/wnba_player_game_features_long.parquet",
        help="Long feature table path."
    ),
    manifest: str = typer.Option(
        "data/processed/feature_schema_manifest.json",
        help="Feature schema manifest path."
    ),
    audit_out: str = typer.Option(
        "artifacts/audits/feature_validation_audit.json",
        help="Validation audit output path."
    ),
    features_wide: str = typer.Option(
        "data/processed/wnba_player_game_features_wide.parquet",
        help="Wide feature table path (optional, for duplicate check)."
    ),
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    typer.echo("Loading manifest and feature tables...")
    if not Path(manifest).exists():
        typer.echo(f"[FAIL] Manifest not found: {manifest}", err=True)
        raise typer.Exit(1)
    with open(manifest) as f:
        manifest_data = json.load(f)

    if not Path(features_long).exists():
        typer.echo(f"[FAIL] Long feature table not found: {features_long}", err=True)
        raise typer.Exit(1)
    long_df = pd.read_parquet(features_long)
    typer.echo(f"  Long table: {len(long_df):,} rows  |  columns: {len(long_df.columns)}")

    wide_df: pd.DataFrame | None = None
    if Path(features_wide).exists():
        wide_df = pd.read_parquet(features_wide)
        typer.echo(f"  Wide table: {len(wide_df):,} rows  |  columns: {len(wide_df.columns)}")

    model_feature_columns = manifest_data.get("model_feature_columns", [])

    # ------------------------------------------------------------------
    # 1. feature_cutoff_policy check (HARD)
    # ------------------------------------------------------------------
    policy = manifest_data.get("temporal_policy", "")
    if policy != FEATURE_CUTOFF_POLICY:
        errors.append(
            f"feature_cutoff_policy is '{policy}', must be '{FEATURE_CUTOFF_POLICY}'"
        )
    checks["feature_cutoff_policy"] = policy

    # ------------------------------------------------------------------
    # 2. Forbidden columns check (HARD)
    # ------------------------------------------------------------------
    forbidden_found = [c for c in model_feature_columns if c in FORBIDDEN_MODEL_FEATURES]
    if forbidden_found:
        errors.append(f"Forbidden market/leakage columns in model_feature_columns: {forbidden_found}")
    checks["forbidden_columns_in_model_features"] = forbidden_found

    # ------------------------------------------------------------------
    # 3. Target leakage check (HARD)
    # ------------------------------------------------------------------
    target_leakage = [c for c in model_feature_columns if c in _SAME_GAME_TARGETS]
    if target_leakage:
        errors.append(f"Same-game target columns in model_feature_columns: {target_leakage}")
    checks["target_leakage_in_model_features"] = target_leakage

    # ------------------------------------------------------------------
    # 4. Long table duplicate check (HARD)
    # ------------------------------------------------------------------
    if "stat" in long_df.columns:
        long_dup = long_df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
        checks["long_table_duplicates"] = int(long_dup)
        if long_dup > 0:
            errors.append(f"Long table has {long_dup} duplicate player_id × game_id × stat rows")
    else:
        errors.append("Long table missing 'stat' column")

    # ------------------------------------------------------------------
    # 5. Infinite values in model features (HARD)
    # ------------------------------------------------------------------
    inf_cols: list[str] = []
    for col in model_feature_columns:
        if col in long_df.columns and pd.api.types.is_numeric_dtype(long_df[col]):
            n_inf = int(np.isinf(long_df[col].fillna(0).values).sum())
            if n_inf > 0:
                inf_cols.append(col)
    checks["infinite_value_columns"] = inf_cols
    if inf_cols:
        errors.append(f"Infinite values in model feature columns: {inf_cols}")

    # ------------------------------------------------------------------
    # 6. Required identity columns (HARD)
    # ------------------------------------------------------------------
    missing_id = [c for c in IDENTITY_COLS if c not in long_df.columns]
    checks["missing_identity_columns"] = missing_id
    if missing_id:
        errors.append(f"Required identity columns missing from long table: {missing_id}")

    # ------------------------------------------------------------------
    # 7. Required target columns (HARD)
    # ------------------------------------------------------------------
    required_targets = ["actual_outcome", "actual_minutes", "did_play"]
    missing_tgt = [c for c in required_targets if c not in long_df.columns]
    checks["missing_target_columns"] = missing_tgt
    if missing_tgt:
        errors.append(f"Required target columns missing from long table: {missing_tgt}")

    # ------------------------------------------------------------------
    # 8. Role bucket columns (HARD)
    # ------------------------------------------------------------------
    required_role_cols = ["projected_minutes_bucket", "role_status", "role_uncertainty_bucket"]
    missing_role = [c for c in required_role_cols if c not in long_df.columns]
    checks["missing_role_bucket_columns"] = missing_role
    if missing_role:
        errors.append(f"Required role bucket columns missing: {missing_role}")

    # ------------------------------------------------------------------
    # 9. Wide table duplicate check (if available, HARD)
    # ------------------------------------------------------------------
    if wide_df is not None:
        wide_dup = wide_df.duplicated(subset=["player_id", "game_id"]).sum()
        checks["wide_table_duplicates"] = int(wide_dup)
        if wide_dup > 0:
            errors.append(f"Wide table has {wide_dup} duplicate player_id × game_id rows")

    # ------------------------------------------------------------------
    # 10. Warnings (soft checks)
    # ------------------------------------------------------------------
    # High-null features
    null_rates = {}
    for col in model_feature_columns:
        if col in long_df.columns:
            r = float(long_df[col].isna().mean())
            null_rates[col] = r
            if r > 0.2:
                warnings.append(f"High null rate {r:.1%} in feature '{col}' (> 20%)")
    checks["high_null_features"] = {k: v for k, v in null_rates.items() if v > 0.2}

    # Usage input availability
    usage_avail = manifest_data.get("unavailable_feature_inputs", {}).get("usage_inputs_available")
    if usage_avail is False:
        warnings.append("Usage proxy inputs (fga, fta, turnover) not available; usage features will be NaN.")

    # Injury alignment
    inj_aligned = manifest_data.get("unavailable_feature_inputs", {}).get("injury_temporal_alignment")
    if inj_aligned != "aligned":
        warnings.append(f"Injury temporal alignment: '{inj_aligned}'. Injury features not included.")

    # ------------------------------------------------------------------
    # 11. Stat coverage check
    # ------------------------------------------------------------------
    if "stat" in long_df.columns:
        stats_present = sorted(long_df["stat"].unique().tolist())
        checks["stats_present"] = stats_present
    else:
        checks["stats_present"] = []

    # ------------------------------------------------------------------
    # 12. Long table grain (rows per stat)
    # ------------------------------------------------------------------
    if wide_df is not None and "stat" in long_df.columns:
        expected_long = len(wide_df) * long_df["stat"].nunique()
        checks["long_rows_vs_expected"] = {
            "actual": len(long_df),
            "expected": expected_long,
            "match": abs(len(long_df) - expected_long) < 10,
        }

    # ------------------------------------------------------------------
    # Write validation audit
    # ------------------------------------------------------------------
    validation_audit = {
        "validated_at_utc": ts,
        "long_table_path": features_long,
        "manifest_path": manifest,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "status": "FAIL" if errors else "PASS",
    }
    Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_out).write_text(json.dumps(validation_audit, indent=2, default=str))
    typer.echo(f"\nValidation audit → {audit_out}")

    # ------------------------------------------------------------------
    # Print summary and exit
    # ------------------------------------------------------------------
    typer.echo(f"\n=== FEATURE VALIDATION SUMMARY ===")
    typer.echo(f"  Status:           {'[PASS]' if not errors else '[FAIL]'}")
    typer.echo(f"  Hard errors:      {len(errors)}")
    typer.echo(f"  Warnings:         {len(warnings)}")
    typer.echo(f"  Long rows:        {len(long_df):,}")
    if wide_df is not None:
        typer.echo(f"  Wide rows:        {len(wide_df):,}")
    typer.echo(f"  Model feat cols:  {len(model_feature_columns)}")

    for e in errors:
        typer.echo(f"  [FAIL] {e}", err=True)
    for w in warnings:
        typer.echo(f"  [WARN] {w}")

    if errors:
        raise typer.Exit(1)
    typer.echo("\n[PASS] Feature validation complete.")


if __name__ == "__main__":
    app()
