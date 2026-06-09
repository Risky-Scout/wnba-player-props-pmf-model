"""Build leakage-safe baseline WNBA player-props feature tables.

Usage:
    python3 scripts/build_features.py \\
        --data-dir data/processed \\
        --out-wide data/processed/wnba_player_game_features_wide.parquet \\
        --out-long data/processed/wnba_player_game_features_long.parquet \\
        --manifest-out data/processed/feature_schema_manifest.json \\
        --audit-out artifacts/audits/feature_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.features.build_features import (
    STATS,
    build_feature_audit,
    build_feature_schema_manifest,
    build_long_table,
    build_wide_table,
    _derive_model_feature_columns,
)
from wnba_props_model.features.feature_contract import assert_no_forbidden_features

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_dir: str = typer.Option("data/processed", help="Directory with canonical parquet files."),
    out_wide: str = typer.Option(
        "data/processed/wnba_player_game_features_wide.parquet",
        help="Wide feature table output path."
    ),
    out_long: str = typer.Option(
        "data/processed/wnba_player_game_features_long.parquet",
        help="Long feature table output path."
    ),
    manifest_out: str = typer.Option(
        "data/processed/feature_schema_manifest.json",
        help="Feature schema manifest output path."
    ),
    audit_out: str = typer.Option(
        "artifacts/audits/feature_audit.json",
        help="Feature audit output path."
    ),
) -> None:
    data = Path(data_dir)

    # ------------------------------------------------------------------
    # Load canonical tables
    # ------------------------------------------------------------------
    typer.echo(f"Loading canonical tables from {data}")

    stats_path = data / "wnba_player_game_stats.parquet"
    games_path = data / "wnba_games.parquet"
    if not stats_path.exists():
        typer.echo(f"[FAIL] Required: {stats_path}", err=True)
        raise typer.Exit(1)
    if not games_path.exists():
        typer.echo(f"[FAIL] Required: {games_path}", err=True)
        raise typer.Exit(1)

    stats_df  = pd.read_parquet(stats_path)
    games_df  = pd.read_parquet(games_path)
    adv_path  = data / "wnba_player_advanced_stats.parquet"
    adv_df    = pd.read_parquet(adv_path) if adv_path.exists() else None
    inj_path  = data / "wnba_injuries.parquet"
    inj_df    = pd.read_parquet(inj_path) if inj_path.exists() else None

    typer.echo(
        f"  stats: {len(stats_df):,} rows  |  games: {len(games_df):,}  |  "
        f"adv: {len(adv_df) if adv_df is not None else 'n/a'}  |  "
        f"injuries: {len(inj_df) if inj_df is not None else 'n/a'}"
    )

    # ------------------------------------------------------------------
    # Build wide feature table
    # ------------------------------------------------------------------
    typer.echo("\nBuilding wide feature table (one row per player_id × game_id)...")
    wide_df, audit_notes = build_wide_table(
        stats_df=stats_df,
        games_df=games_df,
        adv_df=adv_df,
        injuries_df=inj_df,
    )
    typer.echo(f"  Wide rows: {len(wide_df):,}  |  columns: {len(wide_df.columns)}")

    # Verify no duplicate player×game
    dup_count = wide_df.duplicated(subset=["player_id", "game_id"]).sum()
    if dup_count > 0:
        typer.echo(f"[FAIL] Wide table has {dup_count} duplicate player_id × game_id rows!", err=True)
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Derive model feature columns (authoritative allow-list)
    # ------------------------------------------------------------------
    typer.echo("\nDeriving authoritative model_feature_columns...")
    model_feature_columns = _derive_model_feature_columns(wide_df)
    typer.echo(f"  model_feature_columns: {len(model_feature_columns)} features")

    # Hard leakage gate
    typer.echo("  Running leakage guard...")
    assert_no_forbidden_features(model_feature_columns)
    typer.echo("  [PASS] Leakage guard passed.")

    # ------------------------------------------------------------------
    # Build long feature table
    # ------------------------------------------------------------------
    typer.echo("\nBuilding long feature table (one row per player_id × game_id × stat)...")
    long_df = build_long_table(wide_df)
    typer.echo(f"  Long rows: {len(long_df):,}  |  stats: {sorted(long_df['stat'].unique().tolist())}")

    # Verify no duplicate player×game×stat
    long_dup = long_df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
    if long_dup > 0:
        typer.echo(f"[FAIL] Long table has {long_dup} duplicate rows!", err=True)
        raise typer.Exit(1)

    # Verify no infinite values in model features
    import numpy as np
    for col in model_feature_columns:
        if col in wide_df.columns and pd.api.types.is_numeric_dtype(wide_df[col]):
            inf_count = int(np.isinf(wide_df[col].values).sum())
            if inf_count > 0:
                typer.echo(f"[FAIL] Infinite values in model feature '{col}': {inf_count}", err=True)
                raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    typer.echo(f"\nWriting wide table → {out_wide}")
    Path(out_wide).parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_parquet(out_wide, index=False)

    typer.echo(f"Writing long table → {out_long}")
    Path(out_long).parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_long, index=False)

    # ------------------------------------------------------------------
    # Build and write manifest
    # ------------------------------------------------------------------
    source_tables = [
        str(stats_path), str(games_path),
        str(adv_path) if adv_path.exists() else "",
        str(inj_path) if inj_path.exists() else "",
    ]
    manifest = build_feature_schema_manifest(
        wide_df=wide_df,
        long_df=long_df,
        model_feature_columns=model_feature_columns,
        source_tables=[t for t in source_tables if t],
        wide_path=out_wide,
        long_path=out_long,
    )
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).write_text(json.dumps(manifest, indent=2, default=str))
    typer.echo(f"Feature manifest → {manifest_out}")

    # ------------------------------------------------------------------
    # Build and write audit
    # ------------------------------------------------------------------
    audit = build_feature_audit(
        wide_df=wide_df,
        long_df=long_df,
        model_feature_columns=model_feature_columns,
        audit_notes=audit_notes,
    )
    Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_out).write_text(json.dumps(audit, indent=2, default=str))
    typer.echo(f"Feature audit    → {audit_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    typer.echo("\n=== STAGE 3 SUMMARY ===")
    typer.echo(f"  Wide rows:              {len(wide_df):>8,}")
    typer.echo(f"  Long rows:              {len(long_df):>8,}")
    typer.echo(f"  Model feature cols:     {len(model_feature_columns):>8,}")
    typer.echo(f"  Numeric features:       {len(manifest['numeric_feature_columns']):>8,}")
    typer.echo(f"  Categorical features:   {len(manifest['categorical_feature_columns']):>8,}")
    typer.echo(f"  Forbidden check:        PASS")
    typer.echo(f"  Leakage guard:          PASS")
    typer.echo(f"  Duplicate check (wide): PASS (0)")
    typer.echo(f"  Duplicate check (long): PASS (0)")
    typer.echo(f"  Infinite value check:   PASS")
    typer.echo(f"  Usage inputs available: {audit_notes['usage_inputs_available']}")
    typer.echo(f"  Adv stats available:    {audit_notes['advanced_stats_available']}")
    typer.echo(f"  Injury alignment:       {audit_notes['injury_temporal_alignment']}")
    typer.echo(f"  Pace proxy method:      {audit_notes['pace_proxy_method']}")
    typer.echo("\n[PASS] Stage 3 feature build complete.")


if __name__ == "__main__":
    app()
