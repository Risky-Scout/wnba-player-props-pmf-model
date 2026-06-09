"""Ingestion audit script.

Reads the raw parquet files written by pull_bdl_history.py and writes a
JSON audit report covering:
  - row counts per table
  - season breakdown
  - date range
  - missing / null column counts
  - duplicate primary-key counts
  - minutes flag breakdown (null, empty, non_playing, parse_error)
  - zero-minute row count by cause
  - required-column presence check

Usage:
    python scripts/audit_ingestion.py --data-dir data/raw/bdl --out audit_ingestion.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)

_REQUIRED_GAME_COLS = {
    "game_id", "game_date", "season", "status",
    "home_team_id", "away_team_id",
}

_REQUIRED_STAT_COLS = {
    "player_id", "game_id", "game_date", "season", "team_id",
    "minutes", "minutes_raw", "minutes_flag",
    "pts", "reb", "ast", "fg3m", "stl", "blk",
}


def _null_counts(df: pd.DataFrame) -> dict[str, int]:
    nulls = df.isnull().sum()
    return {col: int(count) for col, count in nulls.items() if count > 0}


def _dup_count(df: pd.DataFrame, keys: list[str]) -> int:
    valid_keys = [k for k in keys if k in df.columns]
    if not valid_keys:
        return 0
    return int(df.duplicated(subset=valid_keys).sum())


def audit_games(df: pd.DataFrame) -> dict:
    missing_cols = _REQUIRED_GAME_COLS - set(df.columns)
    return {
        "rows": len(df),
        "seasons": sorted(int(s) for s in df["season"].dropna().unique()),
        "date_min": str(df["game_date"].min()) if "game_date" in df.columns else None,
        "date_max": str(df["game_date"].max()) if "game_date" in df.columns else None,
        "null_counts": _null_counts(df),
        "duplicate_game_ids": _dup_count(df, ["game_id"]),
        "missing_required_columns": sorted(missing_cols),
        "schema_ok": len(missing_cols) == 0,
    }


def audit_player_stats(df: pd.DataFrame) -> dict:
    missing_cols = _REQUIRED_STAT_COLS - set(df.columns)

    # Minutes flag breakdown
    minutes_flag_counts: dict[str, int] = {
        "clean_no_flag": int((df["minutes_flag"].isna()).sum()),
        "non_playing": int((df["minutes_flag"] == "non_playing").sum()),
        "null_value": int((df["minutes_flag"] == "null").sum()),
        "empty_string": int((df["minutes_flag"] == "empty").sum()),
        "parse_error": int((df["minutes_flag"] == "parse_error").sum()),
    }

    # Zero-minute breakdown
    zero_mask = df["minutes"] == 0.0
    zero_by_flag = {
        "zero_min_non_playing_flag": int(
            (zero_mask & (df["minutes_flag"] == "non_playing")).sum()
        ),
        "zero_min_clean_raw_zero": int(
            (zero_mask & df["minutes_flag"].isna() & (df["minutes_raw"] == "0")).sum()
        ),
        "zero_min_null_flag": int(
            (zero_mask & (df["minutes_flag"] == "null")).sum()
        ),
        "zero_min_empty_flag": int(
            (zero_mask & (df["minutes_flag"] == "empty")).sum()
        ),
        "zero_min_parse_error": int(
            (zero_mask & (df["minutes_flag"] == "parse_error")).sum()
        ),
        "zero_min_total": int(zero_mask.sum()),
    }

    # Per-season rows
    rows_by_season = df.groupby("season").size().to_dict()
    rows_by_season = {int(k): int(v) for k, v in rows_by_season.items()}

    return {
        "rows": len(df),
        "unique_players": int(df["player_id"].nunique()),
        "unique_games": int(df["game_id"].nunique()),
        "seasons": sorted(int(s) for s in df["season"].dropna().unique()),
        "rows_by_season": rows_by_season,
        "date_min": str(df["game_date"].min()) if "game_date" in df.columns else None,
        "date_max": str(df["game_date"].max()) if "game_date" in df.columns else None,
        "null_counts": _null_counts(df),
        "duplicate_player_game_pairs": _dup_count(df, ["player_id", "game_id"]),
        "minutes_flag_breakdown": minutes_flag_counts,
        "zero_minute_breakdown": zero_by_flag,
        "missing_required_columns": sorted(missing_cols),
        "schema_ok": len(missing_cols) == 0,
    }


@app.command()
def main(
    data_dir: str = typer.Option("data/raw/bdl", help="Directory containing raw parquet files."),
    out: str = typer.Option("audit_ingestion.json", help="Output JSON path."),
) -> None:
    data_path = Path(data_dir)
    games_path = data_path / "wnba_games.parquet"
    stats_path = data_path / "wnba_player_game_stats.parquet"

    errors: list[str] = []

    games_audit: dict = {}
    if games_path.exists():
        games_df = pd.read_parquet(games_path)
        games_audit = audit_games(games_df)
        typer.echo(f"Games: {games_audit['rows']} rows | seasons {games_audit['seasons']}")
    else:
        errors.append(f"Missing file: {games_path}")
        typer.echo(f"[WARN] {games_path} not found", err=True)

    stats_audit: dict = {}
    if stats_path.exists():
        stats_df = pd.read_parquet(stats_path)
        stats_audit = audit_player_stats(stats_df)
        typer.echo(
            f"Player stats: {stats_audit['rows']} rows | "
            f"{stats_audit['unique_players']} players | "
            f"{stats_audit['unique_games']} games | "
            f"seasons {stats_audit['seasons']}"
        )
        typer.echo(f"Minutes flags: {stats_audit['minutes_flag_breakdown']}")
        typer.echo(f"Zero-minute breakdown: {stats_audit['zero_minute_breakdown']}")
        if not stats_audit["schema_ok"]:
            errors.append(f"Missing columns in player_stats: {stats_audit['missing_required_columns']}")
    else:
        errors.append(f"Missing file: {stats_path}")
        typer.echo(f"[WARN] {stats_path} not found", err=True)

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "games": games_audit,
        "player_stats": stats_audit,
        "errors": errors,
        "status": "FAIL" if errors else "PASS",
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    typer.echo(f"\nAudit written → {out_path}")

    if errors:
        typer.echo(f"\n[FAIL] {len(errors)} error(s):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[PASS] Ingestion audit clean.")


if __name__ == "__main__":
    app()
