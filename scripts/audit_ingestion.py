"""Comprehensive ingestion audit.

Reads both raw (data/raw/bdl) and canonical (data/processed) tables and
writes a unified audit JSON covering:
  - row counts per table
  - season breakdown
  - date ranges
  - missing / null column counts
  - duplicate primary-key counts
  - minutes flag breakdown
  - games vs player-stats mismatch explanation
  - optional endpoint availability
  - schema check against required columns

Usage:
    python3 scripts/audit_ingestion.py \\
        --raw-dir data/raw/bdl \\
        --processed-dir data/processed \\
        --audit-out artifacts/audits/ingestion_audit.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from wnba_props_model.data.audit import (
    audit_games_vs_stats,
    audit_players_vs_stats,
    audit_raw_games,
    audit_raw_injuries,
    audit_raw_odds,
    audit_raw_player_props,
    audit_raw_player_stats,
    audit_raw_players,
    audit_raw_teams,
    audit_teams_vs_games,
)

app = typer.Typer(add_completion=False)

_TABLE_AUDIT_FNS = {
    "wnba_games": audit_raw_games,
    "wnba_player_game_stats": audit_raw_player_stats,
    "wnba_teams": audit_raw_teams,
    "wnba_players": audit_raw_players,
    "wnba_injuries": audit_raw_injuries,
    "wnba_odds": audit_raw_odds,
    "wnba_player_props": audit_raw_player_props,
}

_REQUIRED_TABLES = {"wnba_games", "wnba_player_game_stats"}


def _read(directory: Path, name: str) -> pd.DataFrame | None:
    p = directory / f"{name}.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  [WARN] could not read {p}: {exc}", err=True)
        return None


@app.command()
def main(
    raw_dir: str = typer.Option("data/raw/bdl", help="Raw parquet directory."),
    processed_dir: str = typer.Option("data/processed", help="Canonical parquet directory."),
    audit_out: str = typer.Option(
        "artifacts/audits/ingestion_audit.json", help="Output audit JSON path."
    ),
) -> None:
    raw = Path(raw_dir)
    proc = Path(processed_dir)
    errors: list[str] = []
    table_audits: dict[str, Any] = {}

    typer.echo("=== Raw table audits ===")
    dfs: dict[str, pd.DataFrame] = {}
    for name, fn in _TABLE_AUDIT_FNS.items():
        df = _read(raw, name)
        # Always recompute status_normalized for games so the audit reflects
        # the current normalization logic even if raw was written with old code.
        if df is not None and name == "wnba_games" and "status" in df.columns:
            from wnba_props_model.data.normalize import normalize_game_status
            df = df.copy()
            df["status_normalized"] = df["status"].apply(normalize_game_status)
        if df is None:
            status = "missing_required" if name in _REQUIRED_TABLES else "missing_optional"
            table_audits[name] = {"status": status, "source": "raw"}
            if name in _REQUIRED_TABLES:
                errors.append(f"Required raw table missing: {raw}/{name}.parquet")
                typer.echo(f"  [FAIL] {name}: not found", err=True)
            else:
                typer.echo(f"  [SKIP] {name}: not available")
            continue

        dfs[name] = df
        audit = fn(df)
        audit["status"] = "ok"
        audit["source"] = "raw"
        table_audits[name] = audit
        rows = audit.get("rows", 0)
        seasons = audit.get("seasons", [])
        typer.echo(f"  {name}: rows={rows:,}  seasons={seasons}")

    # Cross-table audits
    typer.echo("\n=== Cross-table checks ===")
    cross_checks: dict[str, Any] = {}

    games_df = dfs.get("wnba_games")
    stats_df = dfs.get("wnba_player_game_stats")
    teams_df = dfs.get("wnba_teams")
    players_df = dfs.get("wnba_players")

    if games_df is not None and stats_df is not None:
        gvs = audit_games_vs_stats(games_df, stats_df)
        cross_checks["games_vs_player_stats"] = gvs
        typer.echo(
            f"  Games: {gvs['total_game_rows']}  "
            f"with_stats: {gvs['games_with_player_stats']}  "
            f"without_stats: {gvs['games_without_player_stats']}"
        )
        if gvs.get("breakdown_by_status"):
            for status, info in gvs["breakdown_by_status"].items():
                typer.echo(
                    f"    status={status:15s} total={info['total']:4d}  "
                    f"with_stats={info['with_player_stats']:4d}  "
                    f"without_stats={info['without_player_stats']:4d}"
                )

    if players_df is not None and stats_df is not None:
        pvs = audit_players_vs_stats(players_df, stats_df)
        cross_checks["players_vs_stats"] = pvs
        typer.echo(
            f"  Players in stats missing from players table: "
            f"{pvs['players_in_stats_missing_from_players_table']}"
        )

    if teams_df is not None and games_df is not None:
        tvg = audit_teams_vs_games(teams_df, games_df)
        cross_checks["teams_vs_games"] = tvg
        typer.echo(
            f"  Team IDs in games missing from teams table: "
            f"{tvg['team_ids_in_games_missing_from_teams_table']}"
        )

    # Canonical table presence check
    typer.echo("\n=== Canonical table check ===")
    canonical_status: dict[str, str] = {}
    for name in _TABLE_AUDIT_FNS:
        p = proc / f"{name}.parquet"
        canonical_status[name] = "present" if p.exists() else "missing"
        typer.echo(f"  {name}: {canonical_status[name]}")

    # Build report
    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "tables": table_audits,
        "cross_checks": cross_checks,
        "canonical_table_status": canonical_status,
        "errors": errors,
        "status": "FAIL" if errors else "PASS",
    }

    out_path = Path(audit_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    typer.echo(f"\nIngestion audit → {out_path}")

    if errors:
        typer.echo(f"\n[FAIL] {len(errors)} error(s):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("[PASS] Ingestion audit complete.")


if __name__ == "__main__":
    app()
