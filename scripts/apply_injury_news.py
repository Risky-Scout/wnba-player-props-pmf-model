"""Inject injury / lineup news and recompute affected player projections.

A thin convenience wrapper around override_projections.py for the common
single-player injury update workflow. Updates the live next-game slate in
place and prints a before/after summary.

Status mapping to minutes impact:
  out          → minutes = 0, redistribute to teammates
  doubtful     → minutes_multiplier = 0.40 (likely out)
  questionable → minutes_multiplier = 0.75 (may be limited)
  limited      → minutes_cap applied (or multiplier = 0.65)
  active       → no change (clears any prior "out" flags downstream)

Usage:
    python scripts/apply_injury_news.py \\
        --player-id 123 \\
        --status out \\
        --game-date 2026-06-18

    python scripts/apply_injury_news.py \\
        --player-id 123 \\
        --status limited \\
        --minutes-cap 22 \\
        --game-date 2026-06-18
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

# Multipliers for each status (applied when no explicit minutes-cap is given)
_STATUS_MULTIPLIERS = {
    "doubtful": 0.40,
    "questionable": 0.75,
    "limited": 0.65,
    "active": 1.0,
}

# Default slate location
_DEFAULT_SLATE = "deliveries/next_game/full_pmfs_wide.parquet"
_DEFAULT_FEATURES = "data/processed/wnba_player_game_features_wide.parquet"
_DEFAULT_MODEL_DIR = "artifacts/models/stage4_baseline"
_DEFAULT_CAL_DIR = "artifacts/models/calibration"


@app.command()
def main(
    player_id: int = typer.Option(..., "--player-id", help="BDL player ID to update."),
    status: str = typer.Option(
        ...,
        "--status",
        help="Injury status: out | doubtful | questionable | limited | active",
    ),
    minutes_cap: float | None = typer.Option(
        None,
        "--minutes-cap",
        help="Hard cap on projected minutes (used with 'limited' status).",
    ),
    game_date: str | None = typer.Option(None, "--game-date", help="ISO game date (YYYY-MM-DD)."),
    slate: str = typer.Option(_DEFAULT_SLATE, "--slate", help="PMF slate parquet to update."),
    features_wide: str = typer.Option(_DEFAULT_FEATURES),
    model_dir: str = typer.Option(_DEFAULT_MODEL_DIR),
    cal_dir: str = typer.Option(_DEFAULT_CAL_DIR),
    raw_props: str | None = typer.Option(None, "--raw-props"),
    in_place: bool = typer.Option(
        True,
        "--in-place/--no-in-place",
        help="Overwrite the slate in place (default). Use --no-in-place to write to deliveries/overrides/.",
    ),
) -> None:
    """Apply a single-player injury update and recompute projections."""
    status_lower = status.lower()

    valid_statuses = {"out", "doubtful", "questionable", "limited", "active"}
    if status_lower not in valid_statuses:
        typer.echo(
            f"[ERROR] Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
            err=True,
        )
        raise typer.Exit(1)

    # Build override dict for this player
    override: dict = {"status": status_lower}

    if status_lower == "out":
        pass  # handled entirely by override_projections.py redistribution logic
    elif minutes_cap is not None:
        override["minutes_cap"] = minutes_cap
        override["status"] = "limited"
    elif status_lower in _STATUS_MULTIPLIERS:
        override["minutes_multiplier"] = _STATUS_MULTIPLIERS[status_lower]

    overrides_json = json.dumps({str(player_id): override})

    # Determine output directory
    if in_place:
        out_dir = str(Path(slate).parent)
    else:
        out_dir = "deliveries/overrides"

    typer.echo(f"Applying: player {player_id} → {status_lower.upper()}")
    if minutes_cap:
        typer.echo(f"  minutes_cap: {minutes_cap}")
    elif "minutes_multiplier" in override:
        typer.echo(f"  minutes_multiplier: {override['minutes_multiplier']}")

    # Delegate to override_projections.py
    cmd = [
        sys.executable, "scripts/override_projections.py",
        "--slate", slate,
        "--features-wide", features_wide,
        "--overrides", overrides_json,
        "--out-dir", out_dir,
        "--model-dir", model_dir,
        "--cal-dir", cal_dir,
    ]
    if game_date:
        cmd += ["--game-date", game_date]
    if raw_props:
        cmd += ["--raw-props", raw_props]

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        typer.echo(f"[ERROR] override_projections.py failed (exit {result.returncode})", err=True)
        raise typer.Exit(result.returncode)

    if in_place:
        typer.echo(f"\nSlate updated in place: {slate}")
    else:
        typer.echo(f"\nRevised slate written to: {out_dir}/full_pmfs_wide.parquet")


if __name__ == "__main__":
    app()
