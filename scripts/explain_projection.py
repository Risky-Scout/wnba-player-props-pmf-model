"""Explain what drives a specific player's projection.

Outputs a plain-English driver explanation with top feature contributions,
minutes change flags, and risk indicators.

Usage:
    # Explain a specific player + stat
    python scripts/explain_projection.py \
        --player-id 341 \
        --stat pts \
        --game-date 2026-06-16

    # Explain all stats for a player
    python scripts/explain_projection.py \
        --player-id 341 \
        --game-date 2026-06-16

    # Explain all players on a team
    python scripts/explain_projection.py \
        --team PHX \
        --game-date 2026-06-16 \
        --out artifacts/explanations/PHX_2026-06-16.json
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.evaluation.explain import build_explanations

app = typer.Typer(add_completion=False)


@app.command()
def main(
    player_id: int | None = typer.Option(None, help="BDL player_id to explain."),
    team: str | None = typer.Option(None, help="Team abbreviation to explain all players."),
    stat: str | None = typer.Option(None, help="Specific stat (pts/reb/ast/fg3m/stl/blk/turnover). Default: all."),
    game_date: str | None = typer.Option(None, help="Game date YYYY-MM-DD (default: tomorrow)."),
    features_wide: str = typer.Option("data/processed/wnba_player_game_features_wide.parquet"),
    pmfs_path: str | None = typer.Option(None, help="PMF parquet. Auto-detected from deliveries/ if not set."),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline"),
    out: str | None = typer.Option(None, help="Output JSON path. Prints to console if not set."),
) -> None:
    """Explain what is driving a player's projection."""
    target = game_date or (date.today() + timedelta(days=1)).isoformat()

    features = pd.read_parquet(features_wide)

    # Load PMFs — look in deliveries/next_game/ by default
    if pmfs_path:
        pmfs = pd.read_parquet(pmfs_path)
    else:
        candidates = [
            Path(f"deliveries/next_game/player_projections_{target}.parquet"),
            Path(f"deliveries/today/full_pmfs_wide.parquet"),
        ]
        pmfs = None
        for c in candidates:
            if c.exists():
                pmfs = pd.read_parquet(c)
                break
        if pmfs is None:
            typer.echo(f"[ERROR] No PMF file found. Run predict_today.py or build_next_game_slate.py first.")
            raise typer.Exit(1)

    # Filter features to target players
    feat_filtered = features.copy()
    if player_id:
        feat_filtered = feat_filtered[feat_filtered["player_id"] == player_id]
    if team:
        feat_filtered = feat_filtered[feat_filtered["team_abbreviation"] == team.upper()]
    if feat_filtered.empty:
        typer.echo(f"[WARN] No matching players found in feature table")
        raise typer.Exit(1)

    # Filter PMFs to matching players
    pids = feat_filtered["player_id"].unique().tolist()
    pmfs_filtered = pmfs[pmfs["player_id"].isin(pids)]
    stats = [stat] if stat else ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    explanations = build_explanations(
        features=feat_filtered,
        pmfs=pmfs_filtered,
        model_dir=model_dir,
        stats=stats,
    )

    if not explanations:
        typer.echo("[WARN] No explanations generated — check player_id and that PMFs exist")
        raise typer.Exit(1)

    # Print formatted output
    for exp in explanations:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {exp['player_name']} | {exp['stat'].upper()} | {target}")
        typer.echo(f"{'='*60}")
        typer.echo(f"  Minutes: {exp['projected_minutes']} min (L5 avg: {exp['minutes_l5_avg']} min)")
        if exp["minutes_change_flag"]:
            typer.echo(f"  ⚠ Minutes change: {exp['minutes_change_vs_l5']:+.1f} min vs. L5 average")
        typer.echo(f"  Projected {exp['stat']}: {exp['projected_mean']:.1f}")
        if exp.get("stat_l5_avg"):
            typer.echo(f"  L5 avg {exp['stat']}: {exp['stat_l5_avg']:.1f}")
        typer.echo(f"  Role: {exp['role_bucket']} | DNP risk: {exp['dnp_risk']} | Injury: {exp['injury_flag']}")
        typer.echo(f"\n  Minutes narrative:")
        typer.echo(f"    {exp['minutes_narrative']}")
        typer.echo(f"\n  {exp['stat'].upper()} narrative:")
        typer.echo(f"    {exp['stat_narrative']}")
        if exp["top_minutes_drivers"]:
            typer.echo(f"\n  Top minutes drivers:")
            for d in exp["top_minutes_drivers"][:3]:
                typer.echo(f"    • {d['label']}: {d['value']:.2f} ({d['direction']})")

    # Write JSON if requested
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(explanations, indent=2))
        typer.echo(f"\nExplanations written → {out}")


if __name__ == "__main__":
    app()
