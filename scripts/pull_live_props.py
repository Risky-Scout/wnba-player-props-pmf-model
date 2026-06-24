"""Pull live player props from BDL for active WNBA games (blueprint §4.3).

Fetches from GET /wnba/v1/odds/player_props?game_id={id} and writes
data/live/bdl_props/game_{id}_live_props.json for each active game.

Usage:
    python scripts/pull_live_props.py \\
        --active-games artifacts/live/active_games.json \\
        --out-dir data/live/bdl_props
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    active_games: str = typer.Option(
        "artifacts/live/active_games.json",
        "--active-games",
    ),
    out_dir: str = typer.Option("data/live/bdl_props", "--out-dir"),
    game_ids: str = typer.Option("", "--game-ids", help="Comma-separated game IDs (overrides active_games file)."),
) -> None:
    """Pull live BDL player props for all active games."""
    api_key = os.environ.get("BDL_API_KEY", "")
    if not api_key:
        typer.echo("[WARN] BDL_API_KEY not set — skipping live props fetch.", err=True)
        raise typer.Exit(0)

    try:
        import requests
    except ImportError:
        typer.echo("[ERROR] requests not installed.", err=True)
        raise typer.Exit(1)

    # Resolve game IDs
    gids: list[int] = []
    if game_ids:
        gids = [int(g.strip()) for g in game_ids.split(",") if g.strip()]
    elif Path(active_games).exists():
        data = json.loads(Path(active_games).read_text())
        if isinstance(data, list):
            gids = [int(g.get("id") or g.get("game_id") or 0) for g in data if g]
        elif isinstance(data, dict):
            gids = [int(g.get("id") or g.get("game_id") or 0) for g in data.get("data", [])]

    gids = [g for g in gids if g]
    if not gids:
        typer.echo("No active game IDs found.")
        raise typer.Exit(0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}

    for gid in gids:
        typer.echo(f"Fetching live props for game {gid} ...")
        try:
            resp = requests.get(
                "https://api.balldontlie.io/wnba/v1/odds/player_props",
                params={"game_id": gid, "per_page": 200},
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("data", data) if isinstance(data, dict) else data

            out_file = out / f"game_{gid}_live_props.json"
            payload = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "game_id": gid,
                "data": raw,
            }
            out_file.write_text(json.dumps(payload, indent=2, default=str))
            typer.echo(f"  → {len(raw)} props → {out_file}")
            time.sleep(0.5)
        except Exception as exc:
            typer.echo(f"  [WARN] game {gid} failed: {exc}", err=True)


if __name__ == "__main__":
    app()
