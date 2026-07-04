"""Find active WNBA games for live tracking.

Queries the BDL API for games scheduled for today and outputs:
  - has_games: true/false (GitHub Actions output)
  - game_ids: space-separated list of active game IDs
  - date: today's date string

Usage:
    python scripts/find_live_games.py [--out-dir artifacts/live]

GitHub Actions sets step outputs via:
    echo "has_games=true" >> $GITHUB_OUTPUT
    echo "game_ids=12345 67890" >> $GITHUB_OUTPUT
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
import pandas as pd

app = typer.Typer(add_completion=False)

# Explicit live/upcoming allowlists (kept for reference and is_live classification).
_LIVE_STATUSES = {
    "in_progress", "live", "halftime", "end_of_period",
    # BDL WNBA quarter-period strings after .lower().replace(" ","_"):
    "1st_qtr", "2nd_qtr", "3rd_qtr", "4th_qtr",
    "1_qtr", "2_qtr", "3_qtr", "4_qtr",
    "q1", "q2", "q3", "q4",
    "ot", "ot1", "ot2", "overtime", "1st_ot", "2nd_ot",
}
_UPCOMING_STATUSES = {"scheduled", "pregame"}
# Terminal statuses — anything NOT in this set (and not empty) is treated as active.
# Denylist is safer than allowlist because BDL may return novel quarter strings.
_TERMINAL_STATUSES = {"final", "final/ot", "final_ot", "canceled", "postponed", "tbd", "post", ""}


@app.command()
def main(
    out_dir: str = typer.Option("artifacts/live", help="Directory for output files."),
    check_date: str | None = typer.Option(
        None, "--date", help="Date to check (YYYY-MM-DD). Defaults to today."
    ),
    include_upcoming: bool = typer.Option(
        True, "--include-upcoming/--live-only",
        help="Include scheduled (upcoming) games in addition to live games.",
    ),
) -> None:
    """Discover active and upcoming WNBA games for live tracking."""
    from wnba_props_model.data.bdl_client import BDLClient, BDLAPIError  # noqa: PLC0415

    target_date = check_date or date.today().isoformat()
    # BDL uses UTC dates. Evening ET games (after 8 PM ET / midnight UTC) are stored
    # under the *next* UTC date. Always query both the given ET date and ET+1 so late
    # games are never missed regardless of when the workflow runs.
    next_date = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()
    dates_to_check = [target_date, next_date]
    typer.echo(f"Checking for WNBA games on {target_date} and {next_date} (UTC+1 boundary)...")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    game_ids: list[int] = []
    games_info: list[dict] = []
    seen_ids: set[int] = set()

    try:
        client = BDLClient()
        all_rows: list[dict] = []
        for qdate in dates_to_check:
            try:
                all_rows.extend(client.list_endpoint("games", {"dates": [qdate], "per_page": 50}))
            except Exception as exc:
                typer.echo(f"[WARN] BDL query for {qdate} failed: {exc}", err=True)

        for row in all_rows:
            status = str(row.get("status") or "").lower().replace(" ", "_")
            gid = row.get("id")
            if gid is None:
                continue
            is_live = status in _LIVE_STATUSES or status not in _TERMINAL_STATUSES
            is_upcoming = status in _UPCOMING_STATUSES
            if (is_live or (include_upcoming and is_upcoming)) and int(gid) not in seen_ids:
                typer.echo(f"  → game {gid}: status={repr(status)} is_live={is_live}")
                seen_ids.add(int(gid))
                game_ids.append(int(gid))
                home = row.get("home_team") or {}
                away = row.get("visitor_team") or {}
                games_info.append({
                    "game_id": int(gid),
                    "date": target_date,
                    "status": status,
                    "home_team": home.get("full_name") or home.get("abbreviation"),
                    "home_team_id": home.get("id"),
                    "away_team": away.get("full_name") or away.get("abbreviation"),
                    "away_team_id": away.get("id"),
                    "home_score": row.get("home_team_score"),
                    "away_score": row.get("visitor_team_score"),
                })
    except BDLAPIError as exc:
        typer.echo(f"[WARN] BDL API error: {exc}", err=True)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[WARN] Unexpected error: {exc}", err=True)

    has_games = len(game_ids) > 0
    game_ids_str = " ".join(str(g) for g in game_ids)

    # Write games info JSON
    info_path = out / "live_games.json"
    info_path.write_text(json.dumps({
        "date": target_date,
        "has_games": has_games,
        "n_games": len(game_ids),
        "game_ids": game_ids,
        "games": games_info,
    }, indent=2))
    typer.echo(f"Games info → {info_path}")

    # Set GitHub Actions outputs
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"has_games={str(has_games).lower()}\n")
            fh.write(f"game_ids={game_ids_str}\n")
            fh.write(f"date={target_date}\n")
    else:
        typer.echo(f"has_games={str(has_games).lower()}")
        typer.echo(f"game_ids={game_ids_str}")
        typer.echo(f"date={target_date}")

    # Also write active_game_ids.txt for workflow compatibility
    ids_txt_path = out / "active_game_ids.txt"
    ids_txt_path.write_text(",".join(str(g) for g in game_ids))

    if not has_games:
        typer.echo(f"No active/upcoming games found on {target_date}")
    else:
        typer.echo(f"Found {len(game_ids)} game(s): {game_ids}")


if __name__ == "__main__":
    app()
