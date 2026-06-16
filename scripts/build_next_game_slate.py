"""Build the next-game player slate for projections.

Identifies upcoming scheduled games (default: next calendar day in ET),
cross-references the active player roster from the feature table, and
produces a filtered feature DataFrame ready for inference.

This enables the model to predict one day BEFORE gameday.

Usage:
    # Predict for tomorrow's games (default)
    python scripts/build_next_game_slate.py \
        --out-dir deliveries/next_game

    # Predict for a specific date
    python scripts/build_next_game_slate.py \
        --game-date 2026-06-16 \
        --out-dir deliveries/next_game

    # Print slate summary only (no prediction)
    python scripts/build_next_game_slate.py --dry-run
"""
from __future__ import annotations

import json
from datetime import date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import typer

ET = ZoneInfo("America/New_York")

app = typer.Typer(add_completion=False)


def _tomorrow_et() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def get_scheduled_games(games_path: str | Path, game_date: str) -> pd.DataFrame:
    """Return all scheduled games on game_date."""
    games = pd.read_parquet(games_path)
    games["game_date_str"] = pd.to_datetime(games["game_date"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    scheduled = games[
        (games["game_date_str"] == game_date) &
        (games["status_normalized"].isin(["scheduled", "unknown"]))
    ].copy()
    return scheduled


def get_active_players_for_slate(
    games: pd.DataFrame,
    features_wide: str | Path,
    injuries: str | Path | None = None,
) -> pd.DataFrame:
    """Return the most recent feature row for each player on a team playing today.

    Strategy: for each team in today's games, take the latest feature row per player
    (most recent game they played). This represents their current form going into
    the next game.
    """
    feat = pd.read_parquet(features_wide)
    feat["game_date"] = pd.to_datetime(feat["game_date"], utc=True, errors="coerce")

    team_ids = set(games["home_team_id"].tolist() + games["visitor_team_id"].tolist())

    # For each player on a playing team, get their most recent feature row
    active = feat[feat["team_id"].isin(team_ids)].copy()
    active = active.sort_values("game_date").groupby("player_id").last().reset_index()

    # Attach game context (game_id, opponent, home_away) for each player
    game_lookup = []
    for _, g in games.iterrows():
        for team_id, opp_id, home_away, opp_abbr in [
            (g["home_team_id"], g["visitor_team_id"], "home", g["visitor_team_abbreviation"]),
            (g["visitor_team_id"], g["home_team_id"], "away", g["home_team_abbreviation"]),
        ]:
            game_lookup.append({
                "team_id": team_id,
                "next_game_id": g["game_id"],
                "next_game_date": g["game_date_str"],
                "opponent_team_id": opp_id,
                "opponent_abbreviation": opp_abbr,
                "home_away": home_away,
            })
    game_ctx = pd.DataFrame(game_lookup)

    active = active.merge(game_ctx, on="team_id", how="inner")

    # Overlay game_date and game_id with the upcoming game values
    active["game_id"] = active["next_game_id"]
    active["game_date"] = active["next_game_date"]

    # Apply injury report: flag players on injury list
    active["injury_flag"] = False
    active["dnp_risk"] = "low"
    if injuries and Path(injuries).exists():
        inj = pd.read_parquet(injuries)
        if "player_id" in inj.columns and "status" in inj.columns:
            out_statuses = {"out", "doubtful"}
            questionable = {"questionable"}
            out_ids = set(inj[inj["status"].str.lower().isin(out_statuses)]["player_id"].tolist())
            q_ids = set(inj[inj["status"].str.lower().isin(questionable)]["player_id"].tolist())
            active.loc[active["player_id"].isin(out_ids | q_ids), "injury_flag"] = True
            active.loc[active["player_id"].isin(out_ids), "dnp_risk"] = "high"
            active.loc[active["player_id"].isin(q_ids), "dnp_risk"] = "moderate"

    # Players with high recent DNP rate are flagged as elevated risk
    if "player_zero_minute_rate_l5" in active.columns:
        high_dnp_mask = active["player_zero_minute_rate_l5"] > 0.3
        active.loc[high_dnp_mask & (active["dnp_risk"] == "low"), "dnp_risk"] = "moderate"

    active["override_applied"] = False
    active["override_source"] = None
    return active


@app.command()
def main(
    game_date: str | None = typer.Option(None, help="Target game date YYYY-MM-DD (default: tomorrow ET)."),
    games_path: str = typer.Option("data/processed/wnba_games.parquet"),
    features_wide: str = typer.Option("data/processed/wnba_player_game_features_wide.parquet"),
    injuries_path: str | None = typer.Option("data/processed/wnba_injuries.parquet"),
    out_dir: str = typer.Option("deliveries/next_game"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print slate summary, do not write files."),
) -> None:
    """Build the next-game feature slate for projections."""
    target = game_date or _tomorrow_et()
    typer.echo(f"Building next-game slate for: {target}")

    games = get_scheduled_games(games_path, target)
    if games.empty:
        typer.echo(f"[WARN] No scheduled games found for {target}")
        raise typer.Exit(0)

    typer.echo(f"Found {len(games)} games:")
    for _, g in games.iterrows():
        typer.echo(f"  {g['home_team_abbreviation']} vs {g['visitor_team_abbreviation']}")

    slate = get_active_players_for_slate(games, features_wide, injuries_path)
    typer.echo(f"\nActive players in slate: {len(slate)}")
    typer.echo(f"Teams: {sorted(slate['team_abbreviation'].unique().tolist())}")

    injury_flagged = slate[slate["injury_flag"]]["player_name"].tolist() if "player_name" in slate.columns else []
    if injury_flagged:
        typer.echo(f"Injury-flagged players: {injury_flagged}")

    high_dnp = slate[slate["dnp_risk"] == "high"]["player_name"].tolist() if "player_name" in slate.columns else []
    if high_dnp:
        typer.echo(f"High DNP risk players: {high_dnp}")

    if dry_run:
        typer.echo("\n[DRY RUN] No files written.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slate_path = out / f"slate_{target}.parquet"
    slate.to_parquet(slate_path, index=False)
    typer.echo(f"\nSlate written → {slate_path}")

    # Write slate manifest JSON
    manifest = {
        "game_date": target,
        "games": [
            {
                "game_id": int(g["game_id"]),
                "home_team": g["home_team_abbreviation"],
                "away_team": g["visitor_team_abbreviation"],
            }
            for _, g in games.iterrows()
        ],
        "total_players": int(len(slate)),
        "injury_flagged": injury_flagged[:20],
        "high_dnp_risk": high_dnp[:10],
    }
    (out / f"slate_manifest_{target}.json").write_text(json.dumps(manifest, indent=2))
    typer.echo(f"Manifest written → {out}/slate_manifest_{target}.json")


if __name__ == "__main__":
    app()
