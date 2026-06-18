"""Generate WNBATeamScorePMFGrid predictions for upcoming games.

Loads the fitted WNBATeamScoreModel and produces one WNBATeamScorePMFGrid
per game on the target date. Writes both parquet and JSON outputs.

Usage:
    python scripts/predict_game_totals.py \
        --games data/processed/wnba_games.parquet \
        --model-dir artifacts/models/game_totals \
        --game-date 2026-06-18 \
        --out-dir deliveries/next_game
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.team_score import WNBATeamScoreModel, WNBATeamScorePMFGrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


def _load_games_for_date(games_path: str, game_date: str) -> pd.DataFrame:
    """Return scheduled (not yet played) games for the target date."""
    games = pd.read_parquet(games_path)
    games["game_date_str"] = (
        pd.to_datetime(games["game_date"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    scheduled = games[games["game_date_str"] == game_date].copy()
    # Include games regardless of status — we want all games on that date
    return scheduled


def _get_pace_adj(games: pd.DataFrame, home_team: str, away_team: str) -> float:
    """Derive log pace adjustment from rolling defensive pace features."""
    rows = games[
        (games.get("home_team", games.get("home_team_abbreviation", pd.Series(dtype=str))) == home_team) |
        (games.get("away_team", games.get("visitor_team_abbreviation", pd.Series(dtype=str))) == away_team)
    ]
    if rows.empty:
        return 0.0
    home_pace_col = "home_opp_points_roll5"
    away_pace_col = "away_opp_points_roll5"
    if home_pace_col not in rows.columns or away_pace_col not in rows.columns:
        return 0.0
    league_avg = float(pd.concat([rows[home_pace_col], rows[away_pace_col]]).mean())
    if league_avg <= 0:
        return 0.0
    last_home = rows[home_pace_col].iloc[-1] if len(rows) else league_avg
    last_away = rows[away_pace_col].iloc[-1] if len(rows) else league_avg
    return float(0.5 * np.log(
        (last_home / league_avg) * (last_away / league_avg) + 1e-9
    ))


@app.command()
def main(
    games: str = typer.Option("data/processed/wnba_games.parquet",
                               help="Games parquet with scheduled games."),
    model_dir: str = typer.Option("artifacts/models/game_totals",
                                   help="Directory containing team_score_model.pkl."),
    game_date: str | None = typer.Option(None,
        help="Target game date YYYY-MM-DD (default: tomorrow ET)."),
    out_dir: str = typer.Option("deliveries/next_game",
                                 help="Output directory."),
) -> None:
    """Predict game total PMFs for all games on the target date."""
    target = game_date or _tomorrow()
    typer.echo(f"Predicting game totals for: {target}")

    model_path = Path(model_dir) / "team_score_model.pkl"
    if not model_path.exists():
        typer.echo(f"[WARN] Model not found at {model_path} — skipping game totals prediction.")
        raise typer.Exit(0)

    model = WNBATeamScoreModel.load(str(model_path))
    typer.echo(f"Loaded model: {model.get_training_summary()}")

    games_df = _load_games_for_date(games, target)
    if games_df.empty:
        typer.echo(f"[WARN] No games found for {target}. Exiting.")
        raise typer.Exit(0)

    # Normalise team column names
    home_col = "home_team_abbreviation" if "home_team_abbreviation" in games_df.columns else "home_team"
    away_col = "visitor_team_abbreviation" if "visitor_team_abbreviation" in games_df.columns else "away_team"

    grids: list[WNBATeamScorePMFGrid] = []
    rows: list[dict] = []

    for _, g in games_df.iterrows():
        home = str(g[home_col])
        away = str(g[away_col])
        game_id = g.get("game_id")
        typer.echo(f"  Predicting: {home} vs {away} (game_id={game_id})")

        try:
            pace_adj = _get_pace_adj(games_df, home, away)
            grid = model.predict(home, away, pace_adj=pace_adj,
                                 game_id=game_id, game_date=target)
            grids.append(grid)

            d = grid.to_dict()
            rows.append({
                "game_id": game_id,
                "game_date": target,
                "home_team": home,
                "away_team": away,
                "home_lambda": d["home_lambda"],
                "away_lambda": d["away_lambda"],
                "home_mean": d["home_mean"],
                "away_mean": d["away_mean"],
                "total_mean": d["total_mean"],
                # Anchor markets at common lines
                "p_over_155_5": grid.total_over(155.5),
                "p_over_160_5": grid.total_over(160.5),
                "p_over_165_5": grid.total_over(165.5),
                "p_over_170_5": grid.total_over(170.5),
                "model_version": WNBATeamScoreModel.VERSION,
            })
        except Exception as exc:
            typer.echo(f"  [ERROR] {home} vs {away}: {exc}")
            continue

    if not rows:
        typer.echo("[WARN] No game totals predictions generated.")
        raise typer.Exit(0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / f"game_totals_{target}.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    typer.echo(f"Game totals parquet → {parquet_path}")

    json_path = out / f"game_totals_{target}.json"
    with open(json_path, "w") as f:
        json.dump([g.to_dict() for g in grids], f, default=str, indent=2)
    typer.echo(f"Game totals JSON → {json_path}")

    typer.echo(f"\nGenerated {len(grids)} game total PMF grids.")
    for grid in grids:
        typer.echo(f"  {grid}")


if __name__ == "__main__":
    app()
