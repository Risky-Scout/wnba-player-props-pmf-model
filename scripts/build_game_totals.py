"""Build and persist the WNBATeamScoreModel (replaces quantile GameTotalsModel).

Usage:
    python scripts/build_game_totals.py \
        --games data/processed/wnba_games.parquet \
        --out-dir artifacts/models/game_totals \
        --time-decay-xi 0.002
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.models.team_score import WNBATeamScoreModel

app = typer.Typer(add_completion=False)

# Columns that map a games parquet to the model API expectations
TEAM_COL_MAP = {
    "home_team_abbreviation": "home_team",
    "visitor_team_abbreviation": "away_team",
    "home_team_score": "home_score",
    "visitor_team_score": "away_score",
}


def _prepare_games_df(games_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise games parquet to the expected team_score model columns."""
    df = games_df.copy()
    for src, dst in TEAM_COL_MAP.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    required = ["home_team", "away_team", "home_score", "away_score", "game_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Games parquet missing required columns: {missing}. "
                         f"Available: {sorted(df.columns.tolist())}")
    # Keep only completed (non-null) games for training
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["home_score"] > 0]
    return df


@app.command()
def main(
    games: str = typer.Option(..., help="Normalized games parquet (wnba_games.parquet)."),
    out_dir: str = typer.Option("artifacts/models/game_totals",
                                 help="Directory to save model artifact."),
    time_decay_xi: float = typer.Option(0.002, "--time-decay-xi",
        help="Dixon-Coles time-decay xi: weight = exp(-xi * days_ago). "
             "Typical: 0.002–0.005."),
) -> None:
    """Fit WNBATeamScoreModel and save to out_dir."""
    games_df = pd.read_parquet(games)
    typer.echo(f"Loaded {len(games_df):,} game rows from {games}")

    train = _prepare_games_df(games_df)
    typer.echo(f"Training on {len(train):,} completed games")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = WNBATeamScoreModel()
    model.fit(train, xi=time_decay_xi)

    model_path = out / "team_score_model.pkl"
    model.save(str(model_path))

    summary = model.get_training_summary()
    summary_path = out / "team_score_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    typer.echo(f"Model saved → {model_path}")
    typer.echo(f"Training summary → {summary_path}")
    typer.echo(
        f"  Teams: {summary.get('n_teams')}, "
        f"Games: {summary.get('n_games')}, "
        f"xi={time_decay_xi}, "
        f"Converged: {summary.get('converged')}, "
        f"NLL: {summary.get('final_nll', 0):.2f}"
    )


if __name__ == "__main__":
    app()
