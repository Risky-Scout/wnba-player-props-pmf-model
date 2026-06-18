"""Build and persist WNBATeamScoreModel (NegBinom) + WNBAWeibullCopulaScoreModel ensemble.

Usage:
    python scripts/build_game_totals.py \
        --games data/processed/wnba_games.parquet \
        --out-dir artifacts/models/game_totals \
        --time-decay-xi 0.002 \
        --ensemble-weight-negbinom 0.5
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.models.team_score import WNBATeamScoreModel, WNBAWeibullCopulaScoreModel

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
    ensemble_weight_negbinom: float = typer.Option(0.5, "--ensemble-weight-negbinom",
        help="Blend weight for NegBinom model in ensemble (0–1). "
             "Remaining weight goes to WeibullCopula."),
) -> None:
    """Fit NegBinom + Weibull Copula game totals ensemble and save to out_dir."""
    games_df = pd.read_parquet(games)
    typer.echo(f"Loaded {len(games_df):,} game rows from {games}")

    train = _prepare_games_df(games_df)
    typer.echo(f"Training on {len(train):,} completed games")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- NegBinom model (primary) -----------------------------------------
    typer.echo("Fitting WNBATeamScoreModel (NegBinom)…")
    negbinom_model = WNBATeamScoreModel()
    negbinom_model.fit(train, xi=time_decay_xi)

    negbinom_path = out / "team_score_negbinom.pkl"
    negbinom_model.save(str(negbinom_path))

    nb_summary = negbinom_model.get_training_summary()
    summary_path = out / "team_score_summary.json"
    with open(summary_path, "w") as f:
        json.dump(nb_summary, f, indent=2, default=str)

    # Keep legacy path for backward compatibility
    legacy_path = out / "team_score_model.pkl"
    negbinom_model.save(str(legacy_path))

    typer.echo(
        f"  NegBinom: Teams={nb_summary.get('n_teams')}, "
        f"Games={nb_summary.get('n_games')}, "
        f"Converged={nb_summary.get('converged')}, "
        f"NLL={nb_summary.get('final_nll', 0):.2f}"
    )

    # --- Weibull Copula model (secondary) ---------------------------------
    typer.echo("Fitting WNBAWeibullCopulaScoreModel…")
    weibull_model = WNBAWeibullCopulaScoreModel()
    weibull_fitted = False
    try:
        weibull_model.fit(train, time_decay_xi=time_decay_xi)
        weibull_path = out / "team_score_weibull.pkl"
        weibull_model.save(str(weibull_path))
        weibull_fitted = True
        typer.echo(f"  WeibullCopula: fitted on {len(train)} games")
    except Exception as exc:
        typer.echo(f"  WeibullCopula fit failed: {exc} — using NegBinom only")
        ensemble_weight_negbinom = 1.0  # fall back to 100% NegBinom

    # --- Ensemble config ---------------------------------------------------
    ensemble_cfg = {
        "negbinom_weight": float(ensemble_weight_negbinom),
        "weibull_weight": float(1.0 - ensemble_weight_negbinom),
        "weibull_fitted": weibull_fitted,
        "time_decay_xi": time_decay_xi,
        "negbinom_path": "team_score_negbinom.pkl",
        "weibull_path": "team_score_weibull.pkl" if weibull_fitted else None,
    }
    ensemble_cfg_path = out / "team_score_ensemble_config.json"
    with open(ensemble_cfg_path, "w") as f:
        json.dump(ensemble_cfg, f, indent=2)

    typer.echo(f"Ensemble config → {ensemble_cfg_path}")
    typer.echo(f"NegBinom model → {negbinom_path}")
    if weibull_fitted:
        typer.echo(f"WeibullCopula model → {out / 'team_score_weibull.pkl'}")
    typer.echo(
        f"Blend weights: NegBinom={ensemble_weight_negbinom:.2f}, "
        f"Weibull={1 - ensemble_weight_negbinom:.2f}"
    )


if __name__ == "__main__":
    app()
