"""Generate WNBATeamScorePMFGrid predictions for upcoming games.

Loads the NegBinom + WeibullCopula ensemble and produces one WNBATeamScorePMFGrid
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

from wnba_props_model.models.simulation import normalize_pmf, convolve_pmfs
from wnba_props_model.models.team_score import (
    WNBATeamScoreModel,
    WNBATeamScorePMFGrid,
    WNBAWeibullCopulaScoreModel,
)

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


def _blend_grids(
    nb_grid: WNBATeamScorePMFGrid,
    wc_grid: WNBATeamScorePMFGrid | None,
    w_nb: float,
) -> WNBATeamScorePMFGrid:
    """Blend two PMF grids with weights w_nb and (1 - w_nb)."""
    if wc_grid is None or w_nb >= 1.0:
        return nb_grid

    w_wc = 1.0 - w_nb
    home_pmf = normalize_pmf(w_nb * nb_grid.home_score_pmf + w_wc * wc_grid.home_score_pmf)
    away_pmf = normalize_pmf(w_nb * nb_grid.away_score_pmf + w_wc * wc_grid.away_score_pmf)
    total_pmf = normalize_pmf(convolve_pmfs(home_pmf, away_pmf))

    ks_h = np.arange(len(home_pmf))
    ks_a = np.arange(len(away_pmf))

    return WNBATeamScorePMFGrid(
        home_team=nb_grid.home_team,
        away_team=nb_grid.away_team,
        home_score_pmf=home_pmf,
        away_score_pmf=away_pmf,
        total_score_pmf=total_pmf,
        home_lambda=float(np.dot(ks_h, home_pmf)),
        away_lambda=float(np.dot(ks_a, away_pmf)),
        game_id=nb_grid.game_id,
        game_date=nb_grid.game_date,
    )


@app.command()
def main(
    games: str = typer.Option("data/processed/wnba_games.parquet",
                               help="Games parquet with scheduled games."),
    model_dir: str = typer.Option("artifacts/models/game_totals",
                                   help="Directory containing model artifacts."),
    game_date: str | None = typer.Option(None,
        help="Target game date YYYY-MM-DD (default: tomorrow ET)."),
    out_dir: str = typer.Option("deliveries/next_game",
                                 help="Output directory."),
) -> None:
    """Predict game total PMFs (ensemble) for all games on the target date."""
    target = game_date or _tomorrow()
    typer.echo(f"Predicting game totals for: {target}")

    model_dir_path = Path(model_dir)

    # --- Load ensemble config (if present) --------------------------------
    ensemble_cfg_path = model_dir_path / "team_score_ensemble_config.json"
    if ensemble_cfg_path.exists():
        with open(ensemble_cfg_path) as f:
            ens_cfg = json.load(f)
        w_nb = float(ens_cfg.get("negbinom_weight", 0.5))
        weibull_trained = ens_cfg.get("weibull_fitted", False)
    else:
        w_nb = 1.0
        weibull_trained = False

    # --- Load NegBinom model (primary, required) --------------------------
    nb_path = model_dir_path / "team_score_negbinom.pkl"
    if not nb_path.exists():
        nb_path = model_dir_path / "team_score_model.pkl"  # legacy fallback
    if not nb_path.exists():
        typer.echo(f"[WARN] NegBinom model not found at {nb_path} — skipping game totals.")
        raise typer.Exit(0)

    nb_model = WNBATeamScoreModel.load(str(nb_path))
    typer.echo(f"Loaded NegBinom model (blend weight={w_nb:.2f})")

    # --- Load Weibull model (optional) ------------------------------------
    wc_model: WNBAWeibullCopulaScoreModel | None = None
    wc_path = model_dir_path / "team_score_weibull.pkl"
    if weibull_trained and wc_path.exists():
        try:
            wc_model = WNBAWeibullCopulaScoreModel.load(str(wc_path))
            typer.echo(f"Loaded WeibullCopula model (blend weight={1 - w_nb:.2f})")
        except Exception as exc:
            typer.echo(f"[WARN] Failed to load WeibullCopula model: {exc} — using NegBinom only")
            w_nb = 1.0

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
            nb_grid = nb_model.predict(home, away, pace_adj=pace_adj,
                                       game_id=game_id, game_date=target)

            # Weibull prediction (if available and both teams are known)
            wc_grid: WNBATeamScorePMFGrid | None = None
            if wc_model is not None and wc_model.can_predict(home, away):
                wc_grid = wc_model.predict(home, away, game_id=game_id, game_date=target)

            grid = _blend_grids(nb_grid, wc_grid, w_nb)
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
                # Common market lines
                "p_over_155_5": grid.total_over(155.5),
                "p_over_160_5": grid.total_over(160.5),
                "p_over_165_5": grid.total_over(165.5),
                "p_over_170_5": grid.total_over(170.5),
                "ensemble_negbinom_weight": w_nb,
                "weibull_used": wc_grid is not None,
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
