"""Format pre-game PMF output into the blueprint §3 JSON schema for wizardofodds.com.

Reads deliveries/tonight/player_projections_{date}.parquet (or full_pmfs_wide.parquet)
and writes the exact JSON structure required by the dashboard contract:

Output files (blueprint §12.1):
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_initial.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_injury_update.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_final.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json  (copy of most recent)

Dashboard fetch URL (blueprint §13.1):
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json

Usage:
    python scripts/format_pregame_output.py \\
        --game-date 2026-06-24 \\
        --pipeline-run pregame_initial \\
        --pmfs deliveries/tonight/full_pmfs_wide.parquet \\
        --out-dir tools/odds-scanner/predictions/WNBA/Pre-Game-Edge
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.pipeline.output_schema import build_pregame_envelope

app = typer.Typer(add_completion=False)

VALID_RUN_TYPES = ("pregame_initial", "pregame_injury_update", "pregame_final")


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="Target game date YYYY-MM-DD."),
    pipeline_run: str = typer.Option(
        "pregame_initial", "--pipeline-run",
        help=f"Run type: {VALID_RUN_TYPES}",
    ),
    pmfs: str = typer.Option(
        "", "--pmfs",
        help="PMF parquet path. Auto-detected from deliveries/ if not set.",
    ),
    out_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA/Pre-Game-Edge",
        "--out-dir",
        help="Output directory for JSON files.",
    ),
    market_props: str = typer.Option(
        "", "--market-props",
        help="Optional: Odds API props parquet for edge / deep links.",
    ),
    injuries: str = typer.Option(
        "", "--injuries",
        help="Optional: injuries JSON or parquet.",
    ),
    games_parquet: str = typer.Option(
        "data/processed/wnba_games.parquet",
        "--games",
        help="Games parquet for spread/total context.",
    ),
) -> None:
    """Format pre-game PMFs into the blueprint JSON schema."""
    if pipeline_run not in VALID_RUN_TYPES:
        typer.echo(f"[WARN] Unknown pipeline_run '{pipeline_run}'. Using pregame_initial.", err=True)
        pipeline_run = "pregame_initial"

    # ── Load PMFs ─────────────────────────────────────────────────────────
    pmf_path = _resolve_pmfs(pmfs, game_date)
    if pmf_path is None or not pmf_path.exists():
        typer.echo(f"[ERROR] No PMF file found for {game_date}. Run predict_today.py first.", err=True)
        raise typer.Exit(1)

    pmfs_df = pd.read_parquet(pmf_path)
    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows from {pmf_path}")

    # ── Load supporting data ───────────────────────────────────────────────
    games_df = _safe_read_parquet(games_parquet)
    market_df = _safe_read_parquet(market_props) if market_props else None
    injuries_df = _load_injuries(injuries)

    # ── Build envelope ─────────────────────────────────────────────────────
    envelope = build_pregame_envelope(
        pmfs_df=pmfs_df,
        game_date=game_date,
        pipeline_run=pipeline_run,
        games_df=games_df,
        market_df=market_df,
        injuries_df=injuries_df,
    )

    n_games = len(envelope.get("games", []))
    n_players = sum(len(g.get("players", [])) for g in envelope.get("games", []))
    typer.echo(f"Built envelope: {n_games} games, {n_players} player records")

    # ── Write files ────────────────────────────────────────────────────────
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_suffix = pipeline_run.replace("pregame_", "")
    dated_path = out / f"{game_date}_{run_suffix}.json"
    latest_path = out / "latest.json"

    payload = json.dumps(envelope, indent=2, default=_json_default)

    dated_path.write_text(payload)
    latest_path.write_text(payload)

    typer.echo(f"Wrote {dated_path}")
    typer.echo(f"Wrote {latest_path}")

    # Print sample
    if envelope.get("games"):
        g0 = envelope["games"][0]
        typer.echo(f"\nGame sample: {g0['home_team']['name']} vs {g0['away_team']['name']}")
        if g0.get("players"):
            p0 = g0["players"][0]
            sp = p0.get("stat_projections", {}).get("points", {})
            typer.echo(f"  {p0['player_name']}: pts_mean={sp.get('mean')} conformal_ci={sp.get('conformal_90_ci')}")


def _resolve_pmfs(pmfs_arg: str, game_date: str) -> Path | None:
    if pmfs_arg:
        return Path(pmfs_arg)
    for candidate in [
        f"deliveries/tonight/full_pmfs_wide.parquet",
        f"deliveries/next_game/full_pmfs_wide.parquet",
        f"deliveries/today/full_pmfs_wide.parquet",
        f"deliveries/tonight/player_projections_{game_date}.parquet",
    ]:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _safe_read_parquet(path_str: str) -> pd.DataFrame | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _load_injuries(injuries_arg: str) -> pd.DataFrame | None:
    if not injuries_arg:
        return None
    p = Path(injuries_arg)
    if not p.exists():
        return None
    try:
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            return pd.DataFrame([raw])
        return pd.read_parquet(p)
    except Exception:
        return None


def _json_default(obj):
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return str(obj)


if __name__ == "__main__":
    app()
