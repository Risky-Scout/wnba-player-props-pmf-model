"""Pull historical WNBA closing lines from The Odds API v4 (historical endpoint).

Used by post_game_scoring.yml to compute true CLV by comparing our pre-game
model probability against the actual closing market price.

The Odds API historical props endpoint costs 10× normal rate; this script
batches calls and writes a Parquet file for use by score_daily_predictions.py.

Output:
    {out_dir}/closing_lines_oddsapi_{game_date}.parquet

Usage:
    python scripts/pull_odds_api_closing_lines.py \\
        --game-date 2026-06-23 \\
        --close-time 2026-06-23T23:30:00Z \\
        --out-dir data/clv_tracking \\
        --api-key $ODDS_API_KEY
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError
from wnba_props_model.models.market import shin_no_vig_two_way_with_z

app = typer.Typer(add_completion=False)


@app.command()
def main(
    game_date: str = typer.Option(
        ..., "--game-date", help="Game date YYYY-MM-DD whose closing lines we want."
    ),
    close_time: str = typer.Option(
        "",
        "--close-time",
        help="ISO UTC datetime for historical snapshot, e.g. '2026-06-23T23:30:00Z'. "
             "Defaults to 23:00:00 UTC (7 PM ET).",
    ),
    out_dir: str = typer.Option("data/clv_tracking", "--out-dir"),
    api_key: str = typer.Option("", envvar="ODDS_API_KEY"),
    region: str = typer.Option("us", "--region"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Pull historical closing lines for CLV calculation."""
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        typer.echo("[WARN] No ODDS_API_KEY — cannot pull closing lines", err=True)
        _write_empty(Path(out_dir), game_date)
        raise typer.Exit(0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    snap_time = close_time if close_time else f"{game_date}T23:00:00Z"
    typer.echo(
        f"[OddsAPI] Pulling closing lines for {game_date} at snapshot={snap_time}"
    )

    try:
        client = OddsAPIClient(api_key=key, region=region)
        raw = client.get_closing_lines_for_date(game_date, close_time_utc=snap_time)
    except OddsAPIError as exc:
        typer.echo(f"[ERROR] Closing line pull failed: {exc}", err=True)
        _write_empty(out, game_date)
        raise typer.Exit(0)

    typer.echo(
        f"[OddsAPI] Got {len(raw):,} outcome rows | "
        f"quota remaining={client.quota_remaining}"
    )

    if dry_run:
        typer.echo("[DRY-RUN] Done. Exiting without writing.")
        raise typer.Exit(0)

    if not raw:
        _write_empty(out, game_date)
        raise typer.Exit(0)

    df = pd.DataFrame(raw)

    # Pivot Over/Under into one row per player-stat-bookmaker
    over_df  = df[df["side"].str.lower().str.startswith("over")].copy()
    under_df = df[df["side"].str.lower().str.startswith("under")].copy()

    key_cols = ["event_id", "game_date", "snapshot_time", "bookmaker",
                "market_key", "player_name", "line", "stat"]

    over_df  = over_df.rename(columns={"odds": "over_odds"})
    under_df = under_df.rename(columns={"odds": "under_odds"})

    merged = over_df[key_cols + ["over_odds", "home_team", "away_team"]].merge(
        under_df[key_cols + ["under_odds"]],
        on=key_cols,
        how="outer",
    )

    merged = merged[merged["stat"].notna()].copy()

    # Shin no-vig probabilities for closing line calibration
    shin_results = merged.apply(
        lambda r: shin_no_vig_two_way_with_z(r.get("over_odds"), r.get("under_odds")),
        axis=1,
        result_type="expand",
    )
    merged["close_prob_over_no_vig"] = shin_results[0]
    merged["close_prob_under_no_vig"] = shin_results[1]
    merged["close_shin_z"] = shin_results[2]
    merged["pulled_at_utc"] = datetime.now(timezone.utc).isoformat()
    merged["source"] = "odds_api_v4_historical"

    out_path = out / f"closing_lines_oddsapi_{game_date}.parquet"
    merged.to_parquet(out_path, index=False)
    typer.echo(f"[OddsAPI] Wrote {len(merged):,} closing-line rows → {out_path}")

    by_stat = merged.groupby("stat").size().sort_values(ascending=False)
    typer.echo("\nRows per stat:")
    for stat, count in by_stat.items():
        typer.echo(f"  {stat:20s}: {count:,}")


def _write_empty(out: Path, game_date: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "event_id", "game_date", "snapshot_time", "home_team", "away_team",
        "bookmaker", "market_key", "stat", "player_name", "line",
        "over_odds", "under_odds", "close_prob_over_no_vig", "close_shin_z",
        "pulled_at_utc", "source",
    ]).to_parquet(out / f"closing_lines_oddsapi_{game_date}.parquet", index=False)


if __name__ == "__main__":
    app()
