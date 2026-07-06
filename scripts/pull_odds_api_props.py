"""Pull today's WNBA player props from The Odds API v4.

Fetches all CORE_PROP_MARKETS (20 market keys: 9 individual stats, 5 combo stats,
3 Q1 quarter props, 3 alternate lines) for every WNBA event on the target date
and writes a normalized parquet that build_edge_report.py consumes.

Includes bookmaker deep links (event / market / outcome level) for wizardofodds.com.

Output:
    {out_dir}/wnba_player_props_oddsapi_{game_date}.parquet
    {out_dir}/wnba_player_props_oddsapi_latest.parquet  (symlink for pipeline)

Usage:
    python scripts/pull_odds_api_props.py \\
        --game-date 2026-06-24 \\
        --out-dir data/processed \\
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

from wnba_props_model.data.odds_api_client import (
    OddsAPIClient,
    OddsAPIError,
    normalize_odds_api_props,
    get_bookmaker_deep_link,
)
from wnba_props_model.models.market import shin_no_vig_two_way_with_z

app = typer.Typer(add_completion=False)


@app.command()
def main(
    game_date: str = typer.Option(
        ..., "--game-date", help="Target date YYYY-MM-DD (predict_today.py's game date)."
    ),
    out_dir: str = typer.Option("data/processed", "--out-dir"),
    api_key: str = typer.Option("", envvar="ODDS_API_KEY"),
    region: str = typer.Option("us", "--region", help="Odds region: us, us2, uk, eu, au"),
    bookmakers: str = typer.Option(
        "",
        "--bookmakers",
        help="Comma-separated bookmaker keys to include. Empty = all in region.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print row count then exit."),
) -> None:
    """Pull WNBA player props from The Odds API and write parquet."""
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        typer.echo("[WARN] No ODDS_API_KEY — cannot pull Odds API props", err=True)
        _write_empty(Path(out_dir), game_date)
        raise typer.Exit(0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    book_list = [b.strip() for b in bookmakers.split(",") if b.strip()] or None
    typer.echo(f"[OddsAPI] Pulling props for {game_date} | region={region} | bookmakers={book_list or 'all'}")

    try:
        client = OddsAPIClient(api_key=key, region=region)
    except OddsAPIError as exc:
        typer.echo(f"[ERROR] OddsAPIClient init failed: {exc}", err=True)
        _write_empty(out, game_date)
        raise typer.Exit(0)

    if not client.is_wnba_active():
        typer.echo("[WARN] basketball_wnba not active in The Odds API — WNBA may be off-season.")

    try:
        raw_rows = client.get_all_props_for_date(
            game_date,
            include_links=True,
            bookmakers=book_list,
        )
    except OddsAPIError as exc:
        typer.echo(f"[ERROR] Props pull failed: {exc}", err=True)
        _write_empty(out, game_date)
        raise typer.Exit(0)

    if not raw_rows:
        typer.echo(f"[WARN] No props returned for {game_date}")
        _write_empty(out, game_date)
        raise typer.Exit(0)

    typer.echo(f"[OddsAPI] Raw outcome rows: {len(raw_rows):,}")
    typer.echo(f"[OddsAPI] Quota remaining: {client.quota_remaining} | used: {client.quota_used}")

    if dry_run:
        typer.echo(f"[DRY-RUN] Would write {len(raw_rows):,} rows. Exiting.")
        raise typer.Exit(0)

    # Normalize to pipeline schema (pivot Over/Under into one row per player-stat-bookmaker)
    df = normalize_odds_api_props(raw_rows)
    typer.echo(f"[OddsAPI] Normalized: {len(df):,} player-stat-line rows")

    # Compute Shin no-vig probabilities
    shin_results = df.apply(
        lambda r: shin_no_vig_two_way_with_z(r.get("over_odds"), r.get("under_odds")),
        axis=1,
        result_type="expand",
    )
    df["market_prob_over_no_vig"] = shin_results[0]
    df["market_prob_under_no_vig"] = shin_results[1]
    df["shin_z"] = shin_results[2]

    # Add deep link column (cascading fallback per blueprint)
    df["deep_link"] = df.apply(get_bookmaker_deep_link, axis=1)

    # Add metadata
    df["pulled_at_utc"] = datetime.now(timezone.utc).isoformat()
    df["source"] = "odds_api_v4"

    # Deduplicate: keep the best line per player-stat (highest over_odds for over bets)
    best = (
        df.dropna(subset=["over_odds", "under_odds"])
        .sort_values("over_odds", ascending=False)
        .drop_duplicates(subset=["event_id", "player_name", "market_key", "line"])
        .reset_index(drop=True)
    )

    out_path = out / f"wnba_player_props_oddsapi_{game_date}.parquet"
    latest_path = out / "wnba_player_props_oddsapi_latest.parquet"

    best.to_parquet(out_path, index=False)
    best.to_parquet(latest_path, index=False)

    # Also write a combined file that build_edge_report.py can consume directly
    # (renames columns to match the BDL prop schema: prop_type, line_value, market)
    compat = _to_bdl_compat(best)
    compat_path = out / "wnba_player_props.parquet"
    compat.to_parquet(compat_path, index=False)

    typer.echo(f"[OddsAPI] Wrote {len(best):,} deduplicated rows:")
    typer.echo(f"  → {out_path}")
    typer.echo(f"  → {latest_path}")
    typer.echo(f"  → {compat_path} (pipeline-compat)")

    # Summary by stat
    by_stat = best.groupby("stat").size().sort_values(ascending=False)
    typer.echo("\nRows per stat:")
    for stat, count in by_stat.items():
        typer.echo(f"  {stat:20s}: {count:,}")

    by_book = best.groupby("bookmaker").size().sort_values(ascending=False).head(10)
    typer.echo("\nRows per bookmaker (top 10):")
    for book, count in by_book.items():
        typer.echo(f"  {book:25s}: {count:,}")


def _to_bdl_compat(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Odds API schema to BDL-compatible pipeline schema.

    build_edge_report.py / normalize_player_props_snapshot() expects:
    game_id, player_id, prop_type, line_value, market{over_odds, under_odds},
    vendor, updated_at.

    Since The Odds API uses player_name not player_id, we use player_name
    as a join key.  game_id is approximated from event_id.
    The pipeline will join on player_name when player_id is unavailable.
    """
    out = pd.DataFrame()
    out["player_name"]  = df.get("player_name", "")
    out["player_id"]    = None   # will be resolved by downstream join
    out["game_id"]      = df.get("event_id", "")
    out["prop_type"]    = df.get("market_key", "")
    out["stat"]         = df.get("stat")
    out["line_value"]   = df.get("line")
    out["over_odds"]    = df.get("over_odds")
    out["under_odds"]   = df.get("under_odds")
    out["market"]       = df.apply(
        lambda r: {"over_odds": r.get("over_odds"), "under_odds": r.get("under_odds")},
        axis=1,
    )
    out["vendor"]            = df.get("bookmaker", "")
    out["updated_at"]        = df.get("last_update")
    out["market_prob_over_no_vig"] = df.get("market_prob_over_no_vig")
    out["shin_z"]            = df.get("shin_z")
    out["deep_link"]         = df.get("deep_link")
    out["event_link"]        = df.get("event_link")
    out["market_link"]       = df.get("market_link")
    out["outcome_link_over"] = df.get("outcome_link_over")
    out["source"]            = "odds_api_v4"
    out["game_date"]         = df.get("game_date")
    return out.reset_index(drop=True)


def _write_empty(out: Path, game_date: str) -> None:
    empty = pd.DataFrame(columns=[
        "player_name", "stat", "line", "over_odds", "under_odds",
        "bookmaker", "market_key", "event_id", "game_date",
        "event_link", "market_link", "outcome_link_over", "outcome_link_under",
        "market_prob_over_no_vig", "shin_z", "deep_link", "source",
    ])
    out.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(out / f"wnba_player_props_oddsapi_{game_date}.parquet", index=False)
    empty.to_parquet(out / "wnba_player_props_oddsapi_latest.parquet", index=False)


if __name__ == "__main__":
    app()
