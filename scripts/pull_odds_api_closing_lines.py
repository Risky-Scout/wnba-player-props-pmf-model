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

import numpy as np
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
    open_time: str = typer.Option(
        "",
        "--open-time",
        help="ISO UTC datetime for the opening snapshot (e.g. '2026-06-23T14:00:00Z'). "
             "When provided, computes line_movement = closing_line - opening_line and "
             "opening_p_over columns. Defaults to 14:00:00 UTC on game_date.",
    ),
    out_dir: str = typer.Option("data/clv_tracking", "--out-dir"),
    api_key: str = typer.Option("", envvar="ODDS_API_KEY"),
    region: str = typer.Option("us", "--region"),
    games_path: str = typer.Option("data/processed/wnba_games.parquet", "--games-path",
                                   help="Canonical games table for event_id -> game_id resolution."),
    roster_path: str = typer.Option("data/processed/wnba_player_game_stats.parquet", "--roster-path",
                                    help="Canonical player-game table for player_name -> player_id resolution."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Pull historical closing lines and CANONICALIZE them to the scorer contract.

    Fail-closed (P0): exits NONZERO on empty output. The scorer requires a
    canonical table (game_id, player_id, stat, market_prob_over_no_vig, line);
    provider-native fields (event_id, player_name, close_prob_over_no_vig) are
    canonicalized here before scoring. Rows whose event or player cannot be
    resolved to canonical IDs by EXACT match are dropped (never fuzzy-matched).
    """
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        typer.echo("[FATAL] No ODDS_API_KEY — cannot pull closing lines", err=True)
        raise typer.Exit(1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    snap_time = close_time if close_time else f"{game_date}T23:00:00Z"
    open_snap_time = open_time if open_time else f"{game_date}T14:00:00Z"
    typer.echo(
        f"[OddsAPI] Pulling closing lines for {game_date} at snapshot={snap_time}"
    )

    try:
        client = OddsAPIClient(api_key=key, region=region)
        raw = client.get_closing_lines_for_date(game_date, close_time_utc=snap_time)
    except OddsAPIError as exc:
        typer.echo(f"[FATAL] Closing line pull failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"[OddsAPI] Got {len(raw):,} outcome rows | "
        f"quota remaining={client.quota_remaining}"
    )

    if dry_run:
        typer.echo("[DRY-RUN] Done. Exiting without writing.")
        raise typer.Exit(0)

    if not raw:
        typer.echo(
            f"[FATAL] Odds API returned zero closing-line rows for {game_date}. "
            "Refusing to write an empty file (fail-closed).",
            err=True,
        )
        raise typer.Exit(1)

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

    # Parts B+F: Pull opening snapshot to compute line_movement (closing - opening).
    # opening_line is used as a lagged feature (prior game day) — no temporal leakage.
    merged["opening_line"] = None
    merged["opening_p_over"] = None
    merged["line_movement"] = None
    try:
        typer.echo(f"[OddsAPI] Pulling opening snapshot at {open_snap_time}")
        raw_open = client.get_closing_lines_for_date(game_date, close_time_utc=open_snap_time)
        if raw_open:
            open_df = pd.DataFrame(raw_open)
            open_over = open_df[open_df["side"].str.lower().str.startswith("over")].copy()
            open_under = open_df[open_df["side"].str.lower().str.startswith("under")].copy()
            open_key_cols = ["event_id", "bookmaker", "market_key", "player_name", "stat"]
            open_over = open_over.rename(columns={"odds": "open_over_odds", "line": "opening_line_val"})
            open_under = open_under.rename(columns={"odds": "open_under_odds"})
            open_merged = open_over[open_key_cols + ["open_over_odds", "opening_line_val"]].merge(
                open_under[open_key_cols + ["open_under_odds"]], on=open_key_cols, how="outer"
            )
            open_shin = open_merged.apply(
                lambda r: shin_no_vig_two_way_with_z(r.get("open_over_odds"), r.get("open_under_odds")),
                axis=1, result_type="expand",
            )
            open_merged["open_p_over"] = open_shin[0]
            open_merged["open_line"] = open_merged["opening_line_val"]
            join_key = ["event_id", "bookmaker", "market_key", "player_name", "stat"]
            merged = merged.merge(
                open_merged[join_key + ["open_p_over", "open_line"]],
                on=join_key, how="left",
            )
            merged["opening_line"] = merged.pop("open_line")
            merged["opening_p_over"] = merged.pop("open_p_over")
            merged["line_movement"] = merged["line"] - merged["opening_line"]
            typer.echo(f"[OddsAPI] Opening snapshot: {len(open_merged):,} rows merged")
    except Exception as exc:
        typer.echo(f"[WARN] Opening snapshot failed (non-fatal): {exc}", err=True)

    # Canonicalize provider-native (event_id/player_name) rows to the scorer
    # contract (game_id/player_id/stat/market_prob_over_no_vig/line). Fail-closed:
    # unmatched rows are dropped, and an empty canonical result is a nonzero exit.
    merged = merged.rename(columns={"close_prob_over_no_vig": "market_prob_over_no_vig"})
    canonical = canonicalize_closing_lines(merged, game_date, games_path, roster_path)
    if canonical.empty:
        typer.echo(
            "[FATAL] Zero rows survived canonicalization (no event/player resolved "
            "to canonical IDs). Refusing to write a table the scorer cannot use.",
            err=True,
        )
        raise typer.Exit(1)

    out_path = out / f"closing_lines_oddsapi_{game_date}.parquet"
    canonical.to_parquet(out_path, index=False)
    typer.echo(f"[OddsAPI] Wrote {len(canonical):,} CANONICAL closing-line rows → {out_path}")
    typer.echo(f"  identity_method: {canonical['identity_method'].value_counts().to_dict()}")

    by_stat = canonical.groupby("stat").size().sort_values(ascending=False)
    typer.echo("\nRows per stat:")
    for stat, count in by_stat.items():
        typer.echo(f"  {stat:20s}: {count:,}")


# WNBA full-name -> abbreviation (stable, 12 teams). Used to bridge Odds API team
# names to canonical team abbreviations for event->game_id resolution.
_WNBA_TEAM_ABBR = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "golden state valkyries": "GSV", "indiana fever": "IND",
    "las vegas aces": "LVA", "los angeles sparks": "LAS", "minnesota lynx": "MIN",
    "new york liberty": "NYL", "phoenix mercury": "PHO", "seattle storm": "SEA",
    "washington mystics": "WAS",
}


def _norm_name(s: object) -> str:
    """Normalize a player name for exact matching (case/space/punctuation-insensitive)."""
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _team_abbr(team: object) -> str:
    t = str(team).strip().lower()
    if t in _WNBA_TEAM_ABBR:
        return _WNBA_TEAM_ABBR[t]
    # already an abbreviation?
    up = str(team).strip().upper()
    return up if len(up) <= 4 else ""


def canonicalize_closing_lines(
    df: pd.DataFrame, game_date: str, games_path: str, roster_path: str
) -> pd.DataFrame:
    """Resolve provider-native closing rows to the canonical scorer schema.

    EXACT matching only (no fuzzy): game via (game_date, {home,away} abbrev),
    player via normalized name within the matched game's rosters. Unmatched rows
    are dropped. Emits one consensus row per (game_id, player_id, stat).
    """
    games_p, roster_p = Path(games_path), Path(roster_path)
    if not games_p.exists() or not roster_p.exists():
        raise FileNotFoundError(
            f"Canonicalization requires games ({games_path}) and roster ({roster_path})."
        )
    games = pd.read_parquet(games_p)
    roster = pd.read_parquet(roster_p)

    # ── game_id resolution: date + unordered team-abbrev pair ──────────────────
    g = games.copy()
    if "game_date" in g.columns:
        g["_gd"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m-%d")
        g = g[g["_gd"] == game_date]
    g["_pair"] = g.apply(
        lambda r: frozenset({str(r.get("home_team_abbreviation", "")).upper(),
                             str(r.get("visitor_team_abbreviation", "")).upper()}), axis=1)
    pair_to_game = {p: gid for p, gid in zip(g["_pair"], g["game_id"]) }

    df = df.copy()
    df["_pair"] = df.apply(
        lambda r: frozenset({_team_abbr(r.get("home_team")), _team_abbr(r.get("away_team"))}), axis=1)
    df["game_id"] = df["_pair"].map(pair_to_game)

    # ── player_id resolution: exact normalized name within the game's rosters ──
    rr = roster[["game_id", "player_id", "player_name"]].dropna().copy() if {
        "game_id", "player_id", "player_name"}.issubset(roster.columns) else pd.DataFrame(
        columns=["game_id", "player_id", "player_name"])
    rr["_nm"] = rr["player_name"].map(_norm_name)
    name_lookup = {(str(gid), nm): pid for gid, nm, pid in zip(rr["game_id"], rr["_nm"], rr["player_id"])}

    df["_nm"] = df["player_name"].map(_norm_name)
    df["player_id"] = [name_lookup.get((str(gid), nm)) for gid, nm in zip(df["game_id"], df["_nm"])]

    df["identity_method"] = np.where(
        df["player_id"].notna(), "exact_roster_name", "unmatched")
    resolved = df[df["game_id"].notna() & df["player_id"].notna() & df["stat"].notna()].copy()
    if resolved.empty:
        return resolved

    # ── one consensus row per (game_id, player_id, stat) ───────────────────────
    for k in ["game_id", "player_id", "stat"]:
        resolved[k] = resolved[k].astype("string")
    agg = {"line": "median", "market_prob_over_no_vig": "median"}
    for c in ["over_odds", "under_odds", "opening_line", "line_movement"]:
        if c in resolved.columns:
            agg[c] = "median"
    consensus = (resolved.groupby(["game_id", "player_id", "stat"], as_index=False)
                 .agg(agg))
    consensus["game_date"] = game_date
    consensus["identity_method"] = "exact_roster_name"
    consensus["source"] = "odds_api_v4_historical"
    consensus["is_closing"] = True
    return consensus


if __name__ == "__main__":
    app()
