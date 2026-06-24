"""Build canonical processed tables from raw BDL parquet files.

Reads from data/raw/bdl/, enriches with provenance and derived columns,
writes canonical parquets to data/processed/, and produces schema_manifest.json.

Derived enrichments applied here (not in normalize.py):
  - games:        has_player_stats, has_odds
  - player_stats: opponent_team_id, opponent_team_abbreviation, is_home,
                  home_away, did_play, started_proxy, and audit flags
  - stat rename:  tov → turnover, pa → pts_ast, pr → pts_reb, etc.

Usage:
    python3 scripts/build_canonical_tables.py \\
        --raw-dir data/raw/bdl \\
        --out-dir data/processed
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.data.schema import (
    ALL_SCHEMAS,
    REQUIRED_TABLES,
    build_schema_manifest,
    validate_table,
)

app = typer.Typer(add_completion=False)

# Raw stat column → canonical stat column
_STAT_RENAMES = {
    "tov": "turnover",
    "pa": "pts_ast",
    "pr": "pts_reb",
    "ra": "reb_ast",
    "pra": "pts_reb_ast",
}

# Raw games column → canonical games column
_GAMES_COL_RENAMES = {
    "away_team_id": "visitor_team_id",
    "home_team_abbr": "home_team_abbreviation",
    "away_team_abbr": "visitor_team_abbreviation",
    "away_score": "visitor_team_score",
    "home_score": "home_team_score",
}

# team_abbreviation alias used in raw player stats
_STATS_COL_RENAMES = {
    "team_abbr": "team_abbreviation",
}


def _read_raw(raw_dir: Path, name: str) -> pd.DataFrame | None:
    p = raw_dir / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _build_games(raw_dir: Path, stats_df: pd.DataFrame | None, odds_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _read_raw(raw_dir, "wnba_games")
    if df is None or df.empty:
        raise RuntimeError("Required raw file wnba_games.parquet not found.")

    # Rename columns to canonical names
    df = df.rename(columns={k: v for k, v in _GAMES_COL_RENAMES.items() if k in df.columns})

    # has_player_stats
    if stats_df is not None and not stats_df.empty and "game_id" in stats_df.columns:
        games_with_stats = set(stats_df["game_id"].dropna().unique())
        df["has_player_stats"] = df["game_id"].isin(games_with_stats)
    else:
        df["has_player_stats"] = False

    # has_odds
    if odds_df is not None and not odds_df.empty and "game_id" in odds_df.columns:
        games_with_odds = set(odds_df["game_id"].dropna().unique())
        df["has_odds"] = df["game_id"].isin(games_with_odds)
    else:
        df["has_odds"] = False

    # has_player_props (placeholder — props not pulled in history)
    df["has_player_props"] = False

    # Always recompute status_normalized, has_final_score, and is_played_game
    # from the raw status column so updates to normalize_game_status propagate
    # without needing to re-pull from the API.
    if "status" in df.columns:
        from wnba_props_model.data.normalize import normalize_game_status
        df["status_normalized"] = df["status"].apply(normalize_game_status)
        df["has_final_score"] = df["status_normalized"] == "final"
        df["is_played_game"] = df["status_normalized"] == "final"
        # Zero out scores for non-final games (BDL returns 0 for upcoming games)
        for score_col in ["home_team_score", "visitor_team_score", "total_score"]:
            if score_col in df.columns:
                df.loc[~df["has_final_score"], score_col] = None

    return df


def _build_player_stats(raw_dir: Path, games_df: pd.DataFrame) -> pd.DataFrame:
    df = _read_raw(raw_dir, "wnba_player_game_stats")
    if df is None or df.empty:
        raise RuntimeError("Required raw file wnba_player_game_stats.parquet not found.")

    # Rename columns
    df = df.rename(columns={k: v for k, v in _STATS_COL_RENAMES.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in _STAT_RENAMES.items() if k in df.columns})

    # Enrich with opponent and home/away from games
    if not games_df.empty:
        game_lookup = games_df.set_index("game_id")[
            ["home_team_id", "visitor_team_id",
             "home_team_abbreviation", "visitor_team_abbreviation"]
        ]
        df = df.join(game_lookup, on="game_id", how="left")

        df["is_home"] = df["team_id"] == df["home_team_id"]
        df["home_away"] = df["is_home"].map({True: "home", False: "away"})

        # opponent_team_id and abbreviation
        df["opponent_team_id"] = df.apply(
            lambda r: r["visitor_team_id"] if r.get("is_home") else r["home_team_id"],
            axis=1,
        )
        df["opponent_team_abbreviation"] = df.apply(
            lambda r: r["visitor_team_abbreviation"]
            if r.get("is_home")
            else r["home_team_abbreviation"],
            axis=1,
        )
        # Drop the joined helper columns we no longer need
        df = df.drop(
            columns=[
                c for c in [
                    "home_team_id", "visitor_team_id",
                    "home_team_abbreviation", "visitor_team_abbreviation",
                ]
                if c in df.columns
            ]
        )
    else:
        df["is_home"] = pd.NA
        df["home_away"] = pd.NA
        df["opponent_team_id"] = pd.NA
        df["opponent_team_abbreviation"] = pd.NA

    # Derived flags
    df["did_play"] = df["minutes"] > 0

    # Started proxy: player with >= team median minutes is assumed a starter
    # Very rough; true lineup data would improve this.
    df["started_proxy"] = False
    if "team_id" in df.columns and "game_id" in df.columns:
        for (game_id, team_id), grp in df.groupby(["game_id", "team_id"]):
            playing = grp[grp["minutes"] > 0]["minutes"]
            if len(playing) < 2:
                continue
            median_min = playing.median()
            df.loc[grp.index, "started_proxy"] = grp["minutes"] >= median_min * 0.85

    # Audit flags
    df["zero_minute_flag"] = df["minutes"] == 0.0
    df["non_playing_flag"] = df.get("minutes_flag", pd.Series(dtype=str)) == "non_playing"
    stat_cols = [
        c for c in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"] if c in df.columns
    ]
    df["stat_line_all_zero_flag"] = (df[stat_cols].sum(axis=1) == 0) if stat_cols else False
    df["missing_team_flag"] = df["team_id"].isna()
    df["missing_opponent_flag"] = df["opponent_team_id"].isna()
    df["missing_game_date_flag"] = df["game_date"].isna()

    return df


@app.command()
def main(
    raw_dir: str = typer.Option("data/raw/bdl", help="Directory with raw parquet files."),
    out_dir: str = typer.Option("data/processed", help="Output directory for canonical tables."),
    manifest_path: str = typer.Option(
        "data/processed/schema_manifest.json", help="Schema manifest output path."
    ),
) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    typer.echo(f"Building canonical tables: {raw} → {out}")

    # Read optional raw tables
    teams_raw = _read_raw(raw, "wnba_teams")
    players_raw = _read_raw(raw, "wnba_players")
    stats_raw = _read_raw(raw, "wnba_player_game_stats")
    odds_raw = _read_raw(raw, "wnba_odds")
    adv_raw = _read_raw(raw, "wnba_player_advanced_stats")
    season_adv_raw = _read_raw(raw, "wnba_player_season_advanced")
    team_adv_raw = _read_raw(raw, "wnba_team_game_advanced")
    player_shot_raw = _read_raw(raw, "wnba_player_shot_locations")
    team_shot_raw = _read_raw(raw, "wnba_team_shot_locations")
    inj_raw = _read_raw(raw, "wnba_injuries")
    standings_raw = _read_raw(raw, "wnba_standings")

    written: dict[str, Path] = {}
    validation_results: list[dict] = []
    seasons: list[int] = []

    errors: list[str] = []

    # -- Games (required) --
    try:
        games_df = _build_games(raw, stats_raw, odds_raw)
        p = out / "wnba_games.parquet"
        games_df.to_parquet(p, index=False)
        written["wnba_games"] = p
        seasons = sorted(int(s) for s in games_df["season"].dropna().unique())
        typer.echo(f"  wnba_games: {len(games_df):,} rows  seasons={seasons}")
        validation_results.append(
            validate_table(games_df, ALL_SCHEMAS["wnba_games"], str(p))
        )
    except Exception as exc:
        errors.append(f"FAIL wnba_games: {exc}")
        typer.echo(f"  [FAIL] wnba_games: {exc}", err=True)

    # -- Player game stats (required) --
    try:
        game_lookup_df = games_df if "games_df" in dir() else pd.DataFrame()
        stats_df = _build_player_stats(raw, game_lookup_df)
        p = out / "wnba_player_game_stats.parquet"
        stats_df.to_parquet(p, index=False)
        written["wnba_player_game_stats"] = p
        typer.echo(
            f"  wnba_player_game_stats: {len(stats_df):,} rows  "
            f"players={stats_df['player_id'].nunique()}"
        )
        validation_results.append(
            validate_table(stats_df, ALL_SCHEMAS["wnba_player_game_stats"], str(p))
        )
    except Exception as exc:
        errors.append(f"FAIL wnba_player_game_stats: {exc}")
        typer.echo(f"  [FAIL] wnba_player_game_stats: {exc}", err=True)

    # -- Teams (optional) --
    if teams_raw is not None:
        p = out / "wnba_teams.parquet"
        teams_raw.to_parquet(p, index=False)
        written["wnba_teams"] = p
        typer.echo(f"  wnba_teams: {len(teams_raw):,} rows")
        validation_results.append(validate_table(teams_raw, ALL_SCHEMAS["wnba_teams"], str(p)))
    else:
        typer.echo("  wnba_teams: NOT AVAILABLE (raw not found)")

    # -- Players (optional) --
    if players_raw is not None:
        p = out / "wnba_players.parquet"
        players_raw.to_parquet(p, index=False)
        written["wnba_players"] = p
        typer.echo(f"  wnba_players: {len(players_raw):,} rows")
        validation_results.append(
            validate_table(players_raw, ALL_SCHEMAS["wnba_players"], str(p))
        )
    else:
        typer.echo("  wnba_players: NOT AVAILABLE (raw not found)")

    # -- Advanced stats (optional) --
    if adv_raw is not None:
        p = out / "wnba_player_advanced_stats.parquet"
        adv_raw.to_parquet(p, index=False)
        written["wnba_player_advanced_stats"] = p
        typer.echo(f"  wnba_player_advanced_stats: {len(adv_raw):,} rows")
        validation_results.append(
            validate_table(adv_raw, ALL_SCHEMAS["wnba_player_advanced_stats"], str(p))
        )

    # -- Injuries (optional) --
    if inj_raw is not None:
        p = out / "wnba_injuries.parquet"
        inj_raw.to_parquet(p, index=False)
        written["wnba_injuries"] = p
        typer.echo(f"  wnba_injuries: {len(inj_raw):,} rows")
        validation_results.append(
            validate_table(inj_raw, ALL_SCHEMAS["wnba_injuries"], str(p))
        )

    # -- Odds (optional) --
    # Enrich with game_date and season from the games table (not in BDL odds response).
    if odds_raw is not None:
        odds_df = odds_raw.copy()
        if "games_df" in dir() and not games_df.empty and "game_id" in games_df.columns:
            date_season = games_df[["game_id", "game_date", "season"]].drop_duplicates("game_id")
            if "game_date" in odds_df.columns:
                odds_df = odds_df.drop(columns=["game_date"])
            if "season" in odds_df.columns:
                odds_df = odds_df.drop(columns=["season"])
            odds_df = odds_df.merge(date_season, on="game_id", how="left")
        p = out / "wnba_odds.parquet"
        odds_df.to_parquet(p, index=False)
        written["wnba_odds"] = p
        vendors = sorted(odds_df["vendor"].dropna().unique().tolist()) if "vendor" in odds_df.columns else []
        typer.echo(f"  wnba_odds: {len(odds_df):,} rows  vendors={vendors}")
        validation_results.append(validate_table(odds_df, ALL_SCHEMAS["wnba_odds"], str(p)))

    # -- Player props (optional, live-only) --
    props_raw = _read_raw(raw, "wnba_player_props")
    if props_raw is not None:
        p = out / "wnba_player_props.parquet"
        props_raw.to_parquet(p, index=False)
        written["wnba_player_props"] = p
        vendors_p = sorted(props_raw["vendor"].dropna().unique().tolist()) if "vendor" in props_raw.columns else []
        typer.echo(f"  wnba_player_props: {len(props_raw):,} rows  vendors={vendors_p}")
        validation_results.append(
            validate_table(props_raw, ALL_SCHEMAS["wnba_player_props"], str(p))
        )

    # -- Player season advanced stats (optional) --
    if season_adv_raw is not None:
        p = out / "wnba_player_season_advanced.parquet"
        season_adv_raw.to_parquet(p, index=False)
        written["wnba_player_season_advanced"] = p
        typer.echo(f"  wnba_player_season_advanced: {len(season_adv_raw):,} rows")

    # -- Team game advanced stats (optional) --
    if team_adv_raw is not None:
        p = out / "wnba_team_game_advanced.parquet"
        team_adv_raw.to_parquet(p, index=False)
        written["wnba_team_game_advanced"] = p
        typer.echo(f"  wnba_team_game_advanced: {len(team_adv_raw):,} rows")

    # -- Player shot locations (optional) --
    if player_shot_raw is not None:
        p = out / "wnba_player_shot_locations.parquet"
        player_shot_raw.to_parquet(p, index=False)
        written["wnba_player_shot_locations"] = p
        typer.echo(f"  wnba_player_shot_locations: {len(player_shot_raw):,} rows")

    # -- Team shot locations (optional) --
    if team_shot_raw is not None:
        p = out / "wnba_team_shot_locations.parquet"
        team_shot_raw.to_parquet(p, index=False)
        written["wnba_team_shot_locations"] = p
        typer.echo(f"  wnba_team_shot_locations: {len(team_shot_raw):,} rows")

    # -- Standings (optional) --
    if standings_raw is not None:
        p = out / "wnba_standings.parquet"
        standings_raw.to_parquet(p, index=False)
        written["wnba_standings"] = p
        typer.echo(f"  wnba_standings: {len(standings_raw):,} rows")
        validation_results.append(
            validate_table(standings_raw, ALL_SCHEMAS["wnba_standings"], str(p))
        )

    # -- Schema manifest --
    manifest = build_schema_manifest(validation_results, seasons, ts)
    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    typer.echo(f"\nSchema manifest → {mpath}")

    # Summary
    failed_validations = [r for r in validation_results if r["status"] == "fail"]
    warned_validations = [r for r in validation_results if r["status"] == "warn"]
    typer.echo(
        f"\nValidation: {len(validation_results)} tables checked | "
        f"{len(failed_validations)} FAIL | {len(warned_validations)} WARN"
    )
    for r in failed_validations:
        typer.echo(f"  FAIL {r['table']}: {r['errors']}", err=True)
    for r in warned_validations:
        typer.echo(f"  WARN {r['table']}: {r['warnings']}")

    # Check required tables
    for tbl in REQUIRED_TABLES:
        if tbl not in written:
            errors.append(f"Required canonical table missing: {tbl}")

    if errors:
        typer.echo(f"\n[FAIL] {len(errors)} error(s):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("\n[PASS] Canonical tables built successfully.")


if __name__ == "__main__":
    app()
