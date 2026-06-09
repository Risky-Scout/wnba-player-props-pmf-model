"""Canonical table schema definitions and validators.

Schemas describe what each canonical table must contain.  Validators return
structured dicts suitable for JSON audit reports.  They never raise on
optional-endpoint absence; missing optional tables produce explicit statuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Schema dataclass
# ---------------------------------------------------------------------------

@dataclass
class TableSchema:
    name: str
    primary_key: list[str]
    required_columns: list[str]
    numeric_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    nonneg_stat_columns: list[str] = field(default_factory=list)
    evaluation_only: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Canonical table schemas
# ---------------------------------------------------------------------------

GAMES_SCHEMA = TableSchema(
    name="wnba_games",
    primary_key=["game_id"],
    required_columns=[
        "game_id", "season", "game_date", "status", "status_normalized",
        "home_team_id", "visitor_team_id",
        "home_team_abbreviation", "visitor_team_abbreviation",
        "has_final_score", "is_played_game",
        "has_player_stats",
        "source", "pull_timestamp_utc",
    ],
    numeric_columns=["home_team_score", "visitor_team_score", "total_score"],
    datetime_columns=["game_date"],
    notes="has_player_stats is derived in build_canonical_tables from joining player_stats.",
)

PLAYER_GAME_STATS_SCHEMA = TableSchema(
    name="wnba_player_game_stats",
    primary_key=["player_id", "game_id"],
    required_columns=[
        "game_id", "game_date", "season",
        "player_id", "player_name",
        "team_id", "team_abbreviation",
        "opponent_team_id",
        "is_home", "home_away",
        "position",
        "minutes", "minutes_raw", "minutes_flag",
        "did_play",
        "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
        "source", "pull_timestamp_utc",
    ],
    numeric_columns=[
        "minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
        "oreb", "dreb", "fga", "fta", "pf",
    ],
    datetime_columns=["game_date"],
    nonneg_stat_columns=["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "minutes"],
    notes="opponent_team_id, is_home, home_away, did_play are derived in build_canonical_tables.",
)

TEAMS_SCHEMA = TableSchema(
    name="wnba_teams",
    primary_key=["team_id"],
    required_columns=["team_id", "team_abbreviation", "team_name", "source", "pull_timestamp_utc"],
    notes="Optional columns: conference, division, city, team_full_name.",
)

PLAYERS_SCHEMA = TableSchema(
    name="wnba_players",
    primary_key=["player_id"],
    required_columns=["player_id", "player_name", "source", "pull_timestamp_utc"],
    notes="Optional: position, height, weight, team_id, draft info.",
)

PLAYER_PROPS_SCHEMA = TableSchema(
    name="wnba_player_props",
    primary_key=[],  # multiple rows per player/game/vendor/stat
    required_columns=[
        "game_id", "player_id", "vendor", "prop_type_raw", "stat", "line",
        "over_odds", "under_odds",
        "source", "pull_timestamp_utc",
    ],
    numeric_columns=["line", "over_odds", "under_odds"],
    evaluation_only=True,
    notes=(
        "Evaluation-only. BDL live-only: historical props not stored. "
        "double_double and triple_double are allowed in this table but "
        "must not be included in the first PMF model. "
        "Must never feed model-only predictive features."
    ),
)

ODDS_SCHEMA = TableSchema(
    name="wnba_odds",
    primary_key=[],  # multiple rows per game/vendor
    required_columns=[
        "odds_id", "game_id", "vendor",
        "spread_home_value", "spread_home_odds", "spread_away_value", "spread_away_odds",
        "moneyline_home_odds", "moneyline_away_odds",
        "total_value", "total_over_odds", "total_under_odds",
        "source", "pull_timestamp_utc",
    ],
    numeric_columns=[
        "spread_home_value", "spread_home_odds", "spread_away_value", "spread_away_odds",
        "moneyline_home_odds", "moneyline_away_odds",
        "total_value", "total_over_odds", "total_under_odds",
    ],
    evaluation_only=True,
    notes=(
        "Evaluation-only. BDL WNBA response is flat (no nested spread/total/moneyline). "
        "game_date and season are joined from wnba_games in build_canonical_tables. "
        "total_value may be used for market-context challenger only, explicitly labeled."
    ),
)

INJURIES_SCHEMA = TableSchema(
    name="wnba_injuries",
    primary_key=[],  # may have multiple entries per player over time
    required_columns=["player_id", "injury_status_normalized", "source", "pull_timestamp_utc"],
    notes="injury_status_normalized must be one of: available, probable, questionable, doubtful, out, inactive, unknown.",
)

ADVANCED_STATS_SCHEMA = TableSchema(
    name="wnba_player_advanced_stats",
    primary_key=["player_id", "game_id"],
    required_columns=["player_id", "game_id", "source", "pull_timestamp_utc"],
    notes="Optional: usage_percentage, pace, off/def ratings, ts_pct, ast_pct, reb_pct.",
)

STANDINGS_SCHEMA = TableSchema(
    name="wnba_standings",
    primary_key=["season", "team_id"],
    required_columns=["season", "team_id", "source", "pull_timestamp_utc"],
    notes="Optional endpoint.",
)

PLAY_BY_PLAY_SCHEMA = TableSchema(
    name="wnba_play_by_play",
    primary_key=[],
    required_columns=["game_id", "source", "pull_timestamp_utc"],
    notes="Optional endpoint. Requires per-game pulling.",
)

SHOT_LOCATIONS_SCHEMA = TableSchema(
    name="wnba_shot_locations",
    primary_key=[],
    required_columns=["game_id", "player_id", "source", "pull_timestamp_utc"],
    notes="Optional endpoint. Requires per-game pulling.",
)

# All schemas indexed by table name
ALL_SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in [
        GAMES_SCHEMA,
        PLAYER_GAME_STATS_SCHEMA,
        TEAMS_SCHEMA,
        PLAYERS_SCHEMA,
        PLAYER_PROPS_SCHEMA,
        ODDS_SCHEMA,
        INJURIES_SCHEMA,
        ADVANCED_STATS_SCHEMA,
        STANDINGS_SCHEMA,
        PLAY_BY_PLAY_SCHEMA,
        SHOT_LOCATIONS_SCHEMA,
    ]
}

# Required tables — pipeline must fail if these are absent
REQUIRED_TABLES = {"wnba_games", "wnba_player_game_stats", "wnba_teams", "wnba_players"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_table(
    df: pd.DataFrame,
    schema: TableSchema,
    table_path: str | None = None,
) -> dict[str, Any]:
    """Validate a canonical DataFrame against its schema.

    Returns a dict with keys: table, status, errors, warnings, stats.
    Status is one of: pass, fail, warn.
    Never raises; all issues are reported in the returned dict.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {"rows": len(df)}

    cols = set(df.columns)

    # Required column check
    missing = sorted(set(schema.required_columns) - cols)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if df.empty:
        return _result(schema.name, errors, warnings, stats, table_path)

    # Primary key uniqueness
    pk = [c for c in schema.primary_key if c in cols]
    if pk:
        n_dups = int(df.duplicated(subset=pk).sum())
        stats["duplicate_pk_rows"] = n_dups
        if n_dups > 0:
            errors.append(f"Duplicate primary-key rows ({pk}): {n_dups}")

    # Numeric columns
    for c in schema.numeric_columns:
        if c not in cols:
            continue
        non_num = int(pd.to_numeric(df[c], errors="coerce").isna().sum()) - int(df[c].isna().sum())
        if non_num > 0:
            warnings.append(f"Column {c!r}: {non_num} non-numeric values")

    # Non-negative stat checks
    for c in schema.nonneg_stat_columns:
        if c not in cols:
            continue
        neg = int((pd.to_numeric(df[c], errors="coerce").fillna(0) < 0).sum())
        if neg > 0:
            errors.append(f"Column {c!r}: {neg} negative values (impossible stat)")
        stats[f"{c}_negative_count"] = neg

    # Minutes sanity (if present)
    if "minutes" in cols:
        over_50 = int((pd.to_numeric(df["minutes"], errors="coerce").fillna(0) > 50).sum())
        stats["minutes_over_50"] = over_50
        if over_50 > 0:
            warnings.append(f"minutes > 50 in {over_50} rows (flag for review)")

    # game_date parseable
    for c in schema.datetime_columns:
        if c not in cols:
            continue
        unparseable = int(pd.to_datetime(df[c], errors="coerce").isna().sum()) - int(
            df[c].isna().sum()
        )
        if unparseable > 0:
            errors.append(f"Column {c!r}: {unparseable} unparseable datetime values")

    # Null counts for required columns
    null_required = {
        c: int(df[c].isna().sum())
        for c in schema.required_columns
        if c in cols and df[c].isna().sum() > 0
    }
    if null_required:
        stats["null_counts_required_cols"] = null_required

    # Injury status normalization check
    if schema.name == "wnba_injuries" and "injury_status_normalized" in cols:
        _valid_statuses = {
            "available", "probable", "questionable", "doubtful", "out", "inactive", "unknown"
        }
        invalid = df["injury_status_normalized"][
            ~df["injury_status_normalized"].isin(_valid_statuses)
        ].dropna()
        if len(invalid) > 0:
            errors.append(
                f"injury_status_normalized has {len(invalid)} invalid values: "
                f"{invalid.unique().tolist()[:5]}"
            )

    # Game status normalization check
    if schema.name == "wnba_games" and "status_normalized" in cols:
        _valid_statuses = {"final", "in_progress", "scheduled", "postponed", "canceled", "unknown"}
        invalid = df["status_normalized"][~df["status_normalized"].isin(_valid_statuses)].dropna()
        if len(invalid) > 0:
            errors.append(
                f"status_normalized has {len(invalid)} invalid values: "
                f"{invalid.unique().tolist()[:5]}"
            )

    return _result(schema.name, errors, warnings, stats, table_path)


def _result(
    name: str,
    errors: list[str],
    warnings: list[str],
    stats: dict[str, Any],
    path: str | None,
) -> dict[str, Any]:
    status = "fail" if errors else ("warn" if warnings else "pass")
    return {
        "table": name,
        "path": str(path) if path else None,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def build_schema_manifest(
    validated: list[dict[str, Any]],
    seasons: list[int],
    pull_timestamp_utc: str,
) -> dict[str, Any]:
    """Build the schema_manifest.json content from validation results."""
    return {
        "schema_version": "2.0",
        "pull_timestamp_utc": pull_timestamp_utc,
        "seasons": seasons,
        "tables": {
            v["table"]: {
                "path": v["path"],
                "status": v["status"],
                "rows": v["stats"].get("rows", 0),
                "errors": v["errors"],
                "warnings": v["warnings"],
            }
            for v in validated
        },
    }
