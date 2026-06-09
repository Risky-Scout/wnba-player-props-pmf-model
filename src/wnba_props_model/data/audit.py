"""Audit report generators for raw and canonical data layers.

All functions return JSON-serializable dicts.  They never raise; missing
optional data produces explicit statuses rather than exceptions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def null_counts(df: pd.DataFrame) -> dict[str, int]:
    return {c: int(n) for c, n in df.isnull().sum().items() if n > 0}


def dup_count(df: pd.DataFrame, keys: list[str]) -> int:
    valid = [k for k in keys if k in df.columns]
    return int(df.duplicated(subset=valid).sum()) if valid else 0


def season_row_counts(df: pd.DataFrame) -> dict[int, int]:
    if "season" not in df.columns:
        return {}
    return {int(k): int(v) for k, v in df.groupby("season").size().items()}


def _safe_min_max(df: pd.DataFrame, col: str) -> tuple[str | None, str | None]:
    if col not in df.columns or df.empty:
        return None, None
    try:
        return str(df[col].min()), str(df[col].max())
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# Per-table raw audits
# ---------------------------------------------------------------------------

def audit_raw_games(df: pd.DataFrame) -> dict[str, Any]:
    date_min, date_max = _safe_min_max(df, "game_date")
    by_status: dict[str, int] = {}
    if "status_normalized" in df.columns:
        by_status = {str(k): int(v) for k, v in df["status_normalized"].value_counts().items()}
    elif "status" in df.columns:
        by_status = {str(k): int(v) for k, v in df["status"].value_counts().items()}
    has_score = int(df["has_final_score"].sum()) if "has_final_score" in df.columns else None
    return {
        "rows": len(df),
        "seasons": sorted(int(s) for s in df["season"].dropna().unique()) if "season" in df.columns else [],
        "rows_by_season": season_row_counts(df),
        "date_min": date_min,
        "date_max": date_max,
        "games_by_status_normalized": by_status,
        "games_with_final_score": has_score,
        "games_without_final_score": (len(df) - has_score) if has_score is not None else None,
        "duplicate_game_ids": dup_count(df, ["game_id"]),
        "null_counts": null_counts(df),
    }


def audit_raw_player_stats(df: pd.DataFrame) -> dict[str, Any]:
    date_min, date_max = _safe_min_max(df, "game_date")
    zero_mask = df["minutes"] == 0.0 if "minutes" in df.columns else pd.Series([], dtype=bool)

    minutes_flags: dict[str, int] = {}
    zero_by_flag: dict[str, int] = {}
    if "minutes_flag" in df.columns:
        minutes_flags = {
            "clean_no_flag": int(df["minutes_flag"].isna().sum()),
            "non_playing": int((df["minutes_flag"] == "non_playing").sum()),
            "null": int((df["minutes_flag"] == "null").sum()),
            "empty": int((df["minutes_flag"] == "empty").sum()),
            "parse_error": int((df["minutes_flag"] == "parse_error").sum()),
        }
        if "minutes_raw" in df.columns:
            zero_by_flag = {
                "non_playing_flag": int((zero_mask & (df["minutes_flag"] == "non_playing")).sum()),
                "clean_raw_zero": int(
                    (zero_mask & df["minutes_flag"].isna() & (df["minutes_raw"] == "0")).sum()
                ),
                "null_flag": int((zero_mask & (df["minutes_flag"] == "null")).sum()),
                "empty_flag": int((zero_mask & (df["minutes_flag"] == "empty")).sum()),
                "parse_error": int((zero_mask & (df["minutes_flag"] == "parse_error")).sum()),
                "total": int(zero_mask.sum()),
            }

    # All-zero stat line
    stat_cols = [c for c in ["pts", "reb", "ast", "fg3m", "stl", "blk"] if c in df.columns]
    all_zero_stats = 0
    if stat_cols and not df.empty:
        all_zero_stats = int((df[stat_cols].sum(axis=1) == 0).sum())

    return {
        "rows": len(df),
        "unique_players": int(df["player_id"].nunique()) if "player_id" in df.columns else None,
        "unique_games": int(df["game_id"].nunique()) if "game_id" in df.columns else None,
        "seasons": sorted(int(s) for s in df["season"].dropna().unique()) if "season" in df.columns else [],
        "rows_by_season": season_row_counts(df),
        "date_min": date_min,
        "date_max": date_max,
        "duplicate_player_game_pairs": dup_count(df, ["player_id", "game_id"]),
        "null_counts": null_counts(df),
        "minutes_flag_breakdown": minutes_flags,
        "zero_minute_breakdown": zero_by_flag,
        "all_zero_stat_rows": all_zero_stats,
    }


def audit_raw_teams(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "unique_team_ids": int(df["team_id"].nunique()) if "team_id" in df.columns else None,
        "duplicate_team_ids": dup_count(df, ["team_id"]),
        "missing_abbreviations": int(df["team_abbreviation"].isna().sum())
        if "team_abbreviation" in df.columns else None,
        "null_counts": null_counts(df),
    }


def audit_raw_players(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "unique_player_ids": int(df["player_id"].nunique()) if "player_id" in df.columns else None,
        "duplicate_player_ids": dup_count(df, ["player_id"]),
        "missing_names": int(df["player_name"].isna().sum()) if "player_name" in df.columns else None,
        "missing_positions": int(df["position"].isna().sum()) if "position" in df.columns else None,
        "null_counts": null_counts(df),
    }


def audit_raw_injuries(df: pd.DataFrame) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    if "injury_status_normalized" in df.columns:
        by_status = {
            str(k): int(v) for k, v in df["injury_status_normalized"].value_counts().items()
        }
    return {
        "rows": len(df),
        "unique_players": int(df["player_id"].nunique()) if "player_id" in df.columns else None,
        "by_status_normalized": by_status,
        "null_counts": null_counts(df),
    }


def audit_raw_odds(df: pd.DataFrame) -> dict[str, Any]:
    books: list[str] = []
    if "book" in df.columns:
        books = sorted(str(b) for b in df["book"].dropna().unique())
    has_total = int(df["total_value"].notna().sum()) if "total_value" in df.columns else None
    has_spread = int(df["spread_value"].notna().sum()) if "spread_value" in df.columns else None
    has_ml = int(df["moneyline_home_odds"].notna().sum()) if "moneyline_home_odds" in df.columns else None
    return {
        "rows": len(df),
        "unique_games": int(df["game_id"].nunique()) if "game_id" in df.columns else None,
        "books": books,
        "rows_with_total": has_total,
        "rows_with_spread": has_spread,
        "rows_with_moneyline": has_ml,
        "duplicate_game_book_rows": dup_count(df, ["game_id", "book"]),
        "null_counts": null_counts(df),
    }


def audit_raw_player_props(df: pd.DataFrame) -> dict[str, Any]:
    stat_counts: dict[str, int] = {}
    if "stat" in df.columns:
        stat_counts = {
            str(k): int(v) for k, v in df["stat"].value_counts().items()
        }
    books: list[str] = []
    if "book" in df.columns:
        books = sorted(str(b) for b in df["book"].dropna().unique())
    return {
        "rows": len(df),
        "unique_players": int(df["player_id"].nunique()) if "player_id" in df.columns else None,
        "unique_games": int(df["game_id"].nunique()) if "game_id" in df.columns else None,
        "stat_counts": stat_counts,
        "books": books,
        "null_counts": null_counts(df),
    }


# ---------------------------------------------------------------------------
# Cross-table mismatch audits
# ---------------------------------------------------------------------------

def audit_games_vs_stats(
    games_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict[str, Any]:
    """Explain why game count > games-with-player-stats count."""
    if games_df.empty or stats_df.empty:
        return {"status": "insufficient_data"}

    games_with_stats = set(stats_df["game_id"].dropna().unique())
    total_games = len(games_df)
    final_games = int(games_df.get("is_played_game", pd.Series(dtype=bool)).sum()) if "is_played_game" in games_df.columns else None

    breakdown: dict[str, int] = {}
    if "status_normalized" in games_df.columns:
        for status, group in games_df.groupby("status_normalized"):
            n_with_stats = group["game_id"].isin(games_with_stats).sum()
            breakdown[str(status)] = {
                "total": len(group),
                "with_player_stats": int(n_with_stats),
                "without_player_stats": len(group) - int(n_with_stats),
            }

    return {
        "total_game_rows": total_games,
        "games_with_player_stats": len(games_with_stats),
        "games_without_player_stats": total_games - len(games_with_stats),
        "final_games": final_games,
        "breakdown_by_status": breakdown,
        "explanation": (
            "Games without player stats are typically: (1) future/scheduled games, "
            "(2) games where BDL player_stats endpoint returned no data yet, "
            "or (3) postponed/canceled games."
        ),
    }


def audit_players_vs_stats(
    players_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict[str, Any]:
    """Find player IDs in stats that are missing from players table."""
    if players_df.empty or stats_df.empty:
        return {"status": "insufficient_data"}

    known_ids = set(players_df["player_id"].dropna().unique())
    stats_ids = set(stats_df["player_id"].dropna().unique())
    missing = sorted(int(x) for x in (stats_ids - known_ids))
    return {
        "players_in_players_table": len(known_ids),
        "players_in_stats": len(stats_ids),
        "players_in_stats_missing_from_players_table": len(missing),
        "missing_player_ids_sample": missing[:20],
    }


def audit_teams_vs_games(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict[str, Any]:
    """Find team IDs in games that are missing from teams table."""
    if teams_df.empty or games_df.empty:
        return {"status": "insufficient_data"}

    known_ids = set(teams_df["team_id"].dropna().unique())
    game_team_ids: set = set()
    for col in ["home_team_id", "visitor_team_id"]:
        if col in games_df.columns:
            game_team_ids |= set(games_df[col].dropna().unique())
    missing = sorted(int(x) for x in (game_team_ids - known_ids))
    return {
        "teams_in_teams_table": len(known_ids),
        "team_ids_in_games": len(game_team_ids),
        "team_ids_in_games_missing_from_teams_table": len(missing),
        "missing_team_ids": missing,
    }
