"""BDL API response normalizers.

Each function accepts a raw list of dicts from the BDL API and returns a
clean, typed pandas DataFrame.  No model features are computed here — this
layer is purely parsing and column renaming.

Provenance columns (source, pull_timestamp_utc) are added by the ingest
layer after normalization, not here.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.constants import (
    INJURY_STATUS_MAP,
    PROP_STAT_NAME_MAP,
    STAT_TO_BDL_COL,
)

# ---------------------------------------------------------------------------
# Minutes parsing  (stable since Stage 1 – do not modify without tests)
# ---------------------------------------------------------------------------

_NON_PLAYING_STRINGS: frozenset[str] = frozenset({
    "--", "-", "dnp", "dnp-cd", "dnp-coach's decision",
    "did not play", "inactive", "out", "scratch", "dnd",
    "did not dress", "not with team", "nwt", "suspension",
    "na", "n/a",
})


def parse_minutes(value: Any) -> float:
    minutes, _ = _parse_minutes_internal(value)
    return minutes


def parse_minutes_flag(value: Any) -> str | None:
    _, flag = _parse_minutes_internal(value)
    return flag


def _parse_minutes_internal(value: Any) -> tuple[float, str | None]:
    if value is None:
        return 0.0, "null"
    if isinstance(value, float) and math.isnan(value):
        return 0.0, "null"
    if isinstance(value, (int, float)):
        return float(value), None
    s = str(value).strip()
    if not s:
        return 0.0, "empty"
    if s.lower() in _NON_PLAYING_STRINGS:
        return 0.0, "non_playing"
    if ":" in s:
        parts = s.split(":", 1)
        try:
            return float(parts[0] or 0) + float(parts[1] or 0) / 60.0, None
        except ValueError:
            return 0.0, "parse_error"
    try:
        return float(s), None
    except ValueError:
        return 0.0, "parse_error"


# ---------------------------------------------------------------------------
# Game status normalization
# ---------------------------------------------------------------------------

_FINAL_TOKENS = frozenset({"final", "f", "f/ot", "final/ot", "post"})
_INPROGRESS_TOKENS = frozenset({"quarter", "half", "overtime", "progress", "halftime", "live"})
_POSTPONED_TOKENS = frozenset({"postpone", "suspend", "delay"})
_CANCELED_TOKENS = frozenset({"cancel", "void", "forfeit"})
_SCHEDULED_TOKENS = frozenset({
    "scheduled", "tbd", "pre game", "pre-game", "pregame",
    "pre",  # BDL WNBA uses "pre" for scheduled/upcoming games
})


def normalize_game_status(status: Any) -> str:
    """Return one of: final, in_progress, scheduled, postponed, canceled, unknown.

    BDL WNBA uses "post" for completed games and "pre" for upcoming games.
    """
    if status is None:
        return "unknown"
    s = str(status).strip()
    if not s:
        return "unknown"
    sl = s.lower()
    if sl in _FINAL_TOKENS or "final" in sl:
        return "final"
    if any(t in sl for t in _INPROGRESS_TOKENS):
        return "in_progress"
    if any(t in sl for t in _POSTPONED_TOKENS):
        return "postponed"
    if any(t in sl for t in _CANCELED_TOKENS):
        return "canceled"
    if any(t in sl for t in _SCHEDULED_TOKENS):
        return "scheduled"
    # Looks like a scheduled time ("7:00 pm et", "2026-05-14T19:30:00")
    if ":" in sl or "pm" in sl or "am" in sl or "et" in sl or "T" in s:
        return "scheduled"
    return "unknown"


# ---------------------------------------------------------------------------
# Injury status normalization
# ---------------------------------------------------------------------------

def normalize_injury_status(status: Any) -> str:
    """Return one of: available, probable, questionable, doubtful, out, inactive, unknown."""
    if status is None:
        return "unknown"
    sl = str(status).strip().lower()
    if not sl:
        return "unknown"
    return INJURY_STATUS_MAP.get(sl, "unknown")


# ---------------------------------------------------------------------------
# Player stat flattening  (stable since Stage 1)
# ---------------------------------------------------------------------------

def flatten_player_stat_row(row: dict[str, Any]) -> dict[str, Any]:
    player = row.get("player") or {}
    team = row.get("team") or {}
    game = row.get("game") or {}

    raw_min = row.get("min")
    minutes_value = parse_minutes(raw_min)
    minutes_flag = parse_minutes_flag(raw_min)

    out = {
        "player_id": player.get("id") or row.get("player_id"),
        "player_name": " ".join(x for x in [player.get("first_name"), player.get("last_name")] if x),
        "team_id": team.get("id") or row.get("team_id"),
        "team_abbr": team.get("abbreviation"),
        "game_id": game.get("id") or row.get("game_id"),
        "game_date": pd.to_datetime(
            game.get("date") or row.get("date"), utc=True, errors="coerce"
        ).date(),
        "season": game.get("season") or row.get("season"),
        "position": player.get("position_abbreviation") or player.get("position"),
        "minutes": minutes_value,
        "minutes_raw": str(raw_min) if raw_min is not None else None,
        "minutes_flag": minutes_flag,
    }
    for stat, col in STAT_TO_BDL_COL.items():
        val = row.get(col)
        out[stat] = 0 if val is None or (isinstance(val, float) and math.isnan(val)) else val
    out["oreb"] = _coerce_int(row.get("oreb"))
    out["dreb"] = _coerce_int(row.get("dreb"))
    out["fga"] = _coerce_int(row.get("fga"))
    out["fta"] = _coerce_int(row.get("fta"))
    out["pf"] = _coerce_int(row.get("pf"))
    out["plus_minus"] = row.get("plus_minus")
    return out


def _coerce_int(v: Any) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def normalize_games(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        home = r.get("home_team") or {}
        away = r.get("visitor_team") or r.get("away_team") or {}
        home_score = r.get("home_team_score") or r.get("home_score")
        away_score = (
            r.get("visitor_team_score")
            or r.get("away_team_score")
            or r.get("away_score")
        )
        status_raw = r.get("status")
        status_norm = normalize_game_status(status_raw)
        is_final = status_norm == "final"
        home_score_n = _to_numeric(home_score) if is_final else None
        away_score_n = _to_numeric(away_score) if is_final else None
        flat.append({
            "game_id": r.get("id"),
            "game_date": pd.to_datetime(r.get("date"), utc=True, errors="coerce"),
            "season": r.get("season"),
            "status": status_raw,
            "status_normalized": status_norm,
            "postseason": bool(r.get("postseason", False)),
            "home_team_id": home.get("id") or r.get("home_team_id"),
            "visitor_team_id": away.get("id") or r.get("visitor_team_id") or r.get("away_team_id"),
            "home_team_abbreviation": home.get("abbreviation"),
            "visitor_team_abbreviation": away.get("abbreviation"),
            "home_team_score": home_score_n,
            "visitor_team_score": away_score_n,
            "total_score": (home_score_n + away_score_n) if is_final else None,
            "has_final_score": is_final,
            "is_played_game": is_final,
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player stats  (aggregate + combos)
# ---------------------------------------------------------------------------

def normalize_player_stats(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_player_stat_row(r) for r in rows])
    if df.empty:
        return df
    stat_cols = list(STAT_TO_BDL_COL)
    for c in stat_cols + ["oreb", "dreb", "fga", "fta", "pf", "minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Combo stats (raw names; canonical names applied in build_canonical_tables)
    df["pa"] = df["pts"] + df["ast"]
    df["pr"] = df["pts"] + df["reb"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["stocks"] = df["stl"] + df["blk"]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def normalize_teams(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        flat.append({
            "team_id": r.get("id"),
            "team_abbreviation": r.get("abbreviation"),
            "team_name": r.get("name"),
            "team_full_name": r.get("full_name"),
            "city": r.get("city"),
            "conference": r.get("conference"),
            "division": r.get("division"),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return df.sort_values("team_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

def normalize_players(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        team = r.get("team") or {}
        flat.append({
            "player_id": r.get("id"),
            "first_name": r.get("first_name"),
            "last_name": r.get("last_name"),
            "player_name": " ".join(
                x for x in [r.get("first_name"), r.get("last_name")] if x
            ),
            "position": r.get("position"),
            "position_abbreviation": r.get("position_abbreviation") or r.get("position"),
            "height": r.get("height"),
            "weight": r.get("weight"),
            "jersey_number": r.get("jersey_number"),
            "college": r.get("college"),
            "country": r.get("country"),
            "draft_year": r.get("draft_year"),
            "draft_round": r.get("draft_round"),
            "draft_number": r.get("draft_number"),
            "team_id": team.get("id") or r.get("team_id"),
            "team_abbreviation": team.get("abbreviation"),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return df.sort_values("player_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

def normalize_injuries(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        player = r.get("player") or {}
        team = r.get("team") or {}
        status_raw = r.get("status")
        flat.append({
            "player_id": player.get("id") or r.get("player_id"),
            "player_name": " ".join(
                x for x in [player.get("first_name"), player.get("last_name")] if x
            ),
            "team_id": team.get("id") or r.get("team_id"),
            "team_abbreviation": team.get("abbreviation"),
            "game_id": r.get("game_id"),
            "report_date": pd.to_datetime(
                r.get("report_date") or r.get("date"), utc=True, errors="coerce"
            ),
            "return_date": pd.to_datetime(
                r.get("return_date") or r.get("return_date_estimate"),
                utc=True, errors="coerce",
            ),
            "injury_status": status_raw,
            "injury_status_normalized": normalize_injury_status(status_raw),
            "injury_description": r.get("description") or r.get("notes"),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return df.sort_values(
        ["report_date", "player_id"], na_position="last"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Game odds
# ---------------------------------------------------------------------------

def normalize_odds(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        game = r.get("game") or {}
        spread = r.get("spread") or {}
        total = r.get("total") or {}
        ml = r.get("moneyline") or {}
        game_date = pd.to_datetime(
            game.get("date") or r.get("date"), utc=True, errors="coerce"
        )
        flat.append({
            "game_id": game.get("id") or r.get("game_id"),
            "game_date": game_date,
            "season": game.get("season") or r.get("season"),
            "book": r.get("book") or r.get("sportsbook"),
            "sportsbook": r.get("sportsbook") or r.get("book"),
            "spread_value": _to_numeric(spread.get("home_spread") or r.get("home_spread")),
            "spread_home_odds": _to_numeric(spread.get("home_odds") or r.get("spread_home_odds")),
            "spread_visitor_odds": _to_numeric(
                spread.get("visitor_odds") or r.get("spread_visitor_odds")
            ),
            "total_value": _to_numeric(total.get("value") or r.get("total_value")),
            "total_over_odds": _to_numeric(total.get("over_odds") or r.get("total_over_odds")),
            "total_under_odds": _to_numeric(total.get("under_odds") or r.get("total_under_odds")),
            "moneyline_home_odds": _to_numeric(ml.get("home_odds") or r.get("moneyline_home_odds")),
            "moneyline_visitor_odds": _to_numeric(
                ml.get("visitor_odds") or r.get("moneyline_visitor_odds")
            ),
            "snapshot_timestamp_utc": pd.to_datetime(
                r.get("snapshot_timestamp") or r.get("updated_at"),
                utc=True, errors="coerce",
            ),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player props  (evaluation-only table)
# ---------------------------------------------------------------------------

def normalize_player_props(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        player = r.get("player") or {}
        game = r.get("game") or {}
        team = r.get("team") or {}
        raw_market = r.get("market") or r.get("prop_type") or r.get("stat_type") or ""
        canonical_stat = PROP_STAT_NAME_MAP.get(str(raw_market).lower().replace(" ", "_"), raw_market)
        flat.append({
            "market_id": r.get("id") or r.get("market_id"),
            "game_id": game.get("id") or r.get("game_id"),
            "game_date": pd.to_datetime(
                game.get("date") or r.get("date"), utc=True, errors="coerce"
            ),
            "season": game.get("season") or r.get("season"),
            "player_id": player.get("id") or r.get("player_id"),
            "player_name": " ".join(
                x for x in [player.get("first_name"), player.get("last_name")] if x
            ),
            "team_id": team.get("id") or r.get("team_id"),
            "team_abbreviation": team.get("abbreviation"),
            "market_raw": raw_market,
            "stat": canonical_stat,
            "line": _to_numeric(r.get("line") or r.get("value")),
            "over_odds": _to_numeric(r.get("over_odds") or r.get("over")),
            "under_odds": _to_numeric(r.get("under_odds") or r.get("under")),
            "book": r.get("book") or r.get("sportsbook"),
            "sportsbook": r.get("sportsbook") or r.get("book"),
            "snapshot_timestamp_utc": pd.to_datetime(
                r.get("snapshot_timestamp") or r.get("updated_at"),
                utc=True, errors="coerce",
            ),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id", "player_id", "stat"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Advanced stats (per-game)
# ---------------------------------------------------------------------------

def normalize_advanced_stats(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        player = r.get("player") or {}
        team = r.get("team") or {}
        game = r.get("game") or {}
        flat.append({
            "game_id": game.get("id") or r.get("game_id"),
            "game_date": pd.to_datetime(
                game.get("date") or r.get("date"), utc=True, errors="coerce"
            ),
            "season": game.get("season") or r.get("season"),
            "player_id": player.get("id") or r.get("player_id"),
            "player_name": " ".join(
                x for x in [player.get("first_name"), player.get("last_name")] if x
            ),
            "team_id": team.get("id") or r.get("team_id"),
            "usage_percentage": _to_numeric(r.get("usage_percentage") or r.get("usg_pct")),
            "pace": _to_numeric(r.get("pace")),
            "offensive_rating": _to_numeric(r.get("offensive_rating") or r.get("off_rtg")),
            "defensive_rating": _to_numeric(r.get("defensive_rating") or r.get("def_rtg")),
            "true_shooting_percentage": _to_numeric(
                r.get("true_shooting_percentage") or r.get("ts_pct")
            ),
            "assist_percentage": _to_numeric(r.get("assist_percentage") or r.get("ast_pct")),
            "rebound_percentage": _to_numeric(r.get("rebound_percentage") or r.get("reb_pct")),
            "pie": _to_numeric(r.get("pie")),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Play-by-play
# ---------------------------------------------------------------------------

def normalize_plays(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        player = r.get("player") or {}
        team = r.get("team") or {}
        flat.append({
            "game_id": r.get("game_id"),
            "event_order": r.get("order") or r.get("event_order") or r.get("id"),
            "period": r.get("period"),
            "clock": r.get("clock"),
            "event_type": r.get("type") or r.get("event_type"),
            "description": r.get("description"),
            "player_id": player.get("id") or r.get("player_id"),
            "team_id": team.get("id") or r.get("team_id"),
            "points_scored": _to_numeric(r.get("pts") or r.get("points_scored")),
            "home_score": _to_numeric(r.get("home_score")),
            "visitor_score": _to_numeric(r.get("visitor_score")),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return df.sort_values(["game_id", "event_order"], na_position="last").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Shot locations
# ---------------------------------------------------------------------------

def normalize_shot_locations(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        player = r.get("player") or {}
        team = r.get("team") or {}
        game = r.get("game") or {}
        flat.append({
            "game_id": game.get("id") or r.get("game_id"),
            "game_date": pd.to_datetime(
                game.get("date") or r.get("date"), utc=True, errors="coerce"
            ),
            "season": game.get("season") or r.get("season"),
            "player_id": player.get("id") or r.get("player_id"),
            "team_id": team.get("id") or r.get("team_id"),
            "x": _to_numeric(r.get("x")),
            "y": _to_numeric(r.get("y")),
            "shot_made": r.get("shot_made") or r.get("made"),
            "shot_type": r.get("shot_type") or r.get("type"),
            "shot_zone": r.get("shot_zone") or r.get("zone"),
            "three_point_flag": bool(r.get("three_point") or r.get("is_three_point", False)),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def normalize_standings(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        team = r.get("team") or {}
        flat.append({
            "season": r.get("season"),
            "team_id": team.get("id") or r.get("team_id"),
            "team_abbreviation": team.get("abbreviation"),
            "conference": r.get("conference"),
            "wins": _to_numeric(r.get("wins") or r.get("w")),
            "losses": _to_numeric(r.get("losses") or r.get("l")),
            "win_pct": _to_numeric(r.get("win_pct") or r.get("pct")),
            "conference_rank": _to_numeric(r.get("conference_rank") or r.get("rank")),
            "games_behind": _to_numeric(r.get("games_behind") or r.get("gb")),
            "home_record": r.get("home_record"),
            "road_record": r.get("road_record") or r.get("away_record"),
            "streak": r.get("streak"),
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return df.sort_values(["season", "conference_rank"], na_position="last").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_rate(numer: pd.Series, denom: pd.Series, floor: float = 1e-6) -> pd.Series:
    return numer.astype(float) / np.maximum(denom.astype(float), floor)


def _to_numeric(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None
