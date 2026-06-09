from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.constants import STAT_TO_BDL_COL

# Canonical strings that indicate a player did not participate in the game.
# All comparisons are done case-insensitively after stripping whitespace.
_NON_PLAYING_STRINGS: frozenset[str] = frozenset({
    "--",
    "-",
    "dnp",
    "dnp-cd",
    "dnp-coach's decision",
    "did not play",
    "inactive",
    "out",
    "scratch",
    "dnd",
    "did not dress",
    "not with team",
    "nwt",
    "suspension",
    "na",
    "n/a",
})


def parse_minutes(value: Any) -> float:
    """Parse a BDL minutes field and return decimal minutes.

    Handles every known BDL format:
      - None / NaN           → 0.0  (flag: null)
      - ""                   → 0.0  (flag: empty)
      - "--" / "DNP" etc.    → 0.0  (flag: non_playing)
      - "12"                 → 12.0
      - "12.5"               → 12.5
      - "12:34"              → 12 + 34/60 ≈ 12.567
      - 12  (int)            → 12.0
      - 12.5 (float)         → 12.5

    Unrecognised strings that cannot be converted to float also return 0.0
    (flag: parse_error).  Use ``parse_minutes_flag`` to retrieve the category.
    """
    minutes, _ = _parse_minutes_internal(value)
    return minutes


def parse_minutes_flag(value: Any) -> str | None:
    """Return the audit flag for a BDL minutes value, or None if clean.

    Flag values:
      None           – parsed successfully (clean numeric or MM:SS)
      "null"         – value was None or NaN
      "empty"        – value was an empty / whitespace-only string
      "non_playing"  – recognised DNP / inactive sentinel
      "parse_error"  – non-empty string that could not be converted to float
    """
    _, flag = _parse_minutes_internal(value)
    return flag


def _parse_minutes_internal(value: Any) -> tuple[float, str | None]:
    """Return (decimal_minutes, audit_flag)."""
    # Null / NaN
    if value is None:
        return 0.0, "null"
    if isinstance(value, float) and math.isnan(value):
        return 0.0, "null"

    # Numeric types pass directly
    if isinstance(value, (int, float)):
        return float(value), None

    s = str(value).strip()

    # Empty string
    if not s:
        return 0.0, "empty"

    # Non-playing sentinels (case-insensitive)
    if s.lower() in _NON_PLAYING_STRINGS:
        return 0.0, "non_playing"

    # MM:SS format
    if ":" in s:
        parts = s.split(":", 1)
        try:
            return float(parts[0] or 0) + float(parts[1] or 0) / 60.0, None
        except ValueError:
            return 0.0, "parse_error"

    # Plain numeric string
    try:
        return float(s), None
    except ValueError:
        return 0.0, "parse_error"


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
        "game_date": pd.to_datetime(game.get("date") or row.get("date"), utc=True, errors="coerce").date(),
        "season": game.get("season") or row.get("season"),
        "position": player.get("position_abbreviation") or player.get("position"),
        "minutes": minutes_value,
        # Audit columns – always present so downstream queries can filter/count.
        "minutes_raw": str(raw_min) if raw_min is not None else None,
        "minutes_flag": minutes_flag,
    }
    for stat, col in STAT_TO_BDL_COL.items():
        out[stat] = 0 if pd.isna(row.get(col)) else row.get(col)
    out["oreb"] = 0 if pd.isna(row.get("oreb")) else row.get("oreb")
    out["dreb"] = 0 if pd.isna(row.get("dreb")) else row.get("dreb")
    out["fga"] = 0 if pd.isna(row.get("fga")) else row.get("fga")
    out["fta"] = 0 if pd.isna(row.get("fta")) else row.get("fta")
    out["pf"] = 0 if pd.isna(row.get("pf")) else row.get("pf")
    out["plus_minus"] = row.get("plus_minus")
    return out


def normalize_player_stats(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_player_stat_row(r) for r in rows])
    if df.empty:
        return df
    stat_cols = list(STAT_TO_BDL_COL)
    for c in stat_cols + ["oreb", "dreb", "fga", "fta", "pf", "minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["fg3m"] = df["fg3m"].fillna(0)
    df["pa"] = df["pts"] + df["ast"]
    df["pr"] = df["pts"] + df["reb"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df["stocks"] = df["stl"] + df["blk"]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id", "team_id", "player_id"]).reset_index(drop=True)


def normalize_games(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        home = r.get("home_team") or {}
        away = r.get("visitor_team") or r.get("away_team") or {}
        home_score = r.get("home_team_score")
        if home_score is None:
            home_score = r.get("home_score")
        away_score = r.get("visitor_team_score")
        if away_score is None:
            away_score = r.get("away_team_score")
        if away_score is None:
            away_score = r.get("away_score")
        flat.append({
            "game_id": r.get("id"),
            "game_date": pd.to_datetime(r.get("date"), utc=True, errors="coerce"),
            "season": r.get("season"),
            "status": r.get("status"),
            "postseason": bool(r.get("postseason", False)),
            "home_team_id": home.get("id") or r.get("home_team_id"),
            "away_team_id": away.get("id") or r.get("visitor_team_id") or r.get("away_team_id"),
            "home_team_abbr": home.get("abbreviation"),
            "away_team_abbr": away.get("abbreviation"),
            "home_score": home_score,
            "away_score": away_score,
        })
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    df["game_total"] = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")
    df["home_team_total"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_team_total"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def safe_rate(numer: pd.Series, denom: pd.Series, floor: float = 1e-6) -> pd.Series:
    return numer.astype(float) / np.maximum(denom.astype(float), floor)
