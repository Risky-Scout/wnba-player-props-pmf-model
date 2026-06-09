"""Tests for Stage 2 canonical schema, normalization, and leakage guards."""
from __future__ import annotations

import pytest
import pandas as pd

from wnba_props_model.constants import FORBIDDEN_MARKET_COLUMNS, PROP_STAT_NAME_MAP
from wnba_props_model.data.normalize import (
    normalize_game_status,
    normalize_injury_status,
    normalize_games,
    normalize_teams,
    normalize_players,
    normalize_injuries,
    normalize_odds,
    normalize_player_props,
    normalize_advanced_stats,
)
from wnba_props_model.data.schema import (
    ALL_SCHEMAS,
    GAMES_SCHEMA,
    PLAYER_GAME_STATS_SCHEMA,
    REQUIRED_TABLES,
    validate_table,
)
from wnba_props_model.data.audit import (
    audit_games_vs_stats,
    audit_players_vs_stats,
    null_counts,
    dup_count,
)
from wnba_props_model.features.feature_contract import (
    assert_no_forbidden_features,
    assert_no_market_columns,
    FORBIDDEN_MODEL_FEATURES,
)


# ===========================================================================
# 1. Game status normalization
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    # Standard text formats
    ("Final", "final"),
    ("final", "final"),
    ("F", "final"),
    ("F/OT", "final"),
    ("Final - OT", "final"),
    # BDL WNBA-specific codes
    ("post", "final"),   # BDL WNBA: game is complete
    ("pre", "scheduled"),  # BDL WNBA: game is upcoming
    ("live", "in_progress"),
    # In-progress
    ("4th Quarter", "in_progress"),
    ("Halftime", "in_progress"),
    ("In Progress", "in_progress"),
    ("1st Quarter", "in_progress"),
    # Scheduled
    ("Scheduled", "scheduled"),
    ("7:00 pm et", "scheduled"),
    ("2026-05-14T19:30:00", "scheduled"),
    # Postponed / canceled
    ("Postponed", "postponed"),
    ("Suspended", "postponed"),
    ("Canceled", "canceled"),
    ("Cancelled", "canceled"),
    # Edge cases
    ("", "unknown"),
    (None, "unknown"),
    ("foobar", "unknown"),
])
def test_normalize_game_status(raw, expected):
    assert normalize_game_status(raw) == expected, f"normalize_game_status({raw!r})"


def test_normalize_games_adds_status_normalized():
    rows = [
        {
            "id": 1, "date": "2024-05-14", "season": 2024, "status": "Final",
            "home_team": {"id": 10, "abbreviation": "LVA"},
            "visitor_team": {"id": 20, "abbreviation": "SEA"},
            "home_team_score": 85, "visitor_team_score": 79,
        },
        {
            "id": 2, "date": "2026-06-10", "season": 2026, "status": "7:00 pm et",
            "home_team": {"id": 10, "abbreviation": "LVA"},
            "visitor_team": {"id": 20, "abbreviation": "SEA"},
            "home_team_score": None, "visitor_team_score": None,
        },
    ]
    df = normalize_games(rows)
    assert "status_normalized" in df.columns
    assert df.loc[df["game_id"] == 1, "status_normalized"].iloc[0] == "final"
    assert df.loc[df["game_id"] == 2, "status_normalized"].iloc[0] == "scheduled"


def test_normalize_games_has_final_score_flags():
    rows = [
        {
            "id": 1, "date": "2024-05-14", "season": 2024, "status": "Final",
            "home_team": {"id": 10, "abbreviation": "LVA"},
            "visitor_team": {"id": 20, "abbreviation": "SEA"},
            "home_team_score": 85, "visitor_team_score": 79,
        },
        {
            "id": 2, "date": "2026-06-10", "season": 2026, "status": "Scheduled",
            "home_team": {"id": 10, "abbreviation": "LVA"},
            "visitor_team": {"id": 20, "abbreviation": "SEA"},
            "home_team_score": None, "visitor_team_score": None,
        },
    ]
    df = normalize_games(rows)
    assert bool(df.loc[df["game_id"] == 1, "has_final_score"].iloc[0]) is True
    assert bool(df.loc[df["game_id"] == 2, "has_final_score"].iloc[0]) is False
    assert bool(df.loc[df["game_id"] == 1, "is_played_game"].iloc[0]) is True
    assert bool(df.loc[df["game_id"] == 2, "is_played_game"].iloc[0]) is False


# ===========================================================================
# 2. Injury status normalization
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    ("Active", "available"),
    ("active", "available"),
    ("Probable", "probable"),
    ("Questionable", "questionable"),
    ("GTD", "questionable"),
    ("day-to-day", "questionable"),
    ("Doubtful", "doubtful"),
    ("Out", "out"),
    ("Injured Reserve", "out"),
    ("Inactive", "inactive"),
    ("Not With Team", "inactive"),
    (None, "unknown"),
    ("", "unknown"),
    ("foobar", "unknown"),
])
def test_normalize_injury_status(raw, expected):
    assert normalize_injury_status(raw) == expected, f"normalize_injury_status({raw!r})"


def test_normalize_injuries_produces_normalized_status():
    rows = [
        {
            "player": {"id": 1, "first_name": "A'ja", "last_name": "Wilson"},
            "team": {"id": 10, "abbreviation": "LVA"},
            "status": "Questionable",
            "description": "Ankle",
        }
    ]
    df = normalize_injuries(rows)
    assert "injury_status_normalized" in df.columns
    assert df["injury_status_normalized"].iloc[0] == "questionable"


# ===========================================================================
# 3. Prop stat name normalization (PROP_STAT_NAME_MAP)
# ===========================================================================

@pytest.mark.parametrize("raw,expected_canonical", [
    ("points", "pts"),
    ("rebounds", "reb"),
    ("assists", "ast"),
    ("threes", "fg3m"),
    ("three_pointers_made", "fg3m"),
    ("steals", "stl"),
    ("blocks", "blk"),
    ("turnovers", "turnover"),
    ("points_rebounds", "pts_reb"),
    ("points_assists", "pts_ast"),
    ("rebounds_assists", "reb_ast"),
    ("points_rebounds_assists", "pts_reb_ast"),
])
def test_prop_stat_name_map(raw, expected_canonical):
    assert PROP_STAT_NAME_MAP[raw] == expected_canonical, f"PROP_STAT_NAME_MAP[{raw!r}]"


def test_normalize_player_props_stat_name():
    rows = [
        {
            "player": {"id": 1, "first_name": "A'ja", "last_name": "Wilson"},
            "game": {"id": 100, "date": "2024-05-14", "season": 2024},
            "team": {"id": 10, "abbreviation": "LVA"},
            "market": "points",
            "line": 22.5,
            "over_odds": -110,
            "under_odds": -110,
            "sportsbook": "DraftKings",
        }
    ]
    df = normalize_player_props(rows)
    assert df["stat"].iloc[0] == "pts"
    assert df["line"].iloc[0] == 22.5


# ===========================================================================
# 4. Leakage guard — forbidden features
# ===========================================================================

def test_forbidden_market_feature_raises():
    with pytest.raises(ValueError, match="market_line"):
        assert_no_forbidden_features(["market_line"])


@pytest.mark.parametrize("col", [
    "line", "over_odds", "under_odds", "market_id", "book",
    "no_vig_prob_over", "edge", "clv", "hit_result", "outcome",
    "total_value", "spread_value", "moneyline_home_odds",
])
def test_each_forbidden_market_column_raises(col):
    with pytest.raises(ValueError):
        assert_no_forbidden_features([col])


@pytest.mark.parametrize("col", [
    "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
    "final_score", "home_score", "home_team_score",
])
def test_box_score_leakage_raises(col):
    with pytest.raises(ValueError):
        assert_no_forbidden_features([col])


def test_clean_feature_list_does_not_raise():
    clean = [
        "minutes_roll5", "pts_per_min_roll5", "rest_days",
        "is_home", "opp_pts_allowed_roll5",
    ]
    assert_no_forbidden_features(clean)  # must not raise


def test_assert_no_market_columns():
    with pytest.raises(ValueError):
        assert_no_market_columns(["line", "minutes_roll5"])


def test_all_forbidden_market_constants_in_feature_guard():
    """Every constant in FORBIDDEN_MARKET_COLUMNS must trigger the leakage guard."""
    for col in FORBIDDEN_MARKET_COLUMNS:
        with pytest.raises(ValueError):
            assert_no_forbidden_features([col])


# ===========================================================================
# 5. Schema definitions — structure checks
# ===========================================================================

def test_all_schemas_have_names():
    for name, schema in ALL_SCHEMAS.items():
        assert schema.name == name


def test_required_tables_are_in_all_schemas():
    for tbl in REQUIRED_TABLES:
        assert tbl in ALL_SCHEMAS, f"Required table {tbl!r} not in ALL_SCHEMAS"


def test_games_schema_required_cols():
    required = set(GAMES_SCHEMA.required_columns)
    expected_subset = {
        "game_id", "season", "game_date", "status_normalized",
        "home_team_id", "visitor_team_id",
        "has_final_score", "is_played_game", "has_player_stats",
        "source", "pull_timestamp_utc",
    }
    assert expected_subset <= required, f"Missing: {expected_subset - required}"


def test_player_stats_schema_required_cols():
    required = set(PLAYER_GAME_STATS_SCHEMA.required_columns)
    expected_subset = {
        "player_id", "game_id", "season", "team_id",
        "opponent_team_id", "is_home", "home_away",
        "minutes", "did_play",
        "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
        "source", "pull_timestamp_utc",
    }
    assert expected_subset <= required, f"Missing: {expected_subset - required}"


# ===========================================================================
# 6. Schema validator
# ===========================================================================

def _make_games_df(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "game_id": i,
            "season": 2024,
            "game_date": pd.Timestamp("2024-05-14"),
            "status": "Final",
            "status_normalized": "final",
            "postseason": False,
            "home_team_id": 10,
            "visitor_team_id": 20,
            "home_team_abbreviation": "LVA",
            "visitor_team_abbreviation": "SEA",
            "home_team_score": 85.0,
            "visitor_team_score": 79.0,
            "total_score": 164.0,
            "has_final_score": True,
            "is_played_game": True,
            "has_player_stats": True,
            "has_odds": False,
            "has_player_props": False,
            "source": "bdl",
            "pull_timestamp_utc": "2026-06-09T00:00:00+00:00",
        })
    return pd.DataFrame(rows)


def test_validate_games_pass():
    df = _make_games_df(5)
    result = validate_table(df, GAMES_SCHEMA)
    assert result["status"] == "pass", result["errors"]


def test_validate_games_fail_missing_col():
    df = _make_games_df(3).drop(columns=["game_id"])
    result = validate_table(df, GAMES_SCHEMA)
    assert result["status"] == "fail"
    assert any("game_id" in e for e in result["errors"])


def test_validate_games_fail_duplicate_pk():
    df = _make_games_df(3)
    df = pd.concat([df, df.head(1)], ignore_index=True)
    result = validate_table(df, GAMES_SCHEMA)
    assert result["status"] == "fail"
    assert any("Duplicate" in e for e in result["errors"])


def test_validate_games_fail_bad_status_normalized():
    df = _make_games_df(3)
    df.loc[0, "status_normalized"] = "ILLEGAL_STATUS"
    result = validate_table(df, GAMES_SCHEMA)
    assert result["status"] == "fail"


def test_validate_negative_stats_fails():
    schema = PLAYER_GAME_STATS_SCHEMA
    rows = [{
        "game_id": 100, "game_date": pd.Timestamp("2024-05-14"), "season": 2024,
        "player_id": 1, "player_name": "A'ja Wilson",
        "team_id": 10, "team_abbreviation": "LVA",
        "opponent_team_id": 20, "is_home": True, "home_away": "home",
        "position": "F", "minutes": 32.0, "minutes_raw": "32:00", "minutes_flag": None,
        "did_play": True,
        "pts": -5,  # impossible
        "reb": 7, "ast": 3, "fg3m": 1, "stl": 1, "blk": 0, "turnover": 2,
        "source": "bdl", "pull_timestamp_utc": "2026-06-09T00:00:00+00:00",
    }]
    df = pd.DataFrame(rows)
    result = validate_table(df, schema)
    assert result["status"] == "fail"
    assert any("negative" in e for e in result["errors"])


# ===========================================================================
# 7. Duplicate primary-key detection
# ===========================================================================

def test_dup_count_detects_dups():
    df = pd.DataFrame({"player_id": [1, 1, 2], "game_id": [100, 100, 100]})
    assert dup_count(df, ["player_id", "game_id"]) == 1


def test_dup_count_no_dups():
    df = pd.DataFrame({"player_id": [1, 2, 3], "game_id": [100, 100, 100]})
    assert dup_count(df, ["player_id", "game_id"]) == 0


def test_dup_count_missing_key_col():
    df = pd.DataFrame({"player_id": [1, 1]})
    # game_id not present — should not crash, return 0
    assert dup_count(df, ["player_id", "game_id"]) == 1  # only player_id used


# ===========================================================================
# 8. Optional endpoint unavailable — no crash
# ===========================================================================

def test_validate_optional_table_skips_gracefully():
    """validate_table on an empty DataFrame should not crash."""
    result = validate_table(pd.DataFrame(), ALL_SCHEMAS["wnba_injuries"])
    assert result["status"] in ("pass", "fail", "warn")
    assert "rows" in result["stats"]
    assert result["stats"]["rows"] == 0


def test_missing_optional_table_produces_skipped_status():
    """Schema validator treats optional tables as 'skipped', not 'fail'."""
    assert "wnba_injuries" not in REQUIRED_TABLES


# ===========================================================================
# 9. Cross-table audit helpers
# ===========================================================================

def test_audit_games_vs_stats_mismatch():
    games = pd.DataFrame({
        "game_id": [1, 2, 3],
        "status_normalized": ["final", "final", "scheduled"],
        "is_played_game": [True, True, False],
    })
    stats = pd.DataFrame({"game_id": [1]})  # only game 1 has stats
    result = audit_games_vs_stats(games, stats)
    assert result["total_game_rows"] == 3
    assert result["games_with_player_stats"] == 1
    assert result["games_without_player_stats"] == 2


def test_audit_players_vs_stats():
    players = pd.DataFrame({"player_id": [1, 2, 3]})
    stats = pd.DataFrame({"player_id": [1, 4]})  # player 4 is unknown
    result = audit_players_vs_stats(players, stats)
    assert result["players_in_stats_missing_from_players_table"] == 1
    assert 4 in result["missing_player_ids_sample"]


# ===========================================================================
# 10. normalize_teams / normalize_players basics
# ===========================================================================

def test_normalize_teams():
    rows = [
        {"id": 1, "abbreviation": "LVA", "name": "Aces", "full_name": "Las Vegas Aces",
         "city": "Las Vegas", "conference": "Western"},
    ]
    df = normalize_teams(rows)
    assert "team_id" in df.columns
    assert df["team_abbreviation"].iloc[0] == "LVA"


def test_normalize_players():
    rows = [
        {"id": 1, "first_name": "A'ja", "last_name": "Wilson",
         "position": "F", "team": {"id": 1, "abbreviation": "LVA"}},
    ]
    df = normalize_players(rows)
    assert "player_id" in df.columns
    assert df["player_name"].iloc[0] == "A'ja Wilson"


def test_normalize_odds():
    rows = [
        {
            "game": {"id": 100, "date": "2024-05-14", "season": 2024},
            "spread": {"home_odds": -110, "visitor_odds": -110, "home_spread": -4.5},
            "total": {"value": 162.5, "over_odds": -110, "under_odds": -110},
            "moneyline": {"home_odds": -185, "visitor_odds": 155},
            "sportsbook": "DraftKings",
        }
    ]
    df = normalize_odds(rows)
    assert df["total_value"].iloc[0] == 162.5
    assert df["spread_value"].iloc[0] == -4.5


def test_normalize_advanced_stats():
    rows = [
        {
            "player": {"id": 1, "first_name": "A'ja", "last_name": "Wilson"},
            "team": {"id": 10, "abbreviation": "LVA"},
            "game": {"id": 100, "date": "2024-05-14", "season": 2024},
            "usage_percentage": 28.5,
            "pace": 96.2,
            "offensive_rating": 112.0,
        }
    ]
    df = normalize_advanced_stats(rows)
    assert df["usage_percentage"].iloc[0] == 28.5
    assert df["game_id"].iloc[0] == 100
