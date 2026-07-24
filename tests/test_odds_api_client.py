"""Tests for The Odds API v4 client and related normalization helpers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.data.odds_api_client import (
    ODDS_API_TO_STAT,
    OddsAPIClient,
    OddsAPIError,
    get_bookmaker_deep_link,
    normalize_odds_api_props,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EVENTS = [
    {
        "id": "evt001",
        "sport_key": "basketball_wnba",
        "home_team": "Las Vegas Aces",
        "away_team": "New York Liberty",
        "commence_time": "2026-06-24T23:00:00Z",
    }
]

FAKE_PROPS_DATA = {
    "id": "evt001",
    "home_team": "Las Vegas Aces",
    "away_team": "New York Liberty",
    "bookmakers": [
        {
            "key": "draftkings",
            "link": "https://sportsbook.draftkings.com/event/wnba-evt001",
            "markets": [
                {
                    "key": "player_points",
                    "last_update": "2026-06-24T22:00:00Z",
                    "link": "https://sportsbook.draftkings.com/market/pts-evt001",
                    "outcomes": [
                        {
                            "name": "Over",
                            "description": "A'ja Wilson",
                            "price": -115,
                            "point": 22.5,
                            "link": "https://sportsbook.draftkings.com/bet/pts-over-evt001",
                        },
                        {
                            "name": "Under",
                            "description": "A'ja Wilson",
                            "price": -105,
                            "point": 22.5,
                            "link": "https://sportsbook.draftkings.com/bet/pts-under-evt001",
                        },
                    ],
                },
                {
                    "key": "player_rebounds",
                    "last_update": "2026-06-24T22:00:00Z",
                    "link": None,
                    "outcomes": [
                        {
                            "name": "Over",
                            "description": "Breanna Stewart",
                            "price": -110,
                            "point": 8.5,
                            "link": None,
                        },
                        {
                            "name": "Under",
                            "description": "Breanna Stewart",
                            "price": -110,
                            "point": 8.5,
                            "link": None,
                        },
                    ],
                },
            ],
        }
    ],
}

FAKE_SCORES = [
    {
        "id": "evt001",
        "sport_key": "basketball_wnba",
        "home_team": "Las Vegas Aces",
        "away_team": "New York Liberty",
        "completed": False,
        "scores": [
            {"name": "Las Vegas Aces", "score": "48"},
            {"name": "New York Liberty", "score": "52"},
        ],
    }
]


def _make_client(mock_get):
    """Build an OddsAPIClient with a mocked _get method."""
    with patch.dict("os.environ", {"ODDS_API_KEY": "test-key-123"}):
        client = OddsAPIClient()
    client._get = mock_get
    return client


# ---------------------------------------------------------------------------
# Market key mapping
# ---------------------------------------------------------------------------

class TestMarketKeyMapping:
    def test_core_stats_mapped(self):
        assert ODDS_API_TO_STAT["player_points"] == "pts"
        assert ODDS_API_TO_STAT["player_rebounds"] == "reb"
        assert ODDS_API_TO_STAT["player_assists"] == "ast"
        assert ODDS_API_TO_STAT["player_threes"] == "fg3m"

    def test_combo_stats_mapped(self):
        # Pipeline uses full canonical names (pts_reb_ast, pts_ast) not short aliases (pra, pa)
        assert ODDS_API_TO_STAT["player_points_rebounds_assists"] == "pts_reb_ast"
        assert ODDS_API_TO_STAT["player_points_assists"] == "pts_ast"
        assert ODDS_API_TO_STAT["player_blocks_steals"] == "stocks"

    def test_alternate_lines_mapped(self):
        assert ODDS_API_TO_STAT["player_points_alternate"] == "pts"
        assert ODDS_API_TO_STAT["player_rebounds_alternate"] == "reb"


# ---------------------------------------------------------------------------
# OddsAPIClient init
# ---------------------------------------------------------------------------

class TestOddsAPIClientInit:
    def test_raises_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove ODDS_API_KEY from env
            import os
            old = os.environ.pop("ODDS_API_KEY", None)
            try:
                with pytest.raises(OddsAPIError, match="No ODDS_API_KEY"):
                    OddsAPIClient(api_key="")
            finally:
                if old is not None:
                    os.environ["ODDS_API_KEY"] = old

    def test_init_from_env(self):
        with patch.dict("os.environ", {"ODDS_API_KEY": "my-key"}):
            client = OddsAPIClient()
        assert client.api_key == "my-key"

    def test_init_from_param(self):
        with patch.dict("os.environ", {}, clear=True):
            client = OddsAPIClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# list_events_for_date
# ---------------------------------------------------------------------------

class TestListEventsForDate:
    def test_returns_events_list(self):
        mock_get = MagicMock(return_value=FAKE_EVENTS)
        client = _make_client(mock_get)
        events = client.list_events_for_date("2026-06-24")
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["id"] == "evt001"

    def test_uses_correct_time_bounds(self):
        mock_get = MagicMock(return_value=[])
        client = _make_client(mock_get)
        client.list_events_for_date("2026-06-24")
        _, kwargs = mock_get.call_args
        # Should have been called with path arg
        args = mock_get.call_args[0]
        assert "events" in args[0]


# ---------------------------------------------------------------------------
# get_event_player_props
# ---------------------------------------------------------------------------

class TestGetEventPlayerProps:
    def test_returns_bookmaker_data(self):
        mock_get = MagicMock(return_value=FAKE_PROPS_DATA)
        client = _make_client(mock_get)
        data = client.get_event_player_props("evt001")
        assert "bookmakers" in data
        assert len(data["bookmakers"]) == 1
        assert data["bookmakers"][0]["key"] == "draftkings"

    def test_include_links_param_sent(self):
        mock_get = MagicMock(return_value=FAKE_PROPS_DATA)
        client = _make_client(mock_get)
        client.get_event_player_props("evt001", include_links=True)
        params = mock_get.call_args[0][1]
        assert params.get("includeLinks") == "true"

    def test_specific_bookmakers_param(self):
        mock_get = MagicMock(return_value=FAKE_PROPS_DATA)
        client = _make_client(mock_get)
        client.get_event_player_props("evt001", bookmakers=["draftkings"])
        params = mock_get.call_args[0][1]
        assert params.get("bookmakers") == "draftkings"


# ---------------------------------------------------------------------------
# get_all_props_for_date
# ---------------------------------------------------------------------------

class TestGetAllPropsForDate:
    def test_returns_flat_rows_per_outcome(self):
        def mock_get(path, params=None):
            if "events" in path and "odds" not in path:
                return FAKE_EVENTS
            return FAKE_PROPS_DATA

        client = _make_client(mock_get)
        rows = client.get_all_props_for_date("2026-06-24")

        # 2 markets × 2 outcomes = 4 rows
        assert len(rows) == 4

    def test_rows_contain_stat_field(self):
        def mock_get(path, params=None):
            if "events" in path and "odds" not in path:
                return FAKE_EVENTS
            return FAKE_PROPS_DATA

        client = _make_client(mock_get)
        rows = client.get_all_props_for_date("2026-06-24")
        stats = {r["stat"] for r in rows}
        assert "pts" in stats
        assert "reb" in stats

    def test_rows_contain_deep_links(self):
        def mock_get(path, params=None):
            if "events" in path and "odds" not in path:
                return FAKE_EVENTS
            return FAKE_PROPS_DATA

        client = _make_client(mock_get)
        rows = client.get_all_props_for_date("2026-06-24")
        # At least one row should have an outcome_link
        links = [r.get("outcome_link") for r in rows if r.get("outcome_link")]
        assert len(links) > 0

    def test_handles_no_events(self):
        mock_get = MagicMock(return_value=[])
        client = _make_client(mock_get)
        rows = client.get_all_props_for_date("2026-06-24")
        assert rows == []

    def test_continues_on_per_event_error(self):
        """Should skip events where props call fails and not crash."""
        call_count = [0]

        def mock_get(path, params=None):
            if "events" in path and "odds" not in path:
                return FAKE_EVENTS
            call_count[0] += 1
            raise OddsAPIError("Simulated 422")

        client = _make_client(mock_get)
        rows = client.get_all_props_for_date("2026-06-24")
        assert rows == []  # error was caught gracefully


# ---------------------------------------------------------------------------
# get_scores
# ---------------------------------------------------------------------------

class TestGetScores:
    def test_returns_scores_list(self):
        mock_get = MagicMock(return_value=FAKE_SCORES)
        client = _make_client(mock_get)
        scores = client.get_scores(days_from=1)
        assert isinstance(scores, list)
        assert scores[0]["completed"] is False

    def test_live_games_filter(self):
        mock_get = MagicMock(return_value=FAKE_SCORES)
        client = _make_client(mock_get)
        scores = client.get_scores()
        live = [s for s in scores if not s.get("completed")]
        assert len(live) == 1


# ---------------------------------------------------------------------------
# normalize_odds_api_props
# ---------------------------------------------------------------------------

class TestNormalizeOddsApiProps:
    def _make_rows(self):
        return [
            {
                "event_id": "evt001", "game_date": "2026-06-24",
                "home_team": "LV", "away_team": "NY",
                "commence_time": "2026-06-24T23:00:00Z",
                "bookmaker": "draftkings", "market_key": "player_points",
                "stat": "pts", "player_name": "A'ja Wilson",
                "side": "Over", "line": 22.5, "odds": -115,
                "last_update": "2026-06-24T22:00:00Z",
                "event_link": "https://example.com/event",
                "market_link": "https://example.com/market",
                "outcome_link": "https://example.com/over",
            },
            {
                "event_id": "evt001", "game_date": "2026-06-24",
                "home_team": "LV", "away_team": "NY",
                "commence_time": "2026-06-24T23:00:00Z",
                "bookmaker": "draftkings", "market_key": "player_points",
                "stat": "pts", "player_name": "A'ja Wilson",
                "side": "Under", "line": 22.5, "odds": -105,
                "last_update": "2026-06-24T22:00:00Z",
                "event_link": "https://example.com/event",
                "market_link": "https://example.com/market",
                "outcome_link": "https://example.com/under",
            },
        ]

    def test_pivots_over_under_into_one_row(self):
        rows = self._make_rows()
        df = normalize_odds_api_props(rows)
        assert len(df) == 1  # 2 outcomes → 1 row after pivot
        assert "over_odds" in df.columns
        assert "under_odds" in df.columns

    def test_over_under_odds_correct(self):
        rows = self._make_rows()
        df = normalize_odds_api_props(rows)
        assert df.iloc[0]["over_odds"] == -115
        assert df.iloc[0]["under_odds"] == -105

    def test_stat_column_preserved(self):
        rows = self._make_rows()
        df = normalize_odds_api_props(rows)
        assert df.iloc[0]["stat"] == "pts"

    def test_empty_input_returns_empty_df(self):
        df = normalize_odds_api_props([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_deep_link_columns_present(self):
        rows = self._make_rows()
        df = normalize_odds_api_props(rows)
        assert "outcome_link_over" in df.columns
        assert df.iloc[0]["outcome_link_over"] == "https://example.com/over"


# ---------------------------------------------------------------------------
# get_bookmaker_deep_link
# ---------------------------------------------------------------------------

class TestGetBookmakerDeepLink:
    def test_prefers_outcome_link(self):
        row = {
            "outcome_link_over": "https://example.com/outcome",
            "market_link": "https://example.com/market",
            "event_link": "https://example.com/event",
        }
        assert get_bookmaker_deep_link(row) == "https://example.com/outcome"

    def test_falls_back_to_market_link(self):
        row = {
            "outcome_link_over": None,
            "market_link": "https://example.com/market",
            "event_link": "https://example.com/event",
        }
        assert get_bookmaker_deep_link(row) == "https://example.com/market"

    def test_falls_back_to_event_link(self):
        row = {
            "outcome_link_over": None,
            "market_link": None,
            "event_link": "https://example.com/event",
        }
        assert get_bookmaker_deep_link(row) == "https://example.com/event"

    def test_returns_none_when_no_links(self):
        row = {"outcome_link_over": None, "market_link": None, "event_link": None}
        assert get_bookmaker_deep_link(row) is None

    def test_works_with_pandas_series(self):
        row = pd.Series({
            "outcome_link_over": "https://example.com/outcome",
            "market_link": None,
            "event_link": None,
        })
        assert get_bookmaker_deep_link(row) == "https://example.com/outcome"


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

class TestHTTPErrorHandling:
    def test_401_raises_odds_api_error(self):
        with patch.dict("os.environ", {"ODDS_API_KEY": "bad-key"}):
            client = OddsAPIClient()

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {}

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(OddsAPIError, match="invalid or expired"):
                client._get("/v4/sports/")

    def test_422_raises_odds_api_error(self):
        with patch.dict("os.environ", {"ODDS_API_KEY": "test-key"}):
            client = OddsAPIClient()

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.headers = {}
        mock_resp.text = "Invalid params"

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(OddsAPIError, match="Bad request"):
                client._get("/v4/sports/")

    def test_quota_headers_parsed(self):
        with patch.dict("os.environ", {"ODDS_API_KEY": "test-key"}):
            client = OddsAPIClient()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "X-Requests-Remaining": "4850",
            "X-Requests-Used": "150",
        }
        mock_resp.json.return_value = []

        with patch.object(client._session, "get", return_value=mock_resp):
            client._get("/v4/sports/")

        assert client.quota_remaining == 4850
        assert client.quota_used == 150


# ---------------------------------------------------------------------------
# get_closing_lines_for_date — historical endpoint + pre-tip snapshot cap
# ---------------------------------------------------------------------------

class TestGetClosingLinesForDate:
    """Regression: closing lines MUST use the historical events endpoint (the live
    endpoint returns nothing for past dates -> silent 0 rows -> broken CLV), and the
    per-event snapshot MUST be capped to just before tip (post-tip requests 404)."""

    LIVE_EVENTS_PATH = "/v4/sports/basketball_wnba/events"

    def _dispatch(self, path, params=None):
        if "/historical/" in path and path.endswith("/events"):
            return {"data": [dict(FAKE_EVENTS[0])]}
        if "/historical/" in path and "/events/" in path and path.endswith("/odds"):
            return {"data": FAKE_PROPS_DATA}
        raise AssertionError(f"unexpected path requested: {path}")

    def test_uses_historical_endpoint_and_caps_snapshot(self):
        mock_get = MagicMock(side_effect=self._dispatch)
        client = _make_client(mock_get)
        rows = client.get_closing_lines_for_date(
            "2026-06-24", close_time_utc="2026-06-24T23:00:00Z")
        # 2 markets x 2 sides from the fixture
        assert len(rows) == 4
        paths = [c.args[0] for c in mock_get.call_args_list]
        assert any("/historical/" in p and p.endswith("/events") for p in paths)
        # The live events endpoint must NOT be used for a past date.
        assert self.LIVE_EVENTS_PATH not in paths
        # tip=23:00 -> requested 23:00 capped to 22:55 (5 min pre-tip)
        assert {r["snapshot_time"] for r in rows} == {"2026-06-24T22:55:00Z"}

    def test_returns_empty_when_no_historical_events(self):
        def dispatch(path, params=None):
            if path.endswith("/events"):
                return {"data": []}
            raise AssertionError("must not fetch odds when there are no events")
        client = _make_client(MagicMock(side_effect=dispatch))
        assert client.get_closing_lines_for_date(
            "2020-01-01", close_time_utc="2020-01-01T23:00:00Z") == []

    def test_pre_tip_snapshot_kept_uncapped(self):
        mock_get = MagicMock(side_effect=self._dispatch)
        client = _make_client(mock_get)
        # requested well before tip -> kept as-is (no cap)
        rows = client.get_closing_lines_for_date(
            "2026-06-24", close_time_utc="2026-06-24T14:00:00Z")
        assert {r["snapshot_time"] for r in rows} == {"2026-06-24T14:00:00Z"}
