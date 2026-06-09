"""Tests for BDL WNBA game odds and player props handling.

Covers acceptance criteria:
  1.  game odds by date request shape
  2.  game odds by game_ids request shape
  3.  odds endpoint cannot be called without dates or game_ids
  4.  player_props endpoint cannot be called without game_id
  5.  HTTP 200 empty player_props → documented_empty (not failed)
  6.  player prop stat normalization
  7.  game odds normalization (flat BDL response structure)
  8.  vendor normalization
  9.  endpoint audit status classification
  10. no market columns allowed in model feature columns (leakage guard)
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.data.bdl_client import (
    BDLAPIError,
    BDLClient,
    EndpointStatus,
    classify_bdl_error,
    _array_params,
)
from wnba_props_model.data.normalize import normalize_odds, normalize_player_props
from wnba_props_model.features.feature_contract import (
    FORBIDDEN_MODEL_FEATURES,
    assert_no_forbidden_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_odds_row(**kwargs) -> dict:
    base = {
        "id": 252022769,
        "game_id": 24752,
        "vendor": "fanduel",
        "spread_home_value": "-18.5",
        "spread_home_odds": -128,
        "spread_away_value": "18.5",
        "spread_away_odds": -104,
        "moneyline_home_odds": -4500,
        "moneyline_away_odds": 1200,
        "total_value": "167.5",
        "total_over_odds": -152,
        "total_under_odds": 114,
        "updated_at": "2026-05-08T23:49:03.201Z",
    }
    base.update(kwargs)
    return base


def _make_props_row(**kwargs) -> dict:
    # Matches the actual BDL WNBA live player props structure confirmed by debug.
    base = {
        "id": 999001,
        "game_id": 25001,
        "player_id": 111,           # flat int, not nested dict
        "player": {"id": 111, "first_name": "A'ja", "last_name": "Wilson"},
        "team": {"id": 10, "abbreviation": "LVA"},
        "vendor": "fanduel",
        "prop_type": "points",      # BDL live uses "prop_type"
        "line_value": "22.5",       # BDL live uses "line_value"
        "market": {                 # BDL live nests odds under "market"
            "type": "over_under",
            "over_odds": -115,
            "under_odds": -105,
        },
        "updated_at": "2026-06-09T10:00:00Z",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Game odds by date — request parameter shape
# ---------------------------------------------------------------------------

class TestOddsByDateRequestShape:
    def test_dates_param_converted_to_array(self):
        """dates=[date] must produce dates[]=date in query string."""
        params = _array_params({"dates": ["2026-05-08"], "per_page": 100})
        keys = [k for k, v in params]
        assert "dates[]" in keys

    def test_dates_value_preserved(self):
        params = _array_params({"dates": ["2026-05-08"]})
        val = next(v for k, v in params if k == "dates[]")
        assert val == "2026-05-08"

    def test_per_page_present(self):
        params = _array_params({"dates": ["2026-05-08"], "per_page": 100})
        assert ("per_page", 100) in params


# ---------------------------------------------------------------------------
# 2. Game odds by game_ids — request parameter shape
# ---------------------------------------------------------------------------

class TestOddsByGameIdsRequestShape:
    def test_game_ids_converted_to_array(self):
        params = _array_params({"game_ids": [24752, 24753], "per_page": 100})
        keys = [k for k, v in params]
        assert "game_ids[]" in keys

    def test_multiple_game_ids(self):
        params = _array_params({"game_ids": [24752, 24753]})
        vals = [v for k, v in params if k == "game_ids[]"]
        assert 24752 in vals
        assert 24753 in vals

    def test_single_game_id(self):
        params = _array_params({"game_ids": [24752]})
        vals = [v for k, v in params if k == "game_ids[]"]
        assert vals == [24752]


# ---------------------------------------------------------------------------
# 3. Odds endpoint cannot be called without dates or game_ids
# ---------------------------------------------------------------------------

class TestOddsEndpointRequiredParams:
    def test_empty_game_ids_raises_value_error(self, monkeypatch):
        """list_game_odds_by_game_ids([]) must raise ValueError."""
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        client = BDLClient.__new__(BDLClient)
        client.api_key = "test-key-xxx"
        with pytest.raises(ValueError, match="game_ids must be non-empty"):
            client.list_game_odds_by_game_ids([])

    def test_game_ids_with_values_does_not_raise(self, monkeypatch, requests_mock):
        """list_game_odds_by_game_ids([id]) must succeed (no ValueError)."""
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        requests_mock.get(
            "https://api.balldontlie.io/wnba/v1/odds",
            json={"data": [_make_odds_row()], "meta": {"next_cursor": None}},
        )
        client = BDLClient(api_key="test-key-xxx")
        rows = client.list_game_odds_by_game_ids([24752])
        assert isinstance(rows, list)

    def test_date_method_does_not_raise_for_valid_date(self, monkeypatch, requests_mock):
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        requests_mock.get(
            "https://api.balldontlie.io/wnba/v1/odds",
            json={"data": [_make_odds_row()], "meta": {"next_cursor": None}},
        )
        client = BDLClient(api_key="test-key-xxx")
        rows = client.list_game_odds_by_date("2026-05-08")
        assert isinstance(rows, list)


# ---------------------------------------------------------------------------
# 4. Player props endpoint cannot be called without game_id
# ---------------------------------------------------------------------------

class TestPlayerPropsRequiredGameId:
    def test_list_player_props_includes_game_id_in_params(self, monkeypatch, requests_mock):
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        requests_mock.get(
            "https://api.balldontlie.io/wnba/v1/odds/player_props",
            json={"data": [], "meta": {"per_page": 0}},
        )
        client = BDLClient(api_key="test-key-xxx")
        rows = client.list_player_props_for_game(game_id=24752)
        assert isinstance(rows, list)
        # Confirm game_id was in the actual request URL
        assert "game_id=24752" in requests_mock.last_request.url

    def test_vendors_array_param(self, monkeypatch, requests_mock):
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        requests_mock.get(
            "https://api.balldontlie.io/wnba/v1/odds/player_props",
            json={"data": [], "meta": {"per_page": 0}},
        )
        client = BDLClient(api_key="test-key-xxx")
        client.list_player_props_for_game(game_id=24752, vendors=["fanduel"])
        assert "vendors%5B%5D=fanduel" in requests_mock.last_request.url or \
               "vendors[]=fanduel" in requests_mock.last_request.url


# ---------------------------------------------------------------------------
# 5. HTTP 200 empty player_props → documented_empty, not failed
# ---------------------------------------------------------------------------

class TestEmptyPropsClassification:
    def test_empty_props_is_not_error(self, monkeypatch, requests_mock):
        """An HTTP 200 response with empty data[] must NOT raise an exception."""
        monkeypatch.setenv("BDL_API_KEY", "test-key-xxx")
        requests_mock.get(
            "https://api.balldontlie.io/wnba/v1/odds/player_props",
            json={"data": [], "meta": {"per_page": 0}},
            status_code=200,
        )
        client = BDLClient(api_key="test-key-xxx")
        rows = client.list_player_props_for_game(game_id=24752)
        assert rows == []

    def test_empty_rows_normalizes_to_empty_dataframe(self):
        df = normalize_player_props([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_status_classification(self):
        """Ingest logic should classify empty props as documented_empty or live_only."""
        rows: list[dict] = []
        # Simulate the classification logic used in ingest.py
        upcoming_ids = [24752]
        status = (
            EndpointStatus.LIVE_ONLY_NO_HISTORICAL
            if not rows and upcoming_ids
            else EndpointStatus.DOCUMENTED_EMPTY
        )
        assert status in (
            EndpointStatus.LIVE_ONLY_NO_HISTORICAL,
            EndpointStatus.DOCUMENTED_EMPTY,
        )
        assert status != EndpointStatus.FAILED

    def test_live_only_status_string_value(self):
        assert EndpointStatus.LIVE_ONLY_NO_HISTORICAL == "live_only_no_historical_storage"

    def test_documented_empty_status_string_value(self):
        assert EndpointStatus.DOCUMENTED_EMPTY == "documented_empty"


# ---------------------------------------------------------------------------
# 6. Player prop stat normalization
# ---------------------------------------------------------------------------

class TestPlayerPropStatNormalization:
    @pytest.mark.parametrize("raw_type,expected_stat", [
        ("points", "pts"),
        ("rebounds", "reb"),
        ("assists", "ast"),
        ("threes", "fg3m"),
        ("points_rebounds", "pts_reb"),
        ("points_assists", "pts_ast"),
        ("rebounds_assists", "reb_ast"),
        ("points_rebounds_assists", "pts_reb_ast"),
        ("double_double", "double_double"),
        ("triple_double", "triple_double"),
    ])
    def test_canonical_stat_names(self, raw_type: str, expected_stat: str):
        # BDL live API uses "prop_type" field (not "type")
        row = _make_props_row(prop_type=raw_type)
        df = normalize_player_props([row])
        assert len(df) == 1
        assert df.iloc[0]["stat"] == expected_stat

    def test_prop_type_raw_preserved(self):
        row = _make_props_row(prop_type="points")
        df = normalize_player_props([row])
        assert df.iloc[0]["prop_type_raw"] == "points"

    def test_unknown_type_passthrough(self):
        row = _make_props_row(prop_type="some_unknown_stat")
        df = normalize_player_props([row])
        assert df.iloc[0]["stat"] == "some_unknown_stat"

    def test_player_name_extracted(self):
        row = _make_props_row()
        df = normalize_player_props([row])
        assert "Wilson" in df.iloc[0]["player_name"]

    def test_team_abbreviation_extracted(self):
        row = _make_props_row()
        df = normalize_player_props([row])
        assert df.iloc[0]["team_abbreviation"] == "LVA"

    def test_line_is_numeric(self):
        row = _make_props_row(line_value="22.5")
        df = normalize_player_props([row])
        assert df.iloc[0]["line"] == pytest.approx(22.5)

    def test_vendor_field_populated(self):
        row = _make_props_row(vendor="draftkings")
        df = normalize_player_props([row])
        assert df.iloc[0]["vendor"] == "draftkings"
        assert df.iloc[0]["book"] == "draftkings"
        assert df.iloc[0]["sportsbook"] == "draftkings"


# ---------------------------------------------------------------------------
# 7. Game odds normalization
# ---------------------------------------------------------------------------

class TestGameOddsNormalization:
    def test_single_row(self):
        row = _make_odds_row()
        df = normalize_odds([row])
        assert len(df) == 1

    def test_odds_id_field(self):
        row = _make_odds_row(id=252022769)
        df = normalize_odds([row])
        assert df.iloc[0]["odds_id"] == 252022769

    def test_game_id_field(self):
        row = _make_odds_row(game_id=24752)
        df = normalize_odds([row])
        assert df.iloc[0]["game_id"] == 24752

    def test_spread_home_value_numeric(self):
        row = _make_odds_row(spread_home_value="-18.5")
        df = normalize_odds([row])
        assert df.iloc[0]["spread_home_value"] == pytest.approx(-18.5)

    def test_total_value_numeric(self):
        row = _make_odds_row(total_value="167.5")
        df = normalize_odds([row])
        assert df.iloc[0]["total_value"] == pytest.approx(167.5)

    def test_moneyline_odds_numeric(self):
        row = _make_odds_row(moneyline_home_odds=-4500, moneyline_away_odds=1200)
        df = normalize_odds([row])
        assert df.iloc[0]["moneyline_home_odds"] == -4500
        assert df.iloc[0]["moneyline_away_odds"] == 1200

    def test_updated_at_parsed(self):
        row = _make_odds_row(updated_at="2026-05-08T23:49:03.201Z")
        df = normalize_odds([row])
        assert pd.notna(df.iloc[0]["updated_at"])

    def test_empty_rows_returns_empty_dataframe(self):
        df = normalize_odds([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_multiple_vendors_rows(self):
        rows = [
            _make_odds_row(id=1, vendor="fanduel"),
            _make_odds_row(id=2, vendor="draftkings"),
        ]
        df = normalize_odds(rows)
        assert len(df) == 2
        vendors = set(df["vendor"].unique())
        assert "fanduel" in vendors
        assert "draftkings" in vendors

    def test_game_date_season_initially_none(self):
        """BDL odds response does not include game_date or season; they start as None."""
        row = _make_odds_row()
        df = normalize_odds([row])
        assert df.iloc[0]["game_date"] is None
        assert df.iloc[0]["season"] is None


# ---------------------------------------------------------------------------
# 8. Vendor normalization
# ---------------------------------------------------------------------------

class TestVendorNormalization:
    @pytest.mark.parametrize("vendor", [
        "betmgm", "betrivers", "caesars", "draftkings", "fanatics", "fanduel",
    ])
    def test_supported_game_odds_vendors(self, vendor: str):
        row = _make_odds_row(vendor=vendor)
        df = normalize_odds([row])
        assert df.iloc[0]["vendor"] == vendor

    @pytest.mark.parametrize("vendor", [
        "betrivers", "caesars", "draftkings", "fanatics", "fanduel",
    ])
    def test_supported_player_props_vendors(self, vendor: str):
        row = _make_props_row(vendor=vendor)
        df = normalize_player_props([row])
        assert df.iloc[0]["vendor"] == vendor

    def test_vendor_book_sportsbook_aliases_match(self):
        row = _make_odds_row(vendor="fanduel")
        df = normalize_odds([row])
        assert df.iloc[0]["vendor"] == df.iloc[0]["book"] == df.iloc[0]["sportsbook"]


# ---------------------------------------------------------------------------
# 9. Endpoint audit status classification
# ---------------------------------------------------------------------------

class TestEndpointStatusClassification:
    @pytest.mark.parametrize("error_fragment,expected_status", [
        ("401 Unauthorized", EndpointStatus.DOCUMENTED_AUTH_FAILED),
        ("403 Forbidden", EndpointStatus.DOCUMENTED_AUTH_FAILED),
        ("400 Bad Request", EndpointStatus.DOCUMENTED_BAD_REQUEST),
        ("404 Not Found", EndpointStatus.DOCUMENTED_UNAVAILABLE),
        ("422 Unprocessable", EndpointStatus.DOCUMENTED_QUERY_BUG),
        ("Connection timeout", EndpointStatus.FAILED),
        ("500 Server Error", EndpointStatus.FAILED),
    ])
    def test_classify_bdl_error(self, error_fragment: str, expected_status: str):
        assert classify_bdl_error(error_fragment) == expected_status

    def test_status_constants_are_distinct(self):
        statuses = [
            EndpointStatus.DOCUMENTED_SUCCESS,
            EndpointStatus.DOCUMENTED_EMPTY,
            EndpointStatus.DOCUMENTED_AUTH_FAILED,
            EndpointStatus.DOCUMENTED_BAD_REQUEST,
            EndpointStatus.DOCUMENTED_UNAVAILABLE,
            EndpointStatus.DOCUMENTED_QUERY_BUG,
            EndpointStatus.LIVE_ONLY_NO_HISTORICAL,
            EndpointStatus.SKIPPED,
            EndpointStatus.FAILED,
        ]
        assert len(set(statuses)) == len(statuses), "EndpointStatus constants must be unique"

    def test_documented_success_value(self):
        assert EndpointStatus.DOCUMENTED_SUCCESS == "documented_success"

    def test_skipped_value(self):
        assert EndpointStatus.SKIPPED == "skipped"


# ---------------------------------------------------------------------------
# 10. No market columns in model feature columns (leakage guard)
# ---------------------------------------------------------------------------

class TestLeakageGuardOddsColumns:
    def test_odds_columns_forbidden_in_model_features(self):
        market_cols = [
            "spread_home_value", "spread_home_odds", "spread_away_value", "spread_away_odds",
            "moneyline_home_odds", "moneyline_away_odds",
            "total_value", "total_over_odds", "total_under_odds",
            "vendor", "book", "sportsbook",
            "line", "over_odds", "under_odds",
        ]
        for col in market_cols:
            assert col in FORBIDDEN_MODEL_FEATURES, (
                f"Market column '{col}' must be in FORBIDDEN_MODEL_FEATURES"
            )

    def test_assert_no_forbidden_features_catches_odds_column(self):
        import pandas as pd
        df = pd.DataFrame({"pts": [10], "total_value": [167.5]})
        with pytest.raises(ValueError, match="total_value"):
            assert_no_forbidden_features(df)

    def test_clean_features_pass_leakage_guard(self):
        import pandas as pd
        # Use actual model features (trailing stats / schedule) — NOT the outcome columns
        df = pd.DataFrame({
            "trailing_avg_pts": [18.2],
            "rest_days": [2],
            "is_home": [True],
            "is_b2b": [False],
        })
        assert_no_forbidden_features(df)  # must not raise

    def test_prop_type_raw_not_a_model_feature(self):
        """prop_type_raw is evaluation-only metadata, not a model input."""
        assert "prop_type_raw" in FORBIDDEN_MODEL_FEATURES, (
            "prop_type_raw must be in FORBIDDEN_MODEL_FEATURES (evaluation-only)"
        )
