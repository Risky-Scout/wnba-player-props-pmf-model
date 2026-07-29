"""Section M acceptance tests for the historical WNBA backfill contract (no network).

Covers: sport key, exact market keys, comma-separated single request, region, BDL prop_type,
BDL plays non-pagination, timestamp provenance, pairing pass/fail, settlement (direct+combo,
integer push, half-point no push), credit budget fail-closed, and no-secret-in-logs.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from wnba_props_model.constants import ODDS_API_MODEL_MARKET_KEYS, ODDS_API_MODEL_MARKETS
from wnba_props_model.data.atomic_quotes import to_raw_side_snapshots
from wnba_props_model.data.bdl_client import WNBA_ENDPOINTS
from wnba_props_model.data.odds_api_client import SPORT_KEY, OddsAPIClient, OddsAPIError
from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs
from wnba_props_model.data.settlement import (
    OVER_WIN,
    PUSH,
    UNDER_WIN,
    SPORTSBOOK_SETTLEMENT_RULES,
    settle_one,
)


# --- 1-4: Odds API request contract ------------------------------------------------
def test_sport_key_is_basketball_wnba():
    assert SPORT_KEY == "basketball_wnba"


def test_exact_official_market_keys():
    assert ODDS_API_MODEL_MARKET_KEYS == (
        "player_points", "player_rebounds", "player_assists", "player_threes",
        "player_steals", "player_blocks", "player_turnovers", "player_blocks_steals",
        "player_points_assists", "player_points_rebounds", "player_rebounds_assists",
        "player_points_rebounds_assists",
    )
    # every key maps to a model stat with a settlement/PMF target
    assert set(ODDS_API_MODEL_MARKETS.values()) == {
        "pts", "reb", "ast", "fg3m", "blk", "stl", "turnover", "stocks",
        "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"}


def test_markets_sent_comma_separated_in_one_request(monkeypatch):
    c = OddsAPIClient(api_key="x")
    captured = {}
    monkeypatch.setattr(c, "_get", lambda path, params=None: captured.update(path=path, params=params) or {})
    c.get_historical_event_odds("evt1", "2024-08-20T22:00:00Z", markets=list(ODDS_API_MODEL_MARKET_KEYS))
    assert captured["params"]["markets"] == ",".join(ODDS_API_MODEL_MARKET_KEYS)
    assert "," in captured["params"]["markets"]           # single comma-separated request
    assert "basketball_wnba" in captured["path"]
    assert "basketball_nba" not in captured["path"]


def test_model_prop_markets_is_frozen_scope_of_12():
    from wnba_props_model.constants import MODEL_PROP_MARKETS, MODEL_PROP_MARKET_KEYS
    assert len(MODEL_PROP_MARKETS) == 12
    assert MODEL_PROP_MARKET_KEYS == ODDS_API_MODEL_MARKET_KEYS
    forbidden = {"player_field_goals", "player_frees_made", "player_frees_attempts",
                 "player_points_q1", "player_double_double", "player_triple_double",
                 "player_fantasy_points", "player_points_alternate"}
    assert forbidden.isdisjoint(set(MODEL_PROP_MARKETS))


def test_runtime_assertion_rejects_wrong_scope():
    from wnba_props_model.data.odds_api_client import assert_model_market_request
    from wnba_props_model.constants import MODEL_PROP_MARKET_KEYS
    assert_model_market_request("basketball_wnba", "us", "american", "iso", list(MODEL_PROP_MARKET_KEYS))
    for bad in [
        ("basketball_nba", "us", "american", "iso", list(MODEL_PROP_MARKET_KEYS)),
        ("basketball_wnba", "us2", "american", "iso", list(MODEL_PROP_MARKET_KEYS)),
        ("basketball_wnba", "us", "decimal", "iso", list(MODEL_PROP_MARKET_KEYS)),
        ("basketball_wnba", "us", "american", "iso", ["player_points"]),
        ("basketball_wnba", "us", "american", "iso", list(MODEL_PROP_MARKET_KEYS) + ["player_field_goals"]),
    ]:
        with pytest.raises(OddsAPIError):
            assert_model_market_request(*bad)


def test_enforce_flag_blocks_non_model_markets(monkeypatch):
    c = OddsAPIClient(api_key="x", enforce_model_markets=True)
    monkeypatch.setattr(c, "_get", lambda path, params=None: {})
    with pytest.raises(OddsAPIError):
        c.get_historical_event_odds("e1", "2024-08-20T22:00:00Z", markets=["player_field_goals"])
    # exact 12 passes the guard
    c.get_historical_event_odds("e1", "2024-08-20T22:00:00Z", markets=list(ODDS_API_MODEL_MARKET_KEYS))


def test_region_is_us_by_default(monkeypatch):
    c = OddsAPIClient(api_key="x")
    assert c.region == "us"
    captured = {}
    monkeypatch.setattr(c, "_get", lambda path, params=None: captured.update(params=params) or {})
    c.get_historical_event_odds("evt1", "2024-08-20T22:00:00Z", markets=["player_points"])
    assert captured["params"]["regions"] == "us"


# --- 5-6: BDL client contract ------------------------------------------------------
def test_bdl_player_props_uses_prop_type_not_type(monkeypatch):
    from wnba_props_model.data.bdl_client import BDLClient
    c = BDLClient(api_key="x")
    captured = {}
    monkeypatch.setattr(c, "iter_endpoint", lambda name, params=None, **k: captured.update(name=name, params=params) or iter(()))
    c.list_player_props_for_game(123, prop_type="points")
    assert captured["params"].get("prop_type") == "points"
    assert "type" not in captured["params"]                # the old ignored key must be gone


def test_bdl_plays_is_non_paginated_no_undocumented_params():
    assert WNBA_ENDPOINTS["plays"].paginated is False


# --- 7-8: timestamp provenance (canonical schema) ----------------------------------
def _atomic_row(**kw):
    base = {
        "source": "odds_api_v4_historical", "sportsbook": "draftkings", "event_id": "e1",
        "player_id": "p1", "prop": "pts", "line": 15.5, "side": "over", "american_odds": -110,
        "snapshot_role": "decision",
        "requested_snapshot_utc": "2024-08-20T11:00:00Z",     # what we requested
        "provider_snapshot_utc": "2024-08-20T10:57:00Z",      # snapshot the API returned
        "market_last_update_utc": "2024-08-20T10:55:12Z",     # the TRUE quote time
        "quote_timestamp_utc": "2024-08-20T10:55:12Z",
        "quote_timestamp_source": "market_last_update",
        "scheduled_tip_utc": "2024-08-20T23:00:00Z",
        "decision_cutoff_utc": "2024-08-20T11:00:00Z",
        "closing_cutoff_utc": "2024-08-20T22:55:00Z",
        "role_cutoff_utc": "2024-08-20T11:00:00Z",
    }
    base.update(kw)
    return base


def test_adapter_uses_actual_quote_timestamp_not_requested():
    raw = to_raw_side_snapshots(pd.DataFrame([_atomic_row()]))
    assert raw.iloc[0]["snapshot_timestamp"] == "2024-08-20T10:55:12Z"   # quote_timestamp_utc
    assert raw.iloc[0]["provider"] == "odds_api"
    assert raw.iloc[0]["decision_timestamp_utc"] == "2024-08-20T11:00:00Z"   # decision role cutoff


def test_adapter_closing_row_carries_closing_cutoff_not_decision_not_null():
    row = _atomic_row(snapshot_role="closing", role_cutoff_utc="2024-08-20T22:55:00Z",
                      market_last_update_utc="2024-08-20T22:54:30Z",
                      quote_timestamp_utc="2024-08-20T22:54:30Z")
    raw = to_raw_side_snapshots(pd.DataFrame([row]))
    cut = raw.iloc[0]["decision_timestamp_utc"]
    assert cut == "2024-08-20T22:55:00Z"          # closing cutoff (tip-5m), NOT decision, NOT null
    assert cut is not None


def test_adapter_blocks_row_with_no_market_timestamp():
    row = _atomic_row(market_last_update_utc=None, quote_timestamp_utc=None,
                      quote_timestamp_source="BLOCKED_NO_MARKET_TIMESTAMP")
    raw = to_raw_side_snapshots(pd.DataFrame([row]))
    assert len(raw) == 0          # dropped, never fabricated from the requested date


def test_closing_quote_after_decision_cutoff_still_pairs_as_closing():
    """A closing quote at tip-5m is AFTER the decision cutoff but must pair under the CLOSING
    role cutoff without any timestamp mutation."""
    tip = "2024-08-20T23:00:00Z"
    rows = []
    for side in ("over", "under"):
        rows.append(_atomic_row(side=side, snapshot_role="closing",
                                 role_cutoff_utc="2024-08-20T22:55:00Z",
                                 market_last_update_utc="2024-08-20T22:54:00Z",
                                 quote_timestamp_utc="2024-08-20T22:54:00Z",
                                 scheduled_tip_utc=tip))
    raw = to_raw_side_snapshots(pd.DataFrame(rows))
    pairs = build_quote_pairs(raw, snapshot_label="closing")
    assert (pairs["quote_pair_status"] == EXACT_PAIR).all()


# --- 9-11: pairing -----------------------------------------------------------------
def _side(book, side, line, ts, pid="p1", eid="e1"):
    return {"provider": "odds_api", "sportsbook": book, "event_id": eid, "player_id": pid,
            "prop": "pts", "line": line, "side": side, "snapshot_timestamp": ts,
            "american_odds": -110, "scheduled_tip_utc": "2024-08-20T23:00:00Z",
            "decision_timestamp_utc": "2024-08-20T11:00:00Z"}


def test_same_book_same_line_pair_passes():
    raw = pd.DataFrame([_side("dk", "over", 15.5, "2024-08-20T10:55:00Z"),
                        _side("dk", "under", 15.5, "2024-08-20T10:55:30Z")])
    pairs = build_quote_pairs(raw, snapshot_label="decision")
    assert (pairs["quote_pair_status"] == EXACT_PAIR).all()


def test_cross_book_and_cross_line_do_not_pair():
    cross_book = pd.DataFrame([_side("dk", "over", 15.5, "2024-08-20T10:55:00Z"),
                               _side("fd", "under", 15.5, "2024-08-20T10:55:00Z")])
    assert (build_quote_pairs(cross_book)["quote_pair_status"] != EXACT_PAIR).all()
    cross_line = pd.DataFrame([_side("dk", "over", 15.5, "2024-08-20T10:55:00Z"),
                               _side("dk", "under", 16.5, "2024-08-20T10:55:00Z")])
    assert (build_quote_pairs(cross_line)["quote_pair_status"] != EXACT_PAIR).all()


def test_post_cutoff_and_post_tip_quotes_fail():
    post_cut = pd.DataFrame([_side("dk", "over", 15.5, "2024-08-20T12:00:00Z"),   # after decision cutoff
                             _side("dk", "under", 15.5, "2024-08-20T12:00:00Z")])
    assert (build_quote_pairs(post_cut)["quote_pair_status"] != EXACT_PAIR).all()
    post_tip = pd.DataFrame([_side("dk", "over", 15.5, "2024-08-20T23:30:00Z"),   # after tip
                             _side("dk", "under", 15.5, "2024-08-20T23:30:00Z")])
    assert (build_quote_pairs(post_tip)["quote_pair_status"] != EXACT_PAIR).all()


# --- 12-14: settlement -------------------------------------------------------------
_RULE = SPORTSBOOK_SETTLEMENT_RULES["wnba_player_prop_standard_v1"]


def _settle(line, actual):
    return settle_one(rule=_RULE, line=line, appeared=True, actual_outcome=actual)["settlement_status"]


def test_direct_settlement_over_under():
    assert _settle(15.5, 20) == OVER_WIN
    assert _settle(15.5, 10) == UNDER_WIN


def test_integer_line_equality_is_push():
    assert _settle(15.0, 15) == PUSH


def test_half_point_line_never_pushes():
    # actual is an integer; a .5 line can never equal it
    for actual in range(0, 40):
        assert _settle(15.5, actual) in (OVER_WIN, UNDER_WIN)


def test_combination_settlement_from_deterministic_sums():
    # pts_reb_ast settles from the sum of BDL outcomes
    pts, reb, ast = 18, 7, 5
    combo = pts + reb + ast   # 30
    assert _settle(29.5, combo) == OVER_WIN
    assert _settle(30.0, combo) == PUSH
    assert _settle(30.5, combo) == UNDER_WIN


# --- 15: credit budget fail-closed -------------------------------------------------
class _Resp:
    def __init__(self, cost):
        self.status_code = 200
        self.headers = {"X-Requests-Last": str(cost), "X-Requests-Used": "1", "X-Requests-Remaining": "9"}
    def json(self):
        return {"data": {}}
    def raise_for_status(self):
        pass


def test_credit_budget_stops_before_overspend(monkeypatch):
    c = OddsAPIClient(api_key="x", max_credits=100)
    monkeypatch.setattr(c._session, "get", lambda *a, **k: _Resp(cost=120))
    c._get("/v4/historical/sports/basketball_wnba/events/e/odds", {"markets": "player_points"})
    assert c.credits_spent_session == 120
    with pytest.raises(OddsAPIError, match="budget reached"):
        c._get("/v4/historical/sports/basketball_wnba/events/e2/odds", {"markets": "player_points"})


# --- 17: no secret in the request audit --------------------------------------------
def test_request_audit_never_contains_api_key(tmp_path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    c = OddsAPIClient(api_key="SUPER_SECRET_KEY", request_audit_path=str(audit))
    monkeypatch.setattr(c._session, "get", lambda *a, **k: _Resp(cost=10))
    c._get("/v4/x", {"markets": "player_points", "apiKey": "SUPER_SECRET_KEY"})
    text = audit.read_text()
    assert "SUPER_SECRET_KEY" not in text
    rec = json.loads(text.splitlines()[0])
    assert "apiKey" not in rec["params"]
    assert rec["x_requests_last"] == 10
