"""Stage 4: durability + resumption tests for the historical backfill (no network).

Proves: interruption after raw-save and after normalization; restart after each with zero
duplicate API calls and zero duplicate rows; deterministic HTTP 404 tombstones; hard budget
fail-closed before a request; idempotent consolidation.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from wnba_props_model.data import atomic_backfill as ab
from wnba_props_model.data.odds_api_client import OddsAPIError

TIP = datetime(2024, 8, 20, 23, 0, 0, tzinfo=timezone.utc)
ROSTER = pd.DataFrame([{"game_id": "g1", "player_id": 1, "player_name": "A Player"}])


def _payload(mkt_last="2024-08-20T10:55:00Z"):
    return {"timestamp": "2024-08-20T10:57:00Z", "previous_timestamp": None, "next_timestamp": None,
            "data": {"id": "e1", "commence_time": "2024-08-20T23:00:00Z",
                     "home_team": "H", "away_team": "A", "bookmakers": [
                         {"key": "draftkings", "last_update": mkt_last, "markets": [
                             {"key": "player_points", "last_update": mkt_last, "outcomes": [
                                 {"name": "Over", "description": "A Player", "point": 15.5, "price": -110},
                                 {"name": "Under", "description": "A Player", "point": 15.5, "price": -110}]}]}]}}


class FakeClient:
    def __init__(self, payload=None, raise_404=False, raise_budget=False):
        self.calls = 0
        self.payload = payload if payload is not None else _payload()
        self.raise_404 = raise_404
        self.raise_budget = raise_budget

    def get_historical_event_odds(self, event_id, date_str, markets=None):
        self.calls += 1
        if self.raise_budget:
            raise OddsAPIError("ODDS_API_MAX_CREDITS budget reached: fail-closed")
        if self.raise_404:
            raise OddsAPIError("HTTP 404 (deterministic, no retry): /odds")
        return self.payload


def _proc(client, tmp, state, **kw):
    return ab.process_snapshot(
        client, event_id="e1", role="decision", tip=TIP, gid="g1", season=2024,
        game_date="2024-08-20", roster_df=ROSTER, raw_dir=tmp / "raw", part_dir=tmp / "part",
        state=state, state_path=tmp / "state.json", collection_ts="2024-08-20T12:00:00Z", **kw)


def test_happy_path_completes_and_persists(tmp_path):
    client = FakeClient()
    state = {}
    r = _proc(client, tmp_path, state)
    assert r["status"] == ab.COMPLETE and r["api_call"] is True and client.calls == 1
    assert (tmp_path / "raw" / "e1_decision.json").exists()
    assert list((tmp_path / "part").rglob("part.parquet"))
    assert state[ab.state_key("e1", "decision")] == ab.COMPLETE


def test_interruption_after_raw_save_then_restart_no_duplicate_api(tmp_path):
    client = FakeClient()
    state = {}
    with pytest.raises(RuntimeError, match="raw_save"):
        _proc(client, tmp_path, state, fault_after="raw_save")
    assert state[ab.state_key("e1", "decision")] == ab.RAW_SAVED
    assert (tmp_path / "raw" / "e1_decision.json").exists()
    assert client.calls == 1
    # restart: reuse cached raw, NO new api call
    r = _proc(client, tmp_path, state)
    assert r["status"] == ab.COMPLETE and r["from_cache"] is True and r["api_call"] is False
    assert client.calls == 1                          # zero duplicate API requests


def test_interruption_after_normalize_then_restart(tmp_path):
    client = FakeClient()
    state = {}
    with pytest.raises(RuntimeError, match="normalize"):
        _proc(client, tmp_path, state, fault_after="normalize")
    assert state[ab.state_key("e1", "decision")] == ab.RAW_SAVED
    r = _proc(client, tmp_path, state)
    assert r["status"] == ab.COMPLETE and client.calls == 1


def test_restart_skips_complete_with_no_api(tmp_path):
    client = FakeClient()
    state = {}
    _proc(client, tmp_path, state)
    assert client.calls == 1
    r = _proc(client, tmp_path, state)
    assert r.get("skipped") and client.calls == 1     # no duplicate request


def test_http_404_tombstone_is_durable(tmp_path):
    client = FakeClient(raise_404=True)
    state = {}
    r = _proc(client, tmp_path, state)
    assert r["status"] == ab.HTTP_404 and client.calls == 1
    # restart: tombstone skipped, no retry
    r2 = _proc(client, tmp_path, state)
    assert r2.get("skipped") and client.calls == 1


def test_budget_fail_closed_propagates(tmp_path):
    client = FakeClient(raise_budget=True)
    with pytest.raises(OddsAPIError, match="budget reached"):
        _proc(client, tmp_path, {})


def test_consolidation_is_idempotent(tmp_path):
    client = FakeClient()
    state = {}
    _proc(client, tmp_path, state)
    df1 = ab.consolidate_partitions(tmp_path / "part")
    # reprocess the same raw (already COMPLETE -> skipped); consolidation stable
    _proc(client, tmp_path, state)
    df2 = ab.consolidate_partitions(tmp_path / "part")
    assert len(df1) == len(df2)
    assert df2["quote_id"].is_unique                  # zero duplicate primary keys
