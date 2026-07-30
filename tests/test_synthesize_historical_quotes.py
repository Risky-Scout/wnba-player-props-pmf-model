"""No-network tests for the historical quote-synthesis helpers and the pairing contract.

Proves the pure date/snapshot helpers are correct and that a synthesized single-side row pair
(once player/game ids resolve against BDL canonical tables) yields an EXACT_PAIR under the
canonical `build_quote_pairs` validator — i.e. the schema the synth script emits is compatible
with the downstream readiness pipeline. Network calls are never made here.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs

_SPEC = importlib.util.spec_from_file_location(
    "synth_hist_quotes",
    Path(__file__).resolve().parent.parent / "scripts" / "synthesize_historical_quotes.py",
)
synth = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(synth)


def test_daterange_is_inclusive():
    days = list(synth._daterange("2024-08-01", "2024-08-03"))
    assert days == ["2024-08-01", "2024-08-02", "2024-08-03"]


def test_tip_parses_and_rejects_bad_input():
    assert synth._tip("2024-08-20T23:00:00Z") is not None
    assert synth._tip("not-a-time") is None
    assert synth._tip("") is None


def test_snapshots_are_strictly_pre_tip_with_correct_offsets():
    tip = datetime(2024, 8, 20, 23, 0, 0, tzinfo=timezone.utc)
    snaps = synth._snapshots(tip, lead_hours=1.0, closing_minutes=5)
    assert snaps["decision"] == "2024-08-20T22:00:00Z"   # tip - 1h
    assert snaps["closing"] == "2024-08-20T22:55:00Z"     # tip - 5m
    for s in snaps.values():
        assert datetime.fromisoformat(s.replace("Z", "+00:00")) < tip


def test_synthesized_rows_form_exact_pair_when_ids_resolve():
    """Mirror the script's flat-store -> pairs transform and assert EXACT_PAIR when a
    single book posts both sides at the same line, pre-decision-cutoff, ids resolved."""
    tip = "2024-08-20T23:00:00Z"
    decision = "2024-08-20T22:00:00Z"
    snap = "2024-08-20T22:00:00Z"   # at the decision cutoff (<= decision, < tip)
    base = {
        "sportsbook": "draftkings", "event_id": "evt1", "game_id": "g1", "player_id": "p1",
        "player_name": "A. Player", "prop": "pts", "line": 15.5,
        "snapshot_time": snap, "decision_timestamp": decision, "scheduled_tip_utc": tip,
    }
    flat = pd.DataFrame([
        {**base, "side": "over", "american_odds": -110},
        {**base, "side": "under", "american_odds": -110},
    ])
    # same rename the script applies before build_quote_pairs
    raw = flat.rename(columns={
        "snapshot_time": "snapshot_timestamp",
        "decision_timestamp": "decision_timestamp_utc",
    }).copy()
    raw["provider"] = "odds_api"
    pairs = build_quote_pairs(raw, snapshot_label="decision")
    assert len(pairs) == 1
    assert pairs.iloc[0]["quote_pair_status"] == EXACT_PAIR


def test_unresolved_player_id_is_not_exact_pair():
    """Fail-closed: with no resolved player_id the pair is AMBIGUOUS_PLAYER, never EXACT."""
    tip = "2024-08-20T23:00:00Z"
    base = {
        "provider": "odds_api", "sportsbook": "draftkings", "event_id": "evt1",
        "player_id": None, "prop": "pts", "line": 15.5,
        "snapshot_timestamp": "2024-08-20T22:00:00Z",
        "decision_timestamp_utc": "2024-08-20T22:00:00Z", "scheduled_tip_utc": tip,
    }
    raw = pd.DataFrame([
        {**base, "side": "over", "american_odds": -110},
        {**base, "side": "under", "american_odds": -110},
    ])
    pairs = build_quote_pairs(raw, snapshot_label="decision")
    assert (pairs["quote_pair_status"] != EXACT_PAIR).all()
