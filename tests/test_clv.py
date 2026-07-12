"""Tests for correct CLV implementation (§4.7 requirements)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.evaluation.clv import (
    AFTER_TIP,
    NOT_AVAILABLE,
    STALE_CLOSE,
    CLVResult,
    append_to_ledger,
    compute_clv_for_bet,
    compute_line_clv,
    compute_same_line_price_clv,
    create_empty_ledger,
    select_closing_quote,
    validate_closing_quote,
)
from wnba_props_model.pipeline.safety import american_to_implied, american_to_no_vig, no_vig_normalize


# ---------------------------------------------------------------------------
# §4.5: American odds conversion tests
# ---------------------------------------------------------------------------

class TestAmericanToImplied:
    def test_positive_odds(self):
        """§4.7 requirement: positive American odds."""
        p = american_to_implied(150.0)
        assert abs(p - 100.0 / 250.0) < 1e-12

    def test_negative_odds(self):
        """§4.7 requirement: negative American odds."""
        p = american_to_implied(-110.0)
        assert abs(p - 110.0 / 210.0) < 1e-12

    def test_even_money(self):
        """§4.7 requirement: even money."""
        p = american_to_implied(100.0)
        assert abs(p - 0.5) < 1e-12

    def test_heavy_favorite_negative(self):
        p = american_to_implied(-300.0)
        assert abs(p - 300.0 / 400.0) < 1e-12


class TestNoVigNormalization:
    def test_sums_to_one(self):
        """§4.7: no-vig normalization summing to one."""
        p_over, p_under = no_vig_normalize(0.5238, 0.5238)
        assert abs(p_over + p_under - 1.0) < 1e-12

    def test_symmetric_is_50_50(self):
        p_over, p_under = american_to_no_vig(-110.0, -110.0)
        assert abs(p_over - 0.5) < 1e-12
        assert abs(p_under - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# §4.5: Same-line price CLV
# ---------------------------------------------------------------------------

class TestSameLinePriceCLV:
    def test_positive_clv_when_market_moved_against_bet(self):
        """
        Entry: over at -110 → no-vig p = 0.5238
        Close: over at -130 → no-vig p ~= 0.5652
        CLV = 0.5652 - 0.5238 = positive (market values over more at close)
        Bettor got over cheaper at entry = value.
        """
        entry_nv = american_to_implied(-110.0) / (american_to_implied(-110.0) + american_to_implied(-110.0))
        close_p_raw_over = american_to_implied(-130.0)
        close_p_raw_under = american_to_implied(110.0)
        close_nv = close_p_raw_over / (close_p_raw_over + close_p_raw_under)
        clv = compute_same_line_price_clv(entry_nv, close_nv)
        assert clv > 0

    def test_negative_clv_when_market_moved_in_favor(self):
        """Market moved to favor the bet side = entry was not great value."""
        entry_nv_over = 0.52
        close_nv_over = 0.47  # market became cheaper for over at close
        clv = compute_same_line_price_clv(entry_nv_over, close_nv_over)
        assert clv < 0

    def test_zero_clv_no_movement(self):
        clv = compute_same_line_price_clv(0.50, 0.50)
        assert abs(clv) < 1e-12

    def test_nan_inputs_return_nan(self):
        clv = compute_same_line_price_clv(float("nan"), 0.50)
        assert math.isnan(clv)


# ---------------------------------------------------------------------------
# §4.5: Side-adjusted line CLV
# ---------------------------------------------------------------------------

class TestLineClv:
    def test_favorable_over_line_movement(self):
        """§4.7: favorable over line movement: line_clv > 0 when close > entry for over."""
        # Entry line 18.5, close line 20.5 → over bettor has better number
        lclv = compute_line_clv(18.5, 20.5, "over")
        assert lclv > 0  # 20.5 - 18.5 = 2.0

    def test_unfavorable_over_line_movement(self):
        """§4.7: unfavorable over line movement: line_clv < 0 when close < entry for over."""
        lclv = compute_line_clv(20.5, 18.5, "over")
        assert lclv < 0

    def test_favorable_under_line_movement(self):
        """§4.7: favorable under line movement: line moved down for under bettor."""
        lclv = compute_line_clv(20.5, 18.5, "under")
        assert lclv > 0  # entry_line - close_line = 20.5 - 18.5 = 2.0

    def test_unfavorable_under_line_movement(self):
        """§4.7: unfavorable under line movement."""
        lclv = compute_line_clv(18.5, 20.5, "under")
        assert lclv < 0

    def test_nan_inputs_return_nan(self):
        assert math.isnan(compute_line_clv(float("nan"), 18.5, "over"))
        assert math.isnan(compute_line_clv(18.5, float("nan"), "under"))

    def test_integer_line_push_same(self):
        """§4.7: integer-line pushes (no movement = zero line CLV)."""
        lclv = compute_line_clv(20.0, 20.0, "over")
        assert abs(lclv) < 1e-12

    def test_half_point_lines(self):
        """§4.7: half-point lines."""
        lclv = compute_line_clv(19.5, 20.5, "over")
        assert abs(lclv - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# §4.4: Closing quote validation
# ---------------------------------------------------------------------------

class TestValidateClosingQuote:
    def test_valid_close_before_tip(self):
        close_pulled = "2026-07-13T21:00:00Z"
        scheduled_start = "2026-07-13T23:00:00Z"
        status = validate_closing_quote(close_pulled, scheduled_start)
        assert status == "valid"

    def test_after_tip_quote_rejected(self):
        """§4.7: quote after tip."""
        close_pulled = "2026-07-13T23:30:00Z"  # after tip
        scheduled_start = "2026-07-13T23:00:00Z"
        status = validate_closing_quote(close_pulled, scheduled_start)
        assert status == AFTER_TIP

    def test_stale_close_flagged(self):
        """§4.7: stale close — more than 24h before tip."""
        close_pulled = "2026-07-11T23:00:00Z"  # 2 days before
        scheduled_start = "2026-07-13T23:00:00Z"
        status = validate_closing_quote(close_pulled, scheduled_start)
        assert status == STALE_CLOSE

    def test_source_timestamp_after_tip_rejected(self):
        """Source update after tip should be rejected."""
        close_pulled = "2026-07-13T22:00:00Z"
        scheduled_start = "2026-07-13T23:00:00Z"
        source_updated = "2026-07-13T23:30:00Z"  # source updated after tip
        status = validate_closing_quote(close_pulled, scheduled_start, source_updated)
        assert status == AFTER_TIP

    def test_none_close_returns_unknown(self):
        status = validate_closing_quote(None, "2026-07-13T23:00:00Z")
        assert status == "unknown"


# ---------------------------------------------------------------------------
# §4.6: Opening, entry, and closing distinction
# ---------------------------------------------------------------------------

class TestComputeClvForBet:
    def _make_entry(self) -> dict:
        return {
            "game_id": 1,
            "player_id": 101,
            "stat": "pts",
            "market_type": "player_prop",
            "vendor": "draftkings",
            "line": 18.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
            "pulled_at_utc": "2026-07-13T12:00:00Z",
        }

    def test_no_closing_quote_returns_not_available(self):
        entry = self._make_entry()
        result = compute_clv_for_bet(entry, None, 0.55, "over")
        assert result.same_line_price_clv == NOT_AVAILABLE
        assert result.line_clv == NOT_AVAILABLE
        assert result.ticket_ev_at_close == NOT_AVAILABLE
        assert result.close_validation_status == "no_closing_quote"

    def test_after_tip_close_rejected(self):
        """§4.7: quote after tip."""
        entry = self._make_entry()
        closing = {
            "line": 18.5,
            "over_odds": -120.0,
            "under_odds": 100.0,
            "pulled_at_utc": "2026-07-13T23:30:00Z",  # after tip
        }
        result = compute_clv_for_bet(
            entry, closing, 0.55, "over",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert result.same_line_price_clv == NOT_AVAILABLE
        assert result.close_validation_status == AFTER_TIP

    def test_same_line_price_clv_computed(self):
        """§4.7: same-line price movement."""
        entry = self._make_entry()
        closing = {
            "line": 18.5,  # same line
            "over_odds": -130.0,  # tighter price = market moved toward over
            "under_odds": 110.0,
            "pulled_at_utc": "2026-07-13T21:00:00Z",
        }
        result = compute_clv_for_bet(
            entry, closing, 0.55, "over",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert result.close_validation_status == "valid"
        assert isinstance(result.same_line_price_clv, float)
        # Market moved toward over (higher p_over at close) = CLV > 0
        assert result.same_line_price_clv > 0

    def test_changed_line_same_line_price_clv_not_available(self):
        """§4.7: changed line without alternate-line distribution."""
        entry = self._make_entry()
        closing = {
            "line": 20.5,  # different line
            "over_odds": -110.0,
            "under_odds": -110.0,
            "pulled_at_utc": "2026-07-13T21:00:00Z",
        }
        result = compute_clv_for_bet(
            entry, closing, 0.55, "over",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert result.same_line_price_clv == NOT_AVAILABLE
        # But line CLV is available
        assert isinstance(result.line_clv, float)
        assert result.line_clv == 20.5 - 18.5

    def test_model_edge_at_entry_computed(self):
        """Model edge is computed even without closing quote."""
        entry = self._make_entry()  # -110/-110 → no-vig 0.5/0.5
        result = compute_clv_for_bet(entry, None, 0.60, "over")
        assert not math.isnan(result.model_edge_at_entry)
        assert abs(result.model_edge_at_entry - 0.10) < 1e-6

    def test_ticket_ev_not_available_without_distribution(self):
        """§4.7: ticket EV requires monotonic closing probability curve."""
        entry = self._make_entry()
        closing = {
            "line": 18.5,
            "over_odds": -120.0,
            "under_odds": 100.0,
            "pulled_at_utc": "2026-07-13T21:00:00Z",
        }
        result = compute_clv_for_bet(
            entry, closing, 0.55, "over",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert result.ticket_ev_at_close == NOT_AVAILABLE

    def test_is_hypothetical_when_no_wager_log(self):
        """§4.6: without wager execution log, label as hypothetical."""
        entry = self._make_entry()
        result = compute_clv_for_bet(entry, None, 0.55, "over")
        assert result.is_hypothetical is True


# ---------------------------------------------------------------------------
# §4.2: Append-only quote ledger
# ---------------------------------------------------------------------------

class TestQuoteLedger:
    def test_create_empty_ledger(self):
        ledger = create_empty_ledger()
        assert isinstance(ledger, pd.DataFrame)
        assert "snapshot_id" in ledger.columns
        assert "game_id" in ledger.columns

    def test_append_deduplicates_exact_duplicates(self):
        """§4.7: duplicate snapshots should be deduplicated."""
        ledger = create_empty_ledger()
        quote = pd.DataFrame([{
            "snapshot_id": "abc",
            "game_id": 1,
            "player_id": 101,
            "stat": "pts",
            "market_type": "player_prop",
            "vendor": "dk",
            "line": 18.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
            "source_updated_at": "2026-07-13T10:00:00Z",
            "pulled_at_utc": "2026-07-13T12:00:00Z",
            "scheduled_start_utc": "2026-07-13T23:00:00Z",
            "is_opening_snapshot": False,
            "is_current_snapshot": True,
            "raw_source_reference": "bdl:1234",
        }])
        ledger = append_to_ledger(ledger, quote)
        ledger = append_to_ledger(ledger, quote)  # duplicate
        assert len(ledger) == 1  # should not duplicate

    def test_append_keeps_different_prices(self):
        """§4.7: different prices must not be deduplicated."""
        ledger = create_empty_ledger()
        quote1 = pd.DataFrame([{
            "snapshot_id": "abc",
            "game_id": 1,
            "player_id": 101,
            "stat": "pts",
            "market_type": "player_prop",
            "vendor": "dk",
            "line": 18.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
            "source_updated_at": "2026-07-13T10:00:00Z",
            "pulled_at_utc": "2026-07-13T12:00:00Z",
            "scheduled_start_utc": "2026-07-13T23:00:00Z",
            "is_opening_snapshot": False,
            "is_current_snapshot": True,
            "raw_source_reference": "bdl:1234",
        }])
        quote2 = quote1.copy()
        quote2["over_odds"] = -120.0  # price changed
        quote2["pulled_at_utc"] = "2026-07-13T14:00:00Z"

        ledger = append_to_ledger(ledger, quote1)
        ledger = append_to_ledger(ledger, quote2)
        assert len(ledger) == 2

    def test_select_closing_quote_before_tip(self):
        """§4.7: deterministic close selection."""
        ledger = pd.DataFrame([
            {
                "game_id": 1,
                "player_id": 101,
                "stat": "pts",
                "market_type": "player_prop",
                "vendor": "dk",
                "line": 18.5,
                "over_odds": -110.0,
                "under_odds": -110.0,
                "pulled_at_utc": "2026-07-13T12:00:00Z",
                "source_updated_at": "2026-07-13T11:00:00Z",
                "scheduled_start_utc": "2026-07-13T23:00:00Z",
            },
            {
                "game_id": 1,
                "player_id": 101,
                "stat": "pts",
                "market_type": "player_prop",
                "vendor": "dk",
                "line": 19.5,
                "over_odds": -115.0,
                "under_odds": -105.0,
                "pulled_at_utc": "2026-07-13T20:00:00Z",  # latest before tip
                "source_updated_at": "2026-07-13T19:00:00Z",
                "scheduled_start_utc": "2026-07-13T23:00:00Z",
            },
        ])
        close = select_closing_quote(
            ledger,
            game_id=1,
            player_id=101,
            stat="pts",
            market_type="player_prop",
            vendor="dk",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert close is not None
        assert close["line"] == 19.5  # latest valid quote

    def test_select_closing_quote_rejects_after_tip(self):
        """§4.7: quote after tip must be excluded from closing selection."""
        ledger = pd.DataFrame([{
            "game_id": 1,
            "player_id": 101,
            "stat": "pts",
            "market_type": "player_prop",
            "vendor": "dk",
            "line": 18.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
            "pulled_at_utc": "2026-07-13T23:30:00Z",  # AFTER tip
            "source_updated_at": "2026-07-13T23:00:00Z",
            "scheduled_start_utc": "2026-07-13T23:00:00Z",
        }])
        close = select_closing_quote(
            ledger,
            game_id=1,
            player_id=101,
            stat="pts",
            market_type="player_prop",
            vendor="dk",
            scheduled_start_utc="2026-07-13T23:00:00Z",
        )
        assert close is None  # no valid closing quote


# ---------------------------------------------------------------------------
# Backtest CLV naming guard
# ---------------------------------------------------------------------------

def test_backtest_clv_does_not_use_clv_terminology():
    """Verify the backtest script uses correct metric names."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "backtest_clv", "scripts/backtest_clv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The main function should reference 'model_edge_agreement' not 'clv_hit_rate'
    result = mod.compute_model_edge_agreement(pd.DataFrame())
    assert "model_edge_vs_open_agreement_rate" in result
    assert "metric_note" in result
    # Must not call it CLV
    assert "NOT_CLV" in result["metric_note"]
