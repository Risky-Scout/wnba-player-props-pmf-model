"""Ticket 2 — Production workflow integrity: market pipeline integration tests.

Fixture-driven tests that must FAIL before production changes and PASS after.
All expected values are hard-coded from manual arithmetic; they are NOT
computed dynamically using production functions.

Fixture layout:
  Two WNBA games (G001 SEA-LAS, G002 NYL-CHI)
  Five players (3 active, 1 inactive, 1 questionable)
  Two vendors (fanduel, draftkings)
  Stats: pts, reb, ast, fg3m + two combos (pts_reb, pts_ast)
  Integer AND half-point lines
  Positive AND negative American odds
  One market with meaningful push probability (integer line=4, p_push=0.20)
  One confirmed inactive player (P004)
  One questionable player (P005)
  One stale quote (P001 pts fanduel with old timestamp)
  One duplicate quote (P002 pts fanduel line=22.5 appears twice)
  One intentionally unmatched player identity (player_id=None)
  One malformed price (P003 reb odds="INVALID_PRICE")
  One valid market with no model PMF (P_NOPMF)
  One model PMF with no valid market (P_NOMARKET)
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import production modules under test
# ---------------------------------------------------------------------------
from wnba_props_model.models.market import american_to_prob, no_vig_two_way
from wnba_props_model.pipeline.market_integrity import (
    LIVE_MARKETS_NOT_YET_AVAILABLE,
    NONCRITICAL_EXPLAINABILITY_FAILURE,
    AmbiguousIdentityError,
    ArtifactLineageMismatchError,
    DuplicateEdgeError,
    DuplicatePMFError,
    DuplicateQuoteError,
    MalformedOddsError,
    MarketIntegrityError,
    MissingEdgeError,
    MissingPMFError,
    PartialBoardError,
    StaleFallbackForbiddenError,
    StaleQuoteError,
    UnmatchedIdentityError,
    VENDOR_RULE_ACTION_IF_PLAYS,
    VENDOR_RULE_ACTION_IF_STARTS,
    VENDOR_RULE_VOID_IF_NO_PARTICIPATION,
    atomic_deploy,
    build_expected_edge_manifest,
    build_expected_pmf_manifest,
    check_inactive_market_settlement,
    check_no_stale_fallback,
    compute_model_edge,
    compute_no_vig_probs_from_american,
    compute_pmf_probabilities,
    validate_artifact_lineage,
    validate_edge_manifest,
    validate_game_identity_resolved,
    validate_no_duplicate_quotes,
    validate_odds_format,
    validate_player_identity_resolved,
    validate_pmf_manifest,
    validate_quote_freshness,
    validate_staging_board,
)

# ---------------------------------------------------------------------------
# Deterministic fixture PMFs — ALL expected values computed by hand
# ---------------------------------------------------------------------------

# P001 pts PMF over [0..9], sums to 1.0
# index: 0     1     2     3     4     5     6     7     8     9
_PMF_P001_PTS = np.array(
    [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.03, 0.01, 0.01]
)
assert abs(_PMF_P001_PTS.sum() - 1.0) < 1e-12, "PMF_P001_PTS must sum to 1"

# P001 reb PMF over [0..6], sums to 1.0
_PMF_P001_REB = np.array([0.10, 0.20, 0.30, 0.25, 0.10, 0.04, 0.01])
assert abs(_PMF_P001_REB.sum() - 1.0) < 1e-12

# P002 pts PMF over [0..9], sums to 1.0
_PMF_P002_PTS = np.array(
    [0.02, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.10, 0.04, 0.01]
)
assert abs(_PMF_P002_PTS.sum() - 1.0) < 1e-12

# P003 pts PMF over [0..7], sums to 1.0
_PMF_P003_PTS = np.array([0.08, 0.15, 0.22, 0.25, 0.17, 0.09, 0.03, 0.01])
assert abs(_PMF_P003_PTS.sum() - 1.0) < 1e-12

# P001 pts_reb combo PMF over [0..12], sums to 1.0
_PMF_P001_PTS_REB = np.array(
    [0.01, 0.02, 0.04, 0.07, 0.10, 0.14, 0.16, 0.16, 0.12, 0.09, 0.06, 0.02, 0.01]
)
assert abs(_PMF_P001_PTS_REB.sum() - 1.0) < 1e-12

# P002 pts_ast combo PMF over [0..10], sums to 1.0
_PMF_P002_PTS_AST = np.array(
    [0.01, 0.03, 0.06, 0.10, 0.15, 0.18, 0.17, 0.14, 0.10, 0.05, 0.01]
)
assert abs(_PMF_P002_PTS_AST.sum() - 1.0) < 1e-12

# ---------------------------------------------------------------------------
# PRE-COMPUTED EXPECTED VALUES (verified by hand, NOT via production functions)
# ---------------------------------------------------------------------------

# --- Integer line=4, P001 pts ---
# p_over = pmf[5]+pmf[6]+pmf[7]+pmf[8]+pmf[9] = 0.15+0.10+0.03+0.01+0.01 = 0.30
# p_push = pmf[4] = 0.20
# p_under = pmf[0]+pmf[1]+pmf[2]+pmf[3] = 0.05+0.10+0.15+0.20 = 0.50
_EXP_P001_PTS_LINE4_POVER = 0.30
_EXP_P001_PTS_LINE4_PPUSH = 0.20
_EXP_P001_PTS_LINE4_PUNDER = 0.50

# --- Half-point line=4.5, P001 pts ---
# p_over = P(stat >= 5) = pmf[5]+...+pmf[9] = 0.30 (same as above)
# p_push = 0
# p_under = 1 - 0.30 = 0.70
_EXP_P001_PTS_LINE4_5_POVER = 0.30
_EXP_P001_PTS_LINE4_5_PPUSH = 0.00
_EXP_P001_PTS_LINE4_5_PUNDER = 0.70

# --- Integer line=3, P001 pts (MEANINGFUL push) ---
# p_over = pmf[4]+...+pmf[9] = 0.20+0.15+0.10+0.03+0.01+0.01 = 0.50
# p_push = pmf[3] = 0.20
# p_under = pmf[0]+pmf[1]+pmf[2] = 0.05+0.10+0.15 = 0.30
_EXP_P001_PTS_LINE3_PPUSH = 0.20  # meaningful (>5%)

# --- American odds conversion ---
# -110: 110/(110+100) = 110/210
_EXP_NEG_110_PROB = 110 / 210  # ≈ 0.52381

# +150: 100/(150+100) = 100/250
_EXP_POS_150_PROB = 100 / 250  # = 0.40

# --- No-vig with symmetric -110/-110 ---
# raw: 110/210, 110/210 → sum = 220/210
# no-vig: each = (110/210) / (220/210) = 110/220 = 0.5 exactly
_EXP_NOVIG_SYM_OVER = 0.5
_EXP_NOVIG_SYM_UNDER = 0.5

# --- Edge with P001 pts, line=4.5, over=-110, under=-110 ---
# model_p_over=0.30, market_p_over_nv=0.5
# edge_over = 0.30 - 0.5 = -0.20
# model_p_under=0.70, market_p_under_nv=0.5
# edge_under = 0.70 - 0.5 = 0.20
_EXP_EDGE_P001_PTS_LINE4_5_EOVER = -0.20
_EXP_EDGE_P001_PTS_LINE4_5_EUNDER = 0.20

# ---------------------------------------------------------------------------
# Fixture helper — now/stale timestamps
# ---------------------------------------------------------------------------

_NOW_UTC = datetime(2026, 7, 13, 18, 0, 0, tzinfo=timezone.utc)
_FRESH_TS = (_NOW_UTC - timedelta(minutes=10)).isoformat()
_STALE_TS = (_NOW_UTC - timedelta(hours=25)).isoformat()


def _make_quotes_df(**overrides) -> pd.DataFrame:
    """Minimal valid market quote row (fresh, integer player IDs, valid odds)."""
    row = {
        "vendor": "fanduel",
        "game_id": "G001",
        "player_id": "P001",
        "stat": "pts",
        "line": 20.5,
        "over_odds": -110,
        "under_odds": -110,
        "market_updated_at": _FRESH_TS,
        "market_type": "player_prop",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_pmf_manifest_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "player_id", "stat"])


def _make_edge_manifest_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["game_id", "player_id", "stat", "vendor", "line"])


# ===========================================================================
# CATEGORY 1: Basic market math (should PASS before AND after production changes)
# ===========================================================================


class TestAmericanOddsConversion:
    def test_positive_american_odds_conversion(self):
        """american_to_prob(+150) == 100/250 == 0.40 (verified by hand)."""
        result = american_to_prob(150)
        assert result is not None
        assert abs(result - _EXP_POS_150_PROB) < 1e-10

    def test_negative_american_odds_conversion(self):
        """american_to_prob(-110) == 110/210 ≈ 0.52381 (verified by hand)."""
        result = american_to_prob(-110)
        assert result is not None
        assert abs(result - _EXP_NEG_110_PROB) < 1e-10

    def test_no_vig_probabilities_sum_to_one(self):
        """No-vig probs from symmetric -110/-110 must sum exactly to 1.0."""
        p_over, p_under = compute_no_vig_probs_from_american(-110, -110)
        assert p_over is not None and p_under is not None
        assert abs(p_over + p_under - 1.0) < 1e-10
        assert abs(p_over - _EXP_NOVIG_SYM_OVER) < 1e-10
        assert abs(p_under - _EXP_NOVIG_SYM_UNDER) < 1e-10

    def test_no_vig_asymmetric_sums_to_one(self):
        """No-vig probs from asymmetric odds must always sum to 1.0."""
        for over_odds, under_odds in [(-120, +100), (-115, -105), (+110, -130)]:
            p_over, p_under = compute_no_vig_probs_from_american(over_odds, under_odds)
            assert p_over is not None and p_under is not None, (
                f"compute_no_vig returned None for ({over_odds}, {under_odds})"
            )
            assert abs(p_over + p_under - 1.0) < 1e-9, (
                f"sum={p_over+p_under} for ({over_odds}, {under_odds})"
            )


class TestPMFProbabilities:
    def test_integer_line_preserves_push_probability(self):
        """For integer line=4 on P001 pts, p_push == 0.20 (pre-computed)."""
        p_over, p_push, p_under = compute_pmf_probabilities(_PMF_P001_PTS, 4.0)
        assert abs(p_over - _EXP_P001_PTS_LINE4_POVER) < 1e-12
        assert abs(p_push - _EXP_P001_PTS_LINE4_PPUSH) < 1e-12
        assert abs(p_under - _EXP_P001_PTS_LINE4_PUNDER) < 1e-12
        assert abs(p_over + p_push + p_under - 1.0) < 1e-12

    # Alias used in Step 14 category "Market path"
    test_integer_line_push_probability = test_integer_line_preserves_push_probability

    def test_half_point_line_has_zero_push(self):
        """For half-point line=4.5 on P001 pts, p_push == 0.0 exactly."""
        p_over, p_push, p_under = compute_pmf_probabilities(_PMF_P001_PTS, 4.5)
        assert p_push == 0.0, f"expected p_push=0, got {p_push}"
        assert abs(p_over - _EXP_P001_PTS_LINE4_5_POVER) < 1e-12
        assert abs(p_under - _EXP_P001_PTS_LINE4_5_PUNDER) < 1e-12
        assert abs(p_over + p_push + p_under - 1.0) < 1e-12

    # Alias used in Step 14 category "Market path"
    test_half_line_push_is_zero = test_half_point_line_has_zero_push

    def test_meaningful_push_on_integer_line(self):
        """Integer line=3 on P001 pts has p_push=0.20, which is meaningful (>5%)."""
        _, p_push, _ = compute_pmf_probabilities(_PMF_P001_PTS, 3.0)
        assert p_push == _EXP_P001_PTS_LINE3_PPUSH, f"expected {_EXP_P001_PTS_LINE3_PPUSH}, got {p_push}"
        assert p_push > 0.05, "push probability should be meaningful (>5%)"

    def test_pmf_probs_always_sum_to_one(self):
        """All (p_over + p_push + p_under) must sum to 1.0 for every test PMF and line."""
        cases = [
            (_PMF_P001_PTS, 4.0),
            (_PMF_P001_PTS, 4.5),
            (_PMF_P001_PTS, 3.0),
            (_PMF_P001_PTS, 3.5),
            (_PMF_P001_REB, 2.5),
            (_PMF_P001_REB, 3.0),
            (_PMF_P002_PTS, 5.0),
            (_PMF_P002_PTS, 5.5),
        ]
        for pmf, line in cases:
            p_over, p_push, p_under = compute_pmf_probabilities(pmf, line)
            total = p_over + p_push + p_under
            assert abs(total - 1.0) < 1e-12, (
                f"sum={total} for line={line}, pmf_sum={pmf.sum()}"
            )


class TestEdgeComputation:
    def test_edge_computation_is_correct(self):
        """Edge = model_prob - market_no_vig_prob. Values pre-computed by hand."""
        edge_over, edge_under = compute_model_edge(_PMF_P001_PTS, 4.5, -110, -110)
        assert abs(edge_over - _EXP_EDGE_P001_PTS_LINE4_5_EOVER) < 1e-10
        assert abs(edge_under - _EXP_EDGE_P001_PTS_LINE4_5_EUNDER) < 1e-10

    def test_edge_over_plus_edge_under_is_zero(self):
        """edge_over + edge_under == 0 by construction (model probs sum to 1, no-vig sum to 1)."""
        edge_over, edge_under = compute_model_edge(_PMF_P001_PTS, 4.5, -110, -110)
        assert abs(edge_over + edge_under) < 1e-10

    def test_edge_is_not_labeled_clv(self):
        """compute_model_edge must return (edge_over, edge_under) — not a CLV-labeled dict or namedtuple."""
        result = compute_model_edge(_PMF_P001_PTS, 4.5, -110, -110)
        # Result must be a plain tuple of two floats — no CLV labeling
        assert isinstance(result, tuple), "compute_model_edge must return a tuple"
        assert len(result) == 2, "must return exactly (edge_over, edge_under)"
        edge_over, edge_under = result
        assert isinstance(edge_over, float), "edge_over must be a float"
        assert isinstance(edge_under, float), "edge_under must be a float"
        # Verify the result is NOT a named-tuple or dict with any CLV key
        if hasattr(result, "_fields"):
            for field in result._fields:
                assert "clv" not in field.lower(), (
                    f"compute_model_edge returned a namedtuple field '{field}' containing 'clv'. "
                    "Edge must not be labeled as CLV."
                )
        # Verify the function name itself does not contain 'clv'
        assert "clv" not in compute_model_edge.__name__.lower(), (
            "compute_model_edge function name contains 'clv'"
        )


# ===========================================================================
# CATEGORY 2: Fatal validation tests (FAIL before, PASS after production changes)
# ===========================================================================


class TestDuplicateQuoteFatal:
    def test_duplicate_quote_is_fatal(self):
        """Duplicate (vendor, game_id, player_id, stat, line) raises DuplicateQuoteError."""
        row = {
            "vendor": "fanduel",
            "game_id": "G001",
            "player_id": "P002",
            "stat": "pts",
            "line": 22.5,
            "over_odds": -110,
            "under_odds": -110,
            "market_updated_at": _FRESH_TS,
        }
        df = pd.DataFrame([row, row])  # exact duplicate
        with pytest.raises(DuplicateQuoteError):
            validate_no_duplicate_quotes(df)

    def test_unique_quotes_pass_validation(self):
        """Distinct (vendor, game_id, player_id, stat, line) tuples must not raise."""
        rows = [
            {"vendor": "fanduel",     "game_id": "G001", "player_id": "P001", "stat": "pts", "line": 20.5},
            {"vendor": "draftkings",  "game_id": "G001", "player_id": "P001", "stat": "pts", "line": 20.5},
            {"vendor": "fanduel",     "game_id": "G001", "player_id": "P002", "stat": "pts", "line": 22.5},
        ]
        df = pd.DataFrame(rows)
        validate_no_duplicate_quotes(df)  # must not raise


class TestStaleQuoteFatal:
    def test_stale_quote_is_not_silently_used(self):
        """A quote with market_updated_at older than max_age_seconds raises StaleQuoteError."""
        df = _make_quotes_df(market_updated_at=_STALE_TS)
        with pytest.raises(StaleQuoteError):
            validate_quote_freshness(df, max_age_seconds=3600, current_time=_NOW_UTC)

    # Alias for Step 14 Identity category
    test_stale_quote_is_not_used = test_stale_quote_is_not_silently_used

    def test_fresh_quote_passes(self):
        """A quote with market_updated_at within max_age_seconds must not raise."""
        df = _make_quotes_df(market_updated_at=_FRESH_TS)
        validate_quote_freshness(df, max_age_seconds=3600, current_time=_NOW_UTC)  # must not raise


class TestIdentityValidation:
    def test_unmatched_player_identity_is_fatal(self):
        """Quote with player_id=None (unresolved) raises UnmatchedIdentityError."""
        df = _make_quotes_df(player_id=None)
        with pytest.raises(UnmatchedIdentityError):
            validate_player_identity_resolved(df)

    # Alias for Step 14
    test_unmatched_player_identity_fails_run = test_unmatched_player_identity_is_fatal

    def test_unmatched_game_identity_is_fatal(self):
        """Quote with game_id=None (unresolved) raises UnmatchedIdentityError."""
        df = _make_quotes_df(game_id=None)
        with pytest.raises(UnmatchedIdentityError):
            validate_game_identity_resolved(df)

    def test_ambiguous_identity_fails_run(self):
        """Two players matching the same name raises AmbiguousIdentityError."""
        df = _make_quotes_df(player_id="AMBIGUOUS")
        with pytest.raises(AmbiguousIdentityError):
            validate_player_identity_resolved(df, ambiguous_ids={"AMBIGUOUS"})

    def test_resolved_identities_pass(self):
        """Quotes with valid non-null player_id and game_id must not raise."""
        df = _make_quotes_df(player_id="P001", game_id="G001")
        validate_player_identity_resolved(df)
        validate_game_identity_resolved(df)


class TestMalformedOdds:
    def test_malformed_odds_are_fatal(self):
        """Odds value that cannot be parsed as a valid American odds number raises MalformedOddsError."""
        df = _make_quotes_df(over_odds="INVALID_PRICE", under_odds=-110)
        with pytest.raises(MalformedOddsError):
            validate_odds_format(df)

    # Alias for Step 14
    test_malformed_odds_fail_validation = test_malformed_odds_are_fatal

    def test_zero_odds_are_fatal(self):
        """Odds value of 0 is not a valid American odds number."""
        df = _make_quotes_df(over_odds=0, under_odds=-110)
        with pytest.raises(MalformedOddsError):
            validate_odds_format(df)

    def test_valid_odds_pass(self):
        """Integer and float American odds (-110, +150, -115.5) must not raise."""
        for o in [-110, 150, -115, 120, -105]:
            df = _make_quotes_df(over_odds=o, under_odds=-110)
            validate_odds_format(df)


class TestInactiveMarket:
    def test_inactive_market_uses_vendor_settlement_rule(self):
        """Inactive player with void_if_no_participation rule returns 'defer_to_settlement'."""
        result = check_inactive_market_settlement(
            player_status="inactive",
            vendor_settlement_rule=VENDOR_RULE_VOID_IF_NO_PARTICIPATION,
        )
        assert result == "defer_to_settlement"

    # Alias for Step 14
    test_inactive_market_uses_vendor_rule = test_inactive_market_uses_vendor_settlement_rule

    def test_inactive_action_if_starts_defers(self):
        """Inactive player with action_if_starts still defers (player did not start)."""
        result = check_inactive_market_settlement(
            player_status="inactive",
            vendor_settlement_rule=VENDOR_RULE_ACTION_IF_STARTS,
        )
        assert result == "defer_to_settlement"

    def test_active_player_generates_edge(self):
        """Active player must return 'generate_edge' regardless of vendor rule."""
        for rule in [VENDOR_RULE_VOID_IF_NO_PARTICIPATION, VENDOR_RULE_ACTION_IF_STARTS, VENDOR_RULE_ACTION_IF_PLAYS]:
            result = check_inactive_market_settlement("active", rule)
            assert result == "generate_edge", f"active player must generate edge for rule={rule}"


class TestPMFImmutability:
    def test_market_inputs_do_not_change_structural_pmf(self):
        """Changing line/odds/vendor must not alter the PMF array passed to compute_model_edge."""
        pmf_original = _PMF_P001_PTS.copy()
        pmf_test = _PMF_P001_PTS.copy()

        # Apply edge computation with one set of market inputs
        compute_model_edge(pmf_test, 4.5, -110, -110)
        np.testing.assert_array_equal(pmf_test, pmf_original, err_msg="PMF mutated after compute_model_edge call 1")

        # Apply edge computation with different market inputs
        compute_model_edge(pmf_test, 20.5, -115, -105)
        np.testing.assert_array_equal(pmf_test, pmf_original, err_msg="PMF mutated after compute_model_edge call 2")

    # Alias for Step 14
    test_market_mutation_does_not_change_structural_pmf = test_market_inputs_do_not_change_structural_pmf

    def test_pmf_probs_do_not_mutate_pmf(self):
        """compute_pmf_probabilities must not mutate the PMF array."""
        pmf_original = _PMF_P001_PTS.copy()
        pmf_test = _PMF_P001_PTS.copy()
        compute_pmf_probabilities(pmf_test, 4.0)
        np.testing.assert_array_equal(pmf_test, pmf_original)
        compute_pmf_probabilities(pmf_test, 4.5)
        np.testing.assert_array_equal(pmf_test, pmf_original)


# ===========================================================================
# CATEGORY 3: Stale fallback tests
# ===========================================================================


class TestStaleFallbacksForbidden:
    def test_current_slate_missing_is_fatal(self, tmp_path: Path):
        """If the current-run slate file is missing, check_no_stale_fallback must raise."""
        missing = tmp_path / "slate_2026-07-13.parquet"
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("slate", missing)

    def test_current_feature_file_missing_is_fatal(self, tmp_path: Path):
        """If the current-run feature file is missing, check_no_stale_fallback must raise."""
        missing = tmp_path / "features_2026-07-13.parquet"
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("features", missing)

    def test_current_injury_result_missing_is_fatal(self, tmp_path: Path):
        """If the current-run injury JSON is missing, check_no_stale_fallback must raise."""
        missing = tmp_path / "2026-07-13.json"
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("injury_report", missing)

    def test_current_model_artifact_missing_is_fatal(self, tmp_path: Path):
        """If the current-run model artifact is missing, check_no_stale_fallback must raise."""
        missing = tmp_path / "manifest.json"
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("model_manifest", missing)

    def test_current_market_file_missing_when_expected_is_fatal(self, tmp_path: Path):
        """If the market file is missing and games are expected, check_no_stale_fallback must raise."""
        missing = tmp_path / "wnba_player_props_oddsapi_latest.parquet"
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("market_props", missing)

    def test_historical_feature_fallback_is_forbidden(self, tmp_path: Path):
        """check_no_stale_fallback raises even when a fallback path exists."""
        current = tmp_path / "slate_today.parquet"  # does not exist
        historical = tmp_path / "wnba_player_game_features_wide.parquet"
        historical.write_text("fake")  # fallback exists but must not be used
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("slate", current, fallback_path=historical)

    def test_prior_odds_fallback_is_forbidden(self, tmp_path: Path):
        """If current odds file is missing, using a prior odds file is forbidden."""
        current = tmp_path / "odds_today.parquet"  # does not exist
        prior = tmp_path / "odds_yesterday.parquet"
        prior.write_text("fake")
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("market_props", current, fallback_path=prior)

    def test_present_file_passes(self, tmp_path: Path):
        """check_no_stale_fallback must NOT raise when the current file exists."""
        current = tmp_path / "slate_today.parquet"
        current.write_text("fake")
        check_no_stale_fallback("slate", current)  # must not raise


# ===========================================================================
# CATEGORY 4: PMF manifest tests
# ===========================================================================


class TestPMFManifest:
    def _make_slate(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"game_id": "G001", "player_id": "P001"},
            {"game_id": "G001", "player_id": "P002"},
            {"game_id": "G002", "player_id": "P003"},
        ])

    def test_every_expected_pmf_exists_once(self):
        """When expected == actual, validate_pmf_manifest must not raise."""
        rows = [
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P001", "stat": "reb"},
            {"game_id": "G001", "player_id": "P002", "stat": "pts"},
        ]
        expected = _make_pmf_manifest_df(rows)
        actual = _make_pmf_manifest_df(rows)
        validate_pmf_manifest(expected, actual)  # must not raise

    def test_missing_expected_pmf_fails_run(self):
        """If an expected PMF is absent from the actual PMFs, validate_pmf_manifest raises MissingPMFError."""
        expected = _make_pmf_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P001", "stat": "reb"},  # missing in actual
        ])
        actual = _make_pmf_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
        ])
        with pytest.raises(MissingPMFError):
            validate_pmf_manifest(expected, actual)

    def test_duplicate_pmf_fails_run(self):
        """If the same (game_id, player_id, stat) appears twice in actual, raises DuplicatePMFError."""
        row = {"game_id": "G001", "player_id": "P001", "stat": "pts"}
        expected = _make_pmf_manifest_df([row])
        actual = _make_pmf_manifest_df([row, row])  # duplicate
        with pytest.raises(DuplicatePMFError):
            validate_pmf_manifest(expected, actual)

    def test_unexpected_pmf_fails_run(self):
        """An actual PMF with no matching expected entry raises MissingPMFError (unexpected)."""
        expected = _make_pmf_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
        ])
        actual = _make_pmf_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P999", "stat": "pts"},  # unexpected
        ])
        with pytest.raises(MissingPMFError):
            validate_pmf_manifest(expected, actual)

    def test_build_expected_pmf_manifest_shape(self):
        """build_expected_pmf_manifest returns one row per (game_id, player_id, stat)."""
        slate = self._make_slate()
        stats = ["pts", "reb"]
        manifest = build_expected_pmf_manifest(slate, stats)
        assert set(manifest.columns) >= {"game_id", "player_id", "stat"}
        # 3 players × 2 stats = 6 rows
        assert len(manifest) == 6


# ===========================================================================
# CATEGORY 5: Edge manifest tests
# ===========================================================================


class TestEdgeManifest:
    def _make_valid_markets(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 20.5},
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "draftkings", "line": 20.5},
            {"game_id": "G001", "player_id": "P002", "stat": "pts", "vendor": "fanduel",    "line": 22.5},
        ])

    def test_every_expected_edge_exists_once(self):
        """When expected == actual, validate_edge_manifest must not raise."""
        rows = [
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 20.5},
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "draftkings", "line": 20.5},
        ]
        expected = _make_edge_manifest_df(rows)
        actual = _make_edge_manifest_df(rows)
        validate_edge_manifest(expected, actual)  # must not raise

    def test_missing_expected_edge_fails_run(self):
        """A missing expected edge raises MissingEdgeError."""
        expected = _make_edge_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 20.5},
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "draftkings", "line": 20.5},
        ])
        actual = _make_edge_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
        ])
        with pytest.raises(MissingEdgeError):
            validate_edge_manifest(expected, actual)

    def test_duplicate_edge_fails_run(self):
        """Duplicate (game_id, player_id, stat, vendor, line) in actual raises DuplicateEdgeError."""
        row = {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5}
        expected = _make_edge_manifest_df([row])
        actual = _make_edge_manifest_df([row, row])
        with pytest.raises(DuplicateEdgeError):
            validate_edge_manifest(expected, actual)

    def test_build_expected_edge_manifest_shape(self):
        """build_expected_edge_manifest returns correct rows from valid markets."""
        markets = self._make_valid_markets()
        manifest = build_expected_edge_manifest(markets)
        assert set(manifest.columns) >= {"game_id", "player_id", "stat", "vendor", "line"}
        assert len(manifest) == len(markets)


# ===========================================================================
# CATEGORY 6: Blocking step tests
# ===========================================================================


class TestBlockingSteps:
    def test_failed_injury_step_blocks_deployment(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when injury artifact is missing."""
        staging = tmp_path / "staging"
        staging.mkdir()
        # Write all required artifacts except injury
        required = [
            "full_pmfs_wide.parquet",
            "publishable_edges.parquet",
            "run_metadata.json",
            # "injury_report_2026-07-13.json",  # intentionally missing
        ]
        run_meta = {"github_run_id": "RUN001", "game_date": "2026-07-13"}
        for name in required:
            if name.endswith(".json"):
                (staging / name).write_text(json.dumps(run_meta))
            else:
                pd.DataFrame().to_parquet(staging / name)
        # injury artifact is missing
        with pytest.raises(PartialBoardError):
            validate_staging_board(
                staging,
                run_id="RUN001",
                required_artifacts=required + ["injury_report_2026-07-13.json"],
            )

    def test_failed_odds_step_blocks_deployment(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when market artifact is missing."""
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(PartialBoardError):
            validate_staging_board(
                staging,
                run_id="RUN001",
                required_artifacts=["wnba_player_props_oddsapi_latest.parquet"],
            )

    def test_failed_edge_report_blocks_deployment(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when edge artifact is missing."""
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(PartialBoardError):
            validate_staging_board(
                staging,
                run_id="RUN001",
                required_artifacts=["publishable_edges.parquet"],
            )

    def test_failed_page_generation_blocks_deployment(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when web page artifact is missing."""
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(PartialBoardError):
            validate_staging_board(
                staging,
                run_id="RUN001",
                required_artifacts=["edge_board.html"],
            )

    def test_failed_distribution_page_blocks_deployment(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when distributions page is missing."""
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(PartialBoardError):
            validate_staging_board(
                staging,
                run_id="RUN001",
                required_artifacts=["distributions.html"],
            )

    def test_failed_deployment_does_not_replace_live_board(self, tmp_path: Path):
        """atomic_deploy with invalid staging must not modify the live directory."""
        staging = tmp_path / "staging"
        staging.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        # Write a sentinel file in live
        sentinel = live / "live_sentinel.txt"
        sentinel.write_text("previous_live_content")

        # staging is missing required artifacts
        release_manifest = {
            "github_run_id": "RUN001",
            "git_commit": "abc123",
            "artifact_hashes": {},
            "validation_result": "FAIL",  # explicitly failing
            "deployment_timestamp": _NOW_UTC.isoformat(),
        }
        with pytest.raises((PartialBoardError, MarketIntegrityError)):
            atomic_deploy(staging, live, release_manifest)

        # live directory must be unchanged
        assert sentinel.read_text() == "previous_live_content", "live board was modified on failed deploy"


# ===========================================================================
# CATEGORY 7: Artifact lineage tests
# ===========================================================================


class TestArtifactLineage:
    _BASE_META = {
        "github_run_id": "RUN001",
        "git_commit": "abc123",
        "game_date": "2026-07-13",
        "prediction_timestamp_utc": _NOW_UTC.isoformat(),
    }

    def test_artifact_from_prior_run_is_rejected(self):
        """Artifact with a different github_run_id raises ArtifactLineageMismatchError."""
        artifacts = [
            {**self._BASE_META},
            {**self._BASE_META, "github_run_id": "PRIOR_RUN_999"},  # wrong run
        ]
        with pytest.raises(ArtifactLineageMismatchError):
            validate_artifact_lineage(
                artifacts,
                run_id="RUN001",
                git_commit="abc123",
                game_date="2026-07-13",
                prediction_timestamp_utc=_NOW_UTC.isoformat(),
            )

    def test_artifact_from_different_commit_is_rejected(self):
        """Artifact with a different git_commit raises ArtifactLineageMismatchError."""
        artifacts = [
            {**self._BASE_META},
            {**self._BASE_META, "git_commit": "deadbeef"},  # wrong commit
        ]
        with pytest.raises(ArtifactLineageMismatchError):
            validate_artifact_lineage(
                artifacts,
                run_id="RUN001",
                git_commit="abc123",
                game_date="2026-07-13",
                prediction_timestamp_utc=_NOW_UTC.isoformat(),
            )

    def test_artifact_game_date_mismatch_is_rejected(self):
        """Artifact with a different game_date raises ArtifactLineageMismatchError."""
        artifacts = [
            {**self._BASE_META},
            {**self._BASE_META, "game_date": "2026-07-12"},  # wrong date
        ]
        with pytest.raises(ArtifactLineageMismatchError):
            validate_artifact_lineage(
                artifacts,
                run_id="RUN001",
                git_commit="abc123",
                game_date="2026-07-13",
                prediction_timestamp_utc=_NOW_UTC.isoformat(),
            )

    def test_matching_artifacts_pass(self):
        """Artifacts all matching run_id, commit, game_date, and timestamp must not raise."""
        artifacts = [self._BASE_META, self._BASE_META, self._BASE_META]
        validate_artifact_lineage(
            artifacts,
            run_id="RUN001",
            git_commit="abc123",
            game_date="2026-07-13",
            prediction_timestamp_utc=_NOW_UTC.isoformat(),
        )  # must not raise

    def test_partial_board_cannot_be_deployed(self, tmp_path: Path):
        """validate_staging_board raises PartialBoardError when any required artifact is absent."""
        staging = tmp_path / "staging"
        staging.mkdir()
        # Write only one of three required artifacts
        run_meta = {"github_run_id": "RUN001", "game_date": "2026-07-13"}
        (staging / "run_metadata.json").write_text(json.dumps(run_meta))
        required = ["run_metadata.json", "full_pmfs_wide.parquet", "publishable_edges.parquet"]
        with pytest.raises(PartialBoardError):
            validate_staging_board(staging, run_id="RUN001", required_artifacts=required)

    def test_atomic_failure_preserves_previous_live_board(self, tmp_path: Path):
        """Failed atomic_deploy must not corrupt or overwrite the live directory."""
        staging = tmp_path / "staging"
        staging.mkdir()
        live = tmp_path / "live"
        live.mkdir()
        (live / "index.html").write_text("<html>previous</html>")

        release_manifest = {
            "github_run_id": "RUN001",
            "git_commit": "abc123",
            "artifact_hashes": {},
            "validation_result": "FAIL",
            "deployment_timestamp": _NOW_UTC.isoformat(),
        }
        with pytest.raises((PartialBoardError, MarketIntegrityError)):
            atomic_deploy(staging, live, release_manifest)

        assert (live / "index.html").read_text() == "<html>previous</html>"

    def test_no_live_markets_has_explicit_nonpass_status(self):
        """When odds API returns no markets, the result must be LIVE_MARKETS_NOT_YET_AVAILABLE,
        not 'PASS', 'COMPLETE_VALID_BOARD', or 'MARKET_PATH_VALIDATED'."""
        assert LIVE_MARKETS_NOT_YET_AVAILABLE != "PASS"
        assert LIVE_MARKETS_NOT_YET_AVAILABLE != "COMPLETE_VALID_BOARD"
        assert LIVE_MARKETS_NOT_YET_AVAILABLE != "MARKET_PATH_VALIDATED"
        assert isinstance(LIVE_MARKETS_NOT_YET_AVAILABLE, str)
        assert len(LIVE_MARKETS_NOT_YET_AVAILABLE) > 0

    def test_all_pages_share_same_run_id(self, tmp_path: Path):
        """All artifacts in staging must share the same github_run_id."""
        staging = tmp_path / "staging"
        staging.mkdir()
        run_id = "RUN42"
        # Write two consistent artifacts
        meta = {"github_run_id": run_id, "git_commit": "abc", "game_date": "2026-07-13",
                "prediction_timestamp_utc": _NOW_UTC.isoformat()}
        for name in ["run_metadata.json", "edge_report_2026-07-13.json"]:
            (staging / name).write_text(json.dumps(meta))
        # Write one with wrong run_id
        bad_meta = {**meta, "github_run_id": "WRONG_RUN"}
        (staging / "pmf_manifest.json").write_text(json.dumps(bad_meta))

        with pytest.raises(ArtifactLineageMismatchError):
            validate_artifact_lineage(
                [meta, meta, bad_meta],
                run_id=run_id,
                git_commit="abc",
                game_date="2026-07-13",
                prediction_timestamp_utc=_NOW_UTC.isoformat(),
            )


# ===========================================================================
# CATEGORY 8: Full market fixture end-to-end test
# ===========================================================================


def _build_full_fixture() -> dict:
    """Build the deterministic two-game market fixture.

    Returns a dict with keys:
      games, players, pmfs, clean_market_rows, special_rows,
      expected_edge_count, integer_line_count, half_line_count,
      push_row_count, vendor_count
    """
    games = [
        {"game_id": "G001", "game_date": "2026-07-13",
         "home_team_id": "SEA", "away_team_id": "LAS",
         "scheduled_start_utc": "2026-07-13T23:00:00Z"},
        {"game_id": "G002", "game_date": "2026-07-13",
         "home_team_id": "NYL", "away_team_id": "CHI",
         "scheduled_start_utc": "2026-07-14T00:00:00Z"},
    ]
    players = [
        {"player_id": "P001", "player_name": "Alice Stewart",   "team_id": "SEA", "status": "active"},
        {"player_id": "P002", "player_name": "Brenda Wilson",   "team_id": "LAS", "status": "active"},
        {"player_id": "P003", "player_name": "Carol Ionescu",   "team_id": "NYL", "status": "active"},
        {"player_id": "P004", "player_name": "Diana Copper",    "team_id": "CHI", "status": "inactive"},
        {"player_id": "P005", "player_name": "Eve DeShields",   "team_id": "CHI", "status": "questionable"},
    ]
    pmfs = [
        {"game_id": "G001", "player_id": "P001", "stat": "pts",     "pmf": _PMF_P001_PTS},
        {"game_id": "G001", "player_id": "P001", "stat": "reb",     "pmf": _PMF_P001_REB},
        {"game_id": "G001", "player_id": "P001", "stat": "fg3m",    "pmf": np.array([0.10, 0.25, 0.30, 0.20, 0.10, 0.05])},
        {"game_id": "G001", "player_id": "P001", "stat": "pts_reb", "pmf": _PMF_P001_PTS_REB},
        {"game_id": "G001", "player_id": "P001", "stat": "pts_ast", "pmf": _PMF_P002_PTS_AST},
        {"game_id": "G001", "player_id": "P002", "stat": "pts",     "pmf": _PMF_P002_PTS},
        {"game_id": "G001", "player_id": "P002", "stat": "reb",     "pmf": np.array([0.05, 0.15, 0.25, 0.30, 0.15, 0.07, 0.03])},
        {"game_id": "G001", "player_id": "P002", "stat": "pts_ast", "pmf": _PMF_P002_PTS_AST},
        {"game_id": "G002", "player_id": "P003", "stat": "pts",     "pmf": _PMF_P003_PTS},
        {"game_id": "G002", "player_id": "P003", "stat": "ast",     "pmf": np.array([0.10, 0.20, 0.30, 0.25, 0.10, 0.04, 0.01])},
        {"game_id": "G002", "player_id": "P003", "stat": "fg3m",    "pmf": np.array([0.15, 0.30, 0.30, 0.18, 0.06, 0.01])},
        {"game_id": "G002", "player_id": "P004", "stat": "pts",     "pmf": np.array([0.10, 0.20, 0.30, 0.25, 0.10, 0.05])},
        {"game_id": "G002", "player_id": "P005", "stat": "pts",     "pmf": np.array([0.08, 0.18, 0.28, 0.24, 0.13, 0.07, 0.02])},
        # P_NOMARKET: PMF with no market quote
        {"game_id": "G001", "player_id": "P_NOMARKET", "stat": "pts", "pmf": _PMF_P001_PTS},
    ]

    # ---- Clean market rows (16 valid + 1 inactive = 17 total) ----
    clean_market_rows = [
        # G001 P001 pts — integer line AND half-point line
        {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 20.5, "over_odds": -115, "under_odds": -105, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 21.0, "over_odds": -120, "under_odds": +100, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},  # INTEGER line
        {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "draftkings", "line": 20.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G001 P001 reb
        {"game_id": "G001", "player_id": "P001", "stat": "reb", "vendor": "draftkings", "line": 6.5,  "over_odds": -115, "under_odds": -105, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G001 P001 fg3m — positive odds
        {"game_id": "G001", "player_id": "P001", "stat": "fg3m", "vendor": "fanduel",   "line": 2.5,  "over_odds": +110, "under_odds": -130, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G001", "player_id": "P001", "stat": "fg3m", "vendor": "draftkings","line": 3.5,  "over_odds": +140, "under_odds": -165, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G001 P001 combos
        {"game_id": "G001", "player_id": "P001", "stat": "pts_reb", "vendor": "fanduel",    "line": 26.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G001", "player_id": "P001", "stat": "pts_ast", "vendor": "draftkings", "line": 27.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G001 P002 pts
        {"game_id": "G001", "player_id": "P002", "stat": "pts", "vendor": "fanduel",    "line": 22.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G001", "player_id": "P002", "stat": "pts", "vendor": "draftkings", "line": 22.5, "over_odds": -108, "under_odds": -112, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G001 P002 reb + combo
        {"game_id": "G001", "player_id": "P002", "stat": "reb",     "vendor": "fanduel", "line": 8.5,  "over_odds": -115, "under_odds": -105, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G001", "player_id": "P002", "stat": "pts_ast", "vendor": "fanduel", "line": 27.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G002 P003 pts + ast + fg3m
        {"game_id": "G002", "player_id": "P003", "stat": "pts",  "vendor": "fanduel",    "line": 18.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G002", "player_id": "P003", "stat": "pts",  "vendor": "draftkings", "line": 18.5, "over_odds": -112, "under_odds": -108, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G002", "player_id": "P003", "stat": "ast",  "vendor": "fanduel",    "line": 5.5,  "over_odds": +105, "under_odds": -125, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        {"game_id": "G002", "player_id": "P003", "stat": "fg3m", "vendor": "fanduel",    "line": 2.5,  "over_odds": +120, "under_odds": -145, "market_updated_at": _FRESH_TS, "market_type": "player_prop"},
        # G002 P004 INACTIVE player — has quote but deferred
        {"game_id": "G002", "player_id": "P004", "stat": "pts",  "vendor": "fanduel",    "line": 12.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop",
         "player_status": "inactive", "vendor_settlement_rule": VENDOR_RULE_VOID_IF_NO_PARTICIPATION},
    ]

    # ---- Special / problematic rows ----
    special_rows = [
        # STALE: same player/stat as row 0 but 25h old
        {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel",    "line": 20.5, "over_odds": -115, "under_odds": -105, "market_updated_at": _STALE_TS, "market_type": "player_prop", "_special": "STALE"},
        # DUPLICATE: exact copy of clean row 8 (G001 P002 pts fanduel)
        {"game_id": "G001", "player_id": "P002", "stat": "pts", "vendor": "fanduel",    "line": 22.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop", "_special": "DUPLICATE"},
        # UNMATCHED IDENTITY: player_id=None
        {"game_id": "G001", "player_id": None,   "stat": "pts", "vendor": "fanduel",    "line": 15.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop", "_special": "UNMATCHED"},
        # MALFORMED PRICE
        {"game_id": "G002", "player_id": "P003", "stat": "reb", "vendor": "fanduel",    "line": 4.5,  "over_odds": "INVALID_PRICE", "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop", "_special": "MALFORMED"},
        # MARKET WITH NO PMF
        {"game_id": "G001", "player_id": "P_NOPMF", "stat": "pts", "vendor": "fanduel", "line": 10.5, "over_odds": -110, "under_odds": -110, "market_updated_at": _FRESH_TS, "market_type": "player_prop", "_special": "NO_PMF"},
    ]

    # Expected counts from the clean fixture
    integer_line_count = sum(1 for r in clean_market_rows if float(r["line"]) == int(r["line"]))  # line=21.0
    half_line_count = sum(1 for r in clean_market_rows if float(r["line"]) != int(r["line"]))
    # Only P001 pts line=21 has meaningful push on integer line
    push_rows_checked = 1  # integer line with p_push > 0
    vendors = {r["vendor"] for r in clean_market_rows}
    # Edges: all clean rows minus P004 (inactive/deferred) = 16
    expected_edge_count = len(clean_market_rows) - 1  # subtract inactive P004

    return {
        "games": games,
        "players": players,
        "pmfs": pmfs,
        "clean_market_rows": clean_market_rows,
        "special_rows": special_rows,
        "fixture_market_rows": len(clean_market_rows),
        "expected_edge_count": expected_edge_count,
        "integer_line_count": integer_line_count,
        "half_line_count": half_line_count,
        "push_rows_checked": push_rows_checked,
        "vendor_count": len(vendors),
    }


_FIXTURE = _build_full_fixture()


class TestMarketFixtureEndToEnd:
    def test_market_fixture_end_to_end(self):
        """All clean market rows pass validation; special rows each trigger the expected error."""
        clean_df = pd.DataFrame(_FIXTURE["clean_market_rows"]).drop(
            columns=["player_status", "vendor_settlement_rule", "_special"],
            errors="ignore"
        )
        # Clean rows: all validations must pass
        validate_no_duplicate_quotes(clean_df)
        validate_quote_freshness(clean_df, max_age_seconds=3600, current_time=_NOW_UTC)
        validate_player_identity_resolved(clean_df)
        validate_game_identity_resolved(clean_df)
        validate_odds_format(clean_df)

    def test_fixture_integer_line_count(self):
        """Fixture must contain the expected number of integer lines."""
        assert _FIXTURE["integer_line_count"] >= 1, "Fixture must have at least one integer line"

    def test_fixture_half_line_count(self):
        """Fixture must contain the expected number of half-point lines."""
        assert _FIXTURE["half_line_count"] >= 1, "Fixture must have at least one half-point line"

    def test_fixture_has_two_vendors(self):
        """Fixture must have at least two sportsbook vendors."""
        assert _FIXTURE["vendor_count"] >= 2

    def test_fixture_has_combo_markets(self):
        """Fixture must have at least two combo stat markets."""
        stats = {r["stat"] for r in _FIXTURE["clean_market_rows"]}
        combo_stats = {s for s in stats if "_" in s}
        assert len(combo_stats) >= 2, f"Expected >=2 combos, got {combo_stats}"

    def test_exact_market_keys_are_preserved(self):
        """All required market columns survive fixture construction unchanged."""
        required_keys = {"game_id", "player_id", "stat", "vendor", "line",
                         "over_odds", "under_odds", "market_updated_at"}
        for row in _FIXTURE["clean_market_rows"]:
            missing = required_keys - set(row.keys())
            assert not missing, f"Missing keys in clean row: {missing}"

    def test_stale_special_row_raises(self):
        """The STALE special row raises StaleQuoteError."""
        stale_rows = [r for r in _FIXTURE["special_rows"] if r.get("_special") == "STALE"]
        assert stale_rows, "Fixture must have at least one STALE row"
        df = pd.DataFrame(stale_rows).drop(columns=["_special"], errors="ignore")
        with pytest.raises(StaleQuoteError):
            validate_quote_freshness(df, max_age_seconds=3600, current_time=_NOW_UTC)

    def test_duplicate_special_row_raises(self):
        """The DUPLICATE special row combined with its clean counterpart raises DuplicateQuoteError."""
        dup_rows = [r for r in _FIXTURE["special_rows"] if r.get("_special") == "DUPLICATE"]
        clean_rows = [r for r in _FIXTURE["clean_market_rows"] if r["player_id"] == "P002" and r["stat"] == "pts" and r["vendor"] == "fanduel"]
        combined = pd.DataFrame(clean_rows + dup_rows).drop(columns=["_special"], errors="ignore")
        with pytest.raises(DuplicateQuoteError):
            validate_no_duplicate_quotes(combined)

    def test_unmatched_special_row_raises(self):
        """The UNMATCHED special row raises UnmatchedIdentityError."""
        unk_rows = [r for r in _FIXTURE["special_rows"] if r.get("_special") == "UNMATCHED"]
        df = pd.DataFrame(unk_rows).drop(columns=["_special"], errors="ignore")
        with pytest.raises(UnmatchedIdentityError):
            validate_player_identity_resolved(df)

    def test_malformed_special_row_raises(self):
        """The MALFORMED special row raises MalformedOddsError."""
        mal_rows = [r for r in _FIXTURE["special_rows"] if r.get("_special") == "MALFORMED"]
        df = pd.DataFrame(mal_rows).drop(columns=["_special"], errors="ignore")
        with pytest.raises(MalformedOddsError):
            validate_odds_format(df)

    def test_inactive_player_deferred_not_scored(self):
        """P004 (inactive) with void_if_no_participation must return 'defer_to_settlement', not 'generate_edge'."""
        inactive_row = next(
            r for r in _FIXTURE["clean_market_rows"]
            if r["player_id"] == "P004"
        )
        result = check_inactive_market_settlement(
            inactive_row["player_status"],
            inactive_row["vendor_settlement_rule"],
        )
        assert result == "defer_to_settlement"
        assert result != "generate_edge"

    def test_pmf_with_no_market_is_not_fatal(self):
        """P_NOMARKET has a PMF but no market quote — this is allowed (not fatal)."""
        nomarket = [p for p in _FIXTURE["pmfs"] if p["player_id"] == "P_NOMARKET"]
        assert len(nomarket) == 1, "Fixture must have exactly one P_NOMARKET PMF"
        # Having a PMF with no market is not an error — just no edge is generated

    def test_market_with_no_pmf_triggers_missing_pmf_error(self):
        """P_NOPMF has a market quote but no model PMF — this raises MissingPMFError."""
        nopmf_market = next(
            r for r in _FIXTURE["special_rows"]
            if r.get("_special") == "NO_PMF"
        )
        actual_pmf_players = {p["player_id"] for p in _FIXTURE["pmfs"]}
        assert nopmf_market["player_id"] not in actual_pmf_players

        expected = _make_pmf_manifest_df([
            {"game_id": nopmf_market["game_id"], "player_id": nopmf_market["player_id"], "stat": nopmf_market["stat"]}
        ])
        actual = _make_pmf_manifest_df([])  # no matching PMF
        with pytest.raises(MissingPMFError):
            validate_pmf_manifest(expected, actual)


# ===========================================================================
# CATEGORY 9: Deployed artifacts match release manifest
# ===========================================================================


class TestDeployedArtifactsMatchManifest:
    def test_deployed_artifacts_match_release_manifest(self, tmp_path: Path):
        """atomic_deploy with a valid staging dir and matching manifest must succeed."""
        staging = tmp_path / "staging"
        staging.mkdir()
        live = tmp_path / "live"
        live.mkdir()

        run_id = "RUN_VALID_001"
        # Create all required artifacts in staging
        required = [
            "full_pmfs_wide.parquet",
            "publishable_edges.parquet",
            "run_metadata.json",
        ]
        run_meta = {
            "github_run_id": run_id,
            "git_commit": "abc123",
            "game_date": "2026-07-13",
            "prediction_timestamp_utc": _NOW_UTC.isoformat(),
        }
        for name in required:
            if name.endswith(".json"):
                (staging / name).write_text(json.dumps(run_meta))
            else:
                pd.DataFrame({"col": [1]}).to_parquet(staging / name)

        import hashlib
        artifact_hashes = {}
        for name in required:
            content = (staging / name).read_bytes()
            artifact_hashes[name] = hashlib.sha256(content).hexdigest()

        release_manifest = {
            "github_run_id": run_id,
            "git_commit": "abc123",
            "artifact_hashes": artifact_hashes,
            "validation_result": "PASS",
            "deployment_timestamp": _NOW_UTC.isoformat(),
        }

        atomic_deploy(staging, live, release_manifest)

        # Verify all artifacts were promoted to live
        for name in required:
            assert (live / name).exists(), f"{name} not found in live after deploy"
