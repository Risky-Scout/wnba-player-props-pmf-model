"""Tests for WNBAPMFGrid — PenaltyBlog-style PMF output object."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.models.pmf_grid import WNBAPMFGrid


def _make_grid(pmf_pts=None, pmf_reb=None) -> WNBAPMFGrid:
    """Fixture: simple uniform-ish PMFs for testing."""
    if pmf_pts is None:
        pmf_pts = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.07, 0.03], dtype=float)
    if pmf_reb is None:
        pmf_reb = np.array([0.10, 0.20, 0.30, 0.25, 0.10, 0.05], dtype=float)
    return WNBAPMFGrid(
        player_id=1,
        player_name="A.Wilson",
        stat_pmfs={"pts": pmf_pts, "reb": pmf_reb},
        projected_minutes=32.0,
        role_bucket="starter",
        game_context={"game_id": 12345, "game_date": "2026-06-17"},
    )


class TestProbabilities:
    def test_over_plus_under_plus_push_eq_1_half_line(self):
        """At a half-line (no push): over + under == 1."""
        g = _make_grid()
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over = g.prob_over("pts", line)
            p_under = g.prob_under("pts", line)
            p_push = g.push_prob("pts", line)
            total = p_over + p_under + p_push
            assert abs(total - 1.0) < 1e-9, f"line={line}: {p_over}+{p_under}+{p_push}={total}"
            assert p_push == 0.0, f"Half-line {line} should have zero push"

    def test_over_plus_under_plus_push_eq_1_integer_line(self):
        """At an integer line (push possible): over + under + push == 1."""
        g = _make_grid()
        for line in [0.0, 1.0, 2.0, 3.0, 4.0]:
            p_over = g.prob_over("pts", line)
            p_under = g.prob_under("pts", line)
            p_push = g.push_prob("pts", line)
            total = p_over + p_under + p_push
            assert abs(total - 1.0) < 1e-9, f"line={line}: {p_over}+{p_under}+{p_push}={total}"

    def test_push_prob_at_integer_is_pmf_atom(self):
        """push_prob(k.0) == pmf[k]."""
        pmf = np.array([0.1, 0.3, 0.4, 0.2], dtype=float)
        g = WNBAPMFGrid(1, "Test", {"pts": pmf})
        for k in range(len(pmf)):
            assert abs(g.push_prob("pts", float(k)) - pmf[k]) < 1e-9

    def test_push_prob_zero_at_half_line(self):
        g = _make_grid()
        assert g.push_prob("pts", 2.5) == 0.0
        assert g.push_prob("pts", 0.5) == 0.0

    def test_prob_exactly(self):
        pmf = np.array([0.1, 0.3, 0.4, 0.2], dtype=float)
        g = WNBAPMFGrid(1, "Test", {"pts": pmf})
        for k in range(len(pmf)):
            assert abs(g.prob_exactly("pts", k) - pmf[k]) < 1e-9

    def test_over_at_line_beyond_support_is_zero(self):
        g = _make_grid()
        assert g.prob_over("pts", 999.5) == 0.0

    def test_under_at_line_below_zero_is_zero(self):
        g = _make_grid()
        assert g.prob_under("pts", 0.0) == 0.0

    def test_normalization_enforced_on_input(self):
        """Input PMF is renormalized even if it doesn't sum to 1."""
        unnorm = np.array([2.0, 4.0, 6.0, 4.0, 2.0], dtype=float)
        g = WNBAPMFGrid(1, "Test", {"pts": unnorm})
        p = g._pmf("pts")
        assert abs(p.sum() - 1.0) < 1e-9


class TestQuarterLine:
    def test_quarter_line_splits_evenly(self):
        """quarter_line_probs at X.25 = 0.5 * probs_at_X + 0.5 * probs_at_X.5."""
        g = _make_grid()
        stat = "pts"
        line = 2.25

        lo, hi = 2.0, 2.5
        expected_win = 0.5 * g.prob_over(stat, lo) + 0.5 * g.prob_over(stat, hi)
        expected_push = 0.5 * g.push_prob(stat, lo) + 0.5 * g.push_prob(stat, hi)
        expected_lose = 0.5 * g.prob_under(stat, lo) + 0.5 * g.prob_under(stat, hi)

        result = g.quarter_line_probs(stat, line)
        assert abs(result["win"] - expected_win) < 1e-9
        assert abs(result["push"] - expected_push) < 1e-9
        assert abs(result["lose"] - expected_lose) < 1e-9

    def test_quarter_line_sums_to_one(self):
        g = _make_grid()
        for line in [0.25, 0.75, 1.25, 1.75, 2.25]:
            r = g.quarter_line_probs("pts", line)
            assert abs(r["win"] + r["push"] + r["lose"] - 1.0) < 1e-9


class TestSummaryStats:
    def test_pmf_mean(self):
        pmf = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # always 3
        g = WNBAPMFGrid(1, "Test", {"pts": pmf})
        assert abs(g.pmf_mean("pts") - 3.0) < 1e-9

    def test_pmf_std_constant(self):
        pmf = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # always 3
        g = WNBAPMFGrid(1, "Test", {"pts": pmf})
        assert abs(g.pmf_std("pts")) < 1e-9

    def test_pmf_median(self):
        # pmf: 0→10%, 1→10%, 2→30%, 3→30%, 4→20%
        pmf = np.array([0.1, 0.1, 0.3, 0.3, 0.2], dtype=float)
        g = WNBAPMFGrid(1, "Test", {"pts": pmf})
        # CDF: 0.1, 0.2, 0.5 → median at 2
        assert g.pmf_median("pts") == 2.0

    def test_percentile_monotone(self):
        g = _make_grid()
        percs = [g.percentile("pts", p) for p in [10, 25, 50, 75, 90]]
        assert percs == sorted(percs)


class TestEdgeKelly:
    def test_edge_positive_when_model_higher(self):
        g = _make_grid()
        model_p = g.prob_over("pts", 2.5)
        assert g.edge("pts", 2.5, model_p - 0.05) > 0

    def test_edge_zero_when_equal(self):
        g = _make_grid()
        model_p = g.prob_over("pts", 2.5)
        assert abs(g.edge("pts", 2.5, model_p)) < 1e-9

    def test_kelly_zero_when_no_edge(self):
        g = _make_grid()
        model_p = g.prob_over("pts", 2.5)
        # market price higher than model
        assert g.kelly_stake("pts", 2.5, model_p + 0.10) == 0.0

    def test_kelly_positive_when_edge_exists(self):
        g = _make_grid()
        model_p = g.prob_over("pts", 2.5)
        if model_p > 0.1:
            assert g.kelly_stake("pts", 2.5, model_p - 0.05) > 0


class TestNarrativeAndDict:
    def test_narrative_contains_player_name(self):
        g = _make_grid()
        n = g.narrative("pts", 3.5)
        assert "A.Wilson" in n
        assert "pts" in n
        assert "%" in n

    def test_to_dict_structure(self):
        g = _make_grid()
        d = g.to_dict()
        assert d["player_name"] == "A.Wilson"
        assert "pts" in d["stats"]
        assert "reb" in d["stats"]
        markets = d["stats"]["pts"]["markets"]
        assert len(markets) > 0
        for m in markets:
            total = m["p_over"] + m["p_under"] + m["p_push"]
            assert abs(total - 1.0) < 1e-6, f"line={m['line']}: {total}"

    def test_missing_stat_raises_key_error(self):
        g = _make_grid()
        with pytest.raises(KeyError):
            g.prob_over("turnover", 1.5)

    def test_has_stat(self):
        g = _make_grid()
        assert g.has_stat("pts")
        assert not g.has_stat("turnover")
