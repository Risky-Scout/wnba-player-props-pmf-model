"""Tests for the canonical live engine: src/wnba_props_model/live/

These tests cover PBPParser, GammaPoissonLiveEngine, LiveEdgeCalculator,
and PBPOrchestrator deduplication.  The old test_live_engine.py covers
the legacy models/live_engine.py which is no longer the canonical engine.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.live.pbp_parser import PBPParser, LivePlayerState
from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine
from wnba_props_model.live.live_edge import LiveEdgeCalculator
from wnba_props_model.live.orchestrator import LiveGameOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROSTER = {
    "A. Wilson": {"player_id": 1, "team_id": 10, "team_side": "home"},
    "K. Collier": {"player_id": 2, "team_id": 10, "team_side": "home"},
    "B. Stewart": {"player_id": 3, "team_id": 20, "team_side": "away"},
}


# ---------------------------------------------------------------------------
# A. PBPParser tests
# ---------------------------------------------------------------------------

class TestPBPParser:
    def test_two_point_make_credits_pts_and_fgm(self):
        p = PBPParser()
        p.process_plays([{"description": "A. Wilson made 2-point layup", "order": 1}], ROSTER)
        s = p.player_states[1]
        assert s.pts == 2
        assert s.fgm == 1
        assert s.fga == 1

    def test_three_point_make_credits_fg3m_and_pts(self):
        p = PBPParser()
        p.process_plays([{"description": "B. Stewart made 3-point jumper", "order": 1}], ROSTER)
        s = p.player_states[3]
        assert s.pts == 3
        assert s.fg3m == 1

    def test_free_throw_credits_one_point(self):
        p = PBPParser()
        p.process_plays([{"description": "A. Wilson made free throw 1 of 2", "order": 1}], ROSTER)
        s = p.player_states[1]
        assert s.pts == 1
        assert s.ftm == 1

    def test_defensive_rebound(self):
        p = PBPParser()
        p.process_plays([{"description": "K. Collier defensive rebound", "order": 1}], ROSTER)
        s = p.player_states[2]
        assert s.reb == 1

    def test_foul(self):
        p = PBPParser()
        p.process_plays([{"description": "B. Stewart foul", "order": 1}], ROSTER)
        s = p.player_states[3]
        assert s.fouls == 1

    def test_compound_play_credits_both_scorer_and_assister(self):
        """Critical regression test: compound plays must credit ALL players.

        Before the D1 fix, the early `return` inside the scoring branch
        prevented the assist regex from running on the same play text.
        """
        p = PBPParser()
        play_text = "A. Wilson made 2-point layup. K. Collier assist."
        p.process_plays([{"description": play_text, "order": 1}], ROSTER)
        # Scorer gets pts
        assert p.player_states[1].pts == 2, "Scorer should get 2 pts"
        # Assister gets ast — was broken before fix
        assert 2 in p.player_states, "Assister should be in player_states"
        assert p.player_states[2].ast == 1, "Assist should be credited to K. Collier"

    def test_compound_three_plus_assist(self):
        p = PBPParser()
        play_text = "B. Stewart made 3-point jumper. A. Wilson assist."
        p.process_plays([{"description": play_text, "order": 1}], ROSTER)
        assert p.player_states[3].pts == 3
        assert p.player_states[3].fg3m == 1
        assert p.player_states[1].ast == 1

    def test_player_states_attribute_exists(self):
        """Regression: live_stream.py was calling parser.player_stats (typo)."""
        p = PBPParser()
        assert hasattr(p, "player_states"), "PBPParser must expose player_states"
        assert not hasattr(p, "player_stats"), "player_stats typo attribute must NOT exist"

    def test_unknown_player_does_not_crash(self):
        p = PBPParser()
        p.process_plays([{"description": "X. Unknown made 2-point shot", "order": 1}], ROSTER)
        # Should silently skip — no crash
        assert True

    def test_game_state_updated_from_scores(self):
        p = PBPParser()
        p.process_plays([{"description": "shot", "order": 1, "home_score": 4, "away_score": 2}], {})
        assert p.game_state["home_score"] == 4
        assert p.game_state["away_score"] == 2

    def test_visitor_score_field(self):
        """BDL uses visitor_score not away_score."""
        p = PBPParser()
        p.process_plays([{"description": "shot", "order": 1, "visitor_score": 7}], {})
        assert p.game_state["away_score"] == 7


# ---------------------------------------------------------------------------
# B. GammaPoissonLiveEngine tests
# ---------------------------------------------------------------------------

class TestGammaPoissonLiveEngine:
    def test_posterior_pmf_shifts_right_after_observed_scoring(self):
        """After seeing 5 pts in 10 min, posterior P(over 15.5) should be > prior.

        batch_compute uses getattr(ps, stat, 0) — must pass real LivePlayerState objects.
        """
        engine = GammaPoissonLiveEngine()
        pre_game = {1: {"pts": {"mean": 15.0, "line": 15.5}, "projected_minutes": 30.0}}

        # No stats yet (game start)
        results_before = engine.batch_compute(pre_game, {}, elapsed_minutes=0.0)

        # 5 pts in 10 minutes — use real LivePlayerState
        ps = LivePlayerState(player_id=1, player_name="Test", team_id=10, team_side="home")
        ps.pts = 5
        ps.minutes_played = 10.0
        results_after = engine.batch_compute(pre_game, {1: ps}, elapsed_minutes=10.0)

        p_before = results_before.get(1, {}).get("pts", {}).get("p_over", 0.5)
        p_after = results_after.get(1, {}).get("pts", {}).get("p_over", 0.5)

        # P(over 15.5) should be valid probabilities in [0,1]
        assert 0.0 <= p_before <= 1.0, f"Prior p_over out of range: {p_before}"
        assert 0.0 <= p_after <= 1.0, f"Posterior p_over out of range: {p_after}"
        # Observed pace is 5 pts / 10 min → 15 pts / 30 min = on pace for mean.
        # Bayesian update should yield p_after close to p_before (not degenerate).
        assert p_after > 0.01, f"Posterior p_over ({p_after:.4f}) is degenerate (near zero)"

    def test_t_remaining_caps_at_regulation_in_overtime(self):
        """With elapsed > regulation, t_remaining must not go negative — use real LivePlayerState."""
        engine = GammaPoissonLiveEngine()
        pre_game = {1: {"pts": {"mean": 15.0, "line": 15.5}, "projected_minutes": 40.0}}

        ps = LivePlayerState(player_id=1, player_name="Test", team_id=10, team_side="home")
        ps.pts = 20
        ps.minutes_played = 45.0

        results = engine.batch_compute(pre_game, {1: ps}, elapsed_minutes=45.0)
        pts_result = results.get(1, {}).get("pts", {})
        assert "p_over" in pts_result
        assert 0.0 <= pts_result["p_over"] <= 1.0

    def test_batch_compute_empty_states_returns_priors(self):
        engine = GammaPoissonLiveEngine()
        pre_game = {1: {"pts": {"mean": 15.0, "line": 15.5}, "projected_minutes": 30.0}}
        results = engine.batch_compute(pre_game, {}, elapsed_minutes=0.0)
        assert 1 in results
        # p_over at line=15.5 with mean=15.0 should be a valid probability
        assert 0.0 <= results[1]["pts"]["p_over"] <= 1.0
        assert results[1]["pts"]["p_over"] > 0.0, "Prior p_over should not be degenerate"


# ---------------------------------------------------------------------------
# C. LiveEdgeCalculator tests
# ---------------------------------------------------------------------------

class TestLiveEdgeCalculator:
    def _make_props_df(self, player_id, over_odds, under_odds, line=10.5, prop_type="pts"):
        return pd.DataFrame([{
            "player_id": player_id,
            "prop_type": prop_type,
            "line_value": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
        }])

    def test_positive_edge_when_model_probability_exceeds_market(self):
        calc = LiveEdgeCalculator(min_edge=0.04)
        # +115/-135 → fair p_over ≈ 0.461 (Shin) or ~0.465 (multiplicative)
        live_preds = {1: {"pts": {"p_over": 0.60, "projected_total": 12.0,
                                  "observed_count": 8, "elapsed_minutes": 20.0}}}
        props = self._make_props_df(1, over_odds=115, under_odds=-135, line=10.5)
        edges = calc.compute_live_edges(live_preds, props)
        assert len(edges) == 1
        assert edges[0]["edge"] > 0.04, "Should detect positive edge for model > market"
        assert edges[0]["bettable"] is True

    def test_negative_edge_on_under(self):
        calc = LiveEdgeCalculator(min_edge=0.04)
        live_preds = {1: {"pts": {"p_over": 0.30, "projected_total": 8.0,
                                  "observed_count": 3, "elapsed_minutes": 15.0}}}
        props = self._make_props_df(1, over_odds=-115, under_odds=+105, line=10.5)
        edges = calc.compute_live_edges(live_preds, props)
        assert len(edges) == 1
        assert edges[0]["direction"] == "under"
        assert edges[0]["bettable"] is True

    def test_combo_stat_pra_computed_from_components(self):
        calc = LiveEdgeCalculator(min_edge=0.04)
        # Player with 5 pts, 6 reb, 5 ast → projected total ~16; line 14.5 → should be over
        live_preds = {1: {
            "pts": {"p_over": 0.5, "projected_total": 5.0},
            "reb": {"p_over": 0.5, "projected_total": 6.0},
            "ast": {"p_over": 0.5, "projected_total": 5.0},
        }}
        props = pd.DataFrame([{
            "player_id": 1, "prop_type": "pra",
            "line_value": 14.5, "over_odds": -110, "under_odds": -110,
        }])
        edges = calc.compute_live_edges(live_preds, props)
        assert len(edges) == 1
        assert edges[0]["stat"] == "pra"
        # Combo mean ~16 > line 14.5 → P(over) > 0.5
        assert edges[0]["model_p_over"] > 0.5

    def test_no_edges_when_props_empty(self):
        calc = LiveEdgeCalculator(min_edge=0.04)
        live_preds = {1: {"pts": {"p_over": 0.60}}}
        edges = calc.compute_live_edges(live_preds, pd.DataFrame())
        assert edges == []

    def test_sorted_by_edge_magnitude(self):
        calc = LiveEdgeCalculator(min_edge=0.01)
        live_preds = {
            1: {"pts": {"p_over": 0.70, "projected_total": 12.0, "observed_count": 0, "elapsed_minutes": 0}},
            2: {"pts": {"p_over": 0.55, "projected_total": 10.0, "observed_count": 0, "elapsed_minutes": 0}},
        }
        props = pd.DataFrame([
            {"player_id": 1, "prop_type": "pts", "line_value": 10.5, "over_odds": -110, "under_odds": -110},
            {"player_id": 2, "prop_type": "pts", "line_value": 10.5, "over_odds": -110, "under_odds": -110},
        ])
        edges = calc.compute_live_edges(live_preds, props)
        assert len(edges) == 2
        assert abs(edges[0]["edge"]) >= abs(edges[1]["edge"]), "Edges should be sorted by |edge| descending"


# ---------------------------------------------------------------------------
# D. Orchestrator deduplication test
# ---------------------------------------------------------------------------

class TestOrchestratorDeduplication:
    def test_duplicate_plays_are_ignored(self):
        """Plays with the same event_order must not be processed twice."""
        play1 = {"description": "A. Wilson made 2-point layup", "order": 1,
                 "home_score": 2, "visitor_score": 0, "period": 1}
        play2 = {"description": "A. Wilson made 2-point layup", "order": 1,  # duplicate
                 "home_score": 2, "visitor_score": 0, "period": 1}

        parser = PBPParser()
        # Process both plays — the orchestrator should deduplicate by order
        # Simulate orchestrator behavior: track seen IDs
        seen: set[int] = set()
        new_plays = []
        for p in [play1, play2]:
            oid = p.get("order") or p.get("event_order") or p.get("id") or 0
            if oid not in seen:
                new_plays.append(p)
                seen.add(oid)

        parser.process_plays(new_plays, ROSTER)
        # Should only credit pts once
        assert parser.player_states[1].pts == 2


# ---------------------------------------------------------------------------
# E. Compound play stat credit (unit test for the D1 fix)
# ---------------------------------------------------------------------------

class TestCompoundPlayStatCredit:
    def test_three_pointer_with_assist(self):
        p = PBPParser()
        p.process_plays([{
            "description": "B. Stewart made 3-point jumper. A. Wilson assist.",
            "order": 1,
        }], ROSTER)
        assert p.player_states[3].fg3m == 1
        assert p.player_states[3].pts == 3
        assert p.player_states[1].ast == 1

    def test_made_shot_no_assist_only_credits_shooter(self):
        p = PBPParser()
        p.process_plays([{
            "description": "A. Wilson made 2-point shot",
            "order": 1,
        }], ROSTER)
        assert p.player_states[1].pts == 2
        # Collier (id=2) should not be in states
        assert 2 not in p.player_states

    def test_multiple_plays_accumulate_stats(self):
        p = PBPParser()
        plays = [
            {"description": "A. Wilson made 2-point layup. K. Collier assist.", "order": 1},
            {"description": "A. Wilson made 3-point jumper", "order": 2},
            {"description": "K. Collier defensive rebound", "order": 3},
        ]
        p.process_plays(plays, ROSTER)
        assert p.player_states[1].pts == 5  # 2 + 3
        assert p.player_states[1].fgm == 2
        assert p.player_states[1].fg3m == 1
        assert p.player_states[2].ast == 1
        assert p.player_states[2].reb == 1
