"""Strengthened Stage 9 feature-classification contract: explicit allowlist rejects market /
injury / forward / outcome / ambiguous columns and approves strictly-lagged pregame features."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stage9_build", Path(__file__).resolve().parent.parent / "scripts" / "build_stage9_modeling_artifacts.py")
S9 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S9)


def _status(name):
    return S9.classify_feature(name)[0]


def test_market_features_rejected():
    for f in ["game_total", "game_spread_home", "implied_team_total", "blowout_risk",
              "close_game_indicator", "predicted_spread_abs", "player_market_p_over_prev"]:
        assert _status(f) == "REJECTED", f


def test_injury_forward_features_rejected():
    for f in ["player_injured_l1", "player_role_elevation", "projected_usage_given_absences",
              "team_top3_scorers_available", "teammate_out_count", "usage_vacated_proxy",
              "team_out_count", "confirmed_starter", "lineup_confirmed"]:
        assert _status(f) == "REJECTED", f


def test_internal_game_script_rejected():
    for f in ["blowout_probability", "close_game_probability", "pregame_win_probability",
              "expected_minutes_given_script", "minutes_upside"]:
        assert _status(f) == "REJECTED", f


def test_outcome_like_rejected():
    for f in ["actual_pts", "did_play", "actual_minutes"]:
        assert _status(f) == "REJECTED", f


def test_ambiguous_unlagged_rejected():
    # season-level aggregate without a lag token -> conservative reject
    for f in ["player_usage_pct", "player_efg_pct", "team_playoff_seed"]:
        assert _status(f) == "REJECTED", f


def test_strictly_lagged_features_approved():
    for f in ["player_pts_mean_l5", "player_reb_mean_l10", "player_minutes_mean_l5",
              "player_pts_per_min_roll5", "opp_pts_allowed_roll5", "player_pts_std_l10"]:
        assert _status(f) == "APPROVED_ESTIMATOR_FEATURE", f


def test_pregame_facts_approved():
    for f in ["rest_days", "is_home", "player_dnp_streak_prior", "recent_starter_rate5",
              "player_games_in_last_7d", "season_phase_ratio"]:
        assert _status(f) == "APPROVED_ESTIMATOR_FEATURE", f
