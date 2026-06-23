"""Unit tests for Enhancement 6: Live Bayesian Engine."""
import math
import pytest
from wnba_props_model.models.live_engine import (
    LiveEngine,
    LivePlayerState,
    GameState,
    GAME_DURATION,
    BLOWOUT_MARGIN,
    build_pregame_ratings_from_pmfs,
)


def _make_pregame() -> dict:
    return {
        1001: {
            "team_id": 11, "position_group": "guard", "minutes": 34.0,
            "pts":      {"projection": 22.5, "variance": 36.0},
            "reb":      {"projection": 4.2,  "variance": 4.0},
            "ast":      {"projection": 6.1,  "variance": 9.0},
            "fg3m":     {"projection": 2.5,  "variance": 2.5},
            "turnover": {"projection": 2.8,  "variance": 3.0},
        },
        1002: {
            "team_id": 12, "position_group": "big", "minutes": 28.0,
            "pts":  {"projection": 14.5, "variance": 25.0},
            "reb":  {"projection": 9.5,  "variance": 16.0},
            "blk":  {"projection": 1.8,  "variance": 2.5},
        },
    }


def _make_engine() -> LiveEngine:
    engine = LiveEngine(_make_pregame())
    engine.initialize_game({
        "game_id": 999, "home_team_id": 11, "away_team_id": 12
    })
    return engine


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

def test_initialize_sets_player_states():
    engine = _make_engine()
    assert 1001 in engine.player_states
    assert 1002 in engine.player_states


def test_initialize_sets_gamma_priors():
    engine = _make_engine()
    s = engine.player_states[1001]
    assert s.alpha_prior.get("pts", 0) > 0
    assert s.beta_prior.get("pts", 0) > 0


def test_initialize_prior_mean():
    """Posterior rate at t=0 should approximately equal pre-game projection / minutes."""
    engine = _make_engine()
    s = engine.player_states[1001]
    alpha = s.alpha_prior["pts"]
    beta  = s.beta_prior["pts"]
    prior_rate = alpha / beta   # per minute
    expected_rate = 22.5 / 34.0
    # Should be within 20% of expected rate
    assert abs(prior_rate - expected_rate) / expected_rate < 0.20


# ---------------------------------------------------------------------------
# Event processing tests
# ---------------------------------------------------------------------------

def test_process_event_updates_score():
    engine = _make_engine()
    engine.process_event({
        "period": 1, "clock": "5:00",
        "home_score": 12, "away_score": 8,
        "stat_credits": {}, "event_type": "score",
    })
    assert engine.game_state.home_score == 12
    assert engine.game_state.away_score == 8


def test_process_event_updates_observed_stats():
    engine = _make_engine()
    engine.process_event({
        "period": 1, "clock": "5:00",
        "home_score": 6, "away_score": 4,
        "stat_credits": {1001: {"pts": 6, "ast": 1}},
        "event_type": "score",
    })
    s = engine.player_states[1001]
    assert s.observed["pts"] == 6
    assert s.observed["ast"] == 1


def test_process_event_tracks_fouls():
    engine = _make_engine()
    engine.process_event({
        "period": 1, "clock": "7:00",
        "home_score": 2, "away_score": 0,
        "event_type": "foul",
        "primary_player_id": 1002,
        "stat_credits": {},
    })
    assert engine.player_states[1002].fouls == 1


def test_process_event_tracks_subs():
    engine = _make_engine()
    engine.process_event({
        "period": 1, "clock": "6:00",
        "home_score": 4, "away_score": 6,
        "event_type": "substitution",
        "secondary_player_id": 1001,  # going out
        "primary_player_id": 9999,    # coming in (not in our map)
        "stat_credits": {},
    })
    assert engine.player_states[1001].is_active is False


# ---------------------------------------------------------------------------
# Bayesian posterior update tests
# ---------------------------------------------------------------------------

def test_posterior_update_increases_alpha():
    engine = _make_engine()
    alpha_before = engine.player_states[1001].alpha_prior.get("pts", 0)
    engine.process_event({
        "period": 1, "clock": "5:00",
        "home_score": 6, "away_score": 4,
        "stat_credits": {1001: {"pts": 6}},
        "event_type": "score",
    })
    alpha_after = engine.player_states[1001].alpha_post.get("pts", 0)
    assert alpha_after > alpha_before


def test_posterior_update_never_goes_negative():
    engine = _make_engine()
    for _ in range(20):
        engine.process_event({
            "period": 2, "clock": "5:00",
            "home_score": 50, "away_score": 42,
            "stat_credits": {1001: {"pts": 3}},
            "event_type": "score",
        })
    s = engine.player_states[1001]
    assert s.alpha_post.get("pts", 0) > 0
    assert s.beta_post.get("pts", 0) > 0


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

def test_compute_live_projection_returns_tuple():
    engine = _make_engine()
    proj, var = engine.compute_live_projection(1001, "pts")
    assert isinstance(proj, float)
    assert isinstance(var, float)
    assert proj >= 0
    assert var >= 0


def test_compute_live_projection_near_pregame_at_start():
    """At t=0 (no events), projection should be close to pre-game mean."""
    engine = _make_engine()
    proj, _ = engine.compute_live_projection(1001, "pts")
    # Without any elapsed time or events, projection should be close to 22.5
    assert 10 < proj < 35, f"Projection {proj} too far from pre-game 22.5"


def test_compute_live_projection_above_observed_after_events():
    """After a player scores 10 pts with half game left, total proj > 10."""
    engine = _make_engine()
    engine.process_event({
        "period": 2, "clock": "0:00",   # end of 2nd quarter = 20 min elapsed
        "home_score": 40, "away_score": 35,
        "stat_credits": {1001: {"pts": 10}},
        "event_type": "score",
    })
    proj, _ = engine.compute_live_projection(1001, "pts")
    assert proj > 10, f"Projection {proj} should exceed observed 10"


# ---------------------------------------------------------------------------
# Over/under probability tests
# ---------------------------------------------------------------------------

def test_compute_live_over_probability_range():
    engine = _make_engine()
    p_over, p_push = engine.compute_live_over_probability(1001, "pts", 22.5)
    assert 0.001 <= p_over <= 0.999
    assert 0.0 <= p_push <= 1.0
    assert p_over + p_push <= 1.001


def test_already_passed_line_returns_high_probability():
    """If player has already exceeded the line, P(over) should be very high."""
    engine = _make_engine()
    engine.process_event({
        "period": 4, "clock": "1:00",
        "home_score": 80, "away_score": 75,
        "stat_credits": {1001: {"pts": 30}},
        "event_type": "score",
    })
    p_over, _ = engine.compute_live_over_probability(1001, "pts", 22.5)
    assert p_over > 0.90, f"Expected very high P(over) after scoring 30, got {p_over}"


def test_unknown_player_returns_default():
    engine = _make_engine()
    p_over, p_push = engine.compute_live_over_probability(9999, "pts", 15.5)
    assert p_over == 0.5
    assert p_push == 0.0


# ---------------------------------------------------------------------------
# Game total tests
# ---------------------------------------------------------------------------

def test_compute_game_total_live():
    engine = _make_engine()
    engine.process_event({
        "period": 2, "clock": "0:00",
        "home_score": 45, "away_score": 40,
        "stat_credits": {}, "event_type": "score",
    })
    proj, p_over = engine.compute_game_total_live(160.5)
    assert isinstance(proj, float)
    assert 0.001 <= p_over <= 0.999


# ---------------------------------------------------------------------------
# Blowout and foul tests
# ---------------------------------------------------------------------------

def test_blowout_reduces_star_minutes():
    engine = _make_engine()
    s = engine.player_states[1001]   # pregame_minutes = 34 (star)

    # Set blowout score
    engine.game_state.home_score = 90
    engine.game_state.away_score = 65
    engine.game_state.elapsed_minutes = 33.0

    t_rem = GAME_DURATION - 33.0
    m_rem_blowout = engine._estimate_remaining_minutes(s, t_rem)

    # Normal case
    engine.game_state.home_score = 85
    engine.game_state.away_score = 82
    m_rem_close = engine._estimate_remaining_minutes(s, t_rem)

    assert m_rem_blowout < m_rem_close, \
        f"Star should play fewer minutes in blowout: {m_rem_blowout:.1f} vs {m_rem_close:.1f}"


def test_foul_trouble_reduces_minutes():
    engine = _make_engine()
    s = engine.player_states[1002]
    s.fouls = 5

    t_rem = 10.0
    m_rem_fouled = engine._estimate_remaining_minutes(s, t_rem)
    s.fouls = 0
    m_rem_clean  = engine._estimate_remaining_minutes(s, t_rem)

    assert m_rem_fouled < m_rem_clean


# ---------------------------------------------------------------------------
# Clock parsing tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clock, period, expected_min, expected_max", [
    ("9:00", 1, 0.9, 1.1),    # 1 minute elapsed
    ("5:00", 1, 4.9, 5.1),    # 5 minutes elapsed
    ("0:00", 1, 9.9, 10.1),   # end of 1st quarter
    ("10:00", 2, 9.9, 10.1),  # start of 2nd quarter (0 into Q2)
    ("0:00", 4, 39.9, 40.1),  # end of game
])
def test_clock_to_elapsed(clock, period, expected_min, expected_max):
    elapsed = LiveEngine._clock_to_elapsed(clock, period)
    assert expected_min <= elapsed <= expected_max, \
        f"clock={clock}, period={period}: elapsed={elapsed:.2f} not in [{expected_min}, {expected_max}]"


# ---------------------------------------------------------------------------
# Snapshot test
# ---------------------------------------------------------------------------

def test_snapshot_is_serialisable():
    import json
    engine = _make_engine()
    engine.process_event({
        "period": 1, "clock": "7:30",
        "home_score": 5, "away_score": 3,
        "stat_credits": {1001: {"pts": 5}},
        "event_type": "score",
    })
    snap = engine.snapshot()
    # Should be JSON-serialisable
    json.dumps(snap)  # should not raise
    assert "elapsed_minutes" in snap
    assert "players" in snap


# ---------------------------------------------------------------------------
# Batch probabilities test
# ---------------------------------------------------------------------------

def test_get_all_live_probabilities():
    engine = _make_engine()
    props = [(1001, "pts", 22.5), (1001, "ast", 6.0), (1002, "reb", 9.5)]
    results = engine.get_all_live_probabilities(props)
    assert len(results) == 3
    for r in results:
        assert "p_over"  in r
        assert "p_under" in r
        assert abs(r["p_over"] + r["p_push"] + r["p_under"] - 1.0) < 0.01
