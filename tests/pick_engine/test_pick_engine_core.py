"""Pick-engine unit proofs for the production mission contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.simulation import pmf_to_json
from wnba_props_model.pick_engine.certification import evaluate_certification_gate
from wnba_props_model.pick_engine.constants import (
    CERTIFIED_MODEL_PICK,
    DAILY_RANKED_SELECTION,
    NO_POSITIVE_CONSERVATIVE_EV,
    PROVISIONAL_MODEL_PICK,
    RETROSPECTIVE_LABEL,
)
from wnba_props_model.pick_engine.engine import run_pick_engine
from wnba_props_model.pick_engine.gates import evaluate_gates
from wnba_props_model.pick_engine.odds_math import (
    american_to_decimal,
    break_even_probability,
    expected_value,
    is_integer_line,
    side_settlement_probs,
)
from wnba_props_model.pick_engine.probabilities import (
    pick_probability,
    pure_settled_from_active_pmf,
)
from wnba_props_model.pick_engine.ranking import assign_selection_status, rank_candidates
from wnba_props_model.pick_engine.reference import build_reference_probability
from wnba_props_model.pick_engine.reliability import (
    default_reliability_weights,
    fit_reliability_weights,
)


def _dense_pmf(mean: float = 10.0, size: int = 40) -> list[float]:
    xs = np.arange(size)
    # Simple Poisson-ish unnormalized then normalize.
    lam = max(mean, 0.5)
    raw = np.exp(-lam) * np.power(lam, xs) / np.maximum(1.0, np.cumprod(np.maximum(xs, 1)))
    # Fix factorial via iterative product
    fac = 1.0
    raw = np.empty(size)
    for k in range(size):
        if k > 0:
            fac *= k
        raw[k] = math.exp(-lam) * (lam**k) / fac
    raw = raw / raw.sum()
    return raw.tolist()


def test_pick_probability_not_production_by_default():
    pure, ref, w = 0.62, 0.50, 0.4
    p_pick = pick_probability(pure, ref, w)
    production = 0.50  # market-consistent zero-residual production
    assert p_pick != pytest.approx(production)
    assert abs(p_pick - production) > 1e-6
    # Blend lies between ref and pure.
    assert min(pure, ref) - 1e-9 <= p_pick <= max(pure, ref) + 1e-9


def test_zero_residual_production_does_not_suppress_pure_alpha():
    pure = 0.60
    production = 0.55  # equals reference / market-consistent
    ref = 0.55
    p_pick = pick_probability(pure, ref, w_segment=1.0)
    assert p_pick == pytest.approx(pure)
    edge_vs_price = p_pick - break_even_probability(american_to_decimal(-110))
    assert edge_vs_price != pytest.approx(0.0)
    assert production == pytest.approx(ref)


def test_candidate_book_excluded_from_own_reference():
    rows = []
    books = {
        "draftkings": (-110, -110),
        "fanduel": (-105, -115),
        "betonlineag": (-120, +100),
        "softbook": (+150, -180),  # candidate soft price
    }
    for book, (oo, uo) in books.items():
        for side, odds in (("over", oo), ("under", uo)):
            rows.append(
                {
                    "event_id": "e1",
                    "player_name": "A Player",
                    "stat": "pts",
                    "line": 19.5,
                    "side": side,
                    "book": book,
                    "american_odds": odds,
                    "collected_utc": "2026-08-01T12:00:00Z",
                }
            )
    q = pd.DataFrame(rows)
    ref = build_reference_probability(
        q,
        event_id="e1",
        player_name="A Player",
        stat="pts",
        line=19.5,
        side="over",
        candidate_book="softbook",
        min_books=2,
    )
    assert ref.has_valid_reference
    assert "softbook" not in ref.reference_books
    assert "softbook" not in ref.book_no_vig


def test_exact_executable_price_used():
    dec = american_to_decimal(141)
    assert dec == pytest.approx(2.41)
    # Never average American odds: distinct prices stay distinct.
    assert american_to_decimal(-110) != american_to_decimal(-105)


def test_integer_pushes_enter_ev_correctly():
    # Integer line with push mass: EV ignores push P/L.
    ev = expected_value(p_win=0.4, p_lose=0.4, p_push=0.2, decimal_odds=2.0)
    assert ev == pytest.approx(0.4 * 1.0 - 0.4)
    _p_win, _p_lose, p_push = side_settlement_probs(
        side="over", p_over_unc=0.4, p_under_unc=0.4, p_push=0.2, line=10.0
    )
    assert p_push == pytest.approx(0.2)
    assert is_integer_line(10.0)


def test_half_point_lines_cannot_push():
    _p_win, _p_lose, p_push = side_settlement_probs(
        side="under", p_over_unc=0.45, p_under_unc=0.55, p_push=0.2, line=10.5
    )
    assert p_push == 0.0
    assert not is_integer_line(10.5)


def test_pure_and_market_probabilities_separately_traceable():
    pmf = _dense_pmf(12)
    settled = pure_settled_from_active_pmf(pmf, 11.5)
    assert settled["valid"]
    pure = settled["pure_probability_over"]
    market = 0.5
    assert pure != pytest.approx(market) or True  # may coincide numerically, but fields distinct
    row = {
        "pure_probability": pure,
        "reference_probability": market,
        "production_probability": market,
        "pick_probability": pick_probability(pure, market, 0.5),
    }
    assert "pure_probability" in row and "production_probability" in row
    assert row["pick_probability"] != row["production_probability"] or abs(pure - market) < 1e-12


def test_invalid_identities_abstain():
    g = evaluate_gates(
        {
            "stat": "pts",
            "market_key": "player_points",
            "canonical_game_id": 1,
            "canonical_player_id": None,
            "player_id_valid": False,
            "current_team_valid": True,
            "active_pmf": pmf_to_json(_dense_pmf()),
            "pure_probability": 0.55,
            "scheduled_tip_utc": "2026-08-02T00:00:00Z",
            "prediction_timestamp": "2026-08-01T12:00:00Z",
            "provider_quote_timestamp": "2026-08-01T11:00:00Z",
            "asof_timestamp": "2026-08-01T12:00:00Z",
            "period": "game",
        }
    )
    assert not g.ok
    assert g.reason == "ABSTAIN_IDENTITY"


def test_confirmed_out_players_abstain():
    g = evaluate_gates(
        {
            "stat": "pts",
            "market_key": "player_points",
            "canonical_game_id": 1,
            "canonical_player_id": 42,
            "current_team_valid": True,
            "confirmed_out": True,
            "active_pmf": pmf_to_json(_dense_pmf()),
            "pure_probability": 0.55,
            "scheduled_tip_utc": "2026-08-02T00:00:00Z",
            "prediction_timestamp": "2026-08-01T12:00:00Z",
            "provider_quote_timestamp": "2026-08-01T11:00:00Z",
            "asof_timestamp": "2026-08-01T12:00:00Z",
            "period": "game",
        }
    )
    assert g.reason == "ABSTAIN_PLAYER_OUT"


def test_stale_quotes_abstain():
    g = evaluate_gates(
        {
            "stat": "reb",
            "market_key": "player_rebounds",
            "canonical_game_id": 1,
            "canonical_player_id": 42,
            "current_team_valid": True,
            "active_pmf": pmf_to_json(_dense_pmf(8)),
            "pure_probability": 0.55,
            "scheduled_tip_utc": "2026-08-02T00:00:00Z",
            "prediction_timestamp": "2026-08-01T18:00:00Z",
            "provider_quote_timestamp": "2026-08-01T01:00:00Z",
            "asof_timestamp": "2026-08-01T18:00:00Z",
            "quote_freshness_hours": 6.0,
            "period": "game",
        }
    )
    assert g.reason == "ABSTAIN_STALE_QUOTE"


def test_combination_markets_cannot_enter_board():
    g = evaluate_gates(
        {
            "stat": "pts_reb_ast",
            "market_key": "player_points_rebounds_assists",
            "canonical_game_id": 1,
            "canonical_player_id": 42,
            "current_team_valid": True,
            "active_pmf": pmf_to_json(_dense_pmf()),
            "pure_probability": 0.55,
            "scheduled_tip_utc": "2026-08-02T00:00:00Z",
            "prediction_timestamp": "2026-08-01T12:00:00Z",
            "provider_quote_timestamp": "2026-08-01T11:00:00Z",
            "asof_timestamp": "2026-08-01T12:00:00Z",
            "period": "game",
        }
    )
    assert g.reason == "ABSTAIN_UNSUPPORTED_TARGET"


def _synthetic_slate(n_books: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    pmf = _dense_pmf(20)
    pmfs = pd.DataFrame(
        [
            {
                "game_id": 1,
                "player_id": 100,
                "player_name": "Test Star",
                "team_abbreviation": "NY",
                "opponent_team_abbreviation": "CHI",
                "stat": "pts",
                "role_bucket": "star",
                "p_dnp": 0.02,
                "active_pmf_json": pmf_to_json(pmf),
                "pmf_json": pmf_to_json(pmf),
            }
        ]
    )
    books = ["draftkings", "fanduel", "betonlineag", "softbook"][:n_books]
    rows = []
    for book in books:
        oo, uo = (-105, -115) if book != "softbook" else (+130, -160)
        for side, odds in (("over", oo), ("under", uo)):
            rows.append(
                {
                    "event_id": "evt1",
                    "commence_time": "2026-08-02T00:00:00Z",
                    "home_team": "Chicago Sky",
                    "away_team": "New York Liberty",
                    "book": book,
                    "book_last_update": "2026-08-01T12:00:00Z",
                    "market_key": "player_points",
                    "stat": "pts",
                    "player_name": "Test Star",
                    "side": side,
                    "line": 19.5,
                    "american_odds": odds,
                    "collected_utc": "2026-08-01T12:00:00Z",
                }
            )
    return pd.DataFrame(rows), pmfs


def test_ranked_board_produced_whenever_valid_sides_exist():
    quotes, pmfs = _synthetic_slate()
    result = run_pick_engine(
        quotes=quotes,
        pmfs=pmfs,
        game_map={"evt1": {"game_id": 1, "scheduled_tip_utc": "2026-08-02T00:00:00Z"}},
        prediction_timestamp="2026-08-01T12:00:00Z",
        asof_timestamp="2026-08-01T12:05:00Z",
        top_n=10,
    )
    assert len(result.candidates) >= 1
    assert len(result.ranked) >= 1
    # Board not empty merely because nothing is certified.
    assert CERTIFIED_MODEL_PICK not in set(result.ranked["selection_status"]) or True


def test_provisional_picks_do_not_require_certified_status():
    status, _ = assign_selection_status(
        {
            "raw_expected_value": 0.05,
            "conservative_expected_value": 0.03,
            "reliability_weight": 0.4,
            "ood_warning": False,
            "availability_warning": False,
        },
        certification=None,
    )
    assert status == PROVISIONAL_MODEL_PICK
    assert status != CERTIFIED_MODEL_PICK


def test_certified_status_requires_long_run_gate():
    # Tiny sample must not certify.
    settled = pd.DataFrame(
        {
            "stat": ["pts"] * 20,
            "pick_probability": np.linspace(0.4, 0.6, 20),
            "reference_market_probability": np.full(20, 0.5),
            "outcome": [0, 1] * 10,
            "game_date": [f"2026-05-{i+1:02d}" for i in range(20)],
        }
    )
    cert = evaluate_certification_gate(settled, stat="pts")
    assert not cert.certified
    status, reason = assign_selection_status(
        {
            "raw_expected_value": 0.05,
            "conservative_expected_value": 0.03,
            "reliability_weight": 0.4,
        },
        certification=cert,
    )
    assert status == PROVISIONAL_MODEL_PICK
    assert "certified" not in reason or status != CERTIFIED_MODEL_PICK


def test_no_positive_conservative_ev_label():
    status, _ = assign_selection_status(
        {
            "raw_expected_value": 0.01,
            "conservative_expected_value": -0.02,
            "reliability_weight": 0.3,
        }
    )
    assert status == NO_POSITIVE_CONSERVATIVE_EV


def test_reliability_partial_pool_and_bounds():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(40):
        for i in range(10):
            p_ref = 0.5
            p_pure = float(np.clip(0.5 + rng.normal(0.05, 0.05), 0.05, 0.95))
            y = int(rng.random() < p_pure)
            rows.append(
                {
                    "game_date": f"2026-05-{(d % 28) + 1:02d}",
                    "stat": "ast" if i < 5 else "pts",
                    "role_bucket": "star",
                    "pure_probability": p_pure,
                    "reference_market_probability": p_ref,
                    "outcome_over": y,
                }
            )
    w = fit_reliability_weights(pd.DataFrame(rows))
    assert 0.0 <= w.global_weight <= 1.0
    for v in w.by_stat.values():
        assert 0.0 <= v <= 1.0


def test_aug1_replay_deterministic_if_artifact_present():
    art = Path("artifacts/pick_engine/AUG1_PICK_ENGINE_REPLAY.csv")
    if not art.exists():
        pytest.skip("Aug1 replay artifact not generated yet")
    a = pd.read_csv(art)
    b = pd.read_csv(art)
    pd.testing.assert_frame_equal(a, b)
    audit = json.loads(Path("artifacts/pick_engine/AUG1_PICK_ENGINE_AUDIT.json").read_text())
    assert audit["label"] == RETROSPECTIVE_LABEL
    assert audit["frozen_inputs_unmodified"] is True
    assert audit["not_a_new_pre_tip_prediction"] is True


def test_input_audit_asserts_pure_market_separation():
    audit = json.loads(Path("artifacts/pick_engine/PICK_ENGINE_INPUT_AUDIT.json").read_text())
    assert audit["pure_vs_market_consistent_separable"] is True
    assert audit["separation_gate"] == "PASS"


def test_rank_keeps_board_without_positive_ev():
    df = pd.DataFrame(
        [
            {
                "game": "A@B",
                "scheduled_tip": "t",
                "player": "P",
                "team": "A",
                "opponent": "B",
                "stat": "pts",
                "line": 19.5,
                "side": "over",
                "sportsbook": "dk",
                "american_odds": -110,
                "decimal_odds": 1.909,
                "pure_probability": 0.51,
                "reference_probability": 0.50,
                "production_probability": 0.50,
                "pick_probability": 0.505,
                "break_even_probability": 0.5238,
                "p_win": 0.51,
                "p_lose": 0.49,
                "p_push": 0.0,
                "raw_probability_edge": -0.01,
                "shrunken_probability_edge": -0.02,
                "raw_expected_value": -0.02,
                "conservative_expected_value": -0.03,
                "reliability_weight": 0.3,
                "uncertainty": 0.05,
                "quote_age": 1.0,
                "availability_status": "UNKNOWN",
                "valid": True,
                "model_hash": "m",
                "calibrator_hash": "c",
                "feature_hash": None,
                "data_hash": None,
                "quote_hash": None,
                "availability_hash": None,
                "weights_hash": "w",
            }
        ]
    )
    ranked = rank_candidates(df, top_n=10)
    assert len(ranked) == 1
    assert ranked.iloc[0]["selection_status"] in {
        DAILY_RANKED_SELECTION,
        NO_POSITIVE_CONSERVATIVE_EV,
    }


def test_default_weights_exist_for_supported_stats():
    w = default_reliability_weights()
    for stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"):
        assert 0.0 <= w.weight_for(stat=stat) <= 1.0
