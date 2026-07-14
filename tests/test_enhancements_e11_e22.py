"""Smoke tests for Enhancements E11–E22 (new blueprint modules).

These tests verify that every new module:
    1. Imports without errors
    2. Executes its primary API with minimal synthetic data
    3. Returns outputs of the expected shape / type
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# E11: CausalTransferEstimator
# ---------------------------------------------------------------------------

def test_causal_transfer_import():
    from wnba_props_model.models.causal_transfer import CausalTransferEstimator
    est = CausalTransferEstimator(n_folds=2, min_obs_treated=3)
    assert est is not None


def test_causal_transfer_get_fallback():
    from wnba_props_model.models.causal_transfer import CausalTransferEstimator
    est = CausalTransferEstimator()
    rate = est.get_transfer_rate(101, 202, "pts", fallback_rate=0.05)
    assert rate == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# E12: WNBA2Vec
# ---------------------------------------------------------------------------

def test_wnba2vec_import():
    torch = pytest.importorskip("torch", reason="torch is optional (pip install .[neural])")
    from wnba_props_model.models.wnba2vec import WNBA2Vec, EmbeddingFeatureInjector, build_player_id_map
    model = WNBA2Vec(n_players=100, embed_dim=16)
    assert model is not None


def test_wnba2vec_forward():
    torch = pytest.importorskip("torch", reason="torch is optional (pip install .[neural])")
    from wnba_props_model.models.wnba2vec import WNBA2Vec
    model = WNBA2Vec(n_players=100, embed_dim=16, n_outcomes=23, context_dim=5)
    off_ids = torch.randint(1, 100, (2, 5))
    def_ids = torch.randint(1, 100, (2, 5))
    ctx = torch.randn(2, 5)
    logits = model(off_ids, def_ids, ctx)
    assert logits.shape == (2, 23)


def test_embedding_injector_synthetic():
    pytest.importorskip("torch", reason="torch is optional (pip install .[neural])")
    from wnba_props_model.models.wnba2vec import EmbeddingFeatureInjector, build_player_id_map
    df = pd.DataFrame({
        "player_id": [1, 2, 3, 1, 2],
        "team_id":   [10, 10, 20, 10, 20],
        "pts": [15, 12, 10, 18, 11],
        "reb": [5, 6, 8, 4, 7],
        "min": [30, 28, 25, 32, 27],
    })
    pid_map = build_player_id_map(df["player_id"].tolist())
    injector = EmbeddingFeatureInjector(model_path=None, player_id_map=pid_map, n_dims=8)
    out = injector.inject(df)
    assert "player_embed_0" in out.columns
    assert "player_embed_7" in out.columns


# ---------------------------------------------------------------------------
# E13: MultiTaskStatRateModel
# ---------------------------------------------------------------------------

def test_multitask_rates_fit_predict():
    from wnba_props_model.models.multitask_rates import MultiTaskStatRateModel
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 10))
    Y_dict = {
        "pts":      rng.poisson(8, n).astype(float),
        "reb":      rng.poisson(4, n).astype(float),
        "ast":      rng.poisson(3, n).astype(float),
        "fg3m":     rng.poisson(1, n).astype(float),
        "stl":      rng.poisson(1, n).astype(float),
        "blk":      rng.poisson(0.5, n).astype(float),
        "turnover": rng.poisson(1.5, n).astype(float),
    }
    model = MultiTaskStatRateModel(shared_iter=10, private_iter=10)
    model.fit(X, Y_dict)
    preds = model.predict(X, stat="pts")
    assert len(preds) == n
    corr = model.get_correlation_matrix()
    assert corr is not None
    assert corr.shape[0] == corr.shape[1]


# ---------------------------------------------------------------------------
# E14: PossessionSimulator
# ---------------------------------------------------------------------------

def test_possession_simulator_basic():
    from wnba_props_model.models.possession_sim import build_simulator_from_features
    player_feats = {
        1: {"usage": 0.25, "proj_minutes": 30},
        2: {"usage": 0.20, "proj_minutes": 28},
        3: {"usage": 0.18, "proj_minutes": 25},
        4: {"usage": 0.15, "proj_minutes": 22},
        5: {"usage": 0.22, "proj_minutes": 35},
    }
    sim = build_simulator_from_features(player_feats, n_simulations=50, rng_seed=42)
    results = sim.simulate_game()
    assert len(results) == 50
    assert "player_stats" in results[0]
    assert "home_score" in results[0]


def test_possession_sim_pmf():
    from wnba_props_model.models.possession_sim import build_simulator_from_features
    player_feats = {1: {"usage": 0.30, "proj_minutes": 32}}
    sim = build_simulator_from_features(player_feats, n_simulations=100, rng_seed=7)
    result = sim.compute_prop_pmf(1, "pts", line=12.5, n_sims=100)
    assert "p_over" in result
    assert 0.0 <= result["p_over"] <= 1.0


# ---------------------------------------------------------------------------
# E16: LinePredictor
# ---------------------------------------------------------------------------

def test_line_predictor_fit_predict():
    from wnba_props_model.models.line_predictor import LinePredictor, LINE_PREDICTOR_FEATURES
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(0, 1, (n, len(LINE_PREDICTOR_FEATURES)))
    y_closing = rng.normal(20.5, 2.0, n)
    lp = LinePredictor(max_iter=10)
    lp.fit(X, y_closing)
    pred, (ci_low, ci_high) = lp.predict_closing_line(X[0])
    assert ci_low < pred < ci_high
    clv = lp.compute_expected_clv(20.5, X[0])
    assert "expected_clv" in clv


# ---------------------------------------------------------------------------
# E17: CausalInjuryModel
# ---------------------------------------------------------------------------

def test_causal_injury_model():
    from wnba_props_model.models.causal_injury import fit_causal_dnp_model, predict_causal_dnp_probability
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "played":              rng.binomial(1, 0.92, n),
        "age":                 rng.normal(25, 4, n),
        "recent_7day_load":    rng.normal(100, 30, n),
        "rest_days":           rng.integers(0, 5, n).astype(float),
        "is_back_to_back":     rng.binomial(1, 0.15, n).astype(float),
    })
    models = fit_causal_dnp_model(df)
    assert "ipw_model" in models
    feats = np.array([25.0, 110.0, 1.0, 0.0])[:len(models["confounder_cols"])]
    # Pad if needed
    feats = np.pad(feats, (0, max(0, len(models["confounder_cols"]) - len(feats))))
    result = predict_causal_dnp_probability(feats.reshape(1, -1), models)
    assert 0.0 <= result["dnp_probability_causal"] <= 1.0


# ---------------------------------------------------------------------------
# E18: SynergyFeatures
# ---------------------------------------------------------------------------

def test_synergy_from_boxscores():
    from wnba_props_model.models.synergy_features import compute_duo_synergy_from_boxscores, add_synergy_features
    rng = np.random.default_rng(3)
    n = 80
    # 2 players sharing 40 games + 40 solo games each
    games_shared = [f"G{i}" for i in range(40)]
    games_a_only = [f"GA{i}" for i in range(40)]
    df = pd.DataFrame({
        "player_id": [1] * 40 + [1] * 40 + [2] * 40,
        "game_id":   games_shared + games_a_only + games_shared,
        "team_id":   [10] * 120,
        "min":       rng.normal(28, 3, 120),
        "pts":       rng.poisson(15, 120).astype(float),
        "reb":       rng.poisson(5, 120).astype(float),
        "ast":       rng.poisson(3, 120).astype(float),
    })
    synergy = compute_duo_synergy_from_boxscores(df, min_games=10)
    wide = df.drop_duplicates(subset=["player_id"]).copy()
    out = add_synergy_features(wide, synergy, top_n=3)
    assert "player_id" in out.columns


# ---------------------------------------------------------------------------
# E19: RotationModel
# ---------------------------------------------------------------------------

def test_rotation_model_samples():
    from wnba_props_model.models.rotation_model import RotationPattern, estimate_scenario_probs
    pattern = RotationPattern(role="starter")
    scenario_probs = estimate_scenario_probs(pregame_win_prob=0.60, blowout_prob=0.15)
    samples = pattern.sample_conditional_minutes(scenario_probs, n_samples=500)
    assert len(samples) > 0
    assert all(0 <= s <= 40 for s in samples)
    assert np.mean(samples) > 15


def test_rotation_model_bimodal_starter():
    from wnba_props_model.models.rotation_model import RotationPattern
    pattern = RotationPattern(role="starter")
    # With blowout probability the distribution should show clear bimodal tendency
    probs = {"close_game": 0.60, "comfortable_win": 0.30, "blowout": 0.10}
    samples = pattern.sample_conditional_minutes(probs, n_samples=2000)
    mean = np.mean(samples)
    assert 25 <= mean <= 38  # Starter mean should be in this range


# ---------------------------------------------------------------------------
# E20: GameTotalConditioned
# ---------------------------------------------------------------------------

def test_game_total_conditioned():
    from wnba_props_model.models.game_total_conditioned import sample_game_total, condition_player_props_on_total
    totals = sample_game_total(market_line=162.5, n_samples=500)
    assert len(totals) == 500
    assert all(t >= 100 for t in totals)

    projections = [
        {"player_id": 1, "team": "home", "pts_projection": 18.0, "ast_projection": 4.0,
         "reb_projection": 6.0},
        {"player_id": 2, "team": "home", "pts_projection": 14.0, "ast_projection": 3.0,
         "reb_projection": 8.0},
    ]
    conditioned = condition_player_props_on_total(projections, game_total=160.0)
    assert len(conditioned) == 2
    assert "pts_projection_conditioned" in conditioned[0]


# ---------------------------------------------------------------------------
# E21: CalibrationMonitor
# ---------------------------------------------------------------------------

def test_calibration_monitor_update_check():
    from wnba_props_model.models.calibration_monitor import CalibrationMonitor
    mon = CalibrationMonitor(window_size=200, alert_threshold=0.01)
    rng = np.random.default_rng(42)
    # Feed a well-calibrated model (uniform PIT)
    for _ in range(100):
        true_val = int(rng.poisson(10))
        pmf = {k: float(rng.dirichlet(np.ones(25))[k]) for k in range(25)}
        mon.update(pmf, true_val)
    result = mon.check_calibration()
    assert result["status"] in ("ok", "alert", "insufficient_data")
    score = mon.rolling_calibration_score()
    if score is not None:
        assert 0 <= score <= 100


def test_multi_stat_calibration_monitor():
    from wnba_props_model.models.calibration_monitor import MultiStatCalibrationMonitor
    mon = MultiStatCalibrationMonitor()
    # Quick summary should work even with no data
    summary = mon.summary()
    assert "stats" in summary
    assert "pts" in summary["stats"]


# ---------------------------------------------------------------------------
# E22: GameRegimeHMM
# ---------------------------------------------------------------------------

def test_game_hmm_import_and_fallback():
    from wnba_props_model.models.game_hmm import GameRegimeHMM
    hmm = GameRegimeHMM()
    # Should work without training (rule-based fallback)
    recent = [
        {"margin": 5, "clock_remaining_secs": 200, "recent_pts_rate": 1.0,
         "cumulative_pts_rate": 1.0, "cumulative_reb_rate": 0.5, "possession_frac": 0.8,
         "timeout_flag": 0}
    ]
    state_id, state_name, state_prob = hmm.infer_current_state(recent)
    assert state_id in range(4)
    assert state_name in ["normal", "high_scoring", "defensive", "garbage"]
    assert 0.0 <= state_prob <= 1.0


def test_game_hmm_blowout_state():
    from wnba_props_model.models.game_hmm import GameRegimeHMM
    hmm = GameRegimeHMM()
    recent = [
        {"margin": 25, "clock_remaining_secs": 120, "recent_pts_rate": 0.9,
         "cumulative_pts_rate": 1.0, "cumulative_reb_rate": 0.5, "possession_frac": 0.95,
         "timeout_flag": 0}
    ]
    state_id, state_name, _ = hmm.infer_current_state(recent)
    assert state_name == "garbage"


def test_game_hmm_adjust_live_rate():
    from wnba_props_model.models.game_hmm import GameRegimeHMM
    hmm = GameRegimeHMM()
    recent = [
        {"margin": 2, "clock_remaining_secs": 300, "recent_pts_rate": 1.3,
         "cumulative_pts_rate": 1.3, "cumulative_reb_rate": 0.5, "possession_frac": 0.6,
         "timeout_flag": 0}
    ]
    adjusted_rate, state_name, prob = hmm.adjust_live_rate("pts", 0.5, recent)
    assert adjusted_rate > 0
    assert state_name in ["normal", "high_scoring", "defensive", "garbage"]


def test_live_engine_with_hmm():
    """E22: LiveEngine respects HMM regime adjustments."""
    from wnba_props_model.models.live_engine import LiveEngine
    from wnba_props_model.models.game_hmm import GameRegimeHMM
    pregame = {
        101: {
            "team_id": 1,
            "position_group": "wing",
            "minutes": 30.0,
            "pts": {"projection": 18.0, "variance": 20.0},
        }
    }
    hmm = GameRegimeHMM()
    engine = LiveEngine(pregame, hmm=hmm)
    engine.initialize_game({"home_team_id": 1, "away_team_id": 2})
    engine.process_event({
        "home_score": 20, "away_score": 18, "period": 2,
        "clock": "05:00",
        "stat_credits": {101: {"pts": 4}},
        "event_type": "score",
    })
    proj, var = engine.compute_live_projection(101, "pts")
    assert proj > 0
