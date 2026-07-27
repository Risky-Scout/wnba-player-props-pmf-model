"""PHASE 0 guards: pure-model information contract + repaired minutes subsystem.

Locks in that:
  * a pure model/candidate cannot carry any market weight/nudge or market-derived feature;
  * the market_prior_lambda blend and CLV head are hard-disabled in pure mode;
  * the inconsistent 42-min IQR cap is gone;
  * conditional-minute regressors train on appearances only (availability not double-counted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.minutes_model import MinutesModel
from wnba_props_model.models.pure_model_contract import (
    FORBIDDEN_MARKET_INPUT_FIELDS,
    MAX_SENSITIVITY_MARKET_WEIGHT,
    NOT_PURE_CANDIDATES,
    MarketLeakageError,
    assert_pure_feature_columns,
    assert_pure_model_config,
    enforce_pure_model_config,
    forbidden_market_columns,
    is_forbidden_market_field,
    is_pure_model,
)


# ---- market-input exclusion -------------------------------------------------
@pytest.mark.parametrize("field", [
    "market_prob_over_no_vig", "over_odds", "under_odds", "closing_line", "line_movement",
    "clv", "game_total", "game_spread_home", "implied_team_total", "predicted_spread",
    "player_market_p_over_prev", "market_line", "prop_line", "line", "consensus_odds",
])
def test_forbidden_market_fields_are_rejected(field):
    assert is_forbidden_market_field(field), field


@pytest.mark.parametrize("field", [
    "player_pts_mean_l5", "minutes_mean", "p_dnp", "opp_pace_ewma10", "player_usage_pct",
    "role_bucket", "stat_variance",
])
def test_pure_model_features_are_allowed(field):
    assert not is_forbidden_market_field(field), field


def test_assert_pure_feature_columns_fails_closed():
    cols = ["player_pts_mean_l5", "minutes_mean", "market_prob_over_no_vig"]
    assert forbidden_market_columns(cols) == ["market_prob_over_no_vig"]
    with pytest.raises(MarketLeakageError, match="market"):
        assert_pure_feature_columns(cols, context="unit")
    # clean frame passes
    assert_pure_feature_columns(["player_pts_mean_l5", "minutes_mean", "p_dnp"])


def test_all_declared_forbidden_fields_detected():
    for f in FORBIDDEN_MARKET_INPUT_FIELDS:
        assert is_forbidden_market_field(f), f


# ---- pure config enforcement ------------------------------------------------
def test_enforce_pure_model_config_zeroes_market_weights():
    cfg = {"market_prior_lambda": 0.1, "market_prior_lambda_display": 0.15,
           "market_probability_weight": 0.3, "use_clv_head": True,
           "use_live_calibrators": True, "stats": ["pts"]}
    pure = enforce_pure_model_config(cfg)
    assert is_pure_model(pure)
    assert pure["market_prior_lambda"] == 0.0
    assert pure["market_prior_lambda_display"] == 0.0
    assert pure["market_probability_weight"] == 0.0
    assert pure["use_clv_head"] is False
    assert pure["use_live_calibrators"] is False
    assert pure["stats"] == ["pts"]  # non-market keys preserved
    assert_pure_model_config(pure)  # normalized config must pass the guard


def test_assert_pure_model_config_detects_leak():
    with pytest.raises(MarketLeakageError):
        assert_pure_model_config({"pure_model": True, "market_prior_lambda": 0.1})
    with pytest.raises(MarketLeakageError):
        assert_pure_model_config({"pure_model": True, "use_clv_head": True})
    # a non-pure cfg is not policed by this guard
    assert_pure_model_config({"pure_model": False, "market_prior_lambda": 0.1})


def test_market_blends_marked_not_pure():
    for c in ("C4_blend", "C5_role_blend", "C6_market_residual"):
        assert c in NOT_PURE_CANDIDATES
    assert MAX_SENSITIVITY_MARKET_WEIGHT == 0.15


def test_shipped_stage4_config_is_pure():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("config/model/stage4_baseline.yaml").read_text())
    assert cfg.get("pure_model") is True
    assert float(cfg.get("market_prior_lambda", 0.0)) == 0.0
    assert float(cfg.get("market_prior_lambda_display", 0.0)) == 0.0
    assert_pure_model_config(cfg, context="stage4_baseline.yaml")


# ---- repaired minutes subsystem --------------------------------------------
def _synthetic_minutes_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f_role": rng.normal(0, 1, n),
        "f_form": rng.normal(0, 1, n),
        "f_pace": rng.normal(0, 1, n),
    })
    # ~25% DNP; appearances play 20-40 min driven by f_role, DNP rows have 0 minutes.
    did_play = (rng.random(n) > 0.25).astype(int)
    cond_minutes = np.clip(28 + 6 * X["f_role"].to_numpy() + rng.normal(0, 3, n), 5, 44)
    minutes = np.where(did_play == 1, cond_minutes, 0.0)
    meta = pd.DataFrame({
        "did_play": did_play,
        "projected_minutes_bucket": "mid",
        "role_uncertainty_bucket": "stable",
    })
    return X, pd.Series(minutes), meta


def test_minutes_regressor_trains_on_appearances_only():
    X, y, meta = _synthetic_minutes_frame()
    cfg = {"min_minutes_sigma": 2.0, "minutes_clip_max": 48.0,
           "hgb_regressor": {"max_iter": 40}}
    appear = MinutesModel(cfg).fit(X, y, meta)
    allrows = MinutesModel({**cfg, "train_minutes_on_appearances_only": False}).fit(X, y, meta)
    assert appear.get_training_summary()["trained_minutes_on_appearances_only"] is True
    mean_appear = appear.predict(X, meta)[0].mean()
    mean_allrows = allrows.predict(X, meta)[0].mean()
    # Including DNP zeros deflates the conditional minute mean; appearance-only is higher.
    assert mean_appear > mean_allrows
    # Conditional mean should sit near the true appearance mean (~28), not the DNP-deflated one.
    assert mean_appear > 24.0


def test_no_42_minute_iqr_cap():
    # A workhorse starter whose q75 should exceed 42 must not have sigma crushed by a 42 cap.
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"f_role": np.full(n, 3.0), "f_form": rng.normal(0, 0.2, n)})
    did_play = np.ones(n, dtype=int)
    minutes = np.clip(rng.normal(40, 5, n), 30, 48)
    meta = pd.DataFrame({"did_play": did_play, "projected_minutes_bucket": "high",
                         "role_uncertainty_bucket": "stable"})
    cfg = {"min_minutes_sigma": 1.0, "minutes_clip_max": 48.0, "hgb_regressor": {"max_iter": 60}}
    m = MinutesModel(cfg).fit(X, pd.Series(minutes), meta)
    q = m.predict_quantiles(X, meta)
    # q75 (index 3) is allowed to exceed 42 (no hard 42 cap in the quantile path).
    assert q[:, 3].max() > 42.0 or minutes.max() <= 42.0
    # predict() sigma is derived from the same uncapped clip so it is not artificially 0.
    _, sigma, _ = m.predict(X, meta)
    assert np.all(sigma >= cfg["min_minutes_sigma"])


def test_quantile_crossing_is_repaired():
    X, y, meta = _synthetic_minutes_frame(n=500, seed=3)
    cfg = {"min_minutes_sigma": 1.0, "minutes_clip_max": 48.0, "hgb_regressor": {"max_iter": 30}}
    m = MinutesModel(cfg).fit(X, y, meta)
    q = m.predict_quantiles(X, meta)
    # q10<=q25<=q50<=q75<=q90 for every row (no crossing after repair).
    assert np.all(np.diff(q, axis=1) >= -1e-9)
    # derived sigma is therefore always well-defined and >= the floor.
    _, sigma, _ = m.predict(X, meta)
    assert np.all(sigma >= cfg["min_minutes_sigma"])
