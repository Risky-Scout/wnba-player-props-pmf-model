"""Availability model tests for Opportunity V2 (section 15)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.availability_model import AvailabilityModelV2


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    active_rate = rng.uniform(0.3, 1.0, n)
    dnp_streak = rng.integers(0, 5, n)
    # higher active_rate -> more likely to play
    p = np.clip(active_rate - 0.05 * dnp_streak, 0.02, 0.99)
    did_play = rng.binomial(1, p).astype(bool)
    X = pd.DataFrame({
        "player_active_rate_ewma": active_rate,
        "player_dnp_streak_prior": dnp_streak,
        "player_days_since_last_game": rng.integers(1, 6, n),
    })
    return X, pd.Series(did_play)


def test_predictions_within_bounds():
    X, y = _data()
    m = AvailabilityModelV2().fit(X, y)
    p = m.predict_active_probability(X)
    assert p.min() >= 0.001 and p.max() <= 0.999


def test_records_feature_contract_and_hash():
    X, y = _data()
    m = AvailabilityModelV2().fit(X, y)
    assert set(m.feature_names_) == set(X.columns)
    assert m.model_hash and len(m.model_hash) == 64


def test_missing_inference_feature_raises():
    X, y = _data()
    m = AvailabilityModelV2().fit(X, y)
    with pytest.raises(ValueError):
        m.predict_active_probability(X.drop(columns=["player_dnp_streak_prior"]))


def test_status_prior_fallback_diagnostic():
    fb = AvailabilityModelV2.status_prior_fallback(pd.Series(["out", "available", "unknown"]))
    assert fb[0] <= 0.01 and fb[1] >= 0.99


def test_higher_active_rate_increases_probability():
    X, y = _data()
    m = AvailabilityModelV2().fit(X, y)
    lo = m.predict_active_probability(pd.DataFrame([{"player_active_rate_ewma": 0.35,
        "player_dnp_streak_prior": 3, "player_days_since_last_game": 2}]))
    hi = m.predict_active_probability(pd.DataFrame([{"player_active_rate_ewma": 0.98,
        "player_dnp_streak_prior": 0, "player_days_since_last_game": 2}]))
    assert hi[0] > lo[0]
