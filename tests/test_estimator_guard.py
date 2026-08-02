"""Strengthened Stage 9 (Sections 6-7): fail-closed estimator guard + adversarial leakage tests.

No large parquets required (synthetic frames) so this runs in CI where data is gitignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.estimator_guard import (
    EstimatorGuardError,
    assert_no_forbidden_names,
    assert_no_identifiers,
    guard_estimator_frame,
)
from wnba_props_model.models.prop_feature_policy import feature_schema_hash

APPROVED = ["player_pts_mean_l5", "player_reb_mean_l10", "player_minutes_mean_l5",
            "opp_pts_allowed_roll5", "rest_days", "is_home"]
HASH = feature_schema_hash(APPROVED)


def _valid(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=n) for c in APPROVED})


def test_valid_approved_frame_passes():
    X = guard_estimator_frame(_valid(), APPROVED, HASH)
    assert X.shape == (20, len(APPROVED))


# --- Section 7: every injected forbidden/leaky field must fail closed ---
@pytest.mark.parametrize("col,val", [
    ("actual_pts", 10.0), ("actual_minutes", 25.0), ("did_play", 1),
    ("settlement_status", 1), ("over_odds", -110), ("under_price", -110),
    ("line", 15.5), ("market_prob_over", 0.5), ("closing_line", 16.0),
    ("final_score", 80), ("future_usage", 0.2),
])
def test_injected_forbidden_or_target_field_fails_closed(col, val):
    bad = _valid().assign(**{col: val})
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(bad, APPROVED, HASH)


def test_injected_current_game_boxscore_fails_closed():
    bad = _valid().assign(actual_reb=5)   # current-game box score
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(bad, APPROVED, HASH)


def test_unexpected_column_fails_closed():
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(_valid().assign(surprise_feature=1.0), APPROVED, HASH)


def test_missing_approved_feature_fails_closed():
    df = _valid().drop(columns=["is_home"])
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(df, APPROVED, HASH)


def test_column_order_mismatch_fails_closed():
    df = _valid()[list(reversed(APPROVED))]
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(df, APPROVED, HASH)


def test_duplicate_column_fails_closed():
    df = _valid()
    df = pd.concat([df, df[[APPROVED[0]]]], axis=1)
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(df, APPROVED, HASH)


def test_schema_hash_mismatch_fails_closed():
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(_valid(), APPROVED, "deadbeefdeadbeef")


def test_identifier_in_estimator_fails_closed():
    with pytest.raises(EstimatorGuardError):
        assert_no_identifiers(APPROVED + ["game_id"])


def test_nonnumeric_value_fails_closed():
    df = _valid().assign(**{APPROVED[0]: ["x"] * 20})
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(df[APPROVED], APPROVED, HASH)


def test_infinite_value_fails_closed():
    df = _valid()
    df.loc[0, APPROVED[0]] = np.inf
    with pytest.raises(EstimatorGuardError):
        guard_estimator_frame(df, APPROVED, HASH)


def test_forbidden_name_secondary_alarm():
    with pytest.raises(EstimatorGuardError):
        assert_no_forbidden_names(APPROVED + ["market_line_x"])
