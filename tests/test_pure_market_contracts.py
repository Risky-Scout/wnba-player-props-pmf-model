"""W0.4: pure_forecast vs market_anchored information contracts.

The pure forecast must be provably free of same-game sportsbook features and lagged
prior-game market features; the market-anchored contract may include them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.features import feature_contract as fc


SAME_GAME = ["implied_team_total", "game_total", "game_spread_home",
             "blowout_risk", "predicted_spread_abs", "close_game_indicator"]
LAGGED = ["player_market_p_over_prev", "player_market_line_prev", "player_line_movement_prev"]


def test_pure_excludes_all_same_game_market_features():
    pure = set(fc.pure_forecast_features())
    for f in SAME_GAME + LAGGED:
        assert f not in pure, f"pure forecast must not contain market feature {f}"


def test_pure_keeps_model_derived_game_script():
    # Model net-rating game-script features are NOT Vegas-derived -> allowed in pure.
    pure = set(fc.pure_forecast_features())
    for f in ["pregame_win_probability", "blowout_probability", "close_game_probability"]:
        assert f in pure, f"model-derived game-script feature {f} should remain in pure"


def test_market_anchored_is_superset_of_pure():
    pure = set(fc.pure_forecast_features())
    anchored = set(fc.market_anchored_features())
    assert pure < anchored
    # The difference is exactly the market-derived features.
    assert (anchored - pure) == (set(SAME_GAME) | set(LAGGED))


def test_assert_pure_forecast_rejects_market_features():
    with pytest.raises(ValueError):
        fc.assert_pure_forecast(fc.pure_forecast_features() + ["game_total"])
    with pytest.raises(ValueError):
        fc.assert_pure_forecast(pd.DataFrame(columns=fc.pure_forecast_features() + ["player_market_line_prev"]))


def test_assert_pure_forecast_accepts_clean_list():
    fc.assert_pure_forecast(fc.pure_forecast_features())  # must not raise


def test_pure_forecast_passes_no_forbidden_features():
    # Pure list must also satisfy the existing leakage gate.
    fc.assert_no_forbidden_features(fc.pure_forecast_features())
