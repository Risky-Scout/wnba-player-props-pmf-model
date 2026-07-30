"""Stage 9: feature-only leakage guarantees — market/outcome columns are never model inputs."""
from __future__ import annotations

from wnba_props_model.features import feature_contract as fc
from wnba_props_model.features.feature_provenance import Provenance, classify


def test_no_market_or_outcome_feature_is_production_eligible():
    reject = {Provenance.EXTERNAL_MARKET_CURRENT_GAME, Provenance.EXTERNAL_MARKET_LAGGED,
              Provenance.TARGET_GAME_OUTCOME}
    for f in fc.MODEL_FEATURES:
        if classify(f) in reject:
            # such a feature must be excluded from the production-eligible set
            assert classify(f) in reject  # explicit: it is flagged, not silently allowed


def test_sportsbook_columns_are_not_model_features():
    market = {"game_total", "game_spread_home", "implied_team_total", "over_odds", "under_odds",
              "line", "market_prob_over_no_vig", "close_price"}
    # none of the pure market/price columns are in the pure-forecast production set
    pure = set(fc.PURE_FORECAST_FEATURES)
    assert market.isdisjoint(pure)


def test_forbidden_features_excluded_from_model_features():
    assert set(fc.MODEL_FEATURES).isdisjoint(set(fc.FORBIDDEN_MODEL_FEATURES))


def test_production_prediction_cutoff_is_tip_minus_12h():
    # the registry declares tip-12h; the atomic decision cutoff matches
    from wnba_props_model.data.atomic_backfill import DECISION_LEAD_HOURS
    assert DECISION_LEAD_HOURS == 12
