"""Invariant tests for the explicit, fail-closed per-stat feature policy (directive S3/S21).

Feature-policy invariants proven here:
* a two-feature map remains two features;
* a one-feature map remains one feature;
* an empty map declares a base-rate model (not the full matrix);
* a missing required feature raises;
* certified (explicit) mode never falls back to the full matrix;
* feature ordering and hashes survive save/load;
* pure models reject current-game market fields.
"""
from __future__ import annotations

import pytest

from wnba_props_model.features.feature_provenance import Provenance, classify, pure_compact_eligible
from wnba_props_model.models.prop_feature_policy import (
    FeaturePolicyError,
    FittedFeatureSpec,
    PropFeaturePolicy,
    feature_schema_hash,
)


def _explicit(stat, required, optional=(), contract="pure_forecast"):
    return PropFeaturePolicy(
        stat=stat, mode="explicit", required_columns=tuple(required),
        optional_columns=tuple(optional), information_contract=contract,
        missing_required_policy="raise", feature_set_id=f"{stat}_test_v1",
    )


AVAIL = ["pred_minutes_mean", "fga_per_min_roll5", "fta_per_min_roll5",
         "reb_per_min_roll5", "extra_unused_a", "extra_unused_b"]


def test_two_feature_policy_stays_two_features():
    p = _explicit("pts", ["pred_minutes_mean", "fga_per_min_roll5"])
    cols = p.resolve_columns(AVAIL)
    assert cols == ["pred_minutes_mean", "fga_per_min_roll5"]


def test_one_feature_policy_stays_one_feature():
    p = _explicit("reb", ["reb_per_min_roll5"])
    assert p.resolve_columns(AVAIL) == ["reb_per_min_roll5"]


def test_empty_policy_is_base_rate_model():
    p = PropFeaturePolicy(
        stat="fg3m", mode="base_rate", required_columns=(), optional_columns=(),
        information_contract="pure_forecast", missing_required_policy="raise",
        feature_set_id="fg3m_base_rate_structured_v1",
    )
    assert p.resolve_columns(AVAIL) == []       # intercept, NOT the full matrix
    assert p.certifiable is True


def test_base_rate_with_columns_is_rejected():
    with pytest.raises(FeaturePolicyError):
        PropFeaturePolicy(
            stat="fg3m", mode="base_rate", required_columns=("x",), optional_columns=(),
            information_contract="pure_forecast", missing_required_policy="raise",
            feature_set_id="bad",
        )


def test_missing_required_feature_raises():
    p = _explicit("pts", ["pred_minutes_mean", "does_not_exist"])
    with pytest.raises(FeaturePolicyError, match="missing required"):
        p.resolve_columns(AVAIL)


def test_optional_columns_included_only_when_available():
    p = _explicit("pts", ["pred_minutes_mean"], optional=["fga_per_min_roll5", "not_here"])
    assert p.resolve_columns(AVAIL) == ["pred_minutes_mean", "fga_per_min_roll5"]


def test_certified_explicit_never_falls_back_to_full_matrix():
    p = _explicit("pts", ["pred_minutes_mean", "fga_per_min_roll5"])
    cols = p.resolve_columns(AVAIL)
    assert set(cols) != set(AVAIL)
    assert len(cols) == 2


def test_legacy_full_diagnostic_is_non_certifiable_full_matrix():
    p = PropFeaturePolicy(
        stat="pts", mode="legacy_full_diagnostic", required_columns=(), optional_columns=(),
        information_contract="external_market_anchored", missing_required_policy="raise",
        feature_set_id="pts_full_379_control",
    )
    assert p.resolve_columns(AVAIL) == AVAIL
    assert p.certifiable is False


def test_pure_forecast_rejects_current_game_market_field():
    with pytest.raises(FeaturePolicyError, match="market-derived"):
        _explicit("pts", ["pred_minutes_mean", "game_total"])


def test_pure_forecast_rejects_lagged_market_field():
    with pytest.raises(FeaturePolicyError, match="market-derived"):
        _explicit("pts", ["pred_minutes_mean", "player_market_line_prev"])


def test_market_anchored_contract_allows_market_field():
    p = _explicit("pts", ["pred_minutes_mean", "game_total"], contract="external_market_anchored")
    assert "game_total" in p.resolve_columns(AVAIL + ["game_total"])


def test_feature_schema_hash_is_order_sensitive():
    assert feature_schema_hash(["a", "b"]) != feature_schema_hash(["b", "a"])
    assert feature_schema_hash(["a", "b"]) == feature_schema_hash(["a", "b"])


def test_fitted_spec_roundtrip_and_inference_verification():
    p = _explicit("pts", ["pred_minutes_mean", "fga_per_min_roll5"])
    cols = p.resolve_columns(AVAIL)
    spec = FittedFeatureSpec.build(p, cols, training_cutoff="2026-06-01", training_row_hash="abc")
    # ordering + hash survive save/load
    spec2 = FittedFeatureSpec.from_dict(spec.to_dict())
    assert spec2 == spec
    spec2.verify_inference(cols)  # exact match: OK
    with pytest.raises(FeaturePolicyError):
        spec2.verify_inference(list(reversed(cols)))  # reordered: fails
    with pytest.raises(FeaturePolicyError):
        spec2.verify_inference(cols + ["extra"])       # extra column: fails


# --- provenance --------------------------------------------------------------------

def test_current_game_market_features_classified_external_market():
    for f in ("game_total", "game_spread_home", "implied_team_total",
              "blowout_risk", "predicted_spread_abs", "close_game_indicator"):
        assert classify(f) is Provenance.EXTERNAL_MARKET_CURRENT_GAME
        assert not pure_compact_eligible(f)


def test_lagged_market_features_classified_lagged_market():
    for f in ("player_market_p_over_prev", "player_market_line_prev", "player_line_movement_prev"):
        assert classify(f) is Provenance.EXTERNAL_MARKET_LAGGED
        assert not pure_compact_eligible(f)


def test_pure_lagged_rate_features_are_pure_eligible():
    for f in ("fga_per_min_roll5", "reb_per_min_roll5", "player_usage_pct_ewma10"):
        assert classify(f) is Provenance.PURE_LAGGED
        assert pure_compact_eligible(f)


def test_forward_context_features_not_pure_eligible():
    for f in ("confirmed_starter", "usage_vacated_proxy", "team_top3_scorers_available"):
        assert classify(f) is Provenance.FORWARD_PREGAME_CONTEXT
        assert not pure_compact_eligible(f)
