"""Every v1 per-stat policy must resolve against the real feature manifest, keep pure
candidates market-free, and honor the fail-closed contract."""
from __future__ import annotations

import pytest

from wnba_props_model.features.feature_contract import MODEL_FEATURES
from wnba_props_model.features.feature_provenance import pure_compact_eligible
from wnba_props_model.models import prop_feature_policies_v1 as pol
from wnba_props_model.models.prop_feature_policy import FeaturePolicyError

MANIFEST = list(MODEL_FEATURES)


@pytest.mark.parametrize("stat", pol.STATS)
def test_compact_core_required_columns_exist_in_manifest(stat):
    p = pol.compact_causal_core(stat)
    missing = [c for c in p.required_columns if c not in MANIFEST]
    assert not missing, f"{stat} required columns absent from manifest: {missing}"


@pytest.mark.parametrize("stat", pol.STATS)
def test_compact_core_is_pure_and_resolves_without_full_fallback(stat):
    p = pol.compact_causal_core(stat)
    cols = p.resolve_columns(MANIFEST)
    assert 0 < len(cols) < len(MANIFEST)          # never the full matrix
    for c in cols:
        assert pure_compact_eligible(c), f"{stat} pure core admitted non-pure feature {c}"


@pytest.mark.parametrize("stat", pol.STATS)
def test_base_rate_yields_no_columns(stat):
    assert pol.base_rate(stat).resolve_columns(MANIFEST) == []


@pytest.mark.parametrize("stat", pol.STATS)
def test_full_control_is_non_certifiable(stat):
    p = pol.full_control(stat)
    assert p.certifiable is False
    assert p.resolve_columns(MANIFEST) == MANIFEST


def test_external_market_context_is_market_anchored_and_allowed():
    p = pol.external_market_context("pts")
    cols = p.resolve_columns(MANIFEST)
    assert "game_total" in cols          # market feature allowed on the anchored contract
    assert p.information_contract == "external_market_anchored"


def test_internal_game_context_excludes_vegas_total():
    p = pol.internal_game_context("pts")
    cols = p.resolve_columns(MANIFEST)
    assert "game_total" not in cols and "game_spread_home" not in cols


def test_pure_core_cannot_contain_market_feature_by_construction():
    # sanity: injecting a market feature into a pure policy raises
    from wnba_props_model.models.prop_feature_policy import PropFeaturePolicy
    with pytest.raises(FeaturePolicyError):
        PropFeaturePolicy(
            stat="pts", mode="explicit", required_columns=("pred_minutes_mean", "game_total"),
            optional_columns=(), information_contract="pure_forecast",
            missing_required_policy="raise", feature_set_id="bad")
