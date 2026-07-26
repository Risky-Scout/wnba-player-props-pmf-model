"""STEP 3 fail-closed runtime guards for the pure_forecast production track.

Proves, at the actual PMF-generation entrypoint, that pure generation FAILS when
market_prior_lambda>0, when a CLV/market head is attached, and when a forbidden market feature
is present; that pure delivery cannot serialize a market-anchored config; and that every pure
run persists information_contract=pure_forecast with verifiable provenance (config hash +
ordered feature list).
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.models import pmf_engine
from wnba_props_model.models.pure_model_contract import (
    INFORMATION_CONTRACT,
    MarketLeakageError,
    assert_no_market_head,
    config_sha256,
    enforce_pure_model_config,
    pure_forecast_provenance,
)


class _StatModelWithCLV:
    def __init__(self):
        self.clv_head = object()  # market-derived head attached


class _CleanStatModel:
    clv_head = None


def _min_frames():
    return pd.DataFrame({"game_id": [1]}), pd.DataFrame({"game_id": [1]})


def test_pure_pmf_generation_fails_on_market_prior_lambda():
    wide, long = _min_frames()
    cfg = {"pure_model": True, "market_prior_lambda": 0.1, "stats": ["pts"]}
    with pytest.raises(MarketLeakageError, match="market_prior_lambda"):
        pmf_engine.build_all_pmfs(wide, long, ["player_pts_mean_l5"], object(),
                                  {"pts": _CleanStatModel()}, {}, cfg)


def test_pure_pmf_generation_fails_on_clv_head():
    wide, long = _min_frames()
    cfg = enforce_pure_model_config({"stats": ["pts"]})
    with pytest.raises(MarketLeakageError, match="clv_head|market"):
        pmf_engine.build_all_pmfs(wide, long, ["player_pts_mean_l5"], object(),
                                  {"pts": _StatModelWithCLV()}, {}, cfg)


def test_pure_pmf_generation_fails_on_forbidden_market_feature():
    wide, long = _min_frames()
    cfg = enforce_pure_model_config({"stats": ["pts"]})
    with pytest.raises(MarketLeakageError, match="market"):
        pmf_engine.build_all_pmfs(wide, long, ["player_pts_mean_l5", "market_prob_over_no_vig"],
                                  object(), {"pts": _CleanStatModel()}, {}, cfg)


def test_non_pure_config_still_allowed_to_blend():
    # A non-pure (diagnostic/sensitivity) config is NOT policed by the pure guard at the entry.
    from wnba_props_model.models.pure_model_contract import assert_pure_model_config
    assert_pure_model_config({"pure_model": False, "market_prior_lambda": 0.1})


def test_assert_no_market_head_detects_and_passes():
    assert_no_market_head({"pts": _CleanStatModel(), "reb": _CleanStatModel()})
    with pytest.raises(MarketLeakageError):
        assert_no_market_head({"pts": _StatModelWithCLV()})


def test_pure_delivery_cannot_serialize_market_anchor():
    with pytest.raises(MarketLeakageError, match="market_anchor"):
        pure_forecast_provenance({"pure_model": True, "market_anchor": "market_line"},
                                 ["player_pts_mean_l5"])


def test_pure_forecast_provenance_is_verifiable():
    cfg = enforce_pure_model_config({"stats": ["pts"], "some_knob": 7})
    prov = pure_forecast_provenance(cfg, ["player_pts_mean_l5", "minutes_mean", "p_dnp"])
    assert prov["information_contract"] == INFORMATION_CONTRACT
    assert prov["market_prior_lambda"] == 0.0
    assert prov["market_probability_weight"] == 0.0
    assert prov["clv_head_enabled"] is False
    assert prov["forbidden_market_columns_present"] == []
    assert prov["config_sha256"] == config_sha256(cfg)
    assert len(prov["ordered_feature_list"]) == 3


def test_provenance_rejects_market_feature_list():
    cfg = enforce_pure_model_config({"stats": ["pts"]})
    with pytest.raises(MarketLeakageError):
        pure_forecast_provenance(cfg, ["player_pts_mean_l5", "over_odds"])
