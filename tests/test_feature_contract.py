import pytest

from wnba_props_model.features.feature_contract import assert_no_forbidden_features


def test_forbidden_market_feature_fails():
    with pytest.raises(ValueError):
        assert_no_forbidden_features(["market_line"])
