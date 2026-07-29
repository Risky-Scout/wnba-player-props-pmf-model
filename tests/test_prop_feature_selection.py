"""Correctness tests for the fail-closed per-prop feature policy.

Two guarantees:
1. With NO feature map, per-stat training is byte-for-byte the current global-feature
   behavior, so the live pipeline (which ships no map) is unchanged.
2. An explicit map is honored *exactly* — no minimum-column floor, no silent fallback to
   the full matrix, and an empty set is a base-rate candidate (not the full matrix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.training import stat_feature_subset


def _X():
    return pd.DataFrame({f"f{i}": np.arange(20, dtype=float) + i for i in range(20)})


def test_no_map_returns_identical_matrix():
    X = _X()
    out = stat_feature_subset(X, "pts", {})                      # no prop_feature_map key
    assert out is X                                              # exact same object -> zero behavior change
    out2 = stat_feature_subset(X, "pts", {"prop_feature_map": {}})
    assert out2 is X
    out3 = stat_feature_subset(X, "pts", {"prop_feature_map": {"reb": ["f0"]}})  # stat not in map
    assert out3 is X


def test_map_applies_subset_in_column_order():
    X = _X()
    cfg = {"prop_feature_map": {"pts": ["f3", "f1", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"]}}
    out = stat_feature_subset(X, "pts", cfg)
    # only mapped columns, preserving X's native column order
    assert list(out.columns) == ["f1", "f3", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"]
    assert len(out) == len(X)


def test_one_feature_map_stays_one_feature():
    X = _X()
    out = stat_feature_subset(X, "pts", {"prop_feature_map": {"pts": ["f3"]}})
    assert list(out.columns) == ["f3"]


def test_two_feature_map_no_floor_fallback():
    """The old <8-column floor used to revert to the full matrix; it must not anymore."""
    X = _X()
    out = stat_feature_subset(X, "pts", {"prop_feature_map": {"pts": ["f1", "f2"]}})
    assert list(out.columns) == ["f1", "f2"]
    assert out is not X


def test_empty_map_is_base_rate_not_full_matrix():
    X = _X()
    out = stat_feature_subset(X, "pts", {"prop_feature_map": {"pts": []}})
    assert list(out.columns) == []              # zero-column base-rate frame
    assert len(out) == len(X)
    assert out is not X                         # NOT the full matrix


def test_missing_required_column_raises_fail_closed():
    X = _X()
    cfg = {"prop_feature_map": {"pts": ["f0", "f1", "nope1", "nope2"]}}
    with pytest.raises(ValueError, match="Refusing to silently fall back"):
        stat_feature_subset(X, "pts", cfg)


def test_missing_column_lenient_mode_drops_unknown():
    X = _X()
    cfg = {"prop_feature_map": {"pts": ["f0", "f1", "nope1"]}, "prop_feature_strict": False}
    out = stat_feature_subset(X, "pts", cfg)
    assert list(out.columns) == ["f0", "f1"]
