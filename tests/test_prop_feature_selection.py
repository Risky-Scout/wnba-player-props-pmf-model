"""Backward-compatibility + correctness tests for per-prop feature selection.

The single most important guarantee: with NO feature map, per-stat training is byte-for-byte
the current global-feature behavior, so the live pipeline (which ships no map) is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    cfg = {"prop_feature_map": {"pts": ["f3", "f1", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"]},
           "prop_feature_min_cols": 8}
    out = stat_feature_subset(X, "pts", cfg)
    # only mapped columns, preserving X's native column order
    assert list(out.columns) == ["f1", "f3", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"]
    assert len(out) == len(X)


def test_too_aggressive_map_falls_back_to_full():
    X = _X()
    cfg = {"prop_feature_map": {"pts": ["f1", "f2"]}, "prop_feature_min_cols": 8}
    out = stat_feature_subset(X, "pts", cfg)
    assert out is X                                              # below floor -> full set, never starve


def test_unknown_mapped_columns_ignored():
    X = _X()
    cfg = {"prop_feature_map": {"pts": ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "nope1", "nope2"]},
           "prop_feature_min_cols": 8}
    out = stat_feature_subset(X, "pts", cfg)
    assert list(out.columns) == ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]
