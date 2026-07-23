"""Tests for the per-prop feature-optimization pack (candidate map + market-superiority evaluator).

Guards: every candidate-map feature is a real, non-forbidden MODEL_FEATURE; the map is
consumed by the existing backward-compatible training hook; the evaluator's metrics and the
scored-rows P(over) bridge are correct.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DIRECT = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_map_features_all_valid_and_not_forbidden():
    from wnba_props_model.features.feature_contract import MODEL_FEATURES, FORBIDDEN_MODEL_FEATURES
    valid, forb = set(MODEL_FEATURES), set(FORBIDDEN_MODEL_FEATURES)
    cand = json.loads((REPO / "config/prop_feature_map_candidate_v1.json").read_text())
    assert set(cand) >= set(DIRECT)
    for stat, feats in cand.items():
        assert len(feats) == len(set(feats)), f"{stat} has duplicate features"
        missing = [f for f in feats if f not in valid]
        forbidden = [f for f in feats if f in forb]
        assert not missing, f"{stat} has non-contract features: {missing}"
        assert not forbidden, f"{stat} references forbidden features: {forbidden}"


def test_candidate_map_is_a_strict_subset_per_prop():
    from wnba_props_model.features.feature_contract import MODEL_FEATURES
    cand = json.loads((REPO / "config/prop_feature_map_candidate_v1.json").read_text())
    for stat, feats in cand.items():
        assert 8 <= len(feats) < len(MODEL_FEATURES), f"{stat} subset size implausible"


def test_map_consumed_by_training_hook():
    import pandas as pd
    from wnba_props_model.models.training import stat_feature_subset
    cand = json.loads((REPO / "config/prop_feature_map_candidate_v1.json").read_text())
    cols = cand["pts"]
    X = pd.DataFrame({c: np.arange(10, dtype=float) for c in cols + ["extra_unused"]})
    out = stat_feature_subset(X, "pts", {"prop_feature_map": cand})
    assert list(out.columns) == [c for c in X.columns if c in set(cols)]
    assert "extra_unused" not in out.columns


def test_evaluator_metrics_direction():
    mod = _load("evalmkt", REPO / "scripts/evaluate_market_superiority.py")
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    good = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.85, 0.15, 0.05])   # closer to truth
    bad = np.array([0.5] * 8)
    m = mod._metrics(y, good, bad)
    assert m["logloss_delta"] < 0 and m["brier_delta"] < 0    # model beats the "market" here
    assert m["auc_delta"] > 0


def test_scored_bridge_p_over():
    # PR 1A: build_scored_candidates no longer reconstructs P(over); it uses the sole creator
    # build_probability_lineage (push-safe settled probability) at the quote line.
    from wnba_props_model.models.probability_lineage import build_probability_lineage
    pmf = np.array([0.1, 0.2, 0.3, 0.25, 0.15])  # mass 0..4
    # Half line 2.5: no push -> settled == unconditional P(Y>=3) = 0.40.
    lin = build_probability_lineage(final_pmf=pmf, line=2.5, prop="pts", role="all")
    assert abs(lin.model_prob_over_final - 0.40) < 1e-9
    lin2 = build_probability_lineage(final_pmf=pmf, line=0.5, prop="pts", role="all")
    assert abs(lin2.model_prob_over_final - 0.90) < 1e-9
