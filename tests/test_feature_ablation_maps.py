"""Tests for the resolved per-prop feature-ablation maps (Plan P2).

The resolved maps are the exact feature lists each ablation candidate feeds to
training.stat_feature_subset. They must be contract-valid and apply the plan's transforms.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIRECT = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "bfam", REPO / "scripts" / "build_feature_ablation_maps.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_resolve_candidate_applies_add_and_remove():
    m = _mod()
    prop_map = {"pts": ["a", "b", "c"]}
    out = m.resolve_candidate(prop_map, {"remove": ["b"], "add": ["z"]})
    assert out["pts"] == ["a", "c", "z"]
    # identity transform
    assert m.resolve_candidate(prop_map, {})["pts"] == ["a", "b", "c"]


def test_resolved_maps_are_contract_valid():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from wnba_props_model.features.feature_contract import MODEL_FEATURES, FORBIDDEN_MODEL_FEATURES
    p = REPO / "config" / "feature_ablation_maps_v1.json"
    assert p.exists(), "run scripts/build_feature_ablation_maps.py first"
    maps = json.loads(p.read_text())["candidates"]
    valid, forb = set(MODEL_FEATURES), set(FORBIDDEN_MODEL_FEATURES)
    assert "G0" in maps and set(maps["G0"]["pts"]) == valid       # G0 = full global contract
    for cand, pm in maps.items():
        for prop in DIRECT:
            assert prop in pm, f"{cand} missing {prop}"
            feats = pm[prop]
            assert len(feats) == len(set(feats)), f"{cand}/{prop} duplicates"
            assert all(f in valid for f in feats), f"{cand}/{prop} non-contract feature"
            assert not any(f in forb for f in feats), f"{cand}/{prop} forbidden feature"


def test_ablation_transforms_applied():
    maps = json.loads((REPO / "config" / "feature_ablation_maps_v1.json").read_text())["candidates"]
    # S2 adds lagged market priors
    assert "player_market_p_over_prev" in maps["S2_stat_specific_plus_lagged_market"]["pts"]
    # S3 removes the game-script family
    assert "game_total" not in maps["S3_minus_game_script"]["pts"]
    # S5 removes fatigue family
    assert "schedule_fatigue_index" not in maps["S5_minus_fatigue"]["pts"]
    # S1 is a strict subset of the global contract
    assert len(maps["S1_stat_specific_ex_market"]["pts"]) < len(maps["G0"]["pts"])
