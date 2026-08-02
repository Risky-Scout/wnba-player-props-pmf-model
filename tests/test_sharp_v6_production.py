"""Consolidation acceptance tests for the authoritative V6 production path."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_predict_slate_exported():
    from wnba_props_model.sharp_v6 import predict_slate
    assert callable(predict_slate)


def test_legacy_packages_marked_research_only():
    import wnba_props_model.sharp_v3 as v3
    import wnba_props_model.sharp_v4 as v4
    import wnba_props_model.sharp_v5 as v5
    assert v3.RESEARCH_ONLY and v3.DEPRECATED and not v3.PRODUCTION
    assert v4.RESEARCH_ONLY and v4.DEPRECATED and not v4.PRODUCTION
    assert v5.RESEARCH_ONLY and v5.DEPRECATED and not v5.PRODUCTION


def test_v6_production_modules_do_not_import_legacy_inference():
    root = REPO / "src/wnba_props_model/sharp_v6"
    forbidden = ("sharp_v3", "sharp_v4", "sharp_v5")
    for p in root.rglob("*.py"):
        if p.name == "distribution.py":
            # distribution is self-contained; still forbid imports
            pass
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), f"{p} imports {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(f in node.module for f in forbidden), f"{p} imports {node.module}"


def test_team_minutes_allocate_to_200():
    from wnba_props_model.sharp_v6.models import allocate_team_minutes
    mu = np.array([32.0, 28.0, 22.0, 18.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 1.0, 0.5])
    out = allocate_team_minutes(mu, 200.0)
    assert abs(out.sum() - 200.0) < 1e-6


def test_q1_team_minutes_allocate_to_50():
    from wnba_props_model.sharp_v6.models import allocate_team_minutes
    mu = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    out = allocate_team_minutes(mu, 50.0)
    assert abs(out.sum() - 50.0) < 1e-6


def test_push_settlement_half_point_zero_push():
    from wnba_props_model.sharp_v6.inference import _settle
    atoms = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    _A, _B, P, so, su = _settle(atoms, 0.0, 2.5)
    assert P == 0.0
    assert abs(so + su - 1.0) < 1e-12


def test_pmf_normalizes_with_overflow():
    from wnba_props_model.sharp_v6.distribution import CountDistribution
    m = CountDistribution(12.0, 4.0).materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= 1e-10


def test_live_features_not_silent_game_id_swap_only():
    from wnba_props_model.sharp_v6.live_features import build_live_feature_rows
    hist = pd.DataFrame({
        "game_id": [1, 2], "player_id": [10, 10],
        "game_date": pd.to_datetime(["2026-07-01", "2026-07-05"]),
        "team_id": [1, 1], "opponent_team_id": [2, 3],
        "player_minutes_mean_season": [28.0, 29.0],
        "player_rest_days": [2, 3], "is_home": [1, 0],
        "player_pts_mean_l5": [15.0, 16.0],
    })
    stats = hist.rename(columns={"player_minutes_mean_season": "minutes"}).assign(
        team_abbreviation="NY", player_name="Test", pts=10, actual_minutes=30
    )
    games = [{
        "id": 99, "date": "2026-07-10", "scheduled_tip_utc": "2026-07-10T19:00:00Z",
        "home_team": {"id": 1, "abbreviation": "NY"},
        "visitor_team": {"id": 4, "abbreviation": "LA"},
        "status": "scheduled",
    }]
    slate, prov = build_live_feature_rows(
        prediction_timestamp="2026-07-10T12:00:00+00:00",
        scheduled_games=games,
        historical_features=hist,
        historical_stats=stats,
    )
    assert len(slate) >= 1
    assert int(slate.iloc[0]["game_id"]) == 99
    assert all(p.source == "rebuilt_from_prior_observations" for p in prov)
    assert "feature_source" in slate.columns


def test_daily_workflow_does_not_retrain():
    yml = (REPO / ".github/workflows/wnba_pmf_daily.yml").read_text()
    assert "wnba_pmf_daily" in yml or "WNBA PMF Daily" in yml
    assert "run_wnba_pmf.py" in yml
    assert "fit_wnba_pmf_bundle" not in yml
    assert "retrain" in yml.lower()


def test_picks_use_model_probability_not_market_blend():
    src = (REPO / "src/wnba_props_model/pick_engine/engine.py").read_text()
    assert "model_probability = float(pure_p)" in src
    assert "diagnostic_pick_blend" in src


def test_main_reconciliation_artifact():
    p = REPO / "artifacts/sharp_v6/FINAL_MAIN_RECONCILIATION.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["status"] == "RECONCILED_WITH_ORIGIN_MAIN"
    assert d["behind_main_commits"] == 0


@pytest.mark.skipif(
    not (REPO / "artifacts/releases/wnba-pmf-production-v1/MANIFEST.json").exists(),
    reason="production bundle not fitted yet",
)
def test_production_bundle_manifest():
    m = json.loads((REPO / "artifacts/releases/wnba-pmf-production-v1/MANIFEST.json").read_text())
    assert m["bundle_id"] == "wnba-pmf-production-v1"
    assert m["retrain_in_daily"] is False
    assert "predict_slate" in m["inference_function"]
