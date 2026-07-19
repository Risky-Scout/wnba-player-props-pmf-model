"""P3 - multi-stat champion package: registry, immutable calibration package, manifest
hashes, and evaluated-vs-deployed parity (six certified markets)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wnba_props_model.pipeline.forecast_publication import apply_market
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf
from wnba_props_model.evaluation.forecasting import pmf_to_array

REPO = Path(__file__).resolve().parent.parent
REGISTRY = json.loads((REPO / "config/stat_registry.json").read_text())
MANIFEST = json.loads((REPO / "config/champion_manifest.json").read_text())
CALIB = json.loads((REPO / "config/certified_forecast_calibration.json").read_text())
CERTIFIED = ["turnover", "pts", "ast", "stl", "stocks", "pts_ast", "reb", "pts_reb", "pts_reb_ast"]


def test_registry_certified_markets_betting_off():
    for m in CERTIFIED:
        assert REGISTRY[m]["forecast_allowed"] is True
        assert REGISTRY[m]["betting_recommendation_allowed"] is False
        assert REGISTRY[m]["forecast_method"]
    for s in ("fg3m", "blk"):
        assert REGISTRY[s]["forecast_allowed"] is False


def test_manifest_hashes_match_artifacts():
    h = hashlib.sha256(json.dumps(CALIB, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert MANIFEST["calibration_hash"] == h
    rh = hashlib.sha256(json.dumps(REGISTRY, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert MANIFEST["registry_hash"] == rh
    assert MANIFEST["status"] == "LIVE_VALIDATED_FORECAST_ONLY"
    assert MANIFEST["feature_schema"] == "schema_v2"
    assert set(MANIFEST["certified_markets"]) == set(CERTIFIED)


def test_turnover_method_is_location():
    assert CALIB["markets"]["turnover"]["method"] == "location"
    assert CALIB["markets"]["turnover"]["pooled"][1] == 1.0


def test_evaluated_vs_deployed_parity_location():
    spec = CALIB["markets"]["turnover"]
    pmf = np.array([0.5, 0.3, 0.15, 0.05])
    d, s = spec["by_role"].get("starter", spec["pooled"])
    assert np.allclose(apply_market(pmf, 2.0, "starter", 30.0, spec),
                       recalibrate_pmf(pmf, float(d), float(s)))
