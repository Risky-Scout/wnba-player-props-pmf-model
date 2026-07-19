"""P3 - production/validation parity + behavior of the multi-stat forecast publisher."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.pipeline.forecast_publication import apply_multistat_forecast, apply_market
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf
from wnba_props_model.evaluation.distribution_calibration import hierarchical_empirical_pmf
from wnba_props_model.evaluation.forecasting import pmf_to_array

REPO = Path(__file__).resolve().parent.parent
CALIB = json.loads((REPO / "config/certified_forecast_calibration.json").read_text())
CERTIFIED = json.loads((REPO / "config/champion_manifest.json").read_text())["certified_markets"]


def _pmf_json(mean, n=40):
    a = np.zeros(n)
    for k in range(n):
        a[k] = np.exp(-((k - mean) ** 2) / (2 * max(mean, 1)))
    a /= a.sum()
    return json.dumps({str(i): float(v) for i, v in enumerate(a) if v > 1e-9})


def _proj():
    rows = []
    for pid in (1, 2):
        for stat, mean in [("pts", 14), ("ast", 3), ("stl", 1), ("blk", 1), ("reb", 5), ("turnover", 2)]:
            rows.append({"player_id": pid, "player": f"P{pid}", "stat": stat,
                         "pmf_json": _pmf_json(mean), "pmf_mean": float(mean),
                         "role_bucket": "starter", "minutes_mean": 30.0})
    return pd.DataFrame(rows)


def test_publisher_outputs_only_certified_markets_plus_combos():
    out = apply_multistat_forecast(_proj(), CALIB, CERTIFIED)
    stats = set(out["stat"].unique())
    assert {"pts", "ast", "stl", "turnover", "stocks", "pts_ast", "reb"} <= stats
    # blk direct is NOT published (blk is only a stocks component); reb IS certified now
    assert "blk" not in stats
    # combos exist for both players
    assert (out["stat"] == "stocks").sum() == 2 and (out["stat"] == "pts_ast").sum() == 2


def test_reb_evaluated_equals_deployed_parity_with_scale():
    """reb parity: publisher must equal recalibrate(hierarchical(...), 0, scale) exactly."""
    from wnba_props_model.pipeline.forecast_publication import _cells_from_json, _minbucket
    spec = CALIB["markets"]["reb"]
    assert spec["method"] == "hierarchical" and spec.get("scale") == 0.9 and spec["certified"] is True
    pmf = pmf_to_array(_pmf_json(5))
    got = apply_market(pmf, 5.0, "starter", 30.0, spec)
    cells = _cells_from_json(spec["cells"])
    hp = hierarchical_empirical_pmf(5.0, f"starter|{_minbucket(30.0)}", cells, max(len(pmf) - 1, 60))
    exp = recalibrate_pmf(hp, 0.0, 0.9)
    assert np.allclose(got, exp)
    # dispersion scale 0.9 sharpens: narrower std than the unscaled hierarchical PMF
    def _sd(p):
        s = np.arange(len(p)); mu = float((s * p).sum())
        return float(np.sqrt(((s - mu) ** 2 * p).sum()))
    assert _sd(got) < _sd(hp)
    assert abs(got.sum() - 1.0) < 1e-9 and np.all(got >= 0)


def test_evaluated_equals_deployed_parity_hierarchical():
    # applying the publisher's hierarchical method must equal calling the shared function directly
    from wnba_props_model.pipeline.forecast_publication import _cells_from_json, _minbucket
    spec = CALIB["markets"]["pts"]
    pmf = pmf_to_array(_pmf_json(14))
    got = apply_market(pmf, 14.0, "starter", 30.0, spec)
    cells = _cells_from_json(spec["cells"])
    exp = hierarchical_empirical_pmf(14.0, f"starter|{_minbucket(30.0)}", cells, max(len(pmf) - 1, 60))
    assert np.allclose(got, exp)


def test_evaluated_equals_deployed_parity_location():
    spec = CALIB["markets"]["turnover"]
    pmf = pmf_to_array(_pmf_json(2))
    got = apply_market(pmf, 2.0, "starter", 30.0, spec)
    d, s = spec["by_role"].get("starter", spec["pooled"])
    assert np.allclose(got, recalibrate_pmf(pmf, float(d), float(s)))


def test_registry_has_all_eleven_with_methods():
    reg = json.loads((REPO / "config/stat_registry.json").read_text())
    for m in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
              "pts_reb", "pts_ast", "pts_reb_ast", "stocks"):
        assert m in reg
    for m in CERTIFIED:
        assert reg[m]["forecast_allowed"] and reg[m]["forecast_method"]
        assert reg[m]["betting_recommendation_allowed"] is False
