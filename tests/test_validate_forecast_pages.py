"""B5/B6/B7: fail-closed forecast-page validator (abstention, provenance lineage, no Kelly)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "vfp", LANE / "scripts" / "validate_forecast_pages.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _write(base: Path, edge, pmf, dist):
    for sub, payload in (("Edge", edge), ("PMF-Distributions", pmf), ("Distributions", dist)):
        d = base / sub; d.mkdir(parents=True, exist_ok=True)
        (d / "latest.json").write_text(json.dumps(payload))


def _good_edge(rel="r1"):
    return {"release_id": rel, "game_date": "2026-07-20", "git_commit": "abc",
            "abstain": True, "publication_mode": "forecast_only",
            "total_props": 0, "over_signals": 0, "under_signals": 0, "props": []}


def _good_forecast(rel="r1", stat="reb"):
    return {"release_id": rel, "game_date": "2026-07-20", "git_commit": "abc",
            "props": [{"player": "A", "stat": stat.upper(), "stat_raw": stat, "line": 2.0,
                       "model_p_over": 0.62, "model_p_under": 0.38,
                       "pmf_full": [[0, 0.4], [1, 0.3], [2, 0.3]]}]}


def test_clean_abstaining_pages_pass(tmp_path):
    m = _mod()
    _write(tmp_path, _good_edge(), _good_forecast(), _good_forecast())
    assert m.validate(tmp_path) == []


def test_nonabstaining_edge_board_fails(tmp_path):
    m = _mod()
    bad = _good_edge(); bad["abstain"] = False; bad["publication_mode"] = "publish"
    _write(tmp_path, bad, _good_forecast(), _good_forecast())
    errs = m.validate(tmp_path)
    assert any("abstain" in e for e in errs)


def test_exposed_edge_rows_or_kelly_fail(tmp_path):
    m = _mod()
    bad = _good_edge(); bad["total_props"] = 3
    bad["props"] = [{"player": "A", "stat": "REB", "kelly_pct": 4.0}]
    _write(tmp_path, bad, _good_forecast(), _good_forecast())
    errs = m.validate(tmp_path)
    assert any("edge row" in e or "nonzero" in e for e in errs)


def test_suppressed_stat_public_fails(tmp_path):
    m = _mod()
    # fg3m is forecast_allowed=false in stat_registry -> must not be public.
    _write(tmp_path, _good_edge(), _good_forecast(stat="fg3m"), _good_forecast(stat="fg3m"))
    errs = m.validate(tmp_path)
    assert any("fg3m" in e for e in errs)


def test_release_lineage_mismatch_fails(tmp_path):
    m = _mod()
    _write(tmp_path, _good_edge("r1"), _good_forecast("r1"), _good_forecast("r2"))
    errs = m.validate(tmp_path)
    assert any("lineage mismatch" in e for e in errs)


def test_invalid_probability_and_pmf_sum_fail(tmp_path):
    m = _mod()
    bad = _good_forecast()
    bad["props"][0]["model_p_over"] = 1.9
    bad["props"][0]["pmf_full"] = [[0, 0.4], [1, 0.3]]   # sums to 0.7
    _write(tmp_path, _good_edge(), bad, _good_forecast())
    errs = m.validate(tmp_path)
    assert any("invalid probability" in e for e in errs)
    assert any("PMF mass" in e for e in errs)


def test_smoke_workflow_is_dry_run_first():
    wf = (LANE / ".github" / "workflows" / "forecast_pages_smoke.yml").read_text()
    assert "workflow_dispatch" in wf and "dry_run" in wf and "validate_forecast_pages.py" in wf
    # Deploy step must be gated on dry_run=false AND deploy=true.
    assert "dry_run == 'false' && github.event.inputs.deploy == 'true'" in wf
