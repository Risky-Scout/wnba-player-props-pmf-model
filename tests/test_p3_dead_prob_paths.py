"""P3 Defect #4 — the invalid `mean/(mean+1)` P(over) fallback and the unconsumed
`p_over_beta` dead path must be gone; P(over) may only come from a PMF at a line."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_no_mean_over_mean_plus_one_fallback():
    src = (REPO / "scripts/predict_today.py").read_text()
    # the specific invalid scalar fallback must not exist
    assert not re.search(r"_mean_col\]\s*/\s*\(_stat_rows\[_mean_col\]\s*\+\s*1", src)
    assert "mean / (mean + 1" not in src


def test_p_over_beta_dead_path_removed():
    src = (REPO / "scripts/predict_today.py").read_text()
    # p_over_beta was never consumed by the production selector -> removed
    assert 'pmfs.loc[_stat_rows.index, "p_over_beta"]' not in src
    assert "p_over_beta" not in src


def test_p_over_beta_not_consumed_by_selector():
    for f in ("scripts/build_edge_report.py", "src/wnba_props_model/pipeline/deliver.py"):
        assert "p_over_beta" not in (REPO / f).read_text()
