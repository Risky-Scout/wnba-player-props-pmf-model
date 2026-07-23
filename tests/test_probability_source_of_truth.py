"""PR 1A B2/B3/B4: single source of truth for the delivered binary probability.

Guards (scoped to what PR 1A migrates - the LIVE delivery creation site):
  * build_probability_lineage is the only constructor of ProbabilityLineage /
    model_prob_over_final;
  * the live delivery layer (deliver.py) creates the probability via the lineage function
    and does not reconstruct it by calling PMF-to-over logic directly;
  * the lineage columns are emitted by delivery.

NOTE: full migration of every historical/evaluator/report consumer to
model_prob_over_final is completed together with the PR 1B/1C quote+proof paths; this test
locks the delivery-side single source established in PR 1A.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "wnba_props_model"


def test_lineage_module_is_the_only_constructor():
    hits = []
    for p in SRC.rglob("*.py"):
        txt = p.read_text()
        if "ProbabilityLineage(" in txt and p.name != "probability_lineage.py":
            hits.append(str(p.relative_to(REPO)))
    assert hits == [], f"ProbabilityLineage constructed outside its module: {hits}"


def test_lineage_field_exists():
    src = (SRC / "models" / "probability_lineage.py").read_text()
    assert "model_prob_over_final" in src
    assert "def build_probability_lineage" in src
    for col in ("model_prob_over_unconditional", "model_prob_push",
                "model_prob_over_settled_from_final_pmf", "probability_track",
                "calibration_status", "probability_lineage_version"):
        assert col in src


def test_deliver_uses_lineage_and_not_direct_pmf_to_over():
    src = (SRC / "pipeline" / "deliver.py").read_text()
    assert "build_probability_lineage" in src
    # The decision-grade delivery layer must not reconstruct the probability directly.
    assert "prob_over_from_pmf(" not in src
    assert "settled_probabilities_from_pmf(" not in src


def test_deliver_emits_lineage_columns():
    src = (SRC / "pipeline" / "deliver.py").read_text()
    for col in ("model_prob_over_final", "model_prob_over_settled_from_final_pmf",
                "probability_track", "calibration_status", "probability_lineage_version",
                "model_prob_push", "binary_score_eligible"):
        assert col in src


def test_deprecated_wrapper_is_documented():
    src = (SRC / "models" / "market.py").read_text()
    assert "DEPRECATED" in src
    assert "settled_probabilities_from_pmf" in src
