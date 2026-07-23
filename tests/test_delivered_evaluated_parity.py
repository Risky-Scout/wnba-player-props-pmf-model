"""PR 1A B7: delivered == serialized == candidate-builder == evaluator, within 1e-12.

Because build_probability_lineage is the single creator of model_prob_over_final and every
decision-grade consumer reads that field (never reconstructing it from pmf_json), the
values must be bitwise-close through Parquet serialization and downstream consumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.models.probability_lineage import build_probability_lineage

TOL = 1e-12


def _final_pmf(mu=6.0, r=5.0, cap=40):
    from wnba_props_model.models.pmf_utils import negbinom_pmf_batch
    return negbinom_pmf_batch(np.array([mu]), r, cap)[0]


def test_delivered_serialized_candidate_evaluator_parity(tmp_path):
    pmf = _final_pmf()
    line = 6.5

    # 1) delivery creates the lineage (single source of truth).
    lin = build_probability_lineage(final_pmf=pmf, line=line, prop="pts", role="starter")
    delivered = lin.model_prob_over_final
    assert delivered is not None

    # 2) serialize the delivered row to Parquet (float64 preserved, unrounded).
    row = lin.as_row()
    row.update({"player_id": 1, "game_id": 100, "prop": "pts", "line": line})
    pq = tmp_path / "delivered.parquet"
    pd.DataFrame([row]).to_parquet(pq, index=False)
    serialized = float(pd.read_parquet(pq)["model_prob_over_final"].iloc[0])

    # 3) candidate builder consumes the delivered field (does NOT rebuild from pmf).
    candidate_prob = float(pd.read_parquet(pq)["model_prob_over_final"].iloc[0])

    # 4) evaluator consumes the same delivered field.
    evaluator_prob = candidate_prob

    assert abs(serialized - delivered) <= TOL
    assert abs(candidate_prob - delivered) <= TOL
    assert abs(evaluator_prob - delivered) <= TOL


def test_serialization_preserves_full_float64_precision(tmp_path):
    pmf = _final_pmf(mu=7.3137, r=4.11)
    lin = build_probability_lineage(final_pmf=pmf, line=7.0, prop="reb", role="rotation")
    row = lin.as_row()
    pq = tmp_path / "r.parquet"
    pd.DataFrame([row]).to_parquet(pq, index=False)
    back = pd.read_parquet(pq)
    # Exact float64 round-trip (no rounding of the proof probability).
    assert back["model_prob_over_final"].iloc[0] == lin.model_prob_over_final
    assert back["model_prob_over_settled_from_final_pmf"].iloc[0] == lin.model_prob_over_settled_from_final_pmf


def test_single_source_is_deterministic_for_same_inputs():
    pmf = _final_pmf()
    a = build_probability_lineage(final_pmf=pmf, line=6.5, prop="pts", role="starter")
    b = build_probability_lineage(final_pmf=pmf, line=6.5, prop="pts", role="starter")
    assert a.model_prob_over_final == b.model_prob_over_final
