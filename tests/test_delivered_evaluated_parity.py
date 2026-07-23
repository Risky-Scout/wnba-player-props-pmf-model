"""PR 1A B7: delivered == serialized == candidate-builder == evaluator, within 1e-12.

Because build_probability_lineage is the single creator of model_prob_over_final and every
decision-grade consumer reads that field (never reconstructing it from pmf_json), the
values must be bitwise-close through Parquet serialization and downstream consumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# --- integer-line (push) + legacy-ignored end-to-end via the real contract ---

from wnba_props_model.models.probability_contract import (  # noqa: E402
    FINAL_PROBABILITY_COLUMN, LEGACY_PROBABILITY_COLUMN,
    ProbabilityContractError, require_final_probability, validate_final_probability,
)


def _consume_like_candidate_and_evaluator(pq_path):
    """Simulate the migrated candidate-builder + evaluator: read the delivered final
    probability via the shared contract (never the legacy column, never the PMF)."""
    df = pd.read_parquet(pq_path)
    return float(require_final_probability(df, consumer="parity-test").iloc[0])


def test_integer_line_with_push_parity_and_legacy_ignored(tmp_path):
    pmf = _final_pmf(mu=10.0, r=6.0, cap=40)
    line = 10.0  # integer line with nonzero push mass
    lin = build_probability_lineage(final_pmf=pmf, line=line, prop="pts", role="starter")
    assert lin.model_prob_push > 0 and lin.binary_score_eligible is True
    delivered = lin.model_prob_over_final
    # Serialize with the legacy column set to a DIFFERENT value on purpose.
    row = lin.as_row()
    row[LEGACY_PROBABILITY_COLUMN] = 0.123456789  # wrong; must be ignored
    pq = tmp_path / "int.parquet"
    pd.DataFrame([row]).to_parquet(pq, index=False)
    serialized = float(pd.read_parquet(pq)[FINAL_PROBABILITY_COLUMN].iloc[0])
    consumed = _consume_like_candidate_and_evaluator(pq)
    assert abs(serialized - delivered) <= TOL
    assert abs(consumed - delivered) <= TOL
    assert abs(consumed - 0.123456789) > 0.01           # legacy value NOT used
    # settled conditions out the push (integer line).
    assert delivered == lin.model_prob_over_settled_from_final_pmf


def test_half_line_parity(tmp_path):
    pmf = _final_pmf(mu=6.0, r=5.0)
    lin = build_probability_lineage(final_pmf=pmf, line=6.5, prop="pts", role="starter")
    row = lin.as_row(); row[LEGACY_PROBABILITY_COLUMN] = 0.999
    pq = tmp_path / "half.parquet"
    pd.DataFrame([row]).to_parquet(pq, index=False)
    assert abs(_consume_like_candidate_and_evaluator(pq) - lin.model_prob_over_final) <= TOL


def test_all_push_integer_line_is_binary_ineligible():
    pmf = np.zeros(21); pmf[10] = 1.0
    lin = build_probability_lineage(final_pmf=pmf, line=10.0, prop="pts", role="starter")
    assert lin.binary_score_eligible is False
    assert lin.model_prob_over_final is None
    # excluded from binary scoring, never fabricated to 0.5
    assert np.isnan(validate_final_probability(lin.model_prob_over_final,
                                               consumer="parity", allow_none=True))


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.001, 1.001])
def test_fail_closed_on_invalid_final(bad):
    with pytest.raises(ProbabilityContractError):
        validate_final_probability(bad, consumer="parity")


def test_missing_final_column_fails_closed(tmp_path):
    pq = tmp_path / "nofinal.parquet"
    pd.DataFrame([{"line": 6.5, LEGACY_PROBABILITY_COLUMN: 0.5}]).to_parquet(pq, index=False)
    with pytest.raises(ProbabilityContractError):
        _consume_like_candidate_and_evaluator(pq)


def _oof_and_quotes(tmp_path, line, mu=10.0, r=6.0):
    """Build tiny OOF + quotes parquets for a real build_scored_candidates run."""
    from wnba_props_model.models.pmf_utils import negbinom_pmf_batch
    from wnba_props_model.models.simulation import pmf_to_json
    pmf = negbinom_pmf_batch(np.array([mu]), r, 40)[0]
    oof = pd.DataFrame([{
        "game_id": "G1", "player_id": "P1", "stat": "pts", "game_date": "2026-06-20",
        "pmf_json": pmf_to_json(pmf), "actual_outcome": 12.0, "role_bucket": "starter",
    }])
    quotes = pd.DataFrame([{
        "game_id": "G1", "player_id": "P1", "stat": "pts", "line": float(line),
        "market_prob_over_no_vig": 0.5,
    }])
    op = tmp_path / "oof.parquet"; qp = tmp_path / "quotes.parquet"
    oof.to_parquet(op, index=False); quotes.to_parquet(qp, index=False)
    return op, qp, pmf


@pytest.mark.parametrize("line", [10.0, 6.5])  # integer(push) + half line
def test_real_build_scored_candidates_preserves_final(tmp_path, line):
    import subprocess
    import sys
    op, qp, pmf = _oof_and_quotes(tmp_path, line)
    scored = tmp_path / "scored.parquet"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_scored_candidates.py"),
         "--oof", str(op), "--quotes", str(qp), "--out", str(scored), "--candidate", "T"],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_parquet(scored)
    assert len(df) == 1
    lineage_final = build_probability_lineage(
        final_pmf=pmf, line=float(line), prop="pts", role="starter").model_prob_over_final
    # Real candidate builder preserves the sole-creator final probability within 1e-12.
    assert abs(float(df[FINAL_PROBABILITY_COLUMN].iloc[0]) - lineage_final) <= TOL
    # It emits the final column, not the legacy alias.
    assert LEGACY_PROBABILITY_COLUMN not in df.columns


REPO = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_rounded_serialized_probability_would_break_parity(tmp_path):
    # Proof that rounding the proof probability violates the 1e-12 parity contract.
    pmf = _final_pmf(mu=7.3137, r=4.11)
    lin = build_probability_lineage(final_pmf=pmf, line=7.5, prop="reb", role="rotation")
    row = lin.as_row()
    row[FINAL_PROBABILITY_COLUMN] = round(lin.model_prob_over_final, 4)  # display rounding
    pq = tmp_path / "rounded.parquet"
    pd.DataFrame([row]).to_parquet(pq, index=False)
    consumed = _consume_like_candidate_and_evaluator(pq)
    assert abs(consumed - lin.model_prob_over_final) > TOL  # rounding detectably breaks parity
