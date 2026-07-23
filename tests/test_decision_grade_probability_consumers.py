"""PR 1A #5: decision-grade consumers must read model_prob_over_final via the shared
fail-closed contract, never the legacy column and never a PMF reconstruction.

These tests exercise the shared contract (require_final_probability / validate_final_probability
/ assert_alias_invariant) that every migrated decision-grade consumer must route through. A
mismatch fixture (final=0.73, legacy=0.19) proves the contract yields 0.73 and that changing
the legacy value never changes the answer.

Scope note: the live delivery creation site (deliver.py) is migrated in PR 1A. Per-consumer
migration of the historical/evaluator/report paths (several Foundation-Lock hash-pinned)
lands with PR 1B/1C; this suite locks the contract they must adopt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.probability_contract import (
    FINAL_PROBABILITY_COLUMN,
    LEGACY_PROBABILITY_COLUMN,
    ProbabilityContractError,
    assert_alias_invariant,
    require_final_probability,
    validate_final_probability,
)


def _mismatch_df():
    return pd.DataFrame({
        FINAL_PROBABILITY_COLUMN: [0.73, 0.61],
        LEGACY_PROBABILITY_COLUMN: [0.19, 0.19],   # intentionally WRONG
        "line": [10.5, 4.5], "stat": ["pts", "reb"],
        "pmf_json": ['{"0":1.0}', '{"0":1.0}'],
    })


def test_contract_reads_final_not_legacy():
    df = _mismatch_df()
    vals = require_final_probability(df, consumer="unit-test").to_numpy()
    assert vals[0] == 0.73 and vals[1] == 0.61      # uses final, never 0.19


def test_changing_legacy_does_not_change_result():
    df = _mismatch_df()
    before = require_final_probability(df, consumer="unit-test").to_numpy().copy()
    df[LEGACY_PROBABILITY_COLUMN] = [0.01, 0.99]    # mutate legacy
    after = require_final_probability(df, consumer="unit-test").to_numpy()
    assert np.array_equal(before, after)


def test_alias_invariant_detects_mismatch():
    with pytest.raises(ProbabilityContractError):
        assert_alias_invariant(_mismatch_df(), consumer="unit-test")


def test_missing_final_field_fails_closed():
    df = pd.DataFrame({LEGACY_PROBABILITY_COLUMN: [0.5]})
    with pytest.raises(ProbabilityContractError) as e:
        require_final_probability(df, consumer="edge_report")
    assert "edge_report" in str(e.value)          # error names the consumer
    assert "no legacy" not in str(e.value).lower() or "absent" in str(e.value).lower()


def test_nan_final_fails_closed():
    df = pd.DataFrame({FINAL_PROBABILITY_COLUMN: [0.5, np.nan]})
    with pytest.raises(ProbabilityContractError):
        require_final_probability(df, consumer="scorer")


def test_inf_final_fails_closed():
    df = pd.DataFrame({FINAL_PROBABILITY_COLUMN: [0.5, np.inf]})
    with pytest.raises(ProbabilityContractError):
        require_final_probability(df, consumer="scorer")


def test_out_of_range_fails_closed_no_silent_clip():
    df = pd.DataFrame({FINAL_PROBABILITY_COLUMN: [0.5, 1.4]})
    with pytest.raises(ProbabilityContractError):
        require_final_probability(df, consumer="scorer")


def test_scalar_validator_fail_closed():
    assert validate_final_probability(0.42, consumer="c") == 0.42
    for bad in (None, float("nan"), float("inf"), -0.01, 1.01, "x"):
        with pytest.raises(ProbabilityContractError):
            validate_final_probability(bad, consumer="c")
    # binary-ineligible (all-push) may carry None only when explicitly allowed.
    assert np.isnan(validate_final_probability(None, consumer="c", allow_none=True))


def test_machine_audit_reports_one_zero_zero():
    # Machine proof across ALL decision-grade consumers (item 8/10): creators=1, reads=0, recon=0.
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(repo / "scripts" / "audit_probability_consumers.py"),
                        "--check"], capture_output=True, text=True, cwd=str(repo))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS 1/0/0" in r.stdout


def test_forbidden_get_fallback_pattern_absent_in_contract_source():
    import inspect
    from wnba_props_model.models import probability_contract as m
    src = inspect.getsource(m)
    # The contract must forbid, not implement, the silent legacy fallback.
    assert 'get("model_prob_over_final", ' not in src
    assert ".get(FINAL_PROBABILITY_COLUMN, " not in src
