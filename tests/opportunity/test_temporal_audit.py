"""Temporal-purity audit tests for Opportunity V2."""
from __future__ import annotations

import pandas as pd

from wnba_props_model.opportunity.audit import audit_temporal_purity

CUT = pd.Timestamp("2026-05-08T22:00:00Z")


def _frame():
    return pd.DataFrame({
        "prediction_cutoff_utc": [CUT, CUT, CUT],
        "availability_available_at_utc": [
            CUT - pd.Timedelta(hours=2), CUT - pd.Timedelta(hours=1), CUT - pd.Timedelta(minutes=5),
        ],
        "player_minutes_ewma": [24.0, 30.0, 12.0],
    })


def test_clean_frame_passes():
    res = audit_temporal_purity(
        _frame(), "prediction_cutoff_utc",
        ["availability_available_at_utc"],
        feature_columns=["player_minutes_ewma"],
    )
    assert res.passed
    assert res.violation_count == 0


def test_future_source_timestamp_fails():
    f = _frame()
    f.loc[2, "availability_available_at_utc"] = CUT + pd.Timedelta(hours=1)
    res = audit_temporal_purity(f, "prediction_cutoff_utc", ["availability_available_at_utc"])
    assert not res.passed
    assert res.violation_count == 1
    assert res.violations_by_column["availability_available_at_utc"] == 1
    assert res.max_future_seconds_by_column["availability_available_at_utc"] == 3600.0
    assert res.sampled_violations


def test_forbidden_market_feature_fails():
    f = _frame()
    f["market_prob_over_no_vig"] = [0.5, 0.5, 0.5]
    res = audit_temporal_purity(
        f, "prediction_cutoff_utc", ["availability_available_at_utc"],
        feature_columns=["player_minutes_ewma", "market_prob_over_no_vig"],
    )
    assert not res.passed
    assert "market_prob_over_no_vig" in res.forbidden_market_columns


def test_null_cutoff_counts_as_violation():
    f = _frame()
    f.loc[0, "prediction_cutoff_utc"] = pd.NaT
    res = audit_temporal_purity(f, "prediction_cutoff_utc", ["availability_available_at_utc"])
    assert not res.passed
    assert res.violation_count >= 1


def test_to_dict_is_json_safe():
    res = audit_temporal_purity(_frame(), "prediction_cutoff_utc", ["availability_available_at_utc"])
    import json
    json.dumps(res.to_dict())  # must not raise
