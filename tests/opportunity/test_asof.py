"""Strict as-of join tests for Opportunity V2 (section 34)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.asof import (
    TemporalLeakageError,
    assert_feature_time_purity,
    strict_asof_join,
)

CUT = pd.Timestamp("2026-05-08T22:00:00Z")


def _left(n_players=1):
    rows = []
    for p in range(1, n_players + 1):
        rows.append({"player_id": p, "prediction_cutoff_utc": CUT})
    return pd.DataFrame(rows)


def _right(ts, player_id=1, status="questionable"):
    return pd.DataFrame([{
        "player_id": player_id,
        "available_at_utc": pd.Timestamp(ts),
        "status_normalized": status,
    }])


def test_snapshot_one_second_before_cutoff_included():
    out = strict_asof_join(_left(), _right(CUT - pd.Timedelta(seconds=1)),
                           by=["player_id"], suffix="av")
    assert bool(out["av_matched"].iloc[0])
    assert out["status_normalized_av"].iloc[0] == "questionable"


def test_snapshot_exactly_at_cutoff_included():
    out = strict_asof_join(_left(), _right(CUT), by=["player_id"], suffix="av")
    assert bool(out["av_matched"].iloc[0])


def test_snapshot_one_second_after_cutoff_excluded():
    out = strict_asof_join(_left(), _right(CUT + pd.Timedelta(seconds=1)),
                           by=["player_id"], suffix="av")
    assert not bool(out["av_matched"].iloc[0])


def test_later_same_day_snapshot_cannot_overwrite_earlier_state():
    right = pd.concat([
        _right(CUT - pd.Timedelta(hours=2), status="questionable"),
        _right(CUT + pd.Timedelta(hours=2), status="out"),  # after cutoff, must be ignored
    ], ignore_index=True)
    out = strict_asof_join(_left(), right, by=["player_id"], suffix="av")
    assert out["status_normalized_av"].iloc[0] == "questionable"


def test_snapshot_from_another_player_cannot_join():
    out = strict_asof_join(_left(), _right(CUT - pd.Timedelta(hours=1), player_id=999),
                           by=["player_id"], suffix="av")
    assert not bool(out["av_matched"].iloc[0])


def test_stale_snapshot_nulled_after_max_age():
    out = strict_asof_join(_left(), _right(CUT - pd.Timedelta(days=30)),
                           by=["player_id"], suffix="av", max_age=pd.Timedelta(days=7))
    assert not bool(out["av_matched"].iloc[0])
    assert pd.isna(out["status_normalized_av"].iloc[0])


def test_left_row_count_never_changes():
    left = _left(5)
    right = pd.concat([_right(CUT - pd.Timedelta(hours=1), player_id=p) for p in range(1, 6)],
                      ignore_index=True)
    out = strict_asof_join(left, right, by=["player_id"], suffix="av")
    assert len(out) == len(left)


def test_duplicate_right_identities_do_not_inflate_rows():
    left = _left(1)
    # two right snapshots for same player both before cutoff -> as-of takes the latest, count stays 1
    right = pd.concat([
        _right(CUT - pd.Timedelta(hours=3), status="questionable"),
        _right(CUT - pd.Timedelta(hours=1), status="probable"),
    ], ignore_index=True)
    out = strict_asof_join(left, right, by=["player_id"], suffix="av")
    assert len(out) == 1
    assert out["status_normalized_av"].iloc[0] == "probable"  # most recent valid


def test_required_raises_when_unmatched():
    with pytest.raises(TemporalLeakageError):
        strict_asof_join(_left(), _right(CUT + pd.Timedelta(hours=1)),
                         by=["player_id"], suffix="av", required=True)


def test_null_cutoff_raises():
    left = _left()
    left.loc[0, "prediction_cutoff_utc"] = pd.NaT
    with pytest.raises(TemporalLeakageError):
        strict_asof_join(left, _right(CUT - pd.Timedelta(hours=1)), by=["player_id"], suffix="av")


def test_order_preserved():
    left = _left(4).sample(frac=1.0, random_state=1).reset_index(drop=True)
    left["prediction_cutoff_utc"] = CUT
    right = pd.concat([_right(CUT - pd.Timedelta(hours=1), player_id=p) for p in range(1, 5)],
                      ignore_index=True)
    out = strict_asof_join(left, right, by=["player_id"], suffix="av")
    assert list(out["player_id"]) == list(left["player_id"])


# --- randomized reference comparison (section 10) ---------------------------

def _reference_asof(left, right, cutoff, avail, by, suffix):
    out_matched, out_status = [], []
    for _, lr in left.iterrows():
        cands = right[(right[by[0]] == lr[by[0]]) & (right[avail] <= lr[cutoff])]
        if len(cands):
            best = cands.sort_values(avail).iloc[-1]
            out_matched.append(True)
            out_status.append(best["status_normalized"])
        else:
            out_matched.append(False)
            out_status.append(np.nan)
    return out_matched, out_status


def test_strict_asof_matches_slow_reference_randomized():
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2026-05-01T00:00:00Z")
    left_rows, right_rows = [], []
    for p in range(1, 8):
        cut = base + pd.Timedelta(hours=int(rng.integers(0, 200)))
        left_rows.append({"player_id": p, "prediction_cutoff_utc": cut})
        for _ in range(int(rng.integers(0, 5))):
            ts = base + pd.Timedelta(hours=int(rng.integers(0, 220)))
            right_rows.append({"player_id": p, "available_at_utc": ts,
                               "status_normalized": rng.choice(["out", "questionable", "available"])})
    left = pd.DataFrame(left_rows)
    right = pd.DataFrame(right_rows)
    out = strict_asof_join(left, right, by=["player_id"], suffix="av")
    ref_m, ref_s = _reference_asof(left, right, "prediction_cutoff_utc", "available_at_utc",
                                   ["player_id"], "av")
    assert list(out["av_matched"]) == ref_m
    for got, exp in zip(out["status_normalized_av"], ref_s):
        if isinstance(exp, float):
            assert pd.isna(got)
        else:
            assert got == exp


def test_assert_feature_time_purity_flags_future_source():
    frame = pd.DataFrame({
        "prediction_cutoff_utc": [CUT, CUT],
        "src_a_utc": [CUT - pd.Timedelta(hours=1), CUT + pd.Timedelta(hours=1)],
    })
    with pytest.raises(TemporalLeakageError):
        assert_feature_time_purity(frame, cutoff_col="prediction_cutoff_utc",
                                   source_timestamp_columns=["src_a_utc"])
    ok = frame.copy()
    ok["src_a_utc"] = [CUT - pd.Timedelta(hours=2), CUT - pd.Timedelta(hours=1)]
    assert_feature_time_purity(ok, cutoff_col="prediction_cutoff_utc",
                               source_timestamp_columns=["src_a_utc"])
