"""PR 1A B6: AST/turnover minutes-offset must REBUILD the PMF, not shift a detached mean.

Verifies the mechanism used by the predict.py minutes-offset block
(rebuild_count_pmf_at_mean): when predicted minutes differ from lagged minutes, the
adjusted mean regenerates the PMF so the distribution, its mean, and line probabilities
all move together and remain internally consistent.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from wnba_props_model.models.market import settled_probabilities_from_pmf
from wnba_props_model.models.pmf_utils import (
    negbinom_pmf_batch,
    rebuild_count_pmf_at_mean,
    validate_pmf_row_integrity,
)


def _mean(pmf):
    k = np.arange(len(pmf), dtype=float)
    return float(np.dot(k, pmf))


def _row(pmf):
    k = np.arange(len(pmf), dtype=float)
    m = float(np.dot(k, pmf)); v = float(np.dot(k ** 2, pmf)) - m ** 2
    return {"pmf_json": json.dumps({str(i): float(p) for i, p in enumerate(pmf) if p > 0}),
            "pmf_mean": m, "pmf_variance": v}


@pytest.mark.parametrize("stat_r", [("ast", 6.0), ("turnover", 8.0)])
def test_minutes_up_shifts_distribution_up(stat_r):
    _, r = stat_r
    base = negbinom_pmf_batch(np.array([4.0]), r, 40)[0]   # ~4 at recent minutes
    line = 4.5
    p_before = settled_probabilities_from_pmf(base, line).p_over_settled

    # predicted minutes >> recent -> higher target mean
    up = rebuild_count_pmf_at_mean(base, 6.0)
    assert abs(up.sum() - 1.0) < 1e-9
    assert _mean(up) > _mean(base) + 0.5                 # mean moved up
    assert not np.allclose(up, base)                     # PMF actually changed
    p_up = settled_probabilities_from_pmf(up, line).p_over_settled
    assert p_up > p_before                               # over-prob rose consistently
    validate_pmf_row_integrity(_row(up))                 # mean matches PMF


@pytest.mark.parametrize("stat_r", [("ast", 6.0), ("turnover", 8.0)])
def test_minutes_down_shifts_distribution_down(stat_r):
    _, r = stat_r
    base = negbinom_pmf_batch(np.array([4.0]), r, 40)[0]
    line = 4.5
    p_before = settled_probabilities_from_pmf(base, line).p_over_settled

    # predicted minutes << recent -> lower target mean
    down = rebuild_count_pmf_at_mean(base, 2.0)
    assert abs(down.sum() - 1.0) < 1e-9
    assert _mean(down) < _mean(base) - 0.5
    p_down = settled_probabilities_from_pmf(down, line).p_over_settled
    assert p_down < p_before
    validate_pmf_row_integrity(_row(down))


def test_detached_mean_shift_would_fail_integrity():
    # A detached mean shift (old buggy behavior) leaves the PMF unchanged but the
    # exported mean moved -> integrity validator must catch it.
    base = negbinom_pmf_batch(np.array([4.0]), 6.0, 40)[0]
    row = _row(base)
    row["pmf_mean"] = 6.0            # detached (PMF still centered at 4)
    with pytest.raises(ValueError):
        validate_pmf_row_integrity(row)


# ── W0.2: the SHARED live+OOF rebuild (apply_minutes_offset_rebuild) ─────────────

def _offset_fixture():
    import pandas as pd
    from wnba_props_model.models.simulation import pmf_to_json
    base = negbinom_pmf_batch(np.array([4.0]), 6.0, 25)[0]  # mean ~4 at lagged minutes
    row = {"player_id": "p1", "game_id": "g1", "stat": "ast",
           "pmf_json": pmf_to_json(base), "pmf_mean": _mean(base),
           "pmf_variance": float(np.dot(np.arange(len(base)) ** 2, base)) - _mean(base) ** 2,
           "stat_mean": _mean(base), "stat_variance": 1.0, "p0": float(base[0]),
           "minutes_mean": 30.0}  # MinutesModel projects 30 vs lagged 20 -> mean scales up
    pmfs_long = pd.DataFrame([row])
    feat = pd.DataFrame([{"player_id": "p1", "game_id": "g1", "player_minutes_mean_l5": 20.0}])
    return pmfs_long, feat


def test_shared_rebuild_moves_pmf_mean_and_stat_mean_together():
    from wnba_props_model.models.pmf_utils import apply_minutes_offset_rebuild
    from wnba_props_model.models.simulation import json_to_pmf, pmf_to_json
    pmfs_long, feat = _offset_fixture()
    before = json_to_pmf(pmfs_long.at[0, "pmf_json"]).copy()
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json, from_json=json_to_pmf,
                                 stats=("ast",))
    after = json_to_pmf(pmfs_long.at[0, "pmf_json"])
    # PMF itself was rebuilt (not a detached shift), and moved up (30 > 20 lagged minutes).
    assert not np.allclose(after[:len(before)], before[:len(after)])
    assert _mean(after) > _mean(before) + 0.5
    # stat_mean == pmf_mean == mean(pmf_json): fully consistent.
    assert abs(float(pmfs_long.at[0, "stat_mean"]) - float(pmfs_long.at[0, "pmf_mean"])) <= 1e-9
    assert abs(float(pmfs_long.at[0, "pmf_mean"]) - _mean(after)) <= 1e-6
    validate_pmf_row_integrity(pmfs_long.loc[0], mean_tol=1e-6)


def test_live_and_oof_paths_produce_identical_pmfs():
    """Both delivery and OOF call the SAME function with the SAME serializers, so the
    rebuilt PMF is byte-identical for identical inputs (W0.2 live == OOF parity)."""
    from wnba_props_model.models.pmf_utils import apply_minutes_offset_rebuild
    from wnba_props_model.models.simulation import json_to_pmf, pmf_to_json
    a, fa = _offset_fixture()
    b, fb = _offset_fixture()
    apply_minutes_offset_rebuild(a, fa, to_json=pmf_to_json, from_json=json_to_pmf, stats=("ast",))
    apply_minutes_offset_rebuild(b, fb, to_json=pmf_to_json, from_json=json_to_pmf, stats=("ast",))
    assert a.at[0, "pmf_json"] == b.at[0, "pmf_json"]
    assert float(a.at[0, "pmf_mean"]) == float(b.at[0, "pmf_mean"])
