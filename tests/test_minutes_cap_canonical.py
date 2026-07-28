"""Phase 3A: ONE canonical minutes cap (48) shared across paths — no hard-coded 45 clip.

apply_minutes_offset_rebuild must clip projected minutes at the canonical MinutesModel contract
cap (DEFAULT_MINUTES_CLIP_MAX == 48.0), not a path-local 45. Boundary values 0/44/45/47/48/above-cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.minutes_model import DEFAULT_MINUTES_CLIP_MAX
from wnba_props_model.models.pmf_utils import (
    apply_minutes_offset_rebuild,
    negbinom_pmf_batch,
)
from wnba_props_model.models.simulation import json_to_pmf, pmf_to_json

_LAGGED = 20.0          # lagged minutes
_BASE_MEAN = 4.0        # base ast mean at lagged minutes -> rate = 0.2 / min
_RATE = _BASE_MEAN / _LAGGED


def _active_mean_for_minutes(model_minutes: float) -> float:
    base = negbinom_pmf_batch(np.array([_BASE_MEAN]), 6.0, 80)[0]
    row = {
        "player_id": "p1", "game_id": "g1", "stat": "ast",
        "pmf_json": pmf_to_json(base), "pmf_mean": _BASE_MEAN, "pmf_variance": 5.0,
        "stat_mean": _BASE_MEAN, "stat_variance": 5.0, "p0": float(base[0]),
        "minutes_mean": float(model_minutes), "p_dnp": 0.0,
        "active_pmf_json": pmf_to_json(base), "active_pmf_mean": _BASE_MEAN,
    }
    pmfs_long = pd.DataFrame([row])
    feat = pd.DataFrame([{"player_id": "p1", "game_id": "g1",
                          "player_minutes_mean_l5": _LAGGED}])
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json,
                                 from_json=json_to_pmf, stats=("ast",))
    return float(pmfs_long.at[0, "active_pmf_mean"])


def test_canonical_cap_is_48():
    assert DEFAULT_MINUTES_CLIP_MAX == 48.0


def test_no_hardcoded_45_clip_in_source():
    import wnba_props_model.models.pmf_utils as m
    src = __import__("inspect").getsource(m.apply_minutes_offset_rebuild)
    assert "0, 45)" not in src and ", 45)" not in src, "hard-coded 45 clip must be removed"


def test_45_is_not_the_cap():
    # If 45 were still the cap, minutes 47 and 48 would clip to 45 and give identical means.
    m45 = _active_mean_for_minutes(45.0)
    m47 = _active_mean_for_minutes(47.0)
    assert m47 > m45 + 0.1, (m45, m47)


@pytest.mark.parametrize("mins", [44.0, 45.0, 47.0, 48.0])
def test_below_cap_scales_linearly(mins):
    got = _active_mean_for_minutes(mins)
    assert abs(got - _RATE * mins) < 0.15, (mins, got, _RATE * mins)


def test_above_cap_clips_to_48():
    m48 = _active_mean_for_minutes(48.0)
    m60 = _active_mean_for_minutes(60.0)
    assert abs(m60 - m48) < 0.15, (m48, m60)          # 60 clips down to 48
    assert abs(m60 - _RATE * 48.0) < 0.15


def test_zero_minutes_does_not_crash():
    # minutes 0 -> target 0 -> code keeps base (no rebuild); must not raise.
    _ = _active_mean_for_minutes(0.0)


def test_explicit_minutes_cap_override_respected():
    base = negbinom_pmf_batch(np.array([_BASE_MEAN]), 6.0, 80)[0]
    row = {
        "player_id": "p1", "game_id": "g1", "stat": "ast",
        "pmf_json": pmf_to_json(base), "pmf_mean": _BASE_MEAN, "pmf_variance": 5.0,
        "stat_mean": _BASE_MEAN, "stat_variance": 5.0, "p0": float(base[0]),
        "minutes_mean": 60.0, "p_dnp": 0.0,
        "active_pmf_json": pmf_to_json(base), "active_pmf_mean": _BASE_MEAN,
    }
    pmfs_long = pd.DataFrame([row])
    feat = pd.DataFrame([{"player_id": "p1", "game_id": "g1",
                          "player_minutes_mean_l5": _LAGGED}])
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json,
                                 from_json=json_to_pmf, stats=("ast",), minutes_cap=40.0)
    # capped at 40 -> mean ~ 0.2 * 40 = 8.0, strictly below the 48-cap result (9.6).
    assert abs(float(pmfs_long.at[0, "active_pmf_mean"]) - _RATE * 40.0) < 0.15
