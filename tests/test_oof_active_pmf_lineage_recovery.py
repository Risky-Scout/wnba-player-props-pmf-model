"""Regression tests for the run-30236023013 recovery (post-merge follow-up).

Two independently proven issues from the failed OOF run:

  1. AST/turnover CONSTRUCTION DEFECT — ``apply_minutes_offset_rebuild`` rebuilt the mixture
     ``pmf_json`` but left ``active_pmf_json`` stale, so active↔mixture were inconsistent even
     at p_dnp==0 (elementwise error up to ~0.6). The fix rebuilds a consistent active so
     ``active ⊕ p_dnp == mixture`` for every offset stat.

  2. PTS SERIALIZATION ROUNDING — the OOF active-PMF lineage invariant compared a JSON-recovered
     active mean against the full-precision ``pmf_mean`` column, tripping 18 false positives on
     wide-support pts rows with p_dnp≈0 (distributions identical, only exported-scalar rounding
     differed). The fix compares like-with-like (both means from the same pmf_json).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.models.availability_pmf import recover_active_pmf
from wnba_props_model.models.pmf_engine import _blend_with_dnp
from wnba_props_model.models.pmf_utils import apply_minutes_offset_rebuild
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json


def _mean(pmf) -> float:
    a = normalize_pmf(np.asarray(pmf, dtype=float))
    return float(np.dot(np.arange(a.size), a))


def _emax(a: np.ndarray, b: np.ndarray) -> float:
    n = max(a.size, b.size)
    aa, bb = np.zeros(n), np.zeros(n)
    aa[: a.size], bb[: b.size] = a, b
    return float(np.max(np.abs(aa - bb)))


def _negbinom(mean: float, size: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1, dtype=float)
    p = size / (size + mean)
    from math import lgamma
    logpmf = np.array([lgamma(kk + size) - lgamma(size) - lgamma(kk + 1) for kk in k])
    logpmf += size * np.log(p) + k * np.log(1 - p)
    return normalize_pmf(np.exp(logpmf))


def _offset_frame(p_dnp: float):
    """AST frame carrying active_pmf_json + p_dnp; MinutesModel projects 30 vs lagged 20."""
    active = _negbinom(4.0, 6.0, 25)
    mixture = _blend_with_dnp(active[None, :].copy(), np.array([p_dnp]))[0]
    row = {
        "player_id": "p1", "game_id": "g1", "stat": "ast",
        "pmf_json": pmf_to_json(mixture), "pmf_mean": _mean(mixture),
        "pmf_variance": float(np.dot(np.arange(mixture.size) ** 2, mixture)) - _mean(mixture) ** 2,
        "stat_mean": _mean(mixture), "stat_variance": 1.0, "p0": float(mixture[0]),
        "minutes_mean": 30.0, "p_dnp": p_dnp,
        "active_pmf_json": pmf_to_json(active), "active_pmf_mean": _mean(active),
        "availability_mixture_pmf_json": pmf_to_json(mixture),
    }
    pmfs_long = pd.DataFrame([row])
    feat = pd.DataFrame([{"player_id": "p1", "game_id": "g1", "player_minutes_mean_l5": 20.0}])
    return pmfs_long, feat


def test_offset_rebuild_keeps_active_consistent_with_mixture_pdnp_positive():
    pmfs_long, feat = _offset_frame(p_dnp=0.15)
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json, from_json=json_to_pmf,
                                 stats=("ast",))
    active = json_to_pmf(pmfs_long.at[0, "active_pmf_json"])
    mixture = json_to_pmf(pmfs_long.at[0, "pmf_json"])
    d = float(pmfs_long.at[0, "p_dnp"])
    # active ⊕ p_dnp reproduces the rebuilt mixture (the invariant that failed on run 30236023013).
    rebuilt = _blend_with_dnp(active[None, :].copy(), np.array([d]))[0]
    assert _emax(rebuilt, mixture) < 1e-6
    # active mean recorded consistently with the active PMF itself.
    assert abs(float(pmfs_long.at[0, "active_pmf_mean"]) - _mean(active)) < 1e-9
    # availability-mixture mirror stays equal to the delivered mixture.
    assert pmfs_long.at[0, "availability_mixture_pmf_json"] == pmfs_long.at[0, "pmf_json"]


def test_offset_rebuild_pdnp0_active_equals_mixture():
    pmfs_long, feat = _offset_frame(p_dnp=0.0)
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json, from_json=json_to_pmf,
                                 stats=("ast",))
    active = json_to_pmf(pmfs_long.at[0, "active_pmf_json"])
    mixture = json_to_pmf(pmfs_long.at[0, "pmf_json"])
    # With no DNP mass the active MUST equal the mixture elementwise (INV6). This was violated by
    # up to ~0.27 before the fix (stale active_pmf_json).
    assert _emax(active, mixture) < 1e-6


def test_offset_rebuild_without_active_column_is_noop_for_active():
    # Legacy frames without active_pmf_json must not crash and must still rebuild the mixture.
    pmfs_long, feat = _offset_frame(p_dnp=0.1)
    pmfs_long = pmfs_long.drop(columns=["active_pmf_json", "active_pmf_mean",
                                        "availability_mixture_pmf_json"])
    before = json_to_pmf(pmfs_long.at[0, "pmf_json"]).copy()
    apply_minutes_offset_rebuild(pmfs_long, feat, to_json=pmf_to_json, from_json=json_to_pmf,
                                 stats=("ast",))
    after = json_to_pmf(pmfs_long.at[0, "pmf_json"])
    assert _mean(after) > _mean(before) + 0.3  # mixture still rebuilt (30 > 20 lagged mins)
    assert "active_pmf_json" not in pmfs_long.columns


# --- PTS serialization false-positive guard (build_oof_pmfs active-PMF lineage invariant) -----

def _lineage_bad_count(pmf_jsons, p_dnps, reference_means):
    active_means = []
    for js, d in zip(pmf_jsons, p_dnps):
        a = recover_active_pmf(js, float(d))
        active_means.append(float(np.dot(np.arange(a.size), a)))
    amn = np.asarray(active_means)
    return int(np.sum(np.asarray(reference_means) > amn + 1e-6))


def test_lineage_invariant_no_false_positive_wide_support_pdnp_zero():
    rng = np.random.default_rng(11)
    jsons, full_prec, json_means, dnps = [], [], [], []
    for _ in range(300):
        pmf = normalize_pmf(rng.random(61) ** 3)  # wide (pts-like) support
        js = pmf_to_json(pmf)
        jsons.append(js); dnps.append(0.0)
        full_prec.append(float(np.dot(np.arange(pmf.size), pmf)))
        m = json_to_pmf(js)
        json_means.append(float(np.dot(np.arange(m.size), m)))
    old_bad = _lineage_bad_count(jsons, dnps, full_prec)   # buggy reference (full precision)
    new_bad = _lineage_bad_count(jsons, dnps, json_means)  # fixed reference (json-consistent)
    assert new_bad == 0
    assert new_bad <= old_bad


def test_lineage_invariant_still_catches_real_violation():
    active = normalize_pmf(np.array([0.15, 0.2, 0.2, 0.18, 0.12, 0.1, 0.05]))
    js = pmf_to_json(active)
    inflated = _mean(active) + 1.0  # impossible for a real DNP fold
    assert _lineage_bad_count([js], [0.0], [inflated]) == 1
