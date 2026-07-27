"""Owner item 4 — end-to-end delivery<->OOF settlement parity with nonzero DNP risk.

Runs a p_dnp>0 row through the REAL ``deliver.build_market_comparison`` and asserts the delivered
``model_prob_over_final`` equals the shared active-PMF settlement (the OOF/replay path) to 1e-12,
and does NOT equal the DNP availability-mixture settlement. Also proves unknown-book and
missing-active-PMF rows fail closed. Fails on the pre-patch behaviour (delivery settling from the
mixture).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.availability_pmf import build_availability_mixture
from wnba_props_model.models.probability_lineage import (
    build_probability_lineage,
    build_settled_probability_from_active_pmf,
)
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json
from wnba_props_model.pipeline import deliver

TOL = 1e-12
LINE = 3.0
P_DNP = 0.20


def _active():
    return normalize_pmf(np.array([0.15, 0.20, 0.20, 0.18, 0.12, 0.10, 0.05]))


def _pmf_mean(pmf):
    a = normalize_pmf(np.asarray(pmf, dtype=float))
    return float(np.dot(np.arange(a.size), a))


def _pmfs_frame(active, p_dnp, *, active_json=True):
    a = active
    mix = build_availability_mixture(a, p_dnp)
    aj = pmf_to_json(a) if active_json else None
    return {
        "game_id": 101, "player_id": 5001, "stat": "pts",
        "player_name": "Test Player", "role_bucket": "starter",
        "model_version": "stage4_test", "game_date": "2026-07-28",
        "pmf_json": pmf_to_json(mix), "pmf_mean": _pmf_mean(mix),
        "active_pmf_json": aj, "active_pmf_mean": (_pmf_mean(a) if active_json else np.nan),
        "availability_mixture_pmf_json": pmf_to_json(mix), "p_dnp": p_dnp,
        "sportsbook_settlement_basis": "active_pmf_push_safe_void_on_dnp",
    }


def _raw_prop(vendor, player_id=5001, game_id=101):
    return {
        "game_id": game_id, "player_id": player_id, "player_name": "Test Player",
        "stat": "pts", "prop_type": "player_points", "line": LINE,
        "over_odds": -110, "under_odds": -110, "vendor": vendor,
    }


def test_delivery_settles_from_active_pmf_void_book():
    active = _active()
    pmfs = pd.DataFrame([_pmfs_frame(active, P_DNP)])
    raw = pd.DataFrame([_raw_prop("draftkings")])
    out = deliver.build_market_comparison(pmfs, raw)
    assert not out.empty
    row = out.iloc[0]

    shared = build_settled_probability_from_active_pmf(
        active_pmf=json_to_pmf(pmf_to_json(active)), line=LINE, prop="pts", role="starter")
    mixture = build_probability_lineage(
        final_pmf=json_to_pmf(pmf_to_json(build_availability_mixture(active, P_DNP))),
        line=LINE, prop="pts", role="starter")

    # Delivery == OOF/shared active settlement (void-on-DNP), to float64 tolerance.
    assert row["model_prob_over_final"] == pytest.approx(shared.model_prob_over_final, abs=TOL)
    # Delivery does NOT equal the DNP-as-under mixture settlement (the pre-patch bug).
    assert abs(row["model_prob_over_final"] - mixture.model_prob_over_final) > 0.05
    assert row["sportsbook_settlement_basis"] == "active_pmf_push_safe_void_on_dnp"
    assert row["sportsbook_rule_id"] == "draftkings"


def test_delivery_parity_survives_parquet_roundtrip(tmp_path):
    active = _active()
    pmfs = pd.DataFrame([_pmfs_frame(active, P_DNP)])
    raw = pd.DataFrame([_raw_prop("fanduel")])
    out = deliver.build_market_comparison(pmfs, raw)
    p = tmp_path / "cmp.parquet"
    out.to_parquet(p, index=False)
    reloaded = pd.read_parquet(p)
    live = float(out.iloc[0]["model_prob_over_final"])
    disk = float(reloaded.iloc[0]["model_prob_over_final"])
    assert disk == pytest.approx(live, abs=TOL)
    # Re-settle from the serialized active PMF -> identical to the delivered number.
    resettled = build_settled_probability_from_active_pmf(
        active_pmf=json_to_pmf(reloaded.iloc[0]["active_pmf_json"]),
        line=LINE, prop="pts", role="starter").model_prob_over_final
    assert resettled == pytest.approx(live, abs=TOL)


def test_delivery_fails_closed_when_active_pmf_missing():
    active = _active()
    pmfs = pd.DataFrame([_pmfs_frame(active, P_DNP, active_json=False)])
    raw = pd.DataFrame([_raw_prop("betmgm")])  # VOID_DNP book but active PMF absent
    out = deliver.build_market_comparison(pmfs, raw)
    row = out.iloc[0]
    assert pd.isna(row["model_prob_over_final"])
    assert row["binary_score_eligible"] in (False, np.False_)
    assert row["calibration_status"] == "active_pmf_missing_fail_closed"


def test_delivery_fails_closed_on_unknown_book_rule():
    active = _active()
    pmfs = pd.DataFrame([_pmfs_frame(active, P_DNP)])
    raw = pd.DataFrame([_raw_prop("some_random_local_book")])
    out = deliver.build_market_comparison(pmfs, raw)
    row = out.iloc[0]
    assert pd.isna(row["model_prob_over_final"])
    assert row["sportsbook_settlement_basis"] == "unknown_book_dnp_rule_fail_closed"
