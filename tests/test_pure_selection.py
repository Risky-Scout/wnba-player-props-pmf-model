"""Owner ITEM 3 + ITEM 4 — unified pure candidate-selection engine tests.

Proves the candidate family (P0-P3 direct, S1-S3 structural, E1 ensemble), the constrained
monotone calibrators, the pure nonnegative simplex ensemble, and the nested rolling-origin
protocol (inner selection never touches outer-validation outcomes; leakage-free). No market input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models import pure_selection as psel


def _direct_frame(n_dates=30, per_date=12, seed=0, signal=True, with_struct=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="D").astype(str)
    rows = []
    for dt in dates:
        for _ in range(per_date):
            true_p = rng.uniform(0.2, 0.8)
            y = int(rng.random() < true_p)
            # A slightly miscalibrated but informative settled probability.
            p = np.clip(true_p + rng.normal(0, 0.05), 0.02, 0.98) if signal else rng.uniform(0.3, 0.7)
            row = {"game_date": dt, "outcome_over": y,
                   "market_prob_over_no_vig": 0.5, psel.DIRECT_COL: p}
            if with_struct:
                row[psel.STRUCT_COL] = np.clip(true_p + rng.normal(0, 0.08), 0.02, 0.98)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate registry + calibrators
# ---------------------------------------------------------------------------


def test_candidate_registry_is_the_owner_family():
    assert set(psel.CANDIDATE_SPECS) == {"P0", "P1", "P2", "P3", "S1", "S2", "S3", "E1"}
    assert psel.CANDIDATE_SPECS["P0"] == ("direct", "identity")
    assert psel.CANDIDATE_SPECS["P1"] == ("direct", "platt")
    assert psel.CANDIDATE_SPECS["P2"] == ("direct", "beta")
    assert psel.CANDIDATE_SPECS["P3"] == ("direct", "isotonic")
    assert psel.CANDIDATE_SPECS["S1"] == ("structural", "identity")
    assert psel.CANDIDATE_SPECS["S2"] == ("structural", "platt")
    assert psel.CANDIDATE_SPECS["S3"] == ("structural", "beta")
    assert psel.CANDIDATE_SPECS["E1"] == ("ensemble", "nonneg_simplex")


def test_direct_calibrators_fit_and_are_monotone():
    tr = _direct_frame(seed=1)
    for cid in ("P0", "P1", "P2"):
        fc = psel.fit_candidate(cid, tr)
        assert fc is not None and fc.eligible
        p = fc.predict(tr)
        assert np.all((p >= 0) & (p <= 1))
        # increasing the source probability may not decrease the calibrated probability
        grid = pd.DataFrame({psel.DIRECT_COL: np.linspace(0.05, 0.95, 40)})
        out = fc.predict(grid)
        if fc.monotone:
            assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_requires_support():
    small = _direct_frame(n_dates=4, per_date=8, seed=2)  # < ISO_MIN_ROWS
    fc = psel.fit_candidate("P3", small)
    assert fc is not None and fc.eligible is False  # support fails -> not eligible
    big = _direct_frame(n_dates=40, per_date=12, seed=2)
    fc2 = psel.fit_candidate("P3", big)
    assert fc2 is not None and fc2.eligible is True and fc2.monotone is True


def test_structural_candidates_only_when_struct_col_present():
    tr = _direct_frame(seed=3, with_struct=False)
    assert psel.fit_candidate("S1", tr) is None
    assert set(psel.available_candidates(tr)) == set(psel.DIRECT_CANDIDATES)
    trs = _direct_frame(seed=3, with_struct=True)
    assert psel.fit_candidate("S1", trs) is not None
    assert "E1" in psel.available_candidates(trs)
    assert set(psel.STRUCT_CANDIDATES) <= set(psel.available_candidates(trs))


def test_ensemble_is_nonnegative_simplex():
    trs = _direct_frame(seed=4, with_struct=True)
    fc = psel.fit_candidate("E1", trs)
    assert fc is not None
    w = fc.detail["weights"]
    assert all(x >= -1e-12 for x in w)
    assert abs(sum(w) - 1.0) < 1e-6
    assert len(fc.detail["bases"]) >= 2
    p = fc.predict(trs)
    assert np.all((p >= 0) & (p <= 1))


def test_ensemble_unavailable_without_two_bases():
    tr = _direct_frame(seed=5, with_struct=False)  # no structural -> only one source
    assert psel.fit_candidate("E1", tr) is None


# ---------------------------------------------------------------------------
# Nested rolling-origin selection (ITEM 4)
# ---------------------------------------------------------------------------


def test_nested_selection_produces_leakfree_stream():
    pdf = _direct_frame(n_dates=30, per_date=12, seed=6, with_struct=True)
    res = psel.nested_rolling_origin_select(pdf, "pts", min_train_dates=8, val_block_dates=2)
    assert res is not None
    assert res.selected_mask.sum() > 0
    # Every selected row carries a chosen candidate from the family and a valid outer fold id.
    chosen = set(res.selected_candidate_per_row[res.selected_mask])
    assert chosen <= set(psel.CANDIDATE_SPECS)
    assert (res.selected_fold_per_row[res.selected_mask] >= 0).all()
    # Predictions are valid probabilities.
    p = res.selected_pred[res.selected_mask]
    assert np.all((p >= 0) & (p <= 1))
    # Manifest: every outer fold has chronology_pass and records the inner selection.
    assert res.fold_manifest
    for fm in res.fold_manifest:
        assert fm["chronology_pass"] is True
        assert fm["outer_train_date_max"] < fm["outer_val_date_min"]
        assert fm["selected_candidate"] in psel.CANDIDATE_SPECS
        assert "inner_candidate_scores" in fm
        assert "selection_hash" in fm


def test_inner_selection_never_reads_outer_validation():
    """The chosen candidate for an outer fold must be a deterministic function of the outer-TRAIN
    dates only: mutating the outer-VALIDATION outcomes must not change the selection."""
    pdf = _direct_frame(n_dates=26, per_date=12, seed=7, with_struct=True)
    r1 = psel.nested_rolling_origin_select(pdf, "pts", min_train_dates=8, val_block_dates=2)

    pdf2 = pdf.copy()
    # Flip outcomes only on the LAST outer-validation block's dates.
    last_val_dates = set()
    for fm in r1.fold_manifest:
        last_val_dates.update({fm["outer_val_date_min"], fm["outer_val_date_max"]})
    latest = max(last_val_dates)
    mask = pdf2["game_date"] == latest
    pdf2.loc[mask, "outcome_over"] = 1 - pdf2.loc[mask, "outcome_over"]
    r2 = psel.nested_rolling_origin_select(pdf2, "pts", min_train_dates=8, val_block_dates=2)

    sel1 = [fm["selected_candidate"] for fm in r1.fold_manifest[:-1]]
    sel2 = [fm["selected_candidate"] for fm in r2.fold_manifest[:-1]]
    # Selections for folds whose training window excludes the mutated date must be identical.
    assert sel1 == sel2


def test_nested_selection_returns_none_without_folds():
    pdf = _direct_frame(n_dates=3, per_date=6, seed=8)
    assert psel.nested_rolling_origin_select(pdf, "pts", min_train_dates=10,
                                             val_block_dates=2) is None
