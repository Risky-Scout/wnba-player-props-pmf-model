"""Leakage-safety tests for grouped expanding-window rolling-origin CV.

Reproduces the prior leave-block-out folds (which trained on future date blocks) and REQUIRES
the chronology check to flag them, and verifies the corrected expanding-window folds always
satisfy max(train_date) < min(validation_date).
"""
from __future__ import annotations

import numpy as np

from wnba_props_model.evaluation.rolling_origin import (
    Fold,
    all_chronology_pass,
    expanding_window_folds,
    fold_manifest,
    nested_select,
)

DATES = [f"2026-05-{d:02d}" for d in range(1, 29)]  # 28 game dates


def test_expanding_window_is_leakage_safe():
    folds = expanding_window_folds(DATES, min_train_dates=10, val_block_dates=3)
    assert folds, "expected at least one fold"
    for f in folds:
        assert f.train_date_max < f.val_date_min, f"fold {f.fold_id} trains on/after its validation"
    assert all_chronology_pass(folds)


def test_expanding_window_never_trains_on_future():
    folds = expanding_window_folds(DATES, min_train_dates=10, val_block_dates=3)
    for f in folds:
        assert max(f.train_dates) < min(f.val_dates)
        # no training date may fall inside or after the validation block
        assert not (set(f.train_dates) & set(f.val_dates))
        assert all(t < min(f.val_dates) for t in f.train_dates)


def _leave_block_out_folds(dates, k=5):
    """The PRIOR (buggy) scheme: k contiguous blocks; validate one, train on ALL others."""
    uniq = sorted(set(dates))
    blocks = [list(c) for c in np.array_split(uniq, k)]
    folds = []
    for i, val in enumerate(blocks):
        train = [d for j, b in enumerate(blocks) if j != i for d in b]
        folds.append(Fold(fold_id=i, train_dates=tuple(train), val_dates=tuple(val)))
    return folds


def test_leave_block_out_is_flagged_as_leaky():
    leaky = _leave_block_out_folds(DATES, k=5)
    # Fold 0 validates the EARLIEST block but trains on later blocks -> chronology must FAIL.
    assert leaky[0].chronology_pass is False
    # The corrected aggregate check must reject the leaky scheme.
    assert all_chronology_pass(leaky) is False


def test_fold_manifest_reports_chronology():
    folds = expanding_window_folds(DATES, min_train_dates=10, val_block_dates=3)
    man = fold_manifest(folds, lambda ds: len(ds))
    assert all(m["chronology_pass"] for m in man)
    assert all(m["train_date_max"] < m["validation_date_min"] for m in man)


def test_nested_select_prefers_lower_inner_loss():
    # score_fn returns lower loss for param==1; nested_select must pick it.
    def score_fn(param, itr, iva):
        return abs(param - 1) + 0.01
    chosen = nested_select(DATES[:20], param_grid=[0, 1, 2], score_fn=score_fn,
                           min_train_dates=8, val_block_dates=3)
    assert chosen == 1
