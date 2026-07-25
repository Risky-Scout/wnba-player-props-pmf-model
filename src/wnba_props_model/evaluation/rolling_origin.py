"""Grouped expanding-window rolling-origin cross-validation (leakage-safe).

Replaces the earlier leave-block-out cross-fit (which trained on future date blocks). The
hard invariant for EVERY outer fold is:

    max(training game_date) < min(validation game_date)

All rows on the same game_date stay together (grouped). The earliest dates form the initial
training window and are never validated with later observations. Supports nested selection:
hyperparameters are chosen with an inner expanding-window run over the outer training dates
only, then the chosen candidate is refit on the full outer training period and scored once
on the untouched outer validation block.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: tuple
    val_dates: tuple

    @property
    def train_date_min(self) -> str:
        return str(min(self.train_dates))

    @property
    def train_date_max(self) -> str:
        return str(max(self.train_dates))

    @property
    def val_date_min(self) -> str:
        return str(min(self.val_dates))

    @property
    def val_date_max(self) -> str:
        return str(max(self.val_dates))

    @property
    def chronology_pass(self) -> bool:
        return self.train_date_max < self.val_date_min


def expanding_window_folds(
    dates: Sequence,
    *,
    min_train_dates: int,
    val_block_dates: int = 1,
) -> list[Fold]:
    """Expanding-window outer folds over the sorted unique game dates.

    Fold i trains on ALL dates strictly before its validation block and validates on the next
    ``val_block_dates`` dates. The first ``min_train_dates`` dates seed the initial training
    window (never validated). Every returned fold satisfies max(train) < min(val).
    """
    uniq = sorted({str(d) for d in dates})
    if len(uniq) <= min_train_dates:
        return []
    folds: list[Fold] = []
    start = min_train_dates
    fid = 0
    while start < len(uniq):
        val = uniq[start:start + val_block_dates]
        train = uniq[:start]
        if not val:
            break
        f = Fold(fold_id=fid, train_dates=tuple(train), val_dates=tuple(val))
        assert f.chronology_pass, f"chronology violation in fold {fid}"
        folds.append(f)
        start += val_block_dates
        fid += 1
    return folds


def _date_hash(dates: Sequence) -> str:
    return hashlib.sha256("|".join(sorted(str(d) for d in dates)).encode()).hexdigest()[:16]


def fold_manifest(folds: list[Fold], date_to_rows: Callable[[set], int]) -> list[dict]:
    """Machine-verifiable manifest; ``date_to_rows`` maps a set of dates to a row count."""
    out = []
    for f in folds:
        out.append({
            "fold_id": f.fold_id,
            "train_date_min": f.train_date_min, "train_date_max": f.train_date_max,
            "validation_date_min": f.val_date_min, "validation_date_max": f.val_date_max,
            "training_rows": int(date_to_rows(set(f.train_dates))),
            "validation_rows": int(date_to_rows(set(f.val_dates))),
            "training_date_hash": _date_hash(f.train_dates),
            "validation_date_hash": _date_hash(f.val_dates),
            "chronology_pass": bool(f.chronology_pass),
        })
    return out


def all_chronology_pass(folds: list[Fold]) -> bool:
    return bool(folds) and all(f.chronology_pass for f in folds)


def nested_select(
    outer_train_dates: Sequence,
    *,
    param_grid: list,
    score_fn: Callable[[object, set, set], float],
    min_train_dates: int,
    val_block_dates: int = 1,
) -> object:
    """Choose the param minimizing summed inner expanding-window validation loss.

    ``score_fn(param, inner_train_dates, inner_val_dates)`` returns a loss (lower better),
    fitting ONLY on inner_train_dates and scoring inner_val_dates. Inner folds are expanding
    windows over the outer training dates (never touching the outer validation block).
    """
    inner = expanding_window_folds(outer_train_dates, min_train_dates=min_train_dates,
                                   val_block_dates=val_block_dates)
    if not inner:
        return param_grid[0]
    best, best_loss = param_grid[0], np.inf
    for p in param_grid:
        loss = 0.0
        n = 0
        for f in inner:
            v = score_fn(p, set(f.train_dates), set(f.val_dates))
            if np.isfinite(v):
                loss += v
                n += 1
        loss = loss / n if n else np.inf
        if loss < best_loss:
            best_loss, best = loss, p
    return best
