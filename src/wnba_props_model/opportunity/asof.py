"""Strict point-in-time as-of joining for Opportunity V2.

A single hardened ``strict_asof_join`` is the ONLY sanctioned way to attach a historical snapshot to
a prediction row. It guarantees the matched snapshot was available at or before the row's
``prediction_cutoff_utc`` and never changes the left row count.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class TemporalLeakageError(RuntimeError):
    """Raised when a join or feature frame would expose post-cutoff information."""


_ROWID = "__opp_asof_rowid__"


def strict_asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: list[str],
    left_cutoff_col: str = "prediction_cutoff_utc",
    right_available_col: str = "available_at_utc",
    suffix: str,
    required: bool = False,
    max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Backward as-of join: attach the latest ``right`` row available at/<= each left cutoff.

    Contract (section 10):
      * left row order + count are preserved exactly;
      * matches with ``available > cutoff`` are rejected (never leak the future);
      * a match exactly at the cutoff is allowed;
      * matches older than ``max_age`` are nulled;
      * adds ``{suffix}_matched`` / ``{suffix}_available_at_utc`` / ``{suffix}_age_hours``;
      * ``required=True`` raises if any row lacks a valid match;
      * right-side duplicates can never inflate the row count.
    """
    if left_cutoff_col not in left.columns:
        raise TemporalLeakageError(f"strict_asof_join: left missing cutoff column {left_cutoff_col!r}")
    left_work = left.copy()
    left_work[_ROWID] = np.arange(len(left_work), dtype=np.int64)

    lc = pd.to_datetime(left_work[left_cutoff_col], utc=True, errors="coerce")
    if bool(lc.isna().any()):
        n = int(lc.isna().sum())
        raise TemporalLeakageError(f"strict_asof_join: {n} left row(s) have a null/invalid {left_cutoff_col}")
    # Pin nanosecond resolution so merge_asof keys match regardless of source precision (pandas>=3).
    left_work[left_cutoff_col] = lc.astype("datetime64[ns, UTC]")

    matched_col = f"{suffix}_matched"
    avail_col = f"{suffix}_available_at_utc"
    age_col = f"{suffix}_age_hours"

    if right is None or len(right) == 0:
        out = left_work.copy()
        out[matched_col] = False
        out[avail_col] = pd.NaT
        out[age_col] = np.nan
        if required:
            raise TemporalLeakageError(f"strict_asof_join[{suffix}]: right is empty but required=True")
        return _restore(out, left, left_work, _ROWID)

    right_work = right.copy()
    if right_available_col not in right_work.columns:
        raise TemporalLeakageError(f"strict_asof_join: right missing {right_available_col!r}")
    ra = pd.to_datetime(right_work[right_available_col], utc=True, errors="coerce")
    mask = ra.notna()
    right_work = right_work.loc[mask].copy()
    # Assign the tz-AWARE series (not ``.values``, which would strip the tz and break merge_asof),
    # pinned to nanosecond resolution to match the left key.
    right_work[right_available_col] = ra.loc[mask].astype("datetime64[ns, UTC]")
    for key in by:
        if key not in left_work.columns or key not in right_work.columns:
            raise TemporalLeakageError(f"strict_asof_join: join key {key!r} missing on a side")

    # Suffix right columns (except keys + available col) to avoid clobbering left columns.
    protected = set(by) | {right_available_col}
    rename = {c: f"{c}_{suffix}" for c in right_work.columns if c not in protected}
    right_work = right_work.rename(columns=rename)

    left_sorted = left_work.sort_values(left_cutoff_col, kind="mergesort")
    right_sorted = right_work.sort_values(right_available_col, kind="mergesort")

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_cutoff_col,
        right_on=right_available_col,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )

    # Enforce availability <= cutoff (merge_asof backward already guarantees this, but we assert).
    avail = pd.to_datetime(merged[right_available_col], utc=True, errors="coerce")
    cutoff = pd.to_datetime(merged[left_cutoff_col], utc=True, errors="coerce")
    future = avail.notna() & (avail > cutoff)
    if bool(future.any()):
        raise TemporalLeakageError(
            f"strict_asof_join[{suffix}]: {int(future.sum())} matched row(s) have "
            f"available_at > cutoff (future leak)")

    age_hours = (cutoff - avail).dt.total_seconds() / 3600.0
    matched = avail.notna()
    if max_age is not None:
        too_old = matched & (age_hours > (max_age.total_seconds() / 3600.0))
        if bool(too_old.any()):
            # Null out stale matches (all suffixed right columns + availability/age).
            stale_idx = merged.index[too_old]
            right_cols = [f"{c}_{suffix}" for c in rename.values()] if False else list(rename.values())
            for c in right_cols:
                if c in merged.columns:
                    merged.loc[stale_idx, c] = np.nan
            merged.loc[stale_idx, right_available_col] = pd.NaT
            avail = pd.to_datetime(merged[right_available_col], utc=True, errors="coerce")
            age_hours = (cutoff - avail).dt.total_seconds() / 3600.0
            matched = avail.notna()

    merged[matched_col] = matched.to_numpy()
    merged[avail_col] = avail.to_numpy()
    merged[age_col] = age_hours.to_numpy()
    if right_available_col not in left.columns and right_available_col in merged.columns:
        merged = merged.drop(columns=[right_available_col])

    if required and not bool(matched.all()):
        n = int((~matched).sum())
        raise TemporalLeakageError(f"strict_asof_join[{suffix}]: {n} row(s) lack a valid match (required=True)")

    out = _restore(merged, left, left_work, _ROWID)
    if len(out) != len(left):
        raise TemporalLeakageError(
            f"strict_asof_join[{suffix}]: row count changed {len(left)} -> {len(out)} "
            "(right-side duplicate identities?)")
    return out


def _restore(merged: pd.DataFrame, left: pd.DataFrame, left_work: pd.DataFrame, rowid: str) -> pd.DataFrame:
    out = merged.sort_values(rowid, kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=[rowid])
    return out


def assert_feature_time_purity(
    frame: pd.DataFrame,
    *,
    cutoff_col: str,
    source_timestamp_columns: Sequence[str],
) -> None:
    """Raise ``TemporalLeakageError`` if any source timestamp exceeds the row's prediction cutoff."""
    if cutoff_col not in frame.columns:
        raise TemporalLeakageError(f"assert_feature_time_purity: missing cutoff column {cutoff_col!r}")
    cutoff = pd.to_datetime(frame[cutoff_col], utc=True, errors="coerce")
    if bool(cutoff.isna().any()):
        raise TemporalLeakageError("assert_feature_time_purity: null prediction cutoff(s) present")
    offenders: dict[str, list[int]] = {}
    for col in source_timestamp_columns:
        if col not in frame.columns:
            continue
        ts = pd.to_datetime(frame[col], utc=True, errors="coerce")
        bad = ts.notna() & (ts > cutoff)
        if bool(bad.any()):
            offenders[col] = frame.index[bad].tolist()[:20]
    if offenders:
        raise TemporalLeakageError(
            f"assert_feature_time_purity: future source timestamps in {list(offenders)}; "
            f"sample offending row ids: { {k: v[:5] for k, v in offenders.items()} }")
