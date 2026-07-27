"""Temporal-purity auditing for Opportunity V2 feature frames.

``audit_temporal_purity`` returns a structured result (never silently passes) quantifying, per source
timestamp column, how many rows would leak future information relative to ``prediction_cutoff_utc``.
The OOF workflow fails when ``passed`` is False.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .contracts import forbidden_market_columns


@dataclass
class TemporalAuditResult:
    passed: bool
    row_count: int
    violation_count: int
    violations_by_column: dict[str, int] = field(default_factory=dict)
    max_future_seconds_by_column: dict[str, float] = field(default_factory=dict)
    sampled_violations: list[dict[str, Any]] = field(default_factory=list)
    forbidden_market_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "row_count": int(self.row_count),
            "violation_count": int(self.violation_count),
            "violations_by_column": {k: int(v) for k, v in self.violations_by_column.items()},
            "max_future_seconds_by_column": {
                k: float(v) for k, v in self.max_future_seconds_by_column.items()
            },
            "sampled_violations": self.sampled_violations,
            "forbidden_market_columns": list(self.forbidden_market_columns),
        }


def audit_temporal_purity(
    frame: pd.DataFrame,
    cutoff_col: str,
    source_timestamp_columns: Sequence[str],
    *,
    sample_limit: int = 25,
    feature_columns: Sequence[str] | None = None,
) -> TemporalAuditResult:
    """Audit a feature frame for future-timestamp leaks and forbidden market inputs.

    A row is a violation for column ``c`` when ``frame[c] > frame[cutoff_col]`` (both parsed UTC).
    When ``feature_columns`` is provided, any forbidden market column among them is reported and
    forces ``passed=False`` (market signals may never be model inputs).
    """
    if cutoff_col not in frame.columns:
        return TemporalAuditResult(
            passed=False, row_count=len(frame), violation_count=len(frame),
            violations_by_column={cutoff_col: len(frame)},
        )
    cutoff = pd.to_datetime(frame[cutoff_col], utc=True, errors="coerce")
    null_cutoffs = int(cutoff.isna().sum())

    violations_by_column: dict[str, int] = {}
    max_future_by_column: dict[str, float] = {}
    sampled: list[dict[str, Any]] = []
    total_violation_rows = pd.Series(False, index=frame.index)

    for col in source_timestamp_columns:
        if col not in frame.columns:
            continue
        ts = pd.to_datetime(frame[col], utc=True, errors="coerce")
        delta = (ts - cutoff).dt.total_seconds()
        bad = ts.notna() & cutoff.notna() & (delta > 0)
        n_bad = int(bad.sum())
        if n_bad:
            violations_by_column[col] = n_bad
            max_future_by_column[col] = float(delta[bad].max())
            total_violation_rows = total_violation_rows | bad
            for ridx in frame.index[bad][:sample_limit]:
                sampled.append({
                    "row": int(frame.index.get_loc(ridx)),
                    "column": col,
                    "cutoff_utc": str(cutoff.loc[ridx]),
                    "source_utc": str(ts.loc[ridx]),
                    "future_seconds": float(delta.loc[ridx]),
                })

    forbidden = forbidden_market_columns(feature_columns) if feature_columns else []

    violation_count = int(total_violation_rows.sum()) + null_cutoffs
    if null_cutoffs:
        violations_by_column[cutoff_col] = null_cutoffs
    passed = (violation_count == 0) and (len(forbidden) == 0)
    return TemporalAuditResult(
        passed=passed,
        row_count=len(frame),
        violation_count=violation_count,
        violations_by_column=violations_by_column,
        max_future_seconds_by_column=max_future_by_column,
        sampled_violations=sampled[:sample_limit],
        forbidden_market_columns=list(forbidden),
    )
