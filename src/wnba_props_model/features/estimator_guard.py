"""Fail-closed estimator-input guard (strengthened Stage 9, Section 6).

Every future participation / minutes / direct-stat / combination model entry point MUST call
``guard_estimator_frame`` instead of implementing its own column filter. The explicit approved
allowlist is the PRIMARY control; a secondary forbidden-name alarm is a defense in depth.

The guard fails closed when: an approved feature is missing; an unexpected column is present;
column order differs; the feature-schema hash differs; duplicate column names exist; nonnumeric
values enter the numeric block; infinite values exist; an identifier enters the estimator
matrix; or any forbidden / target-like field appears.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from wnba_props_model.models.prop_feature_policy import feature_schema_hash

# Secondary defense (alarm): target/market/outcome/postgame terms. NOT an overbroad substring
# rule that silently rejects legitimate features — the registry allowlist is authoritative;
# any hit here is treated as a hard error because such a column must never reach the estimator.
FORBIDDEN_PATTERN = re.compile(
    r"(actual_|(^|_)target(_|$)|outcome|settlement|(^|_)result(_|$)|final_score|did_play|"
    r"actual_minutes|quote|odds|price|sportsbook|bookmaker|market_prob|market_line|no_vig|"
    r"(^|_)line(_|$)|over_under|(^|_)closing(_|$)|postgame|(^|_)future(_|$))", re.I)

# Identifiers / keys / audit fields must remain OUTSIDE the numeric estimator matrix.
IDENTIFIER_COLS = frozenset({
    "game_id", "player_id", "game_date", "season", "player_name", "team_id",
    "team_abbreviation", "opponent_team_id", "opponent_team_abbreviation", "position",
    "home_away", "scheduled_tip_utc", "prediction_cutoff_utc", "feature_available_utc",
})


class EstimatorGuardError(ValueError):
    """Raised when a candidate estimator frame violates the fail-closed contract."""


def assert_no_forbidden_names(cols) -> None:
    bad = [c for c in cols if FORBIDDEN_PATTERN.search(str(c))]
    if bad:
        raise EstimatorGuardError(f"forbidden/target-like columns in estimator input: {bad}")


def assert_no_identifiers(cols) -> None:
    ids = [c for c in cols if c in IDENTIFIER_COLS]
    if ids:
        raise EstimatorGuardError(f"identifier columns must not enter the estimator matrix: {ids}")


def assert_estimator_columns(cols: list[str], approved: list[str], expected_hash: str) -> None:
    cols = list(cols)
    if len(cols) != len(set(cols)):
        dups = sorted({c for c in cols if cols.count(c) > 1})
        raise EstimatorGuardError(f"duplicate estimator column names: {dups}")
    if cols != list(approved):
        missing = [c for c in approved if c not in cols]
        unexpected = [c for c in cols if c not in set(approved)]
        raise EstimatorGuardError(
            f"estimator columns must equal the approved allowlist in order. "
            f"missing={missing} unexpected={unexpected} order_mismatch={cols != list(approved)}")
    got = feature_schema_hash(cols)
    if got != expected_hash:
        raise EstimatorGuardError(f"feature-schema hash mismatch: expected {expected_hash} got {got}")
    assert_no_forbidden_names(cols)
    assert_no_identifiers(cols)


def guard_estimator_frame(df: pd.DataFrame, approved: list[str], expected_hash: str) -> np.ndarray:
    """Validate a candidate estimator frame and return the numeric matrix. Fail-closed."""
    assert_estimator_columns(list(df.columns), approved, expected_hash)
    block = df[approved]
    non_numeric = [c for c in approved if not pd.api.types.is_numeric_dtype(block[c])]
    if non_numeric:
        raise EstimatorGuardError(f"nonnumeric columns in numeric estimator block: {non_numeric}")
    X = block.to_numpy(dtype=float)
    if np.isinf(X).any():
        raise EstimatorGuardError("infinite values present in estimator matrix")
    return X
