"""Regression tests for parse_minutes and parse_minutes_flag.

Every row that BDL has been observed to return – or could plausibly return –
must be covered.  The goal is a zero-crash guarantee: no matter what the
API sends for the ``min`` field, the parser must return a valid float and a
well-defined audit flag rather than raising.
"""
from __future__ import annotations

import math

import pytest

from wnba_props_model.data.normalize import (
    _parse_minutes_internal,
    flatten_player_stat_row,
    parse_minutes,
    parse_minutes_flag,
)


# ---------------------------------------------------------------------------
# Parametrized happy-path and edge-case table
# ---------------------------------------------------------------------------

# (input, expected_minutes, expected_flag)
CASES: list[tuple[object, float, str | None]] = [
    # ---- clean numeric ----
    (12, 12.0, None),
    (12.5, 12.5, None),
    (0, 0.0, None),
    (0.0, 0.0, None),
    (34, 34.0, None),
    # ---- clean string – decimal ----
    ("12", 12.0, None),
    ("12.5", 12.5, None),
    ("0", 0.0, None),
    ("36.8", 36.8, None),
    # ---- MM:SS format ----
    ("12:34", 12 + 34 / 60, None),
    ("30:00", 30.0, None),
    ("0:00", 0.0, None),
    ("9:59", 9 + 59 / 60, None),
    ("40:12", 40 + 12 / 60, None),
    # MM:SS with missing segment (defensive)
    (":30", 0 + 30 / 60, None),
    ("30:", 30.0, None),
    # ---- null / missing ----
    (None, 0.0, "null"),
    (float("nan"), 0.0, "null"),
    # ---- empty string ----
    ("", 0.0, "empty"),
    ("   ", 0.0, "empty"),
    # ---- non-playing sentinels (exact) ----
    ("--", 0.0, "non_playing"),
    ("-", 0.0, "non_playing"),
    ("DNP", 0.0, "non_playing"),
    ("dnp", 0.0, "non_playing"),
    ("DNP-CD", 0.0, "non_playing"),
    ("DNP-Coach's Decision", 0.0, "non_playing"),
    ("Did Not Play", 0.0, "non_playing"),
    ("did not play", 0.0, "non_playing"),
    ("INACTIVE", 0.0, "non_playing"),
    ("Inactive", 0.0, "non_playing"),
    ("inactive", 0.0, "non_playing"),
    ("OUT", 0.0, "non_playing"),
    ("out", 0.0, "non_playing"),
    ("Scratch", 0.0, "non_playing"),
    ("DND", 0.0, "non_playing"),
    ("Did Not Dress", 0.0, "non_playing"),
    ("Not With Team", 0.0, "non_playing"),
    ("NWT", 0.0, "non_playing"),
    ("Suspension", 0.0, "non_playing"),
    ("NA", 0.0, "non_playing"),
    ("N/A", 0.0, "non_playing"),
    ("n/a", 0.0, "non_playing"),
    # ---- whitespace around sentinels ----
    ("  DNP  ", 0.0, "non_playing"),
    ("  --  ", 0.0, "non_playing"),
    # ---- unrecognised garbage → parse_error ----
    ("???", 0.0, "parse_error"),
    ("abc", 0.0, "parse_error"),
    ("12:xx", 0.0, "parse_error"),
]


@pytest.mark.parametrize("value,expected_min,expected_flag", CASES)
def test_parse_minutes_value(value: object, expected_min: float, expected_flag: str | None) -> None:
    result = parse_minutes(value)
    assert isinstance(result, float), f"Expected float, got {type(result)} for {value!r}"
    assert math.isfinite(result), f"Expected finite float, got {result} for {value!r}"
    assert abs(result - expected_min) < 1e-9, (
        f"parse_minutes({value!r}) = {result}, expected {expected_min}"
    )


@pytest.mark.parametrize("value,expected_min,expected_flag", CASES)
def test_parse_minutes_flag(value: object, expected_min: float, expected_flag: str | None) -> None:
    result = parse_minutes_flag(value)
    assert result == expected_flag, (
        f"parse_minutes_flag({value!r}) = {result!r}, expected {expected_flag!r}"
    )


# ---------------------------------------------------------------------------
# Internal consistency: _parse_minutes_internal must always agree with the
# two public helpers.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected_min,expected_flag", CASES)
def test_internal_consistency(value: object, expected_min: float, expected_flag: str | None) -> None:
    minutes, flag = _parse_minutes_internal(value)
    assert minutes == parse_minutes(value)
    assert flag == parse_minutes_flag(value)


# ---------------------------------------------------------------------------
# Never raise: no input should crash the parser.
# ---------------------------------------------------------------------------

CRASH_CANDIDATES = [
    object(),
    b"30:00",
    [],
    {},
    True,
    False,
    complex(1, 2),
]


@pytest.mark.parametrize("value", CRASH_CANDIDATES)
def test_parse_minutes_never_raises(value: object) -> None:
    try:
        result = parse_minutes(value)
        assert isinstance(result, float)
        assert math.isfinite(result)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"parse_minutes raised {type(exc).__name__} for {value!r}: {exc}")


# ---------------------------------------------------------------------------
# flatten_player_stat_row audit fields
# ---------------------------------------------------------------------------

_MINIMAL_ROW = {
    "player": {"id": 1, "first_name": "Alyssa", "last_name": "Thomas"},
    "team": {"id": 10, "abbreviation": "CON"},
    "game": {"id": 100, "date": "2024-05-15", "season": 2024},
    "pts": 12, "reb": 5, "ast": 3, "fg3m": 1,
    "turnover": 2, "stl": 1, "blk": 0,
    "oreb": 1, "dreb": 4, "fga": 10, "fta": 3, "pf": 2,
    "plus_minus": 5,
}


def _row_with_min(min_value: object) -> dict:
    return {**_MINIMAL_ROW, "min": min_value}


def test_flatten_clean_minutes() -> None:
    r = flatten_player_stat_row(_row_with_min("32:15"))
    assert abs(r["minutes"] - (32 + 15 / 60)) < 1e-9
    assert r["minutes_flag"] is None
    assert r["minutes_raw"] == "32:15"


def test_flatten_dash_minutes_produces_flag() -> None:
    r = flatten_player_stat_row(_row_with_min("--"))
    assert r["minutes"] == 0.0
    assert r["minutes_flag"] == "non_playing"
    assert r["minutes_raw"] == "--"


def test_flatten_dnp_minutes_produces_flag() -> None:
    r = flatten_player_stat_row(_row_with_min("DNP"))
    assert r["minutes"] == 0.0
    assert r["minutes_flag"] == "non_playing"


def test_flatten_none_minutes_produces_null_flag() -> None:
    r = flatten_player_stat_row(_row_with_min(None))
    assert r["minutes"] == 0.0
    assert r["minutes_flag"] == "null"
    assert r["minutes_raw"] is None


def test_flatten_empty_minutes_produces_empty_flag() -> None:
    r = flatten_player_stat_row(_row_with_min(""))
    assert r["minutes"] == 0.0
    assert r["minutes_flag"] == "empty"


def test_flatten_minutes_raw_always_present() -> None:
    """minutes_raw and minutes_flag columns must always be in the output dict."""
    for min_val in ["--", "DNP", None, "32:00", "30", ""]:
        r = flatten_player_stat_row(_row_with_min(min_val))
        assert "minutes_raw" in r, f"minutes_raw missing for min={min_val!r}"
        assert "minutes_flag" in r, f"minutes_flag missing for min={min_val!r}"


def test_flatten_numeric_minutes_no_flag() -> None:
    r = flatten_player_stat_row(_row_with_min(28))
    assert r["minutes"] == 28.0
    assert r["minutes_flag"] is None
    assert r["minutes_raw"] == "28"
