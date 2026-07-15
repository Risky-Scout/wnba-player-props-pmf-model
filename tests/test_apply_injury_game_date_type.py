"""Regression test for run 29432147967.

Fatal error:
  ArrowTypeError: ("Expected bytes, got a 'Timestamp' object",
                   'Conversion failed for column game_date with type object')
  at apply_injury_updates.py:469 in main — pmfs_after.to_parquet(slate_path, ...)

Root cause: pd.concat([base_pmfs, new_combo_pmfs]) produces an object-dtype
game_date column with mixed str + pd.Timestamp values when base_pmfs carries
string dates and _build_combo_pmf_rows returns Timestamps. PyArrow cannot
convert a Timestamp inside an object column to bytes.

Fix: coerce game_date to str before writing the parquet.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def _make_pmf_df(game_dates) -> pd.DataFrame:
    """Build a minimal PMF-like DataFrame with the given game_date values."""
    return pd.DataFrame({
        "player_id":   [1, 2, 3],
        "game_id":     [24931, 24931, 24932],
        "stat":        ["pts", "reb", "pts"],
        "game_date":   game_dates,
        "pmf_json":    ['{"5":1.0}', '{"3":1.0}', '{"10":1.0}'],
        "pmf_mean":    [5.0, 3.0, 10.0],
    })


def test_mixed_game_date_types_fail_parquet_write():
    """Confirm the root cause: mixed str+Timestamp causes ArrowTypeError."""
    import pyarrow as pa

    mixed = _make_pmf_df([
        "2026-07-15",
        pd.Timestamp("2026-07-15"),
        "2026-07-15",
    ])
    assert mixed["game_date"].dtype == object

    buf = io.BytesIO()
    with pytest.raises(Exception) as exc_info:
        mixed.to_parquet(buf, index=False)
    err = str(exc_info.value)
    assert "game_date" in err or "Timestamp" in err or "bytes" in err.lower(), (
        f"Expected ArrowTypeError about game_date / Timestamp, got: {err}"
    )


def test_coerce_game_date_to_str_fixes_parquet_write():
    """After coercion, mixed str+Timestamp game_date writes without error."""
    mixed = _make_pmf_df([
        "2026-07-15",
        pd.Timestamp("2026-07-15"),
        "2026-07-15",
    ])

    # Apply the fix from apply_injury_updates.py
    fixed = mixed.copy()
    fixed["game_date"] = fixed["game_date"].apply(
        lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v) if v is not None else v
    )

    buf = io.BytesIO()
    fixed.to_parquet(buf, index=False)  # must not raise
    buf.seek(0)
    result = pd.read_parquet(buf)
    assert result["game_date"].tolist() == ["2026-07-15", "2026-07-15", "2026-07-15"]


def test_coerce_preserves_str_game_dates():
    """String game_dates pass through the coercion unchanged."""
    df = _make_pmf_df(["2026-07-15", "2026-07-15", "2026-07-16"])
    fixed = df.copy()
    fixed["game_date"] = fixed["game_date"].apply(
        lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v) if v is not None else v
    )
    assert fixed["game_date"].tolist() == ["2026-07-15", "2026-07-15", "2026-07-16"]


def test_coerce_handles_none_game_date():
    """None values are preserved through coercion (not converted to 'None')."""
    import numpy as np
    df = _make_pmf_df(["2026-07-15", None, pd.Timestamp("2026-07-15")])
    fixed = df.copy()
    fixed["game_date"] = fixed["game_date"].apply(
        lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v) if v is not None else v
    )
    # Timestamp becomes string, string stays string; None becomes NaN in pandas
    assert fixed["game_date"].iloc[0] == "2026-07-15"
    assert pd.isna(fixed["game_date"].iloc[1])  # None → NaN in pandas object col
    assert fixed["game_date"].iloc[2] == "2026-07-15"
