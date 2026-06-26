"""Elite player projection gate tests (Blueprint R1.4).

These tests refuse to ship if:
  1. A'ja Wilson's mean pts projection < 20 (feature pollution likely persists)
  2. Any FORBIDDEN_MODEL_FEATURES appear in the feature manifest
  3. Any model feature has cross-player std < 0.05 (near-zero-variance / constant)

All three tests are skipped when the required data files are not present (e.g.
during pure unit-test runs where only source code is checked out).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_MANIFEST_PATH = Path("data/processed/feature_schema_manifest.json")
_WIDE_PATH = Path("data/processed/wnba_player_game_features_wide.parquet")
_PMF_PATH = Path("data/model_outputs/stage4_baseline/player_stat_pmfs.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        pytest.skip(f"Manifest not found: {_MANIFEST_PATH}")
    return json.loads(_MANIFEST_PATH.read_text())


@pytest.fixture
def wide_df() -> pd.DataFrame:
    if not _WIDE_PATH.exists():
        pytest.skip(f"Wide features not found: {_WIDE_PATH}")
    return pd.read_parquet(_WIDE_PATH)


@pytest.fixture
def pmf_long() -> pd.DataFrame:
    if not _PMF_PATH.exists():
        pytest.skip(f"PMF output not found: {_PMF_PATH}")
    return pd.read_parquet(_PMF_PATH)


# ---------------------------------------------------------------------------
# R1.4-A: Elite player projection gate
# ---------------------------------------------------------------------------

def test_aja_wilson_pts_projection(pmf_long: pd.DataFrame) -> None:
    """A'ja Wilson must project >= 20 pts in seasons where she's elite (2024+).

    Uses "A'ja" to avoid matching other Wilsons (e.g. Alex Wilson), filters
    to did_play=True to exclude DNP/garbage-time rows, and restricts to
    seasons >= 2024 where she averages 22-27 ppg — making a sub-20 projection
    a reliable signal of feature pollution or calibration regression.
    """
    aja = pmf_long[
        pmf_long["player_name"].str.contains("A'ja", case=False, na=False)
        & (pmf_long["stat"] == "pts")
    ]
    if aja.empty:
        pytest.skip("A'ja Wilson not in current dataset")

    # Exclude DNP games — zero-minute appearances produce near-zero PMF means
    # and are not calibration-relevant predictions.
    if "did_play" in aja.columns:
        aja = aja[aja["did_play"] == True]  # noqa: E712

    # Restrict to elite-era seasons (2024+) where she averages 22-27 ppg.
    # Including 2019-2023 early career (14-22 ppg) dilutes the gate signal.
    if "season" in aja.columns:
        aja = aja[aja["season"] >= 2024]

    if aja.empty:
        pytest.skip("A'ja Wilson has no 2024+ played rows in current dataset")

    mean_proj = float(aja["pmf_mean"].mean())
    assert mean_proj >= 20.0, (
        f"A'ja Wilson pts projection {mean_proj:.1f} < 20.0 (2024+ seasons, played only) — "
        f"feature pollution likely persists (check manifest for rotation_minutes_* "
        f"or opp_*_vs_*_allowed_l5)"
    )


# ---------------------------------------------------------------------------
# R1.4-B: No forbidden features in manifest
# ---------------------------------------------------------------------------

def test_no_forbidden_features_in_manifest(manifest: dict) -> None:
    """Feature manifest must not contain any FORBIDDEN_MODEL_FEATURES."""
    from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

    cols: list[str] = manifest.get("model_feature_columns", [])
    overlap = sorted(set(cols) & FORBIDDEN_MODEL_FEATURES)
    assert not overlap, (
        f"Forbidden features found in manifest ({len(overlap)}): {overlap}"
    )


# ---------------------------------------------------------------------------
# R1.4-C: No near-zero-variance features in manifest
# ---------------------------------------------------------------------------

def test_no_near_zero_variance_features(manifest: dict, wide_df: pd.DataFrame) -> None:
    """No model feature may have cross-player std < 0.05."""
    cols: list[str] = manifest.get("model_feature_columns", [])
    low_var: list[str] = []
    for c in cols:
        if c not in wide_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(wide_df[c]):
            continue
        std = float(wide_df[c].std(skipna=True))
        if std < 0.05:
            low_var.append(f"{c} (std={std:.4f})")

    assert not low_var, (
        f"Near-zero-variance features in manifest ({len(low_var)}): {low_var}"
    )
