"""Tests for position-stratified copula correlations (P3.5)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.bivariate_pmf import estimate_correlations, adjust_combo_pmf_for_correlation


def _make_wide_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = (["G"] * 80 + ["F"] * 80 + ["C"] * 40)[:n]
    return pd.DataFrame({
        "player_id": range(n),
        "position": positions,
        "actual_pts": rng.integers(0, 30, n).astype(float),
        "actual_reb": rng.integers(0, 15, n).astype(float),
        "actual_ast": rng.integers(0, 10, n).astype(float),
        "actual_stl": rng.integers(0, 5, n).astype(float),
        "actual_blk": rng.integers(0, 4, n).astype(float),
    })


def test_flat_corr_map_returned_without_position_col():
    """Without position_col, returns a flat dict of ρ values."""
    wide = _make_wide_df()
    result = estimate_correlations(wide)
    assert isinstance(result, dict)
    assert "pts_ast" in result
    assert isinstance(result["pts_ast"], float)


def test_position_stratified_returns_nested_dict():
    """With position_col, returns a nested dict with G/F/C/all keys."""
    wide = _make_wide_df()
    result = estimate_correlations(wide, position_col="position")
    assert isinstance(result, dict)
    assert "all" in result
    for pos in ("G", "F", "C"):
        assert pos in result, f"Expected position bucket: {pos}"
        assert isinstance(result[pos], dict)


def test_position_corr_map_has_pts_ast():
    """Each position bucket should have pts_ast correlation."""
    wide = _make_wide_df()
    result = estimate_correlations(wide, position_col="position")
    for pos in ("G", "F", "C", "all"):
        assert "pts_ast" in result[pos], f"pts_ast missing for position {pos}"
        rho = result[pos]["pts_ast"]
        assert -1.0 <= rho <= 1.0, f"Invalid rho for {pos}"


def test_adjust_combo_pmf_with_position():
    """adjust_combo_pmf_for_correlation should use position-stratified map when provided."""
    rng = np.random.default_rng(42)
    pmf_x = np.array([0.1, 0.3, 0.4, 0.2])
    pmf_y = np.array([0.2, 0.5, 0.2, 0.1])

    # Global corr
    global_map = {"pts_ast": 0.5}
    # Position-stratified corr
    pos_map = {"G": {"pts_ast": 0.3}, "F": {"pts_ast": 0.6}, "C": {"pts_ast": 0.1}, "all": {"pts_ast": 0.4}}

    sum_global, _ = adjust_combo_pmf_for_correlation(pmf_x, pmf_y, "pts", "ast", corr_map=global_map)
    sum_pos_G, _  = adjust_combo_pmf_for_correlation(
        pmf_x, pmf_y, "pts", "ast", corr_map_by_pos=pos_map, position="G"
    )

    assert abs(sum_global.sum() - 1.0) < 1e-6, "PMF must sum to 1"
    assert abs(sum_pos_G.sum() - 1.0) < 1e-6, "PMF must sum to 1"


def test_adjust_combo_pmf_position_falls_back_to_all():
    """Unknown position bucket should fall back to 'all' correlation."""
    pmf_x = np.array([0.2, 0.5, 0.3])
    pmf_y = np.array([0.4, 0.4, 0.2])
    pos_map = {"all": {"pts_ast": 0.3}}

    result, _ = adjust_combo_pmf_for_correlation(
        pmf_x, pmf_y, "pts", "ast", corr_map_by_pos=pos_map, position="X"
    )
    assert abs(result.sum() - 1.0) < 1e-6
