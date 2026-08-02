"""Explicit feature contracts for the authoritative V6 inference graph.

Self-contained — production code must not import sharp_v3/v4/v5.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

SEED = 20260730
TAIL_TOL = 1e-6
NORM_TOL = 1e-10

ID_COLS = ["game_id", "player_id", "game_date", "season", "team_id", "opponent_team_id"]
TIER_A = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMBOS = ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]
LABEL_COLS = [
    "participation", "actual_minutes", "pts", "reb", "ast", "fg3m", "stl", "blk",
    "turnover", "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast",
    "fgm", "ftm", "fg2m", "fg2a", "fg3a", "fta", "oreb", "dreb", "fga",
]
EMERGENCY_CAP = {
    "pts": 80, "reb": 40, "ast": 30, "fg3m": 18, "stl": 15, "blk": 15,
    "turnover": 18, "fgm": 35, "ftm": 35, "fta": 40, "oreb": 25, "dreb": 30,
    "fg2a": 40, "fg3a": 25, "fg2m": 30, "minutes": 48, "q1_pts": 30, "q1_reb": 15, "q1_ast": 12,
}

_COMMON = [
    r"^player_minutes_", r"^player_cumulative_minutes", r"^cumulative_minutes_",
    r"^player_rest_days$", r"^opp_rest_days$", r"^rest_advantage$",
    r"^days_since_", r"^game_number_in_season$", r"^is_home$", r"^is_starter_prior",
    r"^player_usage_proxy_",
]
_FAMILIES: dict[str, list[str]] = {
    "participation": [
        r"^player_minutes_", r"^player_cumulative_minutes", r"^cumulative_minutes_",
        r"^player_rest_days$", r"^opp_rest_days$", r"^rest_advantage$",
        r"^days_since_", r"^game_number_in_season$", r"^player_games_played",
        r"^player_.*_season_zscore$",
    ],
    "minutes": _COMMON + [r"^player_minutes_.*_(mean|std|form)", r"^player_.*_season_zscore$"],
    "pts": _COMMON + [r"^player_pts_", r"^opp_pts_allowed", r"^opp_pos_pts_allowed",
                      r"^player_fga_", r"^player_fg3a_", r"^player_fta_"],
    "reb": _COMMON + [r"^player_reb_", r"^player_oreb_", r"^player_dreb_", r"^opp_reb_allowed"],
    "ast": _COMMON + [r"^player_ast_", r"^opp_ast_allowed", r"^player_usage_proxy_"],
    "fg3m": _COMMON + [r"^player_fg3m_", r"^player_fg3a_", r"^opp_fg3m_allowed", r"^opp_fg3a_allowed"],
    "stl": _COMMON + [r"^player_stl_", r"^opp_stl_", r"^opp_turnover_forced"],
    "blk": _COMMON + [r"^player_blk_", r"^opp_blk_"],
    "turnover": _COMMON + [r"^player_turnover_", r"^opp_turnover_forced"],
    "fg2a": _COMMON + [r"^player_fga_", r"^player_fg3a_", r"^opp_pts_allowed"],
    "fg3a": _COMMON + [r"^player_fg3a_", r"^opp_fg3a_allowed"],
    "fta": _COMMON + [r"^player_fta_", r"^player_ftr", r"^opp_pts_allowed"],
    "oreb": _COMMON + [r"^player_oreb_", r"^player_reb_", r"^opp_reb_allowed"],
    "dreb": _COMMON + [r"^player_dreb_", r"^player_reb_", r"^opp_reb_allowed"],
    "game_environment": [
        r"^opp_pts_allowed", r"^team_", r"^is_home$", r"^player_rest_days$", r"^opp_rest_days$",
        r"^pace", r"^poss",
    ],
}


def resolve_contract(component: str, all_cols: list[str]) -> list[str]:
    pats = [re.compile(p) for p in _FAMILIES[component]]
    forbidden = set(ID_COLS) | set(LABEL_COLS)
    out: list[str] = []
    for c in all_cols:
        if c in forbidden or c.endswith("_tgt"):
            continue
        if any(p.search(c) for p in pats) and c not in out:
            out.append(c)
    return out


def contract_hash(cols: list[str]) -> str:
    return hashlib.sha256("\n".join(cols).encode()).hexdigest()[:16]


def build_all_contracts(all_cols: list[str]) -> dict[str, dict]:
    contracts = {}
    for comp in _FAMILIES:
        cols = resolve_contract(comp, all_cols)
        contracts[comp] = {
            "n": len(cols),
            "schema_hash": contract_hash(cols),
            "features": cols,
            # Explicit classification — not an unclassified silent NaN-fill policy.
            "missingness": "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT",
            "feature_class_default": "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT",
            "provenance": "pregame T-1.2 lagged features (recovered_v2)",
            "pit_rule": "available_at <= prediction_timestamp",
        }
    return contracts


# Governed constants — purpose / unit / range / sensitivity notes for audits.
GOVERNED_CONSTANTS = {
    "SEED": {
        "value": SEED, "purpose": "deterministic RNG for sims/calibration",
        "unit": "dimensionless", "allowed_range": "fixed", "sensitivity": "reproducibility_only",
    },
    "NORM_TOL": {
        "value": NORM_TOL, "purpose": "PMF mass equality tolerance",
        "unit": "probability", "allowed_range": "[1e-12, 1e-6]", "sensitivity": "invariant_gate",
    },
    "TAIL_TOL": {
        "value": TAIL_TOL, "purpose": "overflow / tail mass floor",
        "unit": "probability", "allowed_range": "[1e-9, 1e-4]", "sensitivity": "tail_events",
    },
    "OREB_SHARE_FALLBACK": {
        "value": 0.28, "purpose": "operational split when oreb/dreb labels absent at fit",
        "unit": "share", "allowed_range": "[0.2, 0.4]",
        "sensitivity": "reb_component_only", "selection": "conservative_operational_constraint",
    },
    "TEAM_REG_MINUTES": {
        "value": 200, "purpose": "regulation team minutes invariant",
        "unit": "minutes", "allowed_range": "200", "sensitivity": "mathematical_invariant",
    },
    "TEAM_Q1_MINUTES": {
        "value": 50, "purpose": "Q1 team minutes invariant",
        "unit": "minutes", "allowed_range": "50", "sensitivity": "mathematical_invariant",
    },
    "FIRST_BASKET_OTHER_MASS": {
        "value": 0.02, "purpose": "residual other/no-basket competing-risk mass",
        "unit": "probability", "allowed_range": "[0.01, 0.05]",
        "sensitivity": "first_basket_only", "selection": "conservative_operational_constraint",
    },
    "ROLE_BAND_BREAKS": {
        "value": [12, 22, 30], "purpose": "minutes role-band boundaries for dispersion",
        "unit": "minutes", "allowed_range": "monotonic increasing in (0, 40)",
        "sensitivity": "dispersion_pooling", "selection": "oof_role_stability",
    },
}


def role_band(df: pd.DataFrame) -> np.ndarray:
    col = "player_minutes_mean_season"
    if col not in df.columns:
        return np.zeros(len(df), int)
    mm = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
    return np.digitize(mm, [12, 22, 30])


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def usable_columns(X: np.ndarray) -> np.ndarray:
    ok = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        f = X[:, j][np.isfinite(X[:, j])]
        ok[j] = f.size > 0 and np.unique(f).size >= 2
    return ok


def prep_matrices(train: pd.DataFrame, other: pd.DataFrame, feat: list[str]):
    leak = set(feat) & set(LABEL_COLS)
    if leak:
        raise ValueError(f"LEAKAGE: {sorted(leak)}")
    Xtr = numeric_matrix(train, feat)
    mask = usable_columns(Xtr)
    used = [c for c, k in zip(feat, mask) if k]
    return Xtr[:, mask], numeric_matrix(other, used), used
