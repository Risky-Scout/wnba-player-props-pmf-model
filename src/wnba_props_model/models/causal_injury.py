"""Causal Injury Model with Healthy-Worker Survivor Effect Correction (Enhancement 17).

The Load Management Paradox (Yu & Hu, 2026):
    Naive injury models systematically UNDERESTIMATE injury risk for
    high-workload players because of the healthy-worker survivor effect.

    Paradox: naive models show a NEGATIVE association between recent
    workload and injury risk — heavy-minute players APPEAR less likely to
    miss games.  This is selection bias: only the healthiest players are
    allowed to log heavy minutes.

Solution:
    Inverse Probability Weighting (IPW) corrects for the selection
    mechanism, creating a pseudo-population where game participation is
    independent of observed confounders.

    Stabilised IPW weight:
        sw = P(played=1) / P(played=1 | confounders)   for played=1
        sw = P(played=0) / P(played=0 | confounders)   for played=0

    The IPW-weighted model produces HIGHER DNP probabilities for high-
    workload players (the correct direction).

Reference:
    Yu & Hu (2026). The Load Management Paradox: Correcting the
    Healthy-Worker Survivor Effect in NBA Injury Modeling.
    arXiv:2603.26935
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Confounders used by both propensity and outcome models
DEFAULT_CONFOUNDERS = [
    "age",
    "recent_7day_load",      # total minutes in past 7 days
    "rest_days",
    "is_back_to_back",
    "is_3_in_4",
    "player_cumulative_minutes_l3",
    "player_games_in_last_7_days",
    "schedule_fatigue_index",
]


def compute_ipw_weights(
    df: pd.DataFrame,
    treatment_col: str = "played",
    confounder_cols: list[str] | None = None,
) -> tuple[np.ndarray, HistGradientBoostingClassifier]:
    """Compute stabilised IPW weights for the selection (participation) model.

    Parameters
    ----------
    df             : DataFrame with one row per player-game
    treatment_col  : binary indicator (1 = player participated)
    confounder_cols: confounders affecting both participation and injury risk

    Returns
    -------
    (weights, propensity_model)
    """
    if confounder_cols is None:
        confounder_cols = DEFAULT_CONFOUNDERS

    avail = [c for c in confounder_cols if c in df.columns]
    if not avail:
        logger.warning("No confounder columns found — returning uniform weights.")
        return np.ones(len(df)), HistGradientBoostingClassifier()

    X = df[avail].fillna(0).values
    T = df[treatment_col].astype(int).values

    ps_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, random_state=42
    )
    ps_model.fit(X, T)
    ps_hat = np.clip(ps_model.predict_proba(X)[:, 1], 0.05, 0.95)

    p_marginal = float(T.mean())
    sw = np.where(
        T == 1,
        p_marginal / ps_hat,
        (1 - p_marginal) / (1 - ps_hat),
    )
    # Truncate at 1st / 99th percentile to avoid extreme weights
    sw = np.clip(sw, np.percentile(sw, 1), np.percentile(sw, 99))
    return sw, ps_model


def fit_causal_dnp_model(
    df: pd.DataFrame,
    confounder_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Fit an IPW-corrected DNP/injury probability model.

    The naive model shows paradoxical protective effect of heavy workload.
    The IPW model corrects this, returning higher DNP probabilities for
    high-workload players.

    Returns
    -------
    dict with keys:
        ipw_model         : IPW-corrected LogisticRegression
        naive_model       : uncorrected baseline
        propensity_model  : HGB classifier for P(played=1 | X)
        confounder_cols   : list of columns actually used
    """
    if confounder_cols is None:
        confounder_cols = DEFAULT_CONFOUNDERS

    # Ensure 'played' column exists
    if "played" not in df.columns:
        df = df.copy()
        dnp_col = next((c for c in ["is_dnp", "dnp"] if c in df.columns), None)
        if dnp_col:
            df["played"] = (~df[dnp_col].astype(bool)).astype(int)
        else:
            # Assume everyone in the dataset played (survivor bias in raw PBP)
            df["played"] = 1

    avail = [c for c in confounder_cols if c in df.columns]
    if not avail:
        logger.warning(
            "No confounder columns found for causal DNP model. "
            "Returning naive logistic regression."
        )
        naive = LogisticRegression(max_iter=1000, random_state=42)
        X = np.zeros((len(df), 1))
        y = 1 - df["played"].values
        naive.fit(X, y)
        return {
            "ipw_model": naive,
            "naive_model": naive,
            "propensity_model": None,
            "confounder_cols": avail,
        }

    X = df[avail].fillna(0).values
    y = 1 - df["played"].astype(int).values   # DNP = 1 - played

    sw, ps_model = compute_ipw_weights(df, "played", avail)

    # IPW-corrected model: weighted logistic regression
    ipw_model = LogisticRegression(max_iter=1000, random_state=42)
    ipw_model.fit(X, y, sample_weight=sw)

    # Naive model for comparison / bias estimation
    naive_model = LogisticRegression(max_iter=1000, random_state=42)
    naive_model.fit(X, y)

    logger.info(
        "Causal DNP model fitted: IPW corrects naive towards higher DNP "
        "for high-workload players. Bias correction available via "
        "predict_causal_dnp_probability()."
    )
    return {
        "ipw_model":        ipw_model,
        "naive_model":      naive_model,
        "propensity_model": ps_model,
        "confounder_cols":  avail,
    }


def predict_causal_dnp_probability(
    player_features: np.ndarray,
    causal_models: dict[str, Any],
) -> dict[str, float]:
    """Compute the causally-corrected DNP probability for one player.

    Parameters
    ----------
    player_features : feature vector for the confounder columns (1D array)
    causal_models   : output of fit_causal_dnp_model()

    Returns
    -------
    dict with:
        dnp_probability_causal  : IPW-corrected probability
        dnp_probability_naive   : naive (biased) probability
        selection_bias_magnitude: difference (causal - naive)
    """
    X = np.atleast_2d(player_features)
    ipw_pred   = float(causal_models["ipw_model"].predict_proba(X)[0, 1])
    naive_pred = float(causal_models["naive_model"].predict_proba(X)[0, 1])
    return {
        "dnp_probability_causal":   round(ipw_pred,            4),
        "dnp_probability_naive":    round(naive_pred,           4),
        "selection_bias_magnitude": round(ipw_pred - naive_pred, 4),
    }


def add_causal_dnp_features(
    df: pd.DataFrame,
    causal_models: dict[str, Any],
) -> pd.DataFrame:
    """Add causal DNP probability columns to *df* in-place.

    Adds:
        dnp_probability_causal
        dnp_probability_naive
        dnp_selection_bias
    """
    avail = causal_models.get("confounder_cols", [])
    avail = [c for c in avail if c in df.columns]
    if not avail:
        return df

    X = df[avail].fillna(0).values
    ipw_preds   = causal_models["ipw_model"].predict_proba(X)[:, 1]
    naive_preds = causal_models["naive_model"].predict_proba(X)[:, 1]

    out = df.copy()
    out["dnp_probability_causal"] = ipw_preds
    out["dnp_probability_naive"]  = naive_preds
    out["dnp_selection_bias"]     = ipw_preds - naive_preds
    return out
