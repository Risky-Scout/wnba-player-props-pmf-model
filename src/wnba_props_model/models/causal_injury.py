"""Causal Injury Model with Healthy-Worker Survivor Correction (Enhancement 17).

Naive DNP-probability models condition on game participation (observations
only exist when a player PLAYS), creating a collider bias: heavy workload
appears paradoxically PROTECTIVE against injury because only the healthiest
players log heavy minutes.

This module corrects the bias using Inverse Probability Weighting (IPW),
producing calibrated DNP probabilities that correctly show a POSITIVE
workload–injury relationship for high-load players.

Method
------
1. Fit a propensity model P(played=1 | age, recent_load, rest_days, b2b, …).
2. Compute stabilised IPW weights: sw = P(played) / P(played | confounders).
3. Fit a weighted DNP logistic regression that creates a pseudo-population
   where game participation is independent of observed confounders.
4. The IPW-corrected model produces HIGHER DNP probabilities for high-workload
   players than the naive model.

Reference
---------
Yu & Hu (2026). The Load Management Paradox: Correcting the Healthy-Worker
Survivor Effect in NBA Injury Modeling. arXiv:2603.26935
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Confounders used by both propensity and outcome models
DEFAULT_CONFOUNDER_COLS = [
    "age",
    "recent_7day_load",       # total minutes last 7 days
    "rest_days",              # days since last game
    "is_back_to_back",        # back-to-back flag
    "player_games_in_last_7_days",
    "player_cumulative_minutes_l3",
    "player_3in4_flag",
    "is_4_in_5",
    "schedule_fatigue_index",
    "cumulative_minutes_l7",
]

# Minimum rows to fit the model
MIN_TRAINING_ROWS = 150


def compute_ipw_weights(
    df: pd.DataFrame,
    treatment_col: str = "played",
    confounder_cols: list[str] | None = None,
) -> tuple[np.ndarray, HistGradientBoostingClassifier]:
    """Compute stabilised IPW weights for the selection model.

    Selection model: P(played=1 | age, recent_load, rest_days, b2b, …)
    Stabilised weight: sw = P(played=1) / P(played=1 | confounders)

    Weights are trimmed to the 1st–99th percentile to avoid extreme values
    from near-deterministic propensity scores.

    Returns
    -------
    sw : np.ndarray of stabilised IPW weights (same length as df)
    ps_model : fitted propensity model (HistGradientBoostingClassifier)
    """
    if confounder_cols is None:
        confounder_cols = DEFAULT_CONFOUNDER_COLS
    avail = [c for c in confounder_cols if c in df.columns]

    if not avail:
        # No confounders available; return uniform weights
        logger.warning("E17: no confounder columns found; returning unit IPW weights")
        return np.ones(len(df)), None

    X = df[avail].fillna(0).values.astype(float)
    T = df[treatment_col].values.astype(int)

    ps_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, random_state=42, min_samples_leaf=10
    )
    ps_model.fit(X, T)
    ps_hat = np.clip(ps_model.predict_proba(X)[:, 1], 0.05, 0.95)

    p_marginal = float(T.mean())
    sw = np.where(
        T == 1,
        p_marginal / ps_hat,
        (1.0 - p_marginal) / (1.0 - ps_hat),
    )
    # Trim to 1st–99th percentile
    lo, hi = np.percentile(sw, [1, 99])
    sw = np.clip(sw, lo, hi)
    return sw, ps_model


def fit_causal_dnp_model(df: pd.DataFrame) -> dict[str, Any]:
    """Fit IPW-corrected DNP / injury probability model.

    Requires a DataFrame with a 'played' column (1 = played, 0 = DNP/injured).
    Falls back to naive model if insufficient data.

    Returns
    -------
    dict with keys: ipw_model, naive_model, propensity_model, n_train
    """
    if len(df) < MIN_TRAINING_ROWS:
        logger.warning("E17: only %d rows for causal DNP model (need %d)", len(df), MIN_TRAINING_ROWS)
        return {"ipw_model": None, "naive_model": None, "propensity_model": None, "n_train": 0}

    # Ensure 'played' column exists
    if "played" not in df.columns:
        if "is_dnp" in df.columns:
            df = df.copy()
            df["played"] = (~df["is_dnp"].astype(bool)).astype(int)
        else:
            logger.warning("E17: neither 'played' nor 'is_dnp' column found; using all-played")
            df = df.copy()
            df["played"] = 1

    sw, ps_model = compute_ipw_weights(df, treatment_col="played")

    avail = [c for c in DEFAULT_CONFOUNDER_COLS if c in df.columns]
    if not avail:
        return {"ipw_model": None, "naive_model": None, "propensity_model": ps_model, "n_train": len(df)}

    X = df[avail].fillna(0).values.astype(float)
    y = 1 - df["played"].values  # DNP = 1

    # IPW-corrected model (weighted logistic regression)
    ipw_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])
    try:
        ipw_pipe.fit(X, y, lr__sample_weight=sw)
    except TypeError:
        # Some sklearn versions don't support step__param; try direct
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(X_scaled, y, sample_weight=sw)
        ipw_pipe = (scaler, lr)

    # Naive model (no IPW, reference)
    naive_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])
    naive_pipe.fit(X, y)

    logger.info(
        "E17 causal DNP model fitted on %d rows; "
        "DNP rate=%.1f%%; mean IPW weight=%.2f",
        len(df), 100 * y.mean(), float(sw.mean()),
    )
    return {
        "ipw_model":        ipw_pipe,
        "naive_model":      naive_pipe,
        "propensity_model": ps_model,
        "feature_cols":     avail,
        "confounder_cols":  avail,   # alias for test compatibility
        "n_train":          len(df),
    }


def predict_causal_dnp_probability(
    player_features: pd.DataFrame | np.ndarray,
    causal_models:   dict[str, Any],
) -> dict[str, float]:
    """Compute IPW-corrected DNP probability for one or more players.

    Parameters
    ----------
    player_features : (n, n_features) array or DataFrame matching feature_cols order.
    causal_models : output of fit_causal_dnp_model().

    Returns
    -------
    dict with dnp_probability_causal, dnp_probability_naive, selection_bias_magnitude
    """
    ipw   = causal_models.get("ipw_model")
    naive = causal_models.get("naive_model")
    fcols = causal_models.get("feature_cols", [])

    if ipw is None or naive is None:
        return {
            "dnp_probability_causal": 0.05,
            "dnp_probability_naive":  0.05,
            "selection_bias_magnitude": 0.0,
        }

    if isinstance(player_features, pd.DataFrame):
        avail = [c for c in fcols if c in player_features.columns]
        X = player_features[avail].fillna(0).values.astype(float)
    else:
        X = np.asarray(player_features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

    # Handle both Pipeline and (scaler, lr) fallback tuples
    def _predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
        if isinstance(model, tuple):
            scaler, lr = model
            return lr.predict_proba(scaler.transform(x))[:, 1]
        return model.predict_proba(x)[:, 1]

    ipw_pred   = float(np.mean(_predict_proba(ipw, X)))
    naive_pred = float(np.mean(_predict_proba(naive, X)))

    return {
        "dnp_probability_causal":    round(ipw_pred, 4),
        "dnp_probability_naive":     round(naive_pred, 4),
        "selection_bias_magnitude":  round(ipw_pred - naive_pred, 4),
    }


def add_causal_dnp_features(
    wide: pd.DataFrame,
    causal_models: dict[str, Any],
) -> pd.DataFrame:
    """Add causal DNP probability columns to the wide feature table."""
    if not causal_models.get("ipw_model"):
        return wide

    fcols = causal_models.get("feature_cols", [])
    avail = [c for c in fcols if c in wide.columns]
    if not avail:
        return wide

    X = wide[avail].fillna(0).values.astype(float)
    results = predict_causal_dnp_probability(X, causal_models)
    wide["dnp_probability_causal"]   = results["dnp_probability_causal"]
    wide["selection_bias_magnitude"] = results["selection_bias_magnitude"]
    return wide
