"""Beta-Binomial model for fg3m (three-point field goals made).

Generative model:
    fg3a ~ NegBinom(mu_attempts, r)    [attempts per game, predicted by HGB]
    fg3_pct ~ Beta(alpha, beta)        [player's 3pt%, MLE from data]
    fg3m | fg3a, pct ~ Binomial(n, pct)
    Marginalizing over pct → fg3m ~ BetaBinomial(n, alpha, beta)

This captures both shot-volume uncertainty (fg3a) and shot-rate uncertainty
(fg3_pct) explicitly, rather than lumping both into a single NegBinom mean.

Leakage audit (STEP F): No rolling windows are computed in this file. All
features in X are pre-shifted via _sr() in build_features.py before being
passed here. y_made and y_attempts are training labels (actual game outcomes),
which is correct. No shift(1) changes needed here.
"""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import beta as scipy_beta
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

_MIN_ATTEMPTS = 0.5    # denominator guard when computing fg3_pct
_MIN_ALPHA    = 0.1    # minimum Beta alpha / beta params
_MIN_SAMPLES  = 30     # minimum positive fg3a rows to fit Beta params


def beta_binomial_pmf_batch(
    expected_n: np.ndarray,
    alpha: float,
    beta_param: float,
    cap: int,
) -> np.ndarray:
    """Batch Beta-Binomial PMF.  Returns shape (n_players, cap+1), rows sum to 1.

    P(k | n, α, β) = C(n,k) · B(k+α, n−k+β) / B(α,β)

    Uses log-space arithmetic for numerical stability.
    n = round(expected_n).clip(0, cap).
    """
    n_players = len(expected_n)
    ns = np.round(np.clip(expected_n, 0, cap)).astype(int)
    alpha = max(float(alpha), _MIN_ALPHA)
    beta_param = max(float(beta_param), _MIN_ALPHA)
    log_beta_ab = _log_beta(alpha, beta_param)

    pmf_mat = np.zeros((n_players, cap + 1), dtype=float)
    k_all = np.arange(cap + 1)

    # Pre-compute log-beta for all k values up to cap
    # B(k+α, n-k+β) needs n per player, so we loop over unique n values
    unique_ns = np.unique(ns)
    for n_val in unique_ns:
        mask = (ns == n_val)
        if n_val == 0:
            pmf_mat[mask, 0] = 1.0
            continue
        k_valid = k_all[k_all <= n_val]
        log_comb = gammaln(n_val + 1) - gammaln(k_valid + 1) - gammaln(n_val - k_valid + 1)
        log_bb = (
            log_comb
            + _log_beta(k_valid + alpha, n_val - k_valid + beta_param)
            - log_beta_ab
        )
        probs = np.exp(np.clip(log_bb, -700, 0))
        probs = np.clip(probs, 0.0, None)
        row_sum = probs.sum()
        if row_sum > 0:
            probs /= row_sum
        pmf_row = np.zeros(cap + 1)
        pmf_row[:len(probs)] = probs
        pmf_mat[mask] = pmf_row[np.newaxis, :]

    # Normalize any remaining floating-point drift
    row_sums = pmf_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return pmf_mat / row_sums


def _log_beta(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    return gammaln(a) + gammaln(b) - gammaln(np.asarray(a) + np.asarray(b))


class BetaBinomialStatModel:
    """Beta-Binomial model for fg3m.

    Fits:
    1. A HistGBR to predict expected fg3a (three-point attempts per game).
    2. A Beta(alpha, beta) distribution on observed fg3_pct = fg3m / fg3a.

    At prediction time, produces a Beta-Binomial PMF for fg3m.
    Falls back to NegBinom if fg3a is unavailable.
    """

    VERSION = "stage4_beta_binomial_v1"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._attempts_model: HistGradientBoostingRegressor | None = None
        self.alpha_: float = 2.0   # sensible Beta prior (40% 3pt shooter)
        self.beta_: float  = 3.0
        self._usable_cols: list[str] = []
        self._fitted = False
        # Expose NegBinom dispersion for fallback PMF construction
        self.dispersion_r: float | None = None
        self._global_var: float = 1.5
        self._global_mean: float = 0.85
        self._role_dispersion: dict[str, float] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y_made: pd.Series,
        y_attempts: pd.Series | None,
        sample_weight: np.ndarray | None = None,
        context_df: pd.DataFrame | None = None,
    ) -> "BetaBinomialStatModel":
        """Fit the Beta-Binomial model.

        Args:
            X:            Feature matrix.
            y_made:       Observed fg3m (made 3-pointers).
            y_attempts:   Observed fg3a (attempted 3-pointers). May be None.
            sample_weight: Per-sample weights.
            context_df:   Original wide table (same index), used for Beta fitting.
        """
        seed  = self.cfg.get("random_seed", 42)
        hgb_kw = self.cfg.get("hgb_regressor", {})

        # Remove all-NaN columns
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        if y_attempts is not None and y_attempts.notna().sum() >= _MIN_SAMPLES:
            # Stage 1: HGB regressor for expected fg3a
            y_att = y_attempts.fillna(0.0).clip(lower=0.0)
            self._attempts_model = HistGradientBoostingRegressor(
                max_iter=hgb_kw.get("max_iter", 200),
                max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
                learning_rate=hgb_kw.get("learning_rate", 0.1),
                min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
                early_stopping=hgb_kw.get("early_stopping", False),
                n_iter_no_change=hgb_kw.get("n_iter_no_change", 10),
                tol=hgb_kw.get("tol", 1e-7),
                random_state=seed,
            )
            self._attempts_model.fit(X, y_att, sample_weight=sample_weight)

            # Stage 2: fit Beta distribution to observed fg3_pct
            valid = (y_attempts > 0) & y_attempts.notna() & y_made.notna()
            if valid.sum() >= _MIN_SAMPLES:
                pct = (y_made[valid] / y_attempts[valid].clip(lower=_MIN_ATTEMPTS)).clip(0.001, 0.999)
                try:
                    _, _, a, b = scipy_beta.fit(pct.values, floc=0, fscale=1)
                    self.alpha_ = max(float(a), _MIN_ALPHA)
                    self.beta_  = max(float(b), _MIN_ALPHA)
                    logger.info(
                        "BetaBinomial fg3m: alpha=%.3f beta=%.3f (n=%d valid pct samples)",
                        self.alpha_, self.beta_, valid.sum(),
                    )
                except Exception as exc:
                    logger.warning("Beta MLE failed: %s — using defaults", exc)
        else:
            logger.warning("fg3a unavailable or insufficient — BetaBinomial falls back to NegBinom for fg3m")
            self._attempts_model = None

        # Always compute NegBinom dispersion for fallback
        y_m = y_made.fillna(0.0)
        self._global_mean = float(y_m.mean()) or 0.85
        self._global_var  = float(y_m.var())  or 1.5
        if self._global_var > self._global_mean and self._global_mean > 0:
            self.dispersion_r = float(
                self._global_mean ** 2 / max(self._global_var - self._global_mean, 1e-9)
            )
        else:
            self.dispersion_r = None

        self._fitted = True
        return self

    def predict_pmf_matrix(self, X: pd.DataFrame, cap: int = 12) -> np.ndarray:
        """Return Beta-Binomial PMF matrix (n, cap+1).  Falls back to NegBinom if no attempt model."""
        if not self._fitted:
            raise RuntimeError("BetaBinomialStatModel not fitted")

        X_aligned = X.reindex(columns=self._usable_cols)
        n_players = len(X_aligned)

        if self._attempts_model is not None:
            expected_n = np.clip(
                self._attempts_model.predict(X_aligned), 0.0, float(cap)
            )
            return beta_binomial_pmf_batch(expected_n, self.alpha_, self.beta_, cap)
        else:
            # Fallback: NegBinom on fg3m directly using stored dispersion
            from wnba_props_model.models.pmf_utils import (  # noqa: PLC0415
                negbinom_pmf_batch, poisson_pmf_batch,
            )
            means = np.full(n_players, self._global_mean)
            if self.dispersion_r is not None:
                return negbinom_pmf_batch(means, self.dispersion_r, cap)
            return poisson_pmf_batch(means, cap)

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Expected fg3m = E[fg3a] * alpha/(alpha+beta)."""
        if not self._fitted:
            raise RuntimeError("BetaBinomialStatModel not fitted")
        X_aligned = X.reindex(columns=self._usable_cols)
        if self._attempts_model is not None:
            exp_n = np.clip(self._attempts_model.predict(X_aligned), 0.0, 15.0)
            mean_pct = self.alpha_ / (self.alpha_ + self.beta_)
            return np.clip(exp_n * mean_pct, self.cfg.get("min_stat_mean", 0.01), None)
        return np.full(len(X_aligned), self._global_mean)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BetaBinomialStatModel":
        return joblib.load(path)
