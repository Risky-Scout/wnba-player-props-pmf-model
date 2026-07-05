"""Beta-Binomial model for 3-point makes (fg3m).

PMF support is bounded at [0, fg3a], physically correct (can't make more 3s
than you attempt). Uses Empirical Bayes shrinkage toward the league 3pt%.

Replaces NegBinom r=1.147 with a bounded distribution, eliminating the
long tail that inflated P(fg3m ≥ 4) for low-volume shooters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln
from sklearn.ensemble import HistGradientBoostingRegressor


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class BetaBinomialFg3mModel:
    """Beta-Binomial model for 3-point makes.

    PMF support is bounded at [0, fg3a], physically correct.
    Uses Empirical Bayes shrinkage toward league 3pt%.
    """

    LEAGUE_3PT_PCT: float = 0.325
    PRIOR_STRENGTH: int = 50  # equivalent prior attempts

    def __init__(self) -> None:
        self.attempts_model: HistGradientBoostingRegressor | None = None
        self.attempts_features_: list[str] = []

    def fit_attempts_model(self, df: pd.DataFrame) -> "BetaBinomialFg3mModel":
        """Fit the fg3a (attempts) regression model."""
        features = [
            "player_fg3a_mean_l10", "player_fg3a_mean_season",
            "player_fg3a_rate_season", "player_usage_rate_season",
            "minutes_mean", "opponent_def_3pt_rate_season",
            "game_pace_estimate", "home_flag", "rest_days",
        ]
        available = [c for c in features if c in df.columns]
        self.attempts_features_ = available
        if available and "fg3a" in df.columns:
            X = df[available].fillna(0).values
            y = df["fg3a"].values
            self.attempts_model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=5,
                min_samples_leaf=10, early_stopping=True,
                n_iter_no_change=10, random_state=42,
            )
            self.attempts_model.fit(X, y)
        return self

    def _compute_alpha_beta(
        self,
        fg3m_made: int,
        fg3a_attempts: int,
    ) -> tuple[float, float]:
        """Compute posterior Beta parameters via Empirical Bayes shrinkage."""
        alpha = self.PRIOR_STRENGTH * self.LEAGUE_3PT_PCT + fg3m_made
        beta = self.PRIOR_STRENGTH * (1 - self.LEAGUE_3PT_PCT) + (fg3a_attempts - fg3m_made)
        return max(float(alpha), 0.5), max(float(beta), 0.5)

    def build_pmf(
        self,
        fg3a_pred: float,
        player_fg3m_total: int,
        player_fg3a_total: int,
        cap: int = 15,
    ) -> np.ndarray:
        """Build Beta-Binomial PMF for fg3m given predicted attempts.

        Parameters
        ----------
        fg3a_pred:          Predicted 3-point attempts for this game
        player_fg3m_total:  Season total 3pm (for EB shrinkage)
        player_fg3a_total:  Season total 3pa (for EB shrinkage)
        cap:                Maximum support for the PMF array
        """
        n = min(max(int(round(fg3a_pred)), 0), cap)
        pmf = np.zeros(cap + 1)
        if n == 0:
            pmf[0] = 1.0
            return pmf
        alpha, beta = self._compute_alpha_beta(player_fg3m_total, player_fg3a_total)
        for k in range(n + 1):
            log_p = (
                betaln(k + alpha, n - k + beta)
                - betaln(alpha, beta)
                + _log_comb(n, k)
            )
            pmf[k] = np.exp(log_p)
        total = pmf.sum()
        if total > 0:
            pmf /= total
        return pmf

    def predict_pmf_matrix(self, df: pd.DataFrame, cap: int = 15) -> np.ndarray:
        """Predict PMF matrix (n × cap+1) for a batch of players.

        Uses attempts_model to predict fg3a, then builds per-player
        Beta-Binomial PMFs using season totals for EB shrinkage.
        Falls back to player_fg3a_mean_season when attempts_model is absent.
        """
        n = len(df)
        pmf_mat = np.zeros((n, cap + 1))

        # Predict fg3a
        if self.attempts_model is not None and self.attempts_features_:
            X = df.reindex(columns=self.attempts_features_).fillna(0).values
            fg3a_pred = self.attempts_model.predict(X)
        elif "player_fg3a_mean_season" in df.columns:
            fg3a_pred = df["player_fg3a_mean_season"].fillna(2.0).values
        else:
            fg3a_pred = np.full(n, 2.0)
        fg3a_pred = np.clip(fg3a_pred, 0.0, float(cap))

        # Season totals for EB shrinkage
        fg3m_totals = (
            df["player_fg3m_season_total"].fillna(0).values.astype(int)
            if "player_fg3m_season_total" in df.columns
            else np.zeros(n, dtype=int)
        )
        fg3a_totals = (
            df["player_fg3a_season_total"].fillna(0).values.astype(int)
            if "player_fg3a_season_total" in df.columns
            else np.zeros(n, dtype=int)
        )

        for i in range(n):
            pmf_mat[i] = self.build_pmf(
                float(fg3a_pred[i]),
                int(fg3m_totals[i]),
                int(fg3a_totals[i]),
                cap=cap,
            )

        return pmf_mat
