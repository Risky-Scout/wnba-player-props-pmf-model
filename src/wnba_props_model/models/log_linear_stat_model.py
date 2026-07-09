"""Dixon-Coles inspired log-linear stat prediction model.

Decomposition (Poisson exposure-rate model):

    log(λ_stat) = log(minutes_mean)      ← exposure / pace offset
                + player_rate_per_min    ← intrinsic skill (HGB-predicted)
                + opp_defense_adj        ← opponent suppression
                + home_adj               ← home court effect

The opponent defense adjustment is:

    opp_defense_adj = log(opp_stat_allowed / league_avg_allowed)

A team that allows 20% fewer pts than the league average contributes
log(0.80) ≈ −0.22 to the Poisson log-rate.  This is directly analogous
to Dixon-Coles' defensive parameter for soccer goals.

The model wraps ``StatRateModel`` for the player skill component and adds
the multiplicative matchup adjustments at inference time.  Training is
identical to StatRateModel — the extra adjustments are data features, not
new learned parameters.

Usage
-----
    cfg = load_yaml("config/model/stage4_baseline.yaml")
    m = LogLinearStatModel("pts", cfg)
    m.fit(X_played, y_pts, context_df=played_ctx)
    λ = m.predict(X_infer, wide_infer_df)   # raw Poisson λ per player
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from wnba_props_model.models.rate_model import StatRateModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum league-average denominator to prevent log(0) explosions
_FLOOR = 1e-3


class LogLinearStatModel:
    """Additive log-rate model with opponent-defense and pace adjustments.

    The HGB ``StatRateModel`` predicts player skill in log space.  This
    class wraps it and applies two *post-prediction* multiplicative
    corrections derived from pre-computed context features:

    1. **Opponent defense**: opp_{stat}_allowed_mean_l5 / league_avg
    2. **Pace**: team_total_score_mean_l5 / league_avg_total_score

    These features must already be present in the feature matrix.
    """

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        self.stat = stat
        self.cfg = cfg
        self._base_model = StatRateModel(stat, cfg)
        self._league_avg_rate: float | None = None
        self._league_avg_pace: float | None = None

    # ------------------------------------------------------------------
    # Training: identical to StatRateModel but stores league averages
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "LogLinearStatModel":
        """Fit the base StatRateModel and compute league-average reference values.

        When ``use_minutes_offset`` is in cfg, the base StatRateModel fits on
        per-minute rate and scales at prediction time by projected minutes.
        The league_avg_rate is set accordingly.
        """
        self._base_model.fit(X, y, context_df=context_df, sample_weight=sample_weight)

        # League average stat rate (stat per minute) from training context
        if context_df is not None and "actual_minutes" in context_df.columns:
            total_stat = float(y.sum())
            total_min  = float(context_df["actual_minutes"].clip(lower=0).sum())
            self._league_avg_rate = total_stat / max(total_min, _FLOOR)
        elif len(y) > 0:
            self._league_avg_rate = float(y.mean())

        # League average pace (total score) for pace normalization
        if context_df is not None:
            for col in (f"team_total_score_mean_l5", "team_total_score_mean_l10"):
                if col in context_df.columns:
                    v = context_df[col].dropna()
                    if len(v) > 0:
                        self._league_avg_pace = float(v.mean())
                        break

        logger.debug(
            "[%s] LogLinearStatModel fitted: league_avg_rate=%.4f, "
            "league_avg_pace=%s",
            self.stat,
            self._league_avg_rate or 0.0,
            f"{self._league_avg_pace:.2f}" if self._league_avg_pace else "N/A",
        )
        return self

    # ------------------------------------------------------------------
    # Inference: base prediction × opponent × pace multipliers
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Return adjusted Poisson λ per row.

        Computes:  λ = base_λ × opp_adj × pace_adj

        where base_λ comes from the wrapped StatRateModel (already in
        expected-count units), and the multipliers are computed from
        context_df columns.  If context_df is None, falls back to the
        base model unmodified.
        """
        base_lambda = self._base_model.predict_mean(X)

        if context_df is None or context_df.empty:
            return base_lambda

        n = len(base_lambda)
        opp_adj  = np.ones(n, dtype=float)
        pace_adj = np.ones(n, dtype=float)

        # --- Opponent defense adjustment ---------------------------------
        opp_col = (
            f"opp_{self.stat}_allowed_mean_l5"
            if self.stat != "turnover"
            else "opp_turnover_forced_mean_l5"
        )
        if opp_col in context_df.columns and self._league_avg_rate is not None:
            # opp_allowed is in stat-count units; we compare to league_avg_rate
            # expressed as stat-per-game (multiply by 40 min per game)
            league_avg_pg = self._league_avg_rate * 40.0
            opp_vals = context_df[opp_col].values.astype(float)
            valid = np.isfinite(opp_vals) & (opp_vals > 0)
            opp_adj[valid] = np.exp(
                np.log(opp_vals[valid] / max(league_avg_pg, _FLOOR))
            ).clip(0.5, 2.0)  # hard clamp: never more than 2× adjustment

        # --- Pace adjustment ---------------------------------------------
        if self._league_avg_pace is not None:
            pace_col = "team_pace_proxy_l5" if "team_pace_proxy_l5" in context_df.columns else "team_total_score_mean_l5"
            if pace_col in context_df.columns:
                pace_vals = context_df[pace_col].values.astype(float)
                valid = np.isfinite(pace_vals) & (pace_vals > 0)
                pace_adj[valid] = np.exp(
                    np.log(pace_vals[valid] / max(self._league_avg_pace, _FLOOR))
                ).clip(0.8, 1.25)  # pace adjustment is smaller: ±25% max

        adjusted = base_lambda * opp_adj * pace_adj
        return np.clip(adjusted, 0.0, None)

    # ------------------------------------------------------------------
    # Inference alias: callers use predict_mean() not predict()
    # ------------------------------------------------------------------

    def predict_mean(self, X: pd.DataFrame, role_series=None) -> np.ndarray:
        """Return adjusted Poisson λ per row.

        role_series is accepted for API symmetry with StatRateModel but unused.

        Passes *X* itself as context_df so the Dixon-Coles opponent-defense
        and pace adjustments fire automatically — opp/pace columns are already
        part of the feature matrix, so no separate context_df is needed.
        """
        return self.predict(X, context_df=X)

    # ------------------------------------------------------------------
    # Delegate dispersion + variance attributes to wrapped model
    # ------------------------------------------------------------------

    @property
    def _global_var(self) -> float:
        """Global variance of the training target (delegated to base model)."""
        return self._base_model._global_var

    @property
    def dispersion_r(self) -> float | None:
        """NegBinom dispersion r (delegated to base model; None → Poisson)."""
        return self._base_model.dispersion_r

    @property
    def _dispersion_r(self) -> float | None:
        return self._base_model._dispersion_r

    def get_dispersion(self, role: str = "all") -> float | None:
        """Return per-role (or global) NegBinom dispersion r."""
        return self._base_model.get_dispersion(role)

    @property
    def _role_dispersion(self) -> dict[str, float | None]:
        return getattr(self._base_model, "_role_dispersion", {})

    @property
    def _usable_cols(self) -> list[str]:
        return getattr(self._base_model, "_usable_cols", [])

    def get_training_summary(self) -> dict:
        """Delegate to the underlying StatRateModel's training summary."""
        s = self._base_model.get_training_summary()
        s["model_type"] = "LogLinearStatModel"
        return s
