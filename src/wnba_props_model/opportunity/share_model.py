"""Team-coherent opportunity-share model for Opportunity V2.

Predicts each active player's share of a team opportunity total. Shares are learned as a log-ratio
CONTEXT DELTA away from an active-roster-renormalized baseline share, then exponentiated and
renormalized WITHIN each team-game-cutoff so they are nonnegative and sum to 1. Cross-team
normalization is impossible by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_EPS = 1e-6


class TeamOpportunityShareModel:
    VERSION = "opportunity_v2_share_v1"

    def __init__(
        self,
        share_name: str,
        target_numerator_col: str,
        team_total_col: str,
        baseline_share_col: str,
        feature_columns: list[str],
        *,
        active_prob_col: str = "p_active",
        random_state: int = 42,
    ) -> None:
        self.share_name = share_name
        self.target_numerator_col = target_numerator_col
        self.team_total_col = team_total_col
        self.baseline_share_col = baseline_share_col
        self.feature_columns = list(feature_columns)
        self.active_prob_col = active_prob_col
        self._random_state = random_state
        self._model: HistGradientBoostingRegressor | None = None
        self._fitted = False

    def _renormalized_baseline(self, frame: pd.DataFrame, group_columns: tuple[str, ...]) -> np.ndarray:
        b = pd.to_numeric(frame[self.baseline_share_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if self.active_prob_col in frame.columns:
            pa = pd.to_numeric(frame[self.active_prob_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        else:
            pa = pd.Series(1.0, index=frame.index)
        weighted = (b * pa).to_numpy(dtype=float)
        denom = pd.Series(weighted, index=frame.index).groupby(
            [frame[c] for c in group_columns]).transform("sum").to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            bstar = np.where(denom > 0, weighted / denom, 0.0)
        return bstar

    def fit(
        self,
        frame: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
        *,
        group_columns: tuple[str, ...] = ("game_id", "team_id", "prediction_cutoff_utc"),
    ) -> "TeamOpportunityShareModel":
        for c in (self.target_numerator_col, self.team_total_col, self.baseline_share_col):
            if c not in frame.columns:
                raise KeyError(f"TeamOpportunityShareModel.fit: missing {c!r}")
        for c in group_columns:
            if c not in frame.columns:
                raise KeyError(f"TeamOpportunityShareModel.fit: missing group column {c!r}")
        num = pd.to_numeric(frame[self.target_numerator_col], errors="coerce").fillna(0.0)
        tot = pd.to_numeric(frame[self.team_total_col], errors="coerce")
        valid = tot.notna() & (tot > 0)
        s = np.zeros(len(frame))
        s[valid.to_numpy()] = (num[valid] / tot[valid]).to_numpy()
        bstar = self._renormalized_baseline(frame, group_columns)
        delta = np.log(np.clip(s, 0, None) + _EPS) - np.log(bstar + _EPS)
        X = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        fit_mask = valid.to_numpy() & np.isfinite(delta)
        model = HistGradientBoostingRegressor(loss="squared_error", random_state=self._random_state)
        sw = None if sample_weight is None else np.asarray(sample_weight)[fit_mask]
        model.fit(X[fit_mask], delta[fit_mask], sample_weight=sw)
        self._model = model
        self._fitted = True
        return self

    def predict_team_normalized_shares(
        self,
        frame: pd.DataFrame,
        *,
        group_columns: tuple[str, ...] = ("game_id", "team_id", "prediction_cutoff_utc"),
    ) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("TeamOpportunityShareModel.predict before fit")
        for c in group_columns:
            if c not in frame.columns:
                raise RuntimeError(f"predict_team_normalized_shares: roster/group column {c!r} unavailable")
        bstar = self._renormalized_baseline(frame, group_columns)
        X = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        delta_hat = self._model.predict(X)
        w = bstar * np.exp(np.clip(delta_hat, -10.0, 10.0))
        # Zero out players with (near-)zero active probability so they cannot absorb share.
        if self.active_prob_col in frame.columns:
            pa = pd.to_numeric(frame[self.active_prob_col], errors="coerce").fillna(1.0).to_numpy()
            w = np.where(pa < 1e-4, 0.0, w)
        w_series = pd.Series(w, index=frame.index)
        denom = w_series.groupby([frame[c] for c in group_columns]).transform("sum")
        shares = np.where(denom.to_numpy() > 0, w / denom.to_numpy(), 0.0)
        return shares
