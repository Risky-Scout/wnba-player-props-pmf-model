"""Team-game environment model for Opportunity V2 (expected possessions / opportunity totals).

Predicts nonnegative team count targets from strictly-lagged team features using Poisson-loss
gradient boosting, then reconciles the two teams' possession estimates for a game by averaging.
Targets without adequate historical coverage are marked unavailable rather than fabricated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_MIN_COVERAGE = 50  # minimum non-null target rows to fit a target honestly


def box_possessions(fga: np.ndarray, fta: np.ndarray, oreb: np.ndarray, tov: np.ndarray) -> np.ndarray:
    """Box-score possession estimate: FGA + 0.44*FTA - OREB + TOV."""
    return np.asarray(fga, float) + 0.44 * np.asarray(fta, float) - np.asarray(oreb, float) + np.asarray(tov, float)


class TeamEnvironmentModelV2:
    VERSION = "opportunity_v2_team_environment_v1"

    TARGETS = (
        "possessions", "fga", "fg3a", "fta", "fg_misses", "turnovers",
        "rim_attempts", "potential_assists", "rebound_chances", "touches",
    )

    def __init__(self, *, reconcile_possessions: bool = True, random_state: int = 42) -> None:
        self._models: dict[str, HistGradientBoostingRegressor] = {}
        self._feature_columns: list[str] = []
        self._target_available: dict[str, bool] = {}
        self._reconcile = reconcile_possessions
        self._random_state = random_state
        self._fitted = False

    def fit(
        self,
        team_game_frame: pd.DataFrame,
        feature_columns: list[str],
        sample_weight: np.ndarray | None = None,
    ) -> "TeamEnvironmentModelV2":
        missing = [c for c in feature_columns if c not in team_game_frame.columns]
        if missing:
            raise KeyError(f"TeamEnvironmentModelV2.fit: missing feature columns {missing}")
        self._feature_columns = list(feature_columns)
        X = team_game_frame[self._feature_columns].apply(pd.to_numeric, errors="coerce")
        for target in self.TARGETS:
            if target not in team_game_frame.columns:
                self._target_available[target] = False
                continue
            y = pd.to_numeric(team_game_frame[target], errors="coerce")
            valid = y.notna() & (y >= 0)
            if int(valid.sum()) < _MIN_COVERAGE:
                self._target_available[target] = False
                continue
            model = HistGradientBoostingRegressor(loss="poisson", random_state=self._random_state)
            sw = None if sample_weight is None else np.asarray(sample_weight)[valid.to_numpy()]
            model.fit(X[valid], y[valid], sample_weight=sw)
            self._models[target] = model
            self._target_available[target] = True
        self._fitted = True
        return self

    @property
    def target_available(self) -> dict[str, bool]:
        return dict(self._target_available)

    def predict(self, team_game_features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TeamEnvironmentModelV2.predict before fit")
        X = team_game_features[self._feature_columns].apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame(index=team_game_features.index)
        for target, model in self._models.items():
            out[target] = np.clip(model.predict(X), 0.0, None)
        for target in self.TARGETS:
            out[f"{target}_available"] = bool(self._target_available.get(target, False))

        if self._reconcile and "possessions" in out.columns and \
                {"game_id", "team_id", "opponent_team_id"}.issubset(team_game_features.columns):
            ctx = team_game_features[["game_id", "team_id", "opponent_team_id"]].copy()
            ctx["possessions"] = out["possessions"].to_numpy()
            # Map each team's opponent possession estimate within the same game.
            key = ctx.set_index(["game_id", "team_id"])["possessions"]
            opp = ctx.apply(lambda r: key.get((r["game_id"], r["opponent_team_id"]), np.nan), axis=1)
            reconciled = np.where(opp.notna().to_numpy(),
                                  (out["possessions"].to_numpy() + opp.to_numpy()) / 2.0,
                                  out["possessions"].to_numpy())
            out["possessions"] = reconciled
        return out
