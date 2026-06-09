from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from wnba_props_model.constants import DOMAIN_MAX, QUANTILES
from wnba_props_model.models.base import QuantileModelBundle
from wnba_props_model.models.simulation import normalize_pmf


GAME_TOTAL_FEATURES = [
    "home_team_points_roll3", "home_team_points_roll5", "home_team_points_roll10",
    "home_opp_points_roll3", "home_opp_points_roll5", "home_opp_points_roll10",
    "away_team_points_roll3", "away_team_points_roll5", "away_team_points_roll10",
    "away_opp_points_roll3", "away_opp_points_roll5", "away_opp_points_roll10",
    "home_rest_days", "away_rest_days", "postseason",
]


def pmf_from_total_quantiles(q: dict[float, float], domain_max: int = DOMAIN_MAX["game_total"], draws: int = 50000, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    qs = np.array(sorted(q))
    vals = np.maximum.accumulate([q[x] for x in qs])
    u = rng.uniform(qs.min(), qs.max(), size=draws)
    samples = np.interp(u, qs, vals)
    # Add light discrete noise because game totals are integer sums of many scoring events.
    samples = np.rint(samples + rng.normal(0, 1.75, size=draws)).astype(int)
    samples = np.clip(samples, 0, domain_max)
    return normalize_pmf(np.bincount(samples, minlength=domain_max + 1).astype(float))


@dataclass
class GameTotalsModel:
    features: list[str]
    quantile_bundle: QuantileModelBundle | None = None

    def fit(self, df: pd.DataFrame) -> "GameTotalsModel":
        features = [f for f in self.features if f in df.columns]
        self.features = features
        self.quantile_bundle = QuantileModelBundle("game_total", features, QUANTILES).fit(df, "game_total")
        return self

    def predict_pmfs(self, df: pd.DataFrame, draws: int = 50000) -> list[np.ndarray]:
        if self.quantile_bundle is None:
            raise RuntimeError("GameTotalsModel not fit")
        qs = self.quantile_bundle.predict_quantiles(df)
        return [pmf_from_total_quantiles(q, draws=draws) for q in qs]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "GameTotalsModel":
        return joblib.load(path)
