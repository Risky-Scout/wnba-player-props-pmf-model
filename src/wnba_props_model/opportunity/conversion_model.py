"""Hierarchical Beta conversion model for Opportunity V2.

Estimates a conversion probability (successes/attempts) with nested empirical-Bayes shrinkage and
RETAINS Beta uncertainty: the returned posterior concentration grows with observed exposure so
low-sample players stay appropriately wide. Fit inside each OOF fold on training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline_rates import HierarchicalLaggedRate

_EPS = 1e-9


@dataclass
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class HierarchicalBetaConversionModel:
    VERSION = "opportunity_v2_conversion_v1"

    def __init__(self) -> None:
        self._rate = HierarchicalLaggedRate()
        self._prior_strength = 0.0
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        successes_col: str,
        attempts_col: str,
        hierarchy: list[tuple[str, ...]],
        prior_strength: float,
    ) -> "HierarchicalBetaConversionModel":
        self._prior_strength = float(prior_strength)
        self._rate.fit(frame, successes_col, attempts_col, hierarchy, prior_strength)
        self._fitted = True
        return self

    def predict_posterior(self, frame: pd.DataFrame) -> list[BetaPosterior]:
        if not self._fitted:
            raise RuntimeError("HierarchicalBetaConversionModel.predict_posterior before fit")
        rate, exposure = self._rate.predict_with_exposure(frame)
        rate = np.clip(rate, _EPS, 1.0 - _EPS)
        # Concentration: observed attempts at the finest matched group + prior strength. This keeps
        # posteriors wide for sparse groups and tight for well-observed ones.
        concentration = np.asarray(exposure, dtype=float) + self._prior_strength
        concentration = np.clip(concentration, self._prior_strength + _EPS, None)
        out: list[BetaPosterior] = []
        for p, c in zip(rate, concentration):
            out.append(BetaPosterior(alpha=float(p * c), beta=float((1.0 - p) * c)))
        return out

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([bp.mean for bp in self.predict_posterior(frame)], dtype=float)
