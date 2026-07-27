"""Hierarchical, exposure-weighted empirical-Bayes rate estimator for Opportunity V2.

Estimates a per-unit rate (numerator/denominator) with nested shrinkage toward progressively coarser
parents (league -> position -> role -> position*role -> team*role -> player). Must be fit INSIDE each
OOF fold on training rows only; never on the complete feature table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _key_series(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    """Stable tuple key for a group level (as a string to survive mixed dtypes / NaN)."""
    if not cols:
        return pd.Series(["__global__"] * len(frame), index=frame.index)
    parts = [frame[c].astype("string").fillna("<NA>") for c in cols]
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep="||")
    return key


class HierarchicalLaggedRate:
    """Nested empirical-Bayes shrinkage estimator: ``r_g = (sum y + k*r_parent) / (sum e + k)``."""

    def __init__(self) -> None:
        self._levels: list[tuple[str, ...]] = []
        self._level_rates: list[dict[str, float]] = []
        self._level_exposure: list[dict[str, float]] = []
        self._global_rate: float = 0.0
        self._num_col: str = ""
        self._den_col: str = ""
        self._prior_strength: float = 0.0
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        numerator_col: str,
        denominator_col: str,
        hierarchy: list[tuple[str, ...]],
        prior_strength: float,
    ) -> "HierarchicalLaggedRate":
        if numerator_col not in frame.columns or denominator_col not in frame.columns:
            raise KeyError(f"HierarchicalLaggedRate.fit: missing {numerator_col!r}/{denominator_col!r}")
        num = pd.to_numeric(frame[numerator_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        den = pd.to_numeric(frame[denominator_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.any(den < 0) or np.any(num < 0):
            raise ValueError("HierarchicalLaggedRate.fit: negative numerator/denominator")
        self._num_col, self._den_col = numerator_col, denominator_col
        self._prior_strength = float(prior_strength)
        self._levels = [tuple(h) for h in hierarchy]

        total_den = float(den.sum())
        self._global_rate = float(num.sum() / total_den) if total_den > 0 else 0.0

        # Parent prediction per training row, refined level by level.
        parent_pred = np.full(len(frame), self._global_rate, dtype=float)
        self._level_rates = []
        self._level_exposure = []
        for cols in self._levels:
            keys = _key_series(frame, cols)
            df = pd.DataFrame({"key": keys.to_numpy(), "num": num, "den": den, "parent": parent_pred})
            grp = df.groupby("key", sort=False)
            agg = grp.agg(sum_num=("num", "sum"), sum_den=("den", "sum"),
                          par=("parent", lambda s: float(np.average(
                              s, weights=df.loc[s.index, "den"]) if df.loc[s.index, "den"].sum() > 0
                              else s.mean())))
            k = self._prior_strength
            rates = (agg["sum_num"] + k * agg["par"]) / (agg["sum_den"] + k)
            rate_map = rates.to_dict()
            exp_map = agg["sum_den"].to_dict()
            self._level_rates.append(rate_map)
            self._level_exposure.append(exp_map)
            # Update parent_pred for the NEXT level using this level's estimate.
            parent_pred = keys.map(rate_map).fillna(self._global_rate).to_numpy(dtype=float)
        self._fitted = True
        return self

    def _predict_pair(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("HierarchicalLaggedRate.predict before fit")
        rate = np.full(len(frame), self._global_rate, dtype=float)
        exposure = np.zeros(len(frame), dtype=float)
        for cols, rate_map, exp_map in zip(self._levels, self._level_rates, self._level_exposure):
            keys = _key_series(frame, cols)
            mapped = keys.map(rate_map)
            hit = mapped.notna().to_numpy()
            rate[hit] = mapped[hit].to_numpy(dtype=float)
            exp_mapped = keys.map(exp_map)
            exposure[hit] = exp_mapped[hit].fillna(0.0).to_numpy(dtype=float)
        return rate, exposure

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._predict_pair(frame)[0]

    def predict_with_exposure(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (rate, exposure) where exposure is the finest-matched-group total denominator."""
        return self._predict_pair(frame)
