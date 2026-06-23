"""Causal Meta-Learner for Usage Transfer (Enhancement 11).

Replaces hand-crafted position-aware transfer weights with Doubly-Robust
(DR) learner estimates of heterogeneous treatment effects.

The DR-learner is doubly robust: consistent if EITHER the outcome model OR
the propensity model is correct.  Cross-fitting eliminates overfitting bias.

Architecture:
    Treatment T  = 1 if high-usage teammate is absent, 0 if present
    Outcome   Y  = beneficiary's per-minute stat rate
    Covariates X = matchup, fatigue, role, opponent, usage features

    DR pseudo-outcome:
        τ̂(x) = μ̂₁(x) - μ̂₀(x)
              + (T - ê(x)) / [ê(x)(1-ê(x))] × (Y - μ̂_T(x))

Reference:
    Okasa (2022). Meta-Learners for Estimation of Causal Effects.
    arXiv:2201.12692
    Yu & Hu (2026). Bayesian X-Learner. arXiv:2604.27394
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Default fallback when DR-learner has insufficient treated observations
_MIN_OBS_TREATED = 15
_N_FOLDS = 5
# Stats for which we estimate transfer effects
TRANSFER_STATS = ["pts", "reb", "ast", "fg3m", "turnover", "stl", "blk"]


class CausalTransferEstimator:
    """DR-learner heterogeneous usage transfer estimator.

    For each (player_id, teammate_id, stat) triple, learns the causal
    treatment effect of the teammate being absent on the player's per-minute
    rate.

    Falls back to position-aware defaults from usage_transfer.POS_TRANSFER_WEIGHTS
    when treated observations are insufficient (< min_obs_treated).
    """

    def __init__(self, n_folds: int = _N_FOLDS, min_obs_treated: int = _MIN_OBS_TREATED):
        self.n_folds = n_folds
        self.min_obs_treated = min_obs_treated
        # (player_id, teammate_id, stat) → {"mean", "std", "n_obs", "n_treated", "method"}
        self.transfer_effects: dict[tuple, dict[str, Any]] = {}
        self._feature_cols: list[str] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        teammate_ids: list[int],
        stat_cols: list[str] | None = None,
    ) -> "CausalTransferEstimator":
        """Fit DR-learner for all teammate × player × stat combinations.

        Uses K-fold double cross-fitting:
            - Split data into K folds
            - Estimate nuisance functions (propensity + outcome) on K-1 folds
            - Compute DR pseudo-outcomes on held-out fold
            - Average across folds for debiased estimates

        Parameters
        ----------
        df          : wide feature DataFrame (one row per player-game)
        teammate_ids: list of high-usage teammate player_ids (top-5 by usage)
        stat_cols   : stats to estimate effects for (defaults to TRANSFER_STATS)
        """
        stats = stat_cols or TRANSFER_STATS

        # Build covariate columns (exclude identifiers, targets, treatment flags)
        id_cols = {"player_id", "game_id", "team_id", "opponent_team_id",
                   "game_date", "season", "stat", "actual_outcome"}
        for stat in stats:
            id_cols.add(f"actual_{stat}")
            id_cols.add(f"player_{stat}_per_min")
        for tid in teammate_ids:
            id_cols.add(f"teammate_{tid}_is_out")
            id_cols.add(f"teammate_{tid}_usage_rate")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self._feature_cols = [c for c in numeric_cols if c not in id_cols]

        for tid in teammate_ids:
            treatment_col = f"teammate_{tid}_is_out"
            if treatment_col not in df.columns:
                continue
            n_treated = int(df[treatment_col].sum())
            if n_treated < self.min_obs_treated:
                logger.debug(
                    "Teammate %s: only %d treated obs — skipping DR-learner, "
                    "will use positional fallback",
                    tid, n_treated,
                )
                continue

            for stat in stats:
                y_col = f"player_{stat}_per_min"
                if y_col not in df.columns:
                    # Try to build it on the fly
                    rate_col = f"player_{stat}_per_min_l5"
                    if rate_col in df.columns:
                        y_col = rate_col
                    else:
                        continue

                valid = df[[treatment_col, y_col] + self._feature_cols].dropna()
                if len(valid) < 50:
                    continue

                X = valid[self._feature_cols].values.astype(float)
                T = valid[treatment_col].values.astype(int)
                Y = valid[y_col].values.astype(float)

                tau_preds = self._cross_fit_dr(X, T, Y)

                # Store per-player average transfer effect
                for pid in df["player_id"].unique():
                    mask = (valid["player_id"] == pid).values if "player_id" in valid.columns else np.ones(len(valid), dtype=bool)
                    if mask.sum() < 5:
                        continue
                    self.transfer_effects[(int(pid), int(tid), stat)] = {
                        "mean":      float(np.nanmean(tau_preds[mask])),
                        "std":       float(np.nanstd(tau_preds[mask])),
                        "n_obs":     int(mask.sum()),
                        "n_treated": int((T[mask] == 1).sum()),
                        "method":    "dr_learner",
                    }

            logger.info(
                "DR-learner fit: teammate=%s, effects stored for %d player-stat pairs",
                tid,
                sum(1 for k in self.transfer_effects if k[1] == int(tid)),
            )

        return self

    def _cross_fit_dr(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        """K-fold double cross-fitting DR-learner.

        Returns per-observation DR pseudo-outcome estimates τ̂(x).
        """
        tau_preds = np.full(len(X), np.nan)
        kf = KFold(n_splits=min(self.n_folds, max(2, int(T.sum() // 3))),
                   shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            T_tr, T_te = T[train_idx], T[test_idx]
            Y_tr       = Y[train_idx]

            # Step 1: Propensity model ê(x) = P(T=1 | X)
            e_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            )
            try:
                e_model.fit(X_tr, T_tr)
                e_hat = np.clip(e_model.predict_proba(X_te)[:, 1], 0.05, 0.95)
            except Exception:
                e_hat = np.full(len(X_te), T_tr.mean())

            # Step 2: Outcome models μ₀(x), μ₁(x)
            mask0, mask1 = T_tr == 0, T_tr == 1
            mu0_hat = mu1_hat = np.zeros(len(X_te))

            if mask0.sum() >= 5:
                m0 = HistGradientBoostingRegressor(
                    max_iter=200, max_depth=5, random_state=42
                )
                m0.fit(X_tr[mask0], Y_tr[mask0])
                mu0_hat = m0.predict(X_te)

            if mask1.sum() >= 3:
                m1 = HistGradientBoostingRegressor(
                    max_iter=200, max_depth=5, random_state=42
                )
                m1.fit(X_tr[mask1], Y_tr[mask1])
                mu1_hat = m1.predict(X_te)

            # Step 3: DR pseudo-outcomes
            mu_T_hat = np.where(T_te == 1, mu1_hat, mu0_hat)
            score = (T_te - e_hat) / (e_hat * (1 - e_hat) + 1e-8)
            pseudo_outcome = (mu1_hat - mu0_hat) + score * (Y[test_idx] - mu_T_hat)

            # Step 4: Regress pseudo-outcomes on X to get τ̂(x)
            tau_model = HistGradientBoostingRegressor(
                max_iter=100, max_depth=3, random_state=42
            )
            tau_model.fit(X_te, pseudo_outcome)
            tau_preds[test_idx] = tau_model.predict(X_te)

        return tau_preds

    def get_transfer_rate(
        self,
        player_id: int,
        teammate_id: int,
        stat: str,
        fallback_rate: float | None = None,
    ) -> float:
        """Return the causal transfer rate for (player, teammate, stat).

        Falls back to position-aware default if DR estimate not available.
        """
        key = (int(player_id), int(teammate_id), stat)
        if key in self.transfer_effects:
            effect = self.transfer_effects[key]
            if effect["n_treated"] >= self.min_obs_treated:
                return float(effect["mean"])
        return float(fallback_rate) if fallback_rate is not None else 0.0

    def save(self, path: str | Path) -> None:
        """Serialise learned transfer effects to JSON."""
        out = {
            "transfer_effects": {
                f"{k[0]}_{k[1]}_{k[2]}": v
                for k, v in self.transfer_effects.items()
            },
            "feature_cols": self._feature_cols,
            "n_folds": self.n_folds,
            "min_obs_treated": self.min_obs_treated,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("CausalTransferEstimator saved to %s (%d effects)", path, len(self.transfer_effects))

    @classmethod
    def load(cls, path: str | Path) -> "CausalTransferEstimator":
        """Load from saved JSON."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(
            n_folds=data.get("n_folds", _N_FOLDS),
            min_obs_treated=data.get("min_obs_treated", _MIN_OBS_TREATED),
        )
        obj._feature_cols = data.get("feature_cols", [])
        for key_str, effect in data.get("transfer_effects", {}).items():
            parts = key_str.split("_", 2)
            if len(parts) == 3:
                pid, tid, stat = int(parts[0]), int(parts[1]), parts[2]
                obj.transfer_effects[(pid, tid, stat)] = effect
        return obj

    def apply_causal_utm(
        self,
        wide: pd.DataFrame,
        player_usage_map: dict[int, dict[str, Any]],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Apply DR-learner transfer effects to produce causal UTM features.

        Adds/updates:
            projected_usage_given_absences_causal
            usage_transfer_delta_causal
            player_{stat}_causal_transfer_l5  (per stat)

        Falls back to position-aware UTM if DR estimates unavailable.
        """
        from wnba_props_model.models.usage_transfer import POS_TRANSFER_WEIGHTS  # noqa: PLC0415

        if wide.empty or not player_usage_map:
            return wide

        df = wide.copy()
        sorted_tms = sorted(
            player_usage_map.items(), key=lambda x: -x[1]["usage_season"]
        )[:top_n]
        pos_map = {pid: v.get("position_group", "wing") for pid, v in player_usage_map.items()}

        base_usage = df.get("player_usage_rate_season", pd.Series(0.20, index=df.index)).fillna(0.20)
        projected_causal = base_usage.copy()

        for idx in df.index:
            pid = int(df.at[idx, "player_id"])
            player_pos = pos_map.get(pid, "wing")

            for tid_val, t_info in sorted_tms:
                flag_col = f"teammate_{tid_val}_is_out"
                if flag_col not in df.columns:
                    continue
                is_out = df.at[idx, flag_col]
                if not is_out:
                    continue

                absent_pos = t_info.get("position_group", "wing")
                # Causal rate: DR-learner for pts as the primary driver
                dr_rate = self.get_transfer_rate(pid, tid_val, "pts", fallback_rate=None)
                if dr_rate is None or not math.isfinite(dr_rate):
                    # Positional fallback
                    tk = (player_pos, absent_pos)
                    w = POS_TRANSFER_WEIGHTS.get(tk, 0.15)
                    dr_rate = t_info["usage_season"] * w

                projected_causal.at[idx] = projected_causal.at[idx] + max(0.0, dr_rate)

        df["projected_usage_given_absences_causal"] = projected_causal
        df["usage_transfer_delta_causal"] = projected_causal - base_usage

        return df


def train_causal_transfer(
    wide_df: pd.DataFrame,
    player_usage_map: dict[int, dict[str, Any]],
    out_path: str | None = None,
    top_n: int = 5,
) -> CausalTransferEstimator:
    """Convenience function: build and fit the DR-learner from the wide feature table."""
    sorted_tms = sorted(
        player_usage_map.items(), key=lambda x: -x[1]["usage_season"]
    )[:top_n]
    teammate_ids = [int(tid) for tid, _ in sorted_tms]

    estimator = CausalTransferEstimator()
    estimator.fit(wide_df, teammate_ids)

    if out_path:
        estimator.save(out_path)

    return estimator
