"""Causal Meta-Learner for Usage Transfer (Enhancement 11).

Replaces position-aware default weights in usage_transfer.py with
DR-learner (Doubly Robust) causal estimates of heterogeneous treatment
effects (HTE).

Theory
------
For each (beneficiary_player, absent_teammate, stat) triple:
  Treatment T = 1 if teammate is absent, 0 if present.
  Outcome    Y = beneficiary's per-minute stat rate in that game.
  Covariates X = matchup features, fatigue, role, opponent context.

DR-learner (Chernozhukov et al. 2018 double cross-fitting):
  1. Propensity model  e_hat(X) = P(T=1|X)   [injury selection model]
  2. Outcome models    mu_0(X) = E[Y|X,T=0],  mu_1(X) = E[Y|X,T=1]
  3. Pseudo-outcome    psi = (mu1 - mu0) + (T - e_hat)/(e_hat*(1-e_hat)) * (Y - mu_T)
  4. Regress psi on X to get tau_hat(x)

The DR estimator is doubly robust: consistent if EITHER the propensity
model OR the outcome model is correctly specified.  This beats S-learners
(which treat treatment as one feature among hundreds) and T-learners
(which ignore the propensity).

Falls back to position-aware defaults from usage_transfer.py when fewer
than min_obs_treated treated observations exist for a pair.

References
----------
Okasa (2022). Meta-Learners for Estimation of Causal Effects: Finite Sample
Cross-Fit Performance. arXiv:2201.12692
Acharki et al. (2022). Comparison of meta-learners for estimating multi-valued
treatment heterogeneous effects. arXiv:2205.14714
Uehara (2026). Bayesian X-Learner. arXiv:2604.27394
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Stats for which we estimate causal transfer
CAUSAL_STATS = ["pts", "reb", "ast", "fg3m", "turnover", "stl", "blk"]

# Minimum treated observations required to use DR-learner (else fall back)
MIN_OBS_TREATED = 15

# Identity / leakage columns always excluded from covariate matrix
_EXCLUDE_FROM_COVARIATES = {
    "player_id", "game_id", "game_date", "team_id", "opponent_team_id",
    "player_name", "stat", "actual_outcome", "actual_pts", "actual_reb",
    "actual_ast", "actual_fg3m", "actual_stl", "actual_blk", "actual_turnover",
    "actual_minutes", "feature_build_timestamp_utc", "feature_cutoff_policy",
}


class CausalTransferEstimator:
    """Estimate heterogeneous usage transfer using DR-learner meta-learner.

    For each teammate-player pair and each stat, estimates the CAUSAL
    transfer effect: how much more (or less) of `stat` does the beneficiary
    produce PER MINUTE when `teammate` is absent?

    Parameters
    ----------
    n_folds : int
        Number of cross-fitting folds (default 5).
    min_obs_treated : int
        Minimum absent-game observations needed to trust DR estimates.
    """

    def __init__(self, n_folds: int = 5, min_obs_treated: int = MIN_OBS_TREATED):
        self.n_folds = n_folds
        self.min_obs_treated = min_obs_treated
        self.transfer_effects: dict[tuple, dict[str, Any]] = {}
        self._is_fitted = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        teammate_ids: list[int],
        stat_cols: list[str] | None = None,
    ) -> "CausalTransferEstimator":
        """Fit DR-learner for all teammate–player–stat triples.

        Parameters
        ----------
        df : wide feature DataFrame (one row per player-game)
            Must contain per-minute rate columns: player_{stat}_per_min_l5
            and teammate_{tid}_is_out binary columns.
        teammate_ids : list of teammate player IDs to model (top-N by usage)
        stat_cols : stat names to estimate transfer effects for
        """
        if stat_cols is None:
            stat_cols = CAUSAL_STATS

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        base_covariates = [
            c for c in numeric_cols
            if c not in _EXCLUDE_FROM_COVARIATES
            and not c.startswith("actual_")
            and not c.startswith("teammate_")  # avoid treatment leakage
        ]

        n_fitted = 0
        for tid in teammate_ids:
            treatment_col = f"teammate_{tid}_is_out"
            if treatment_col not in df.columns:
                continue

            n_treated = int(df[treatment_col].sum())
            if n_treated < self.min_obs_treated:
                logger.debug(
                    "E11: teammate %s has only %d treated obs (need %d) — using defaults",
                    tid, n_treated, self.min_obs_treated,
                )
                continue

            for stat in stat_cols:
                y_col = f"player_{stat}_per_min_l5"
                if y_col not in df.columns:
                    # Try alternate naming
                    y_col_alt = f"player_{stat}_per_min_l10"
                    if y_col_alt in df.columns:
                        y_col = y_col_alt
                    else:
                        continue

                excl = _EXCLUDE_FROM_COVARIATES | {treatment_col, y_col}
                X_cols = [c for c in base_covariates if c not in excl]
                if not X_cols:
                    continue

                valid = df[[treatment_col, y_col] + X_cols].dropna()
                if len(valid) < self.min_obs_treated * 3:
                    continue

                X = valid[X_cols].values.astype(float)
                T = valid[treatment_col].values.astype(int)
                Y = valid[y_col].values.astype(float)

                tau_preds = np.full(len(valid), np.nan)
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

                for train_idx, test_idx in kf.split(X):
                    X_tr, X_te = X[train_idx], X[test_idx]
                    T_tr, T_te = T[train_idx], T[test_idx]
                    Y_tr = Y[train_idx]

                    # Step 1: Propensity model e_hat(X)
                    try:
                        e_model = GradientBoostingClassifier(
                            n_estimators=100, max_depth=3, random_state=42,
                            min_samples_leaf=5,
                        )
                        e_model.fit(X_tr, T_tr)
                        e_hat = np.clip(e_model.predict_proba(X_te)[:, 1], 0.05, 0.95)
                    except Exception:
                        e_hat = np.full(len(test_idx), T.mean())
                        e_hat = np.clip(e_hat, 0.05, 0.95)

                    # Step 2: Outcome models mu_0, mu_1
                    mask0 = T_tr == 0
                    mask1 = T_tr == 1
                    if mask0.sum() < 5 or mask1.sum() < 3:
                        continue

                    mu0_model = HistGradientBoostingRegressor(
                        max_iter=200, max_depth=5, random_state=42,
                        min_samples_leaf=5,
                    )
                    mu1_model = HistGradientBoostingRegressor(
                        max_iter=200, max_depth=5, random_state=42,
                        min_samples_leaf=5,
                    )
                    mu0_model.fit(X_tr[mask0], Y_tr[mask0])
                    mu1_model.fit(X_tr[mask1], Y_tr[mask1])

                    mu0_hat = mu0_model.predict(X_te)
                    mu1_hat = mu1_model.predict(X_te)

                    # Step 3: DR pseudo-outcomes
                    mu_T_hat = np.where(T_te == 1, mu1_hat, mu0_hat)
                    score = (T_te - e_hat) / (e_hat * (1.0 - e_hat) + 1e-8)
                    pseudo = (mu1_hat - mu0_hat) + score * (Y[test_idx] - mu_T_hat)

                    # Step 4: Regress pseudo-outcome on X → tau_hat
                    try:
                        tau_model = HistGradientBoostingRegressor(
                            max_iter=100, max_depth=3, random_state=42,
                            min_samples_leaf=5,
                        )
                        tau_model.fit(X_te, pseudo)
                        tau_preds[test_idx] = tau_model.predict(X_te)
                    except Exception:
                        tau_preds[test_idx] = pseudo

                # Store per-player average transfer effects
                pids = valid["player_id"].values if "player_id" in valid.columns else np.zeros(len(valid))
                for pid in np.unique(pids):
                    mask_pid = pids == pid
                    if mask_pid.sum() < 5:
                        continue
                    tau_pid = tau_preds[mask_pid]
                    n_treated_pid = int(T[mask_pid].sum())
                    key = (int(pid), int(tid), stat)
                    self.transfer_effects[key] = {
                        "mean":       float(np.nanmean(tau_pid)),
                        "std":        float(np.nanstd(tau_pid)),
                        "n_obs":      int(mask_pid.sum()),
                        "n_treated":  n_treated_pid,
                        "method":     "dr_learner",
                    }
                    n_fitted += 1

        self._is_fitted = True
        logger.info("E11 CausalTransferEstimator: fitted %d player-teammate-stat effects", n_fitted)
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def get_transfer_rate(
        self,
        player_id: int,
        teammate_id: int,
        stat: str,
        fallback_rate: float | None = None,
    ) -> float:
        """Return causal transfer rate (per-minute delta) for a player-teammate-stat.

        Falls back to `fallback_rate` (position-aware default) when
        insufficient treated observations exist.
        """
        key = (int(player_id), int(teammate_id), stat)
        if key in self.transfer_effects:
            eff = self.transfer_effects[key]
            if eff["n_treated"] >= self.min_obs_treated:
                return float(eff["mean"])
        return fallback_rate if fallback_rate is not None else 0.0

    def enrich_usage_transfer_features(
        self,
        wide: pd.DataFrame,
        player_usage_map: dict[int, dict],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Replace UTM transfer deltas with DR-learner causal estimates.

        For each player-row, for each top-N teammate and each stat, computes:
            causal_transfer_{tid}_{stat} = tau_hat(player, teammate, stat) * minutes_remaining

        If the DR-learner has insufficient data for a pair, falls back to the
        position-aware default from usage_transfer.py.

        Parameters
        ----------
        wide : the wide feature DataFrame (post usage_transfer_features)
        player_usage_map : {player_id: {usage_season, position_group, ...}}
        top_n : number of top teammates to compute causal transfers for
        """
        from wnba_props_model.models.usage_transfer import POS_TRANSFER_WEIGHTS  # noqa: PLC0415

        if not self._is_fitted or not player_usage_map:
            return wide

        sorted_teammates = sorted(
            player_usage_map.items(), key=lambda x: -x[1]["usage_season"]
        )[:top_n]

        new_cols: dict[str, list] = {}
        stats = ["pts", "reb", "ast", "fg3m", "turnover"]

        for stat in stats:
            for tid, t_info in sorted_teammates:
                col = f"causal_transfer_{tid}_{stat}"
                vals = []
                for _, row in wide.iterrows():
                    pid = int(row.get("player_id", 0))
                    is_out = float(row.get(f"teammate_{tid}_is_out", 0))
                    if not is_out:
                        vals.append(0.0)
                        continue

                    # Position-aware fallback
                    player_pos = player_usage_map.get(pid, {}).get("position_group", "wing")
                    absent_pos = t_info.get("position_group", "wing")
                    default_weight = POS_TRANSFER_WEIGHTS.get((player_pos, absent_pos), 0.15)
                    fallback = t_info["usage_season"] * default_weight

                    causal = self.get_transfer_rate(pid, tid, stat, fallback_rate=fallback)
                    vals.append(float(causal))
                new_cols[col] = vals

        if new_cols:
            wide = pd.concat([wide, pd.DataFrame(new_cols, index=wide.index)], axis=1)

        # Aggregate: total causal boost across all absent teammates
        for stat in stats:
            agg_col = f"causal_transfer_total_{stat}"
            stat_cols = [f"causal_transfer_{tid}_{stat}" for tid, _ in sorted_teammates
                         if f"causal_transfer_{tid}_{stat}" in wide.columns]
            if stat_cols:
                wide[agg_col] = wide[stat_cols].fillna(0).sum(axis=1)

        logger.info("E11: causal transfer features added for %d stats × %d teammates",
                    len(stats), len(sorted_teammates))
        return wide

    def summary(self) -> dict[str, Any]:
        """Return summary of fitted transfer effects."""
        if not self._is_fitted:
            return {"fitted": False}
        n = len(self.transfer_effects)
        effects = list(self.transfer_effects.values())
        means = [e["mean"] for e in effects if e["method"] == "dr_learner"]
        return {
            "fitted": True,
            "n_effects": n,
            "mean_transfer": float(np.mean(means)) if means else 0.0,
            "pct_dr_learner": len(means) / max(n, 1),
        }


# ---------------------------------------------------------------------------
# Convenience factory function (called from build_features.py)
# ---------------------------------------------------------------------------

def train_causal_transfer(
    df: "pd.DataFrame",
    player_usage_map: dict,
    top_n: int = 5,
    n_folds: int = 5,
) -> CausalTransferEstimator:
    """Fit a CausalTransferEstimator on the top-N highest-usage players.

    Parameters
    ----------
    df : wide feature DataFrame (one row per player-game)
    player_usage_map : {player_id: {usage_season, position_group, ...}}
    top_n : number of top-usage teammates to model causal effects for
    n_folds : DR-learner cross-fitting folds

    Returns
    -------
    Fitted CausalTransferEstimator
    """
    sorted_players = sorted(
        player_usage_map.items(), key=lambda x: -x[1].get("usage_season", 0.0)
    )
    teammate_ids = [int(pid) for pid, _ in sorted_players[:top_n]]
    estimator = CausalTransferEstimator(n_folds=n_folds)
    estimator.fit(df, teammate_ids)
    return estimator
