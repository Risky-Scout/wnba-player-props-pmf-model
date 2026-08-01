"""Chronological reliability weights for pick-probability shrinkage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.pick_engine.probabilities import EPS, pick_probability

MIN_ROWS_LEAF = 80
MIN_DATES_LEAF = 10
MIN_ROWS_STAT = 40
MIN_DATES_STAT = 6


@dataclass
class ReliabilityWeights:
    """Partial-pooling reliability weights by segment."""

    global_weight: float
    by_stat: dict[str, float] = field(default_factory=dict)
    by_stat_role: dict[str, float] = field(default_factory=dict)
    by_horizon: dict[str, float] = field(default_factory=dict)
    n_training_rows: int = 0
    training_date_min: str | None = None
    training_date_max: str | None = None
    weights_hash: str = ""
    method: str = "chronological_oof_logit_shrinkage_v1"

    def weight_for(
        self,
        *,
        stat: str,
        role: str | None = None,
        horizon: str | None = None,
    ) -> float:
        """Resolve leaf -> parent -> global with shrink when under-sampled keys absent."""
        if horizon and horizon in self.by_horizon:
            return float(self.by_horizon[horizon])
        key = f"{stat}|{role}" if role else None
        if key and key in self.by_stat_role:
            return float(self.by_stat_role[key])
        if stat in self.by_stat:
            return float(self.by_stat[stat])
        return float(self.global_weight)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip_w(w: float) -> float:
    return float(min(1.0, max(0.0, w)))


def _fit_w_grid(
    p_pure: np.ndarray,
    p_ref: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Choose w in [0,1] minimizing log loss of pick_probability vs outcomes."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_w, best_ll = 0.0, float("inf")
    y = y.astype(float)
    for w in grid:
        preds = np.array(
            [pick_probability(float(pp), float(pr), float(w)) for pp, pr in zip(p_pure, p_ref)]
        )
        preds = np.clip(preds, EPS, 1.0 - EPS)
        ll = float(np.mean(-(y * np.log(preds) + (1.0 - y) * np.log(1.0 - preds))))
        if ll < best_ll - 1e-15 or (abs(ll - best_ll) <= 1e-15 and w < best_w):
            best_ll, best_w = ll, float(w)
    return _clip_w(best_w)


def _partial_pool(child_w: float, parent_w: float, n: int, n_min: int) -> float:
    """Shrink a child weight toward its parent when n is inadequate."""
    if n >= n_min:
        return _clip_w(child_w)
    alpha = n / max(n_min, 1)
    return _clip_w(alpha * child_w + (1.0 - alpha) * parent_w)


def fit_reliability_weights(
    frame: pd.DataFrame,
    *,
    pure_col: str = "pure_probability",
    ref_col: str = "reference_market_probability",
    outcome_col: str = "outcome_over",
    date_col: str = "game_date",
    stat_col: str = "stat",
    role_col: str = "role_bucket",
    horizon_col: str | None = None,
) -> ReliabilityWeights:
    """Fit chronological OOF reliability weights with partial pooling.

    Expects historical rows where pure and reference probabilities were known before
    the outcome. Fits on all provided rows assuming the caller already enforced
    chronological eligibility (no future leakage into features).
    """
    need = {pure_col, ref_col, outcome_col, date_col, stat_col}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"reliability frame missing columns: {sorted(missing)}")

    df = frame.dropna(subset=[pure_col, ref_col, outcome_col, date_col]).copy()
    if df.empty:
        weights = ReliabilityWeights(global_weight=0.0, n_training_rows=0)
        weights.weights_hash = _hash_weights(weights)
        return weights

    p_pure = df[pure_col].to_numpy(float)
    p_ref = df[ref_col].to_numpy(float)
    y = df[outcome_col].to_numpy(float)
    w_global = _fit_w_grid(p_pure, p_ref, y)

    by_stat: dict[str, float] = {}
    for stat, g in df.groupby(stat_col, sort=False):
        n_dates = g[date_col].nunique()
        raw = _fit_w_grid(
            g[pure_col].to_numpy(float),
            g[ref_col].to_numpy(float),
            g[outcome_col].to_numpy(float),
        )
        by_stat[str(stat)] = _partial_pool(raw, w_global, len(g), MIN_ROWS_STAT)
        if n_dates < MIN_DATES_STAT:
            by_stat[str(stat)] = _partial_pool(
                by_stat[str(stat)], w_global, n_dates, MIN_DATES_STAT
            )

    by_stat_role: dict[str, float] = {}
    if role_col in df.columns:
        for (stat, role), g in df.groupby([stat_col, role_col], sort=False):
            parent = by_stat.get(str(stat), w_global)
            raw = _fit_w_grid(
                g[pure_col].to_numpy(float),
                g[ref_col].to_numpy(float),
                g[outcome_col].to_numpy(float),
            )
            key = f"{stat}|{role}"
            by_stat_role[key] = _partial_pool(raw, parent, len(g), MIN_ROWS_LEAF)
            if g[date_col].nunique() < MIN_DATES_LEAF:
                by_stat_role[key] = _partial_pool(
                    by_stat_role[key], parent, g[date_col].nunique(), MIN_DATES_LEAF
                )

    by_horizon: dict[str, float] = {}
    if horizon_col and horizon_col in df.columns:
        for horizon, g in df.groupby(horizon_col, sort=False):
            raw = _fit_w_grid(
                g[pure_col].to_numpy(float),
                g[ref_col].to_numpy(float),
                g[outcome_col].to_numpy(float),
            )
            by_horizon[str(horizon)] = _partial_pool(raw, w_global, len(g), MIN_ROWS_STAT)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    weights = ReliabilityWeights(
        global_weight=_clip_w(w_global),
        by_stat=by_stat,
        by_stat_role=by_stat_role,
        by_horizon=by_horizon,
        n_training_rows=int(len(df)),
        training_date_min=str(dates.min().date()) if dates.notna().any() else None,
        training_date_max=str(dates.max().date()) if dates.notna().any() else None,
    )
    weights.weights_hash = _hash_weights(weights)
    return weights


def _hash_weights(weights: ReliabilityWeights) -> str:
    payload = json.dumps(weights.as_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def default_reliability_weights() -> ReliabilityWeights:
    """Conservative defaults when historical OOF evidence is unavailable.

    Global weight > 0 so pure alpha is not fully suppressed; leaf segments shrink
    to global until enough chronological evidence exists.
    """
    # Mild trust in pure vs reference until OOF fit replaces these.
    by_stat = {
        "pts": 0.35,
        "reb": 0.30,
        "ast": 0.40,
        "fg3m": 0.25,
        "stl": 0.20,
        "blk": 0.20,
        "turnover": 0.25,
    }
    w = ReliabilityWeights(
        global_weight=0.30,
        by_stat=by_stat,
        n_training_rows=0,
        training_date_min=None,
        training_date_max=None,
        method="default_partial_pool_prior_v1",
    )
    w.weights_hash = _hash_weights(w)
    return w


def load_or_fit_reliability_weights(
    path: str | Path | None,
    historical: pd.DataFrame | None = None,
) -> ReliabilityWeights:
    path_obj = Path(path) if path else None
    if path_obj and path_obj.exists():
        data = json.loads(path_obj.read_text())
        w = ReliabilityWeights(
            global_weight=float(data["global_weight"]),
            by_stat={str(k): float(v) for k, v in data.get("by_stat", {}).items()},
            by_stat_role={str(k): float(v) for k, v in data.get("by_stat_role", {}).items()},
            by_horizon={str(k): float(v) for k, v in data.get("by_horizon", {}).items()},
            n_training_rows=int(data.get("n_training_rows", 0)),
            training_date_min=data.get("training_date_min"),
            training_date_max=data.get("training_date_max"),
            weights_hash=str(data.get("weights_hash", "")),
            method=str(data.get("method", "loaded")),
        )
        if not w.weights_hash:
            w.weights_hash = _hash_weights(w)
        return w
    if historical is not None and not historical.empty:
        return fit_reliability_weights(historical)
    return default_reliability_weights()


def save_reliability_weights(weights: ReliabilityWeights, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights.as_dict(), indent=2, sort_keys=True) + "\n")


def conservative_probability(
    pick_prob: float,
    *,
    reliability_weight: float,
    uncertainty: float,
) -> float:
    """Conservative lower probability bound from reliability + uncertainty components.

    Preferred path is logit shrinkage (already in pick_probability). This bound is an
    additional haircut for conservative EV: move toward 0.5 by uncertainty*(1-w).
    """
    w = _clip_w(reliability_weight)
    u = float(max(0.0, uncertainty))
    p = float(pick_prob)
    # Shrink toward break-even-neutral 0.5; never invent a fixed arbitrary haircut alone.
    shrink = min(1.0, u * (1.0 - w))
    return float((1.0 - shrink) * p + shrink * 0.5)


def uncertainty_components(
    *,
    calibration_uncertainty: float = 0.0,
    segment_reliability: float = 1.0,
    role_uncertainty: float = 0.0,
    availability_uncertainty: float = 0.0,
    ood_uncertainty: float = 0.0,
    quote_freshness_penalty: float = 0.0,
    model_disagreement: float = 0.0,
) -> dict[str, float]:
    comps = {
        "calibration_uncertainty": float(max(0.0, calibration_uncertainty)),
        "segment_reliability": float(min(1.0, max(0.0, segment_reliability))),
        "role_uncertainty": float(max(0.0, role_uncertainty)),
        "availability_uncertainty": float(max(0.0, availability_uncertainty)),
        "ood_uncertainty": float(max(0.0, ood_uncertainty)),
        "quote_freshness_penalty": float(max(0.0, quote_freshness_penalty)),
        "model_disagreement": float(max(0.0, model_disagreement)),
    }
    # Combine as RMS of adverse components (not an invented fixed haircut).
    adverse = [
        comps["calibration_uncertainty"],
        1.0 - comps["segment_reliability"],
        comps["role_uncertainty"],
        comps["availability_uncertainty"],
        comps["ood_uncertainty"],
        comps["quote_freshness_penalty"],
        comps["model_disagreement"],
    ]
    comps["uncertainty_total"] = float(math.sqrt(sum(a * a for a in adverse) / len(adverse)))
    return comps
