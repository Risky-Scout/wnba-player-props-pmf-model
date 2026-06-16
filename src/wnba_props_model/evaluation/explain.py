"""Projection explainability for WNBA PMF model.

Provides per-player, per-stat driver explanations answering:
  - What is driving this player's projected minutes?
  - What is driving their stat projection?
  - What changed since their last game?
  - Are there any flags (injury, DNP risk, recent trend)?

Uses HGB feature importances (model-level) combined with player-specific
feature value comparisons vs. their own rolling average and the league baseline.

PenaltyBlog principle: probabilities should be interpretable — a model that
can't be explained cannot be trusted or debugged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProjectionExplanation:
    """Full explanation for a single player × stat projection."""
    player_id: int
    player_name: str
    game_date: str
    stat: str

    # Minutes drivers
    projected_minutes: float
    minutes_l5_avg: float
    minutes_l1_last: float
    minutes_change_vs_l5: float
    minutes_change_flag: bool          # True if |change| > 5 min

    # Stat projection
    projected_mean: float
    stat_l5_avg: float | None
    stat_change_vs_l5: float | None

    # Top feature drivers (from model feature importances × player deviation from league mean)
    top_minutes_drivers: list[dict]    # [{"feature": str, "value": float, "direction": str, "importance": float}]
    top_stat_drivers: list[dict]

    # Risk flags
    dnp_risk: str                      # low / moderate / high
    injury_flag: bool
    role_bucket: str
    n_games_sample: int                # how many games drove the rolling features

    # Plain-English narrative
    minutes_narrative: str
    stat_narrative: str

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "game_date": self.game_date,
            "stat": self.stat,
            "projected_minutes": round(self.projected_minutes, 1),
            "minutes_l5_avg": round(self.minutes_l5_avg, 1),
            "minutes_change_vs_l5": round(self.minutes_change_vs_l5, 1),
            "minutes_change_flag": self.minutes_change_flag,
            "projected_mean": round(self.projected_mean, 2),
            "stat_l5_avg": round(self.stat_l5_avg, 2) if self.stat_l5_avg is not None else None,
            "stat_change_vs_l5": round(self.stat_change_vs_l5, 2) if self.stat_change_vs_l5 is not None else None,
            "top_minutes_drivers": self.top_minutes_drivers,
            "top_stat_drivers": self.top_stat_drivers,
            "dnp_risk": self.dnp_risk,
            "injury_flag": self.injury_flag,
            "role_bucket": self.role_bucket,
            "n_games_sample": self.n_games_sample,
            "minutes_narrative": self.minutes_narrative,
            "stat_narrative": self.stat_narrative,
        }


# Human-readable labels for common model features
_FEATURE_LABELS: dict[str, str] = {
    "player_minutes_mean_l5": "avg minutes (L5 games)",
    "player_minutes_mean_l10": "avg minutes (L10 games)",
    "player_minutes_last1": "minutes last game",
    "player_minutes_mean_season": "season avg minutes",
    "projected_minutes_proxy": "projected minutes proxy",
    "player_zero_minute_rate_l5": "DNP rate (L5)",
    "starter_rate_l5": "starter rate (L5)",
    "player_usage_proxy_l5": "usage rate (L5)",
    "rest_days": "rest days since last game",
    "is_back_to_back": "back-to-back game flag",
    "team_pace_proxy": "team pace",
    "opp_defensive_rating_proxy": "opponent defensive rating",
    "player_pts_mean_l5": "avg pts (L5)",
    "player_reb_mean_l5": "avg reb (L5)",
    "player_ast_mean_l5": "avg ast (L5)",
    "player_fg3m_mean_l5": "avg 3PM (L5)",
    "player_stl_mean_l5": "avg stl (L5)",
    "player_blk_mean_l5": "avg blk (L5)",
    "player_turnover_mean_l5": "avg tov (L5)",
    "player_pts_per_minute_l5": "pts/min (L5)",
    "player_reb_per_minute_l5": "reb/min (L5)",
    "player_ast_per_minute_l5": "ast/min (L5)",
}


def _feature_label(col: str) -> str:
    return _FEATURE_LABELS.get(col, col.replace("_", " "))


def _direction_label(deviation: float) -> str:
    if deviation > 0.5:
        return "above average"
    elif deviation < -0.5:
        return "below average"
    return "near average"


def _load_feature_importances(model_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load HGB feature importances for minutes and stat models."""
    import joblib
    importances: dict[str, dict[str, np.ndarray]] = {"minutes": {}, "stats": {}}
    minutes_path = model_dir / "minutes_model.pkl"
    if minutes_path.exists():
        try:
            mm = joblib.load(minutes_path)
            if hasattr(mm, "model") and hasattr(mm.model, "feature_importances_"):
                importances["minutes"]["__model__"] = mm.model.feature_importances_
        except Exception:
            pass
    for stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"):
        for prefix, fname in [("rate", f"rate_{stat}.pkl"), ("hurdle_cls", f"hurdle_{stat}_cls.pkl")]:
            p = model_dir / fname
            if p.exists():
                try:
                    m = joblib.load(p)
                    inner = getattr(m, "model", getattr(m, "classifier", None))
                    if inner and hasattr(inner, "feature_importances_"):
                        importances["stats"][stat] = inner.feature_importances_
                except Exception:
                    pass
    return importances


def _top_drivers(
    feature_row: pd.Series,
    importances: np.ndarray,
    feature_cols: list[str],
    league_means: dict[str, float],
    n: int = 5,
) -> list[dict]:
    """Return top-n driving features for a prediction, with player value vs. league mean."""
    if importances is None or len(importances) != len(feature_cols):
        return []
    drivers = []
    for i, col in enumerate(feature_cols):
        val = float(feature_row.get(col, np.nan))
        if np.isnan(val):
            continue
        league_avg = league_means.get(col, np.nan)
        deviation = val - league_avg if not np.isnan(league_avg) else 0.0
        drivers.append({
            "feature": col,
            "label": _feature_label(col),
            "value": round(val, 3),
            "league_avg": round(float(league_avg), 3) if not np.isnan(league_avg) else None,
            "deviation": round(deviation, 3),
            "direction": _direction_label(deviation),
            "importance": round(float(importances[i]), 4),
        })
    drivers.sort(key=lambda x: x["importance"], reverse=True)
    return drivers[:n]


def _minutes_narrative(row: pd.Series, drivers: list[dict], change: float, change_flag: bool) -> str:
    parts = []
    proj = float(row.get("projected_minutes_proxy") or row.get("player_minutes_mean_l5") or 0)
    l5 = float(row.get("player_minutes_mean_l5") or 0)
    role = str(row.get("role_status", "unknown"))

    parts.append(f"Projected {proj:.1f} min (L5 avg: {l5:.1f} min, role: {role}).")

    if change_flag:
        direction = "increase" if change > 0 else "decrease"
        parts.append(f"⚠ Significant {direction} of {abs(change):.1f} min vs. recent average.")

    if drivers:
        top = drivers[0]
        parts.append(f"Top driver: {top['label']} ({top['value']:.2f}, {top['direction']} league avg).")

    if float(row.get("player_zero_minute_rate_l5") or 0) > 0.2:
        rate = float(row.get("player_zero_minute_rate_l5", 0))
        parts.append(f"DNP risk elevated: sat out {rate:.0%} of last 5 games.")

    if row.get("is_back_to_back"):
        parts.append("Back-to-back game — minutes may be reduced.")

    return " ".join(parts)


def _stat_narrative(stat: str, row: pd.Series, proj_mean: float, drivers: list[dict]) -> str:
    stat_l5_col = f"player_{stat}_mean_l5"
    stat_pm_col = f"player_{stat}_per_minute_l5"
    l5 = float(row.get(stat_l5_col) or 0)
    pm = float(row.get(stat_pm_col) or 0)

    parts = [f"Projected {proj_mean:.1f} {stat} (L5 avg: {l5:.1f})."]

    if pm > 0:
        parts.append(f"Rate: {pm:.3f} {stat}/min (L5).")

    if drivers:
        top = drivers[0]
        parts.append(f"Top driver: {top['label']} ({top['value']:.2f}, {top['direction']}).")

    return " ".join(parts)


def explain_player_projection(
    player_id: int,
    stat: str,
    feature_row: pd.Series,
    pmf_row: pd.Series,
    league_means: dict[str, float],
    feature_cols: list[str],
    importances: dict[str, dict[str, np.ndarray]],
    game_date: str = "",
) -> ProjectionExplanation:
    """Build full explanation for one player × stat projection."""
    proj_minutes = float(feature_row.get("projected_minutes_proxy") or feature_row.get("player_minutes_mean_l5") or 0)
    l5_minutes = float(feature_row.get("player_minutes_mean_l5") or proj_minutes)
    l1_minutes = float(feature_row.get("player_minutes_last1") or proj_minutes)
    change = proj_minutes - l5_minutes
    change_flag = abs(change) > 5.0

    # L5 stat average
    stat_l5 = float(feature_row.get(f"player_{stat}_mean_l5") or 0) or None
    proj_mean = float(pmf_row.get("mean") or 0)
    stat_change = (proj_mean - stat_l5) if stat_l5 is not None else None

    # Feature importance drivers
    min_imp = importances.get("minutes", {}).get("__model__")
    stat_imp = importances.get("stats", {}).get(stat)

    top_min = _top_drivers(feature_row, min_imp, feature_cols, league_means)
    top_stat = _top_drivers(feature_row, stat_imp, feature_cols, league_means)

    n_sample = int(feature_row.get("player_minutes_l5_support") or feature_row.get("player_minutes_l10_support") or 0)

    min_narr = _minutes_narrative(feature_row, top_min, change, change_flag)
    stat_narr = _stat_narrative(stat, feature_row, proj_mean, top_stat)

    return ProjectionExplanation(
        player_id=player_id,
        player_name=str(feature_row.get("player_name", player_id)),
        game_date=game_date,
        stat=stat,
        projected_minutes=proj_minutes,
        minutes_l5_avg=l5_minutes,
        minutes_l1_last=l1_minutes,
        minutes_change_vs_l5=change,
        minutes_change_flag=change_flag,
        projected_mean=proj_mean,
        stat_l5_avg=stat_l5,
        stat_change_vs_l5=stat_change,
        top_minutes_drivers=top_min,
        top_stat_drivers=top_stat,
        dnp_risk=str(feature_row.get("dnp_risk", "unknown")),
        injury_flag=bool(feature_row.get("injury_flag", False)),
        role_bucket=str(feature_row.get("projected_minutes_bucket") or feature_row.get("role_status", "unknown")),
        n_games_sample=n_sample,
        minutes_narrative=min_narr,
        stat_narrative=stat_narr,
    )


def build_explanations(
    features: pd.DataFrame,
    pmfs: pd.DataFrame,
    model_dir: str | Path = "artifacts/models/stage4_baseline",
    stats: list[str] | None = None,
    top_players: int | None = None,
) -> list[dict]:
    """Build explanations for all players × stats in the slate."""
    import json as _json
    model_dir = Path(model_dir)
    stats = stats or ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    # Load model feature columns
    manifest_path = model_dir / "feature_manifest.json"
    feature_cols: list[str] = []
    if manifest_path.exists():
        manifest = _json.loads(manifest_path.read_text())
        feature_cols = manifest.get("model_feature_columns", [])

    # Load importances
    importances = _load_feature_importances(model_dir)

    # League means from features (all historical rows)
    league_means: dict[str, float] = {}
    for col in feature_cols:
        if col in features.columns:
            league_means[col] = float(features[col].mean(skipna=True))

    # Filter to unique players (latest row per player)
    feat_idx = features.sort_values("game_date", errors="ignore").groupby("player_id").last().reset_index()

    game_date = str(features["game_date"].max()) if "game_date" in features.columns else ""

    if top_players:
        feat_idx = feat_idx.head(top_players)

    explanations = []
    for _, feat_row in feat_idx.iterrows():
        pid = int(feat_row["player_id"])
        for stat in stats:
            pmf_row_df = pmfs[(pmfs["player_id"] == pid) & (pmfs["stat"] == stat)]
            if pmf_row_df.empty:
                continue
            pmf_row = pmf_row_df.iloc[0]
            exp = explain_player_projection(
                player_id=pid,
                stat=stat,
                feature_row=feat_row,
                pmf_row=pmf_row,
                league_means=league_means,
                feature_cols=feature_cols,
                importances=importances,
                game_date=game_date,
            )
            explanations.append(exp.to_dict())

    return explanations
