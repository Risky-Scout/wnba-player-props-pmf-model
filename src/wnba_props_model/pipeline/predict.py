from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wnba_props_model.constants import DIRECT_STATS
from wnba_props_model.features.build_features import build_player_training_table
from wnba_props_model.models.base import QuantileModelBundle
from wnba_props_model.models.simulation import build_combo_pmfs, pmf_to_json, simulate_count_pmf


def predict_player_pmfs(
    feature_df: pd.DataFrame,
    model_dir: str | Path = "artifacts/models/player_props",
    draws: int = 50000,
) -> pd.DataFrame:
    model_dir = Path(model_dir)
    features = (model_dir / "features.txt").read_text(encoding="utf-8").splitlines()
    minutes = QuantileModelBundle.load(str(model_dir / "minutes_quantile_bundle.pkl"))
    minutes_q = minutes.predict_quantiles(feature_df)

    rows = []
    rate_models = {
        stat: QuantileModelBundle.load(str(model_dir / f"rate_{stat}_quantile_bundle.pkl"))
        for stat in DIRECT_STATS
        if stat not in ("stl", "blk") and (model_dir / f"rate_{stat}_quantile_bundle.pkl").exists()
    }
    hurdle_models = {
        stat: joblib.load(model_dir / f"hurdle_{stat}.pkl")
        for stat in ("stl", "blk")
        if (model_dir / f"hurdle_{stat}.pkl").exists()
    }

    rate_quantiles = {stat: bundle.predict_quantiles(feature_df) for stat, bundle in rate_models.items()}
    hurdle_pmfs = {stat: model.predict_pmf(feature_df) for stat, model in hurdle_models.items()}

    for i, base in feature_df.reset_index(drop=True).iterrows():
        component_pmfs = {}
        for stat in DIRECT_STATS:
            if stat in rate_quantiles:
                component_pmfs[stat] = simulate_count_pmf(stat, minutes_q[i], rate_quantiles[stat][i], n_draws=draws)
            elif stat in hurdle_pmfs:
                component_pmfs[stat] = hurdle_pmfs[stat][i]
        combo_pmfs = build_combo_pmfs(component_pmfs)
        for stat, pmf in {**component_pmfs, **combo_pmfs}.items():
            ks = np.arange(len(pmf))
            rows.append({
                "game_id": base.get("game_id"),
                "game_date": base.get("game_date"),
                "player_id": base.get("player_id"),
                "player_name": base.get("player_name"),
                "team_id": base.get("team_id"),
                "stat": stat,
                "pmf_json": pmf_to_json(pmf),
                "mean": float(np.dot(ks, pmf)),
                "median": int(np.searchsorted(np.cumsum(pmf), 0.5)),
                "mode": int(np.argmax(pmf)),
                "p0": float(pmf[0]),
                "role_bucket": base.get("role_bucket"),
                "model_version": "wnba_pmf_v0.1",
                "cal_source": "raw_or_calibrated",
                "lineup_status": "estimated_no_bdl_lineups",
            })
    return pd.DataFrame(rows)


def build_features_for_prediction(player_stats: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_player_training_table(player_stats, games)
