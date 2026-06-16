# DEPRECATED: legacy quantile path superseded by Stage 4 HGB engine.
# Retained for audit purposes. Do not use for production inference.
# Use pipeline/predict.py → pmf_engine.build_all_pmfs() instead.
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from wnba_props_model.constants import DIRECT_STATS
from wnba_props_model.features.build_features import build_player_training_table
from wnba_props_model.features.feature_contract import MODEL_FEATURES, SPARSE_EVENT_FEATURES, assert_no_forbidden_features
from wnba_props_model.models.base import train_minutes_model, train_rate_model
from wnba_props_model.models.hurdle import SparseHurdleModel


def available_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in MODEL_FEATURES if c in df.columns]
    assert_no_forbidden_features(feats)
    return feats


def train_player_models(
    player_stats_path: str | Path,
    games_path: str | Path | None = None,
    out_dir: str | Path = "artifacts/models/player_props",
) -> dict[str, Path]:
    stats = pd.read_parquet(player_stats_path)
    games = pd.read_parquet(games_path) if games_path else None
    train_df = build_player_training_table(stats, games)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out / "training_table.parquet", index=False)

    feats = available_features(train_df)
    (out / "features.txt").write_text("\n".join(feats), encoding="utf-8")

    paths: dict[str, Path] = {}
    minutes = train_minutes_model(train_df.dropna(subset=["minutes"]), feats)
    minutes_path = out / "minutes_quantile_bundle.pkl"
    minutes.save(str(minutes_path))
    paths["minutes"] = minutes_path

    for stat in DIRECT_STATS:
        if stat in ("stl", "blk"):
            h = SparseHurdleModel(stat, [f for f in feats if f in train_df.columns] + [f for f in SPARSE_EVENT_FEATURES if f in train_df.columns])
            h.fit(train_df.dropna(subset=[stat]))
            p = out / f"hurdle_{stat}.pkl"
            joblib.dump(h, p)
        else:
            bundle = train_rate_model(train_df.dropna(subset=[stat, "minutes"]), stat, feats)
            p = out / f"rate_{stat}_quantile_bundle.pkl"
            bundle.save(str(p))
        paths[stat] = p
    return paths
