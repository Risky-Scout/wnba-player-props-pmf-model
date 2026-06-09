from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from wnba_props_model.constants import SUPPORTED_STATS
from wnba_props_model.pipeline.train import train_player_models
from wnba_props_model.pipeline.predict import build_features_for_prediction, predict_player_pmfs


def _date_windows(dates: pd.Series, min_training_days: int, window_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values().reset_index(drop=True)
    if unique.empty:
        return []
    start = unique.min() + pd.Timedelta(days=min_training_days)
    end = unique.max()
    windows = []
    cur = start
    while cur <= end:
        wend = min(cur + pd.Timedelta(days=window_days - 1), end)
        windows.append((cur, wend))
        cur = wend + pd.Timedelta(days=1)
    return windows


def _outcomes_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat in SUPPORTED_STATS:
        if stat in df.columns:
            tmp = df[["game_id", "player_id", stat]].copy()
            tmp["stat"] = stat
            tmp["outcome"] = tmp[stat]
            rows.append(tmp[["game_id", "player_id", "stat", "outcome"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_walk_forward_oof_pmfs(
    player_stats_path: str | Path,
    games_path: str | Path | None = None,
    out_path: str | Path = "data/processed/oof_pmfs.parquet",
    min_training_days: int = 365,
    window_days: int = 28,
    draws: int = 5000,
) -> Path:
    """Build leakage-safe walk-forward OOF PMFs.

    This is intentionally slower than production refit. It is the calibration truth source.
    Use 5k draws for backfill speed, then refit production with 50k draws.
    """
    stats = pd.read_parquet(player_stats_path)
    games = pd.read_parquet(games_path) if games_path else None
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    windows = _date_windows(stats["game_date"], min_training_days, window_days)

    all_rows = []
    for start, end in windows:
        train_stats = stats[stats["game_date"] < start].copy()
        val_stats = stats[(stats["game_date"] >= start) & (stats["game_date"] <= end)].copy()
        if len(train_stats) < 500 or val_stats.empty:
            continue

        if games is not None and not games.empty:
            games["game_date"] = pd.to_datetime(games["game_date"])
            train_games = games[games["game_date"] < start].copy()
            context_games = games[games["game_date"] <= end].copy()
        else:
            train_games = None
            context_games = None

        with TemporaryDirectory() as td:
            td_path = Path(td)
            tr_stats_path = td_path / "train_stats.parquet"
            tr_games_path = td_path / "train_games.parquet"
            train_stats.to_parquet(tr_stats_path, index=False)
            if train_games is not None:
                train_games.to_parquet(tr_games_path, index=False)
            model_dir = td_path / "models"
            train_player_models(tr_stats_path, tr_games_path if train_games is not None else None, model_dir)

            # Build features with all history through validation end, then keep validation rows.
            context_stats = stats[stats["game_date"] <= end].copy()
            features = build_features_for_prediction(context_stats, context_games)
            val_keys = val_stats[["game_id", "player_id"]].drop_duplicates()
            val_features = features.merge(val_keys, on=["game_id", "player_id"], how="inner")
            pmfs = predict_player_pmfs(val_features, model_dir=model_dir, draws=draws)
            outcomes = _outcomes_long(val_features)
            scored = pmfs.merge(outcomes, on=["game_id", "player_id", "stat"], how="inner")
            scored["fold_start"] = start
            scored["fold_end"] = end
            all_rows.append(scored[[
                "fold_start", "fold_end", "game_id", "game_date", "player_id", "player_name",
                "team_id", "stat", "pmf_json", "outcome", "role_bucket", "model_version"
            ]])

    if not all_rows:
        raise RuntimeError("No OOF rows produced. Check date range/min_training_days.")
    out_df = pd.concat(all_rows, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    return out
