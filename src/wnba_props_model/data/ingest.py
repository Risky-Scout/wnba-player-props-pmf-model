from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from wnba_props_model.data.bdl_client import BDLClient
from wnba_props_model.data.normalize import normalize_games, normalize_player_stats


def pull_season_history(
    seasons: Iterable[int],
    out_dir: str | Path = "data/raw/bdl",
    client: BDLClient | None = None,
) -> dict[str, Path]:
    """Pull BDL WNBA games and player game stats for modeling.

    Requires a BDL key with access to WNBA game/player stat endpoints.
    """
    client = client or BDLClient()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_games: list[dict] = []
    all_stats: list[dict] = []
    for season in seasons:
        all_games.extend(client.list_endpoint("games", {"seasons": [season]}))
        all_stats.extend(client.list_endpoint("player_stats", {"seasons": [season]}))

    games_df = normalize_games(all_games)
    stats_df = normalize_player_stats(all_stats)

    games_path = out / "wnba_games.parquet"
    stats_path = out / "wnba_player_game_stats.parquet"
    games_df.to_parquet(games_path, index=False)
    stats_df.to_parquet(stats_path, index=False)
    return {"games": games_path, "player_stats": stats_path}


def pull_live_market_snapshot(game_ids: list[int], out_dir: str | Path = "data/raw/bdl/live") -> Path:
    client = BDLClient()
    rows = []
    for game_id in game_ids:
        rows.extend(client.list_endpoint("player_props", {"game_id": game_id}))
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "wnba_player_props_snapshot.parquet"
    df.to_parquet(path, index=False)
    return path
