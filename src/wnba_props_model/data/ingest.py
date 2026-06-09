"""BDL ingestion layer.

Attempts to pull every relevant WNBA endpoint, normalizes results, writes
raw parquet files, and returns a structured endpoint availability report.

Design rules:
- Required endpoints (games, player_stats) raise on failure.
- Optional endpoints are caught and recorded as unavailable/failed/empty.
- Endpoints that need per-game iteration (player_props, plays, shot_locations)
  are marked skipped_needs_game_id in historical pulls.
- Every raw parquet gets source + pull_timestamp_utc columns.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient
from wnba_props_model.data.normalize import (
    normalize_advanced_stats,
    normalize_games,
    normalize_injuries,
    normalize_odds,
    normalize_player_props,
    normalize_player_stats,
    normalize_players,
    normalize_shot_locations,
    normalize_standings,
    normalize_teams,
)


# ---------------------------------------------------------------------------
# Endpoint availability record
# ---------------------------------------------------------------------------

def _ep_record(
    *,
    attempted: bool,
    status: str,
    row_count: int = 0,
    seasons_attempted: list[int] | None = None,
    error_message: str | None = None,
    fallback_required: bool = False,
    raw_output_path: str | None = None,
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "status": status,
        "row_count": row_count,
        "seasons_attempted": seasons_attempted or [],
        "error_message": error_message,
        "fallback_required": fallback_required,
        "raw_output_path": raw_output_path,
    }


def _try_pull(
    client: BDLClient,
    endpoint_name: str,
    params: dict | None = None,
) -> tuple[list[dict], str | None]:
    """Return (rows, error_message). Never raises."""
    try:
        rows = client.list_endpoint(endpoint_name, params or {})
        return rows, None
    except BDLAPIError as exc:
        return [], str(exc)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------

def pull_full_history(
    seasons: Iterable[int],
    out_dir: str | Path = "data/raw/bdl",
    client: BDLClient | None = None,
) -> dict[str, Any]:
    """Pull all available WNBA BDL endpoints for the given seasons.

    Returns a dict with:
      "paths":    {table_name: Path}
      "endpoints": {endpoint_name: endpoint_availability_record}
      "pull_timestamp_utc": str
      "seasons": list[int]
    """
    client = client or BDLClient()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seasons_list = sorted(set(int(s) for s in seasons))
    ts = datetime.now(timezone.utc).isoformat()

    paths: dict[str, Path] = {}
    availability: dict[str, Any] = {}

    # Helper to write a DataFrame
    def _write(df: pd.DataFrame, name: str) -> Path:
        df["source"] = "bdl"
        df["pull_timestamp_utc"] = ts
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        return p

    # ------------------------------------------------------------------
    # 1. Teams  (no season filter, paginated=False)
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "teams")
    if err:
        availability["teams"] = _ep_record(
            attempted=True, status="failed", error_message=err, fallback_required=True,
        )
    elif not rows:
        availability["teams"] = _ep_record(attempted=True, status="empty", fallback_required=True)
    else:
        df = normalize_teams(rows)
        p = _write(df, "wnba_teams")
        paths["teams"] = p
        availability["teams"] = _ep_record(
            attempted=True, status="success", row_count=len(df),
            raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 2. Players  (all active players; no reliable season filter)
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "players_active")
    if err or not rows:
        rows2, err2 = _try_pull(client, "players")
        if err2 or not rows2:
            availability["players"] = _ep_record(
                attempted=True, status="failed" if (err2 or err) else "empty",
                error_message=err2 or err, fallback_required=True,
            )
            rows = []
        else:
            rows = rows2
            err = None
    if rows:
        df = normalize_players(rows)
        p = _write(df, "wnba_players")
        paths["players"] = p
        availability["players"] = _ep_record(
            attempted=True, status="success", row_count=len(df),
            raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 3. Games  (REQUIRED)
    # ------------------------------------------------------------------
    all_game_rows: list[dict] = []
    game_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "games", {"seasons": [season]})
        if err:
            game_err = err
            break
        all_game_rows.extend(rows)
    if game_err and not all_game_rows:
        raise BDLAPIError(f"Required endpoint 'games' failed: {game_err}")
    df_games = normalize_games(all_game_rows)
    p = _write(df_games, "wnba_games")
    paths["games"] = p
    availability["games"] = _ep_record(
        attempted=True, status="success", row_count=len(df_games),
        seasons_attempted=seasons_list, raw_output_path=str(p),
    )

    # ------------------------------------------------------------------
    # 4. Player stats  (REQUIRED, GOAT)
    # ------------------------------------------------------------------
    all_stat_rows: list[dict] = []
    stat_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "player_stats", {"seasons": [season]})
        if err:
            stat_err = err
            break
        all_stat_rows.extend(rows)
    if stat_err and not all_stat_rows:
        raise BDLAPIError(f"Required endpoint 'player_stats' failed: {stat_err}")
    df_stats = normalize_player_stats(all_stat_rows)
    p = _write(df_stats, "wnba_player_game_stats")
    paths["player_stats"] = p
    availability["player_stats"] = _ep_record(
        attempted=True, status="success", row_count=len(df_stats),
        seasons_attempted=seasons_list, raw_output_path=str(p),
    )

    # ------------------------------------------------------------------
    # 5. Advanced stats  (optional, GOAT)
    # ------------------------------------------------------------------
    all_adv_rows: list[dict] = []
    adv_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "player_game_advanced_stats", {"seasons": [season]})
        if err:
            adv_err = err
            break
        all_adv_rows.extend(rows)
    if adv_err:
        availability["player_game_advanced_stats"] = _ep_record(
            attempted=True, status="unavailable" if "403" in (adv_err or "") else "failed",
            error_message=adv_err, seasons_attempted=seasons_list, fallback_required=True,
        )
    elif not all_adv_rows:
        availability["player_game_advanced_stats"] = _ep_record(
            attempted=True, status="empty", seasons_attempted=seasons_list,
        )
    else:
        df_adv = normalize_advanced_stats(all_adv_rows)
        p = _write(df_adv, "wnba_player_advanced_stats")
        paths["player_game_advanced_stats"] = p
        availability["player_game_advanced_stats"] = _ep_record(
            attempted=True, status="success", row_count=len(df_adv),
            seasons_attempted=seasons_list, raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 6. Odds  (optional, GOAT)
    # ------------------------------------------------------------------
    all_odds_rows: list[dict] = []
    odds_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "odds", {"seasons": [season]})
        if err:
            odds_err = err
            break
        all_odds_rows.extend(rows)
    if odds_err:
        availability["odds"] = _ep_record(
            attempted=True, status="unavailable" if "403" in (odds_err or "") else "failed",
            error_message=odds_err, seasons_attempted=seasons_list, fallback_required=True,
        )
    elif not all_odds_rows:
        availability["odds"] = _ep_record(
            attempted=True, status="empty", seasons_attempted=seasons_list,
        )
    else:
        df_odds = normalize_odds(all_odds_rows)
        p = _write(df_odds, "wnba_odds")
        paths["odds"] = p
        availability["odds"] = _ep_record(
            attempted=True, status="success", row_count=len(df_odds),
            seasons_attempted=seasons_list, raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 7. Standings  (optional)
    # ------------------------------------------------------------------
    all_standings_rows: list[dict] = []
    for season in seasons_list:
        rows, err = _try_pull(client, "standings", {"season": season})
        if err:
            availability["standings"] = _ep_record(
                attempted=True, status="failed", error_message=err, seasons_attempted=seasons_list,
            )
            all_standings_rows = []
            break
        all_standings_rows.extend(rows)
    if all_standings_rows:
        df_standings = normalize_standings(all_standings_rows)
        p = _write(df_standings, "wnba_standings")
        paths["standings"] = p
        availability["standings"] = _ep_record(
            attempted=True, status="success", row_count=len(df_standings),
            seasons_attempted=seasons_list, raw_output_path=str(p),
        )
    elif "standings" not in availability:
        availability["standings"] = _ep_record(
            attempted=True, status="empty", seasons_attempted=seasons_list,
        )

    # ------------------------------------------------------------------
    # 8. Player injuries  (optional, current only – no season filter)
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "player_injuries")
    if err:
        availability["player_injuries"] = _ep_record(
            attempted=True, status="failed", error_message=err, fallback_required=True,
        )
    elif not rows:
        availability["player_injuries"] = _ep_record(attempted=True, status="empty")
    else:
        df_inj = normalize_injuries(rows)
        p = _write(df_inj, "wnba_injuries")
        paths["player_injuries"] = p
        availability["player_injuries"] = _ep_record(
            attempted=True, status="success", row_count=len(df_inj),
            raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 9. Endpoints skipped because they require per-game iteration
    # ------------------------------------------------------------------
    for ep in ("player_props", "plays", "player_shot_locations"):
        availability[ep] = _ep_record(
            attempted=False,
            status="skipped",
            error_message="Requires per-game iteration. Run targeted pull for specific game_ids.",
        )

    return {
        "paths": paths,
        "endpoints": availability,
        "pull_timestamp_utc": ts,
        "seasons": seasons_list,
    }


# ---------------------------------------------------------------------------
# Legacy thin wrapper (used by old pull_bdl_history.py CLI)
# ---------------------------------------------------------------------------

def pull_season_history(
    seasons: Iterable[int],
    out_dir: str | Path = "data/raw/bdl",
    client: BDLClient | None = None,
) -> dict[str, Path]:
    result = pull_full_history(seasons, out_dir=out_dir, client=client)
    return {k: v for k, v in result["paths"].items()}


def pull_live_market_snapshot(
    game_ids: list[int],
    out_dir: str | Path = "data/raw/bdl/live",
    client: BDLClient | None = None,
) -> Path:
    client = client or BDLClient()
    ts = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for game_id in game_ids:
        r, _ = _try_pull(client, "player_props", {"game_id": game_id})
        rows.extend(r)
    df = normalize_player_props(rows)
    df["source"] = "bdl"
    df["pull_timestamp_utc"] = ts
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "wnba_player_props_snapshot.parquet"
    df.to_parquet(p, index=False)
    return p
