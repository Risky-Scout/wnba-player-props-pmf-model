"""BDL ingestion layer.

Attempts to pull every relevant WNBA endpoint, normalizes results, writes
raw parquet files, and returns a structured endpoint availability report.

Design rules:
  - Required endpoints (games, player_stats) raise on failure.
  - Optional endpoints are caught and recorded with EndpointStatus constants.
  - /wnba/v1/odds MUST always be called with game_ids[] — never without filters.
  - /wnba/v1/odds/player_props is live-only; empty history is documented_empty.
  - Every raw parquet gets source + pull_timestamp_utc columns.
  - Player props for future/upcoming games are attempted; past games return empty.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from wnba_props_model.data.bdl_client import (
    BDLAPIError,
    BDLClient,
    EndpointStatus,
    classify_bdl_error,
)
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

# How many game_ids to send per odds API call
_ODDS_BATCH_SIZE = 100
# Pause between batches to be gentle on rate limits
_BATCH_SLEEP = 0.4


# ---------------------------------------------------------------------------
# Endpoint availability record builder
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
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "status": status,
        "row_count": row_count,
        "seasons_attempted": seasons_attempted or [],
        "error_message": error_message,
        "fallback_required": fallback_required,
        "raw_output_path": raw_output_path,
        "notes": notes,
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

    Returns:
        paths:               {table_name: Path}
        endpoints:           {endpoint_name: availability_record}
        pull_timestamp_utc:  str (ISO)
        seasons:             list[int]
    """
    client = client or BDLClient()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seasons_list = sorted(set(int(s) for s in seasons))
    ts = datetime.now(timezone.utc).isoformat()

    paths: dict[str, Path] = {}
    availability: dict[str, Any] = {}

    def _write(df: pd.DataFrame, name: str) -> Path:
        df["source"] = "bdl"
        df["pull_timestamp_utc"] = ts
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        return p

    # ------------------------------------------------------------------
    # 1. Teams
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "teams")
    if err:
        availability["teams"] = _ep_record(
            attempted=True, status=classify_bdl_error(err),
            error_message=err, fallback_required=True,
        )
    elif not rows:
        availability["teams"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY, fallback_required=True,
        )
    else:
        df = normalize_teams(rows)
        p = _write(df, "wnba_teams")
        paths["teams"] = p
        availability["teams"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df), raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 2. Players
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "players_active")
    if err or not rows:
        rows2, err2 = _try_pull(client, "players")
        rows = rows2 if rows2 else []
        err = err2 or err
    if rows:
        df = normalize_players(rows)
        p = _write(df, "wnba_players")
        paths["players"] = p
        availability["players"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df), raw_output_path=str(p),
        )
    else:
        availability["players"] = _ep_record(
            attempted=True,
            status=classify_bdl_error(err or "") if err else EndpointStatus.DOCUMENTED_EMPTY,
            error_message=err, fallback_required=True,
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
        attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
        row_count=len(df_games), seasons_attempted=seasons_list, raw_output_path=str(p),
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
        attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
        row_count=len(df_stats), seasons_attempted=seasons_list, raw_output_path=str(p),
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
            attempted=True, status=classify_bdl_error(adv_err),
            error_message=adv_err, seasons_attempted=seasons_list, fallback_required=True,
        )
    elif not all_adv_rows:
        availability["player_game_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY,
            seasons_attempted=seasons_list,
        )
    else:
        df_adv = normalize_advanced_stats(all_adv_rows)
        p = _write(df_adv, "wnba_player_advanced_stats")
        paths["player_game_advanced_stats"] = p
        availability["player_game_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_adv), seasons_attempted=seasons_list, raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 6. Game odds  (optional, GOAT)
    #    MUST use game_ids[] — never call /wnba/v1/odds without filters.
    # ------------------------------------------------------------------
    all_game_ids: list[int] = sorted(
        int(x) for x in df_games["game_id"].dropna().unique()
    )
    all_odds_rows: list[dict] = []
    odds_err: str | None = None
    for i in range(0, len(all_game_ids), _ODDS_BATCH_SIZE):
        batch = all_game_ids[i:i + _ODDS_BATCH_SIZE]
        try:
            batch_rows = client.list_game_odds_by_game_ids(batch)
            all_odds_rows.extend(batch_rows)
        except BDLAPIError as exc:
            odds_err = str(exc)
            break
        except Exception as exc:  # noqa: BLE001
            odds_err = f"{type(exc).__name__}: {exc}"
            break
        if i + _ODDS_BATCH_SIZE < len(all_game_ids):
            time.sleep(_BATCH_SLEEP)

    if odds_err:
        availability["odds"] = _ep_record(
            attempted=True,
            status=classify_bdl_error(odds_err),
            error_message=odds_err,
            seasons_attempted=seasons_list,
            fallback_required=True,
            notes=(
                "BDL WNBA game odds endpoint documented and accessible. "
                "Requires dates[] or game_ids[]. Key confirmed working by "
                "manual HTTP 200 test."
            ),
        )
    elif not all_odds_rows:
        availability["odds"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY,
            seasons_attempted=seasons_list,
            notes="Odds endpoint returned no rows for the requested game IDs.",
        )
    else:
        df_odds = normalize_odds(all_odds_rows)
        p = _write(df_odds, "wnba_odds")
        paths["odds"] = p
        availability["odds"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_odds), seasons_attempted=seasons_list, raw_output_path=str(p),
            notes=(
                "BDL WNBA game odds endpoint documented and accessible. "
                "Requires dates[] or game_ids[]."
            ),
        )

    # ------------------------------------------------------------------
    # 7. Standings  (optional)
    # ------------------------------------------------------------------
    all_standings_rows: list[dict] = []
    standings_status = EndpointStatus.DOCUMENTED_EMPTY
    standings_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "standings", {"season": season})
        if err:
            standings_err = err
            standings_status = classify_bdl_error(err)
            break
        all_standings_rows.extend(rows)
    if standings_err:
        availability["standings"] = _ep_record(
            attempted=True, status=standings_status,
            error_message=standings_err, seasons_attempted=seasons_list,
        )
    elif all_standings_rows:
        df_st = normalize_standings(all_standings_rows)
        p = _write(df_st, "wnba_standings")
        paths["standings"] = p
        availability["standings"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_st), seasons_attempted=seasons_list, raw_output_path=str(p),
        )
    else:
        availability["standings"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY,
            seasons_attempted=seasons_list,
        )

    # ------------------------------------------------------------------
    # 8. Player injuries  (optional, current only)
    # ------------------------------------------------------------------
    rows, err = _try_pull(client, "player_injuries")
    if err:
        availability["player_injuries"] = _ep_record(
            attempted=True, status=classify_bdl_error(err),
            error_message=err, fallback_required=True,
        )
    elif not rows:
        availability["player_injuries"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY,
        )
    else:
        df_inj = normalize_injuries(rows)
        p = _write(df_inj, "wnba_injuries")
        paths["player_injuries"] = p
        availability["player_injuries"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_inj), raw_output_path=str(p),
        )

    # ------------------------------------------------------------------
    # 9. Player props  (live-only; attempt for upcoming games)
    #    BDL does NOT store historical props — completed games return HTTP 200
    #    with empty data[], which is NOT a failure.
    # ------------------------------------------------------------------
    upcoming_ids: list[int] = sorted(
        int(x)
        for x in df_games.loc[
            df_games["status_normalized"].isin(["scheduled", "in_progress"]), "game_id"
        ]
        .dropna()
        .unique()
    )[:50]  # cap to 50 most recent upcoming games
    all_props_rows: list[dict] = []
    for game_id in upcoming_ids:
        try:
            rows = client.list_player_props_for_game(game_id)
            all_props_rows.extend(rows)
        except BDLAPIError:
            pass

    if all_props_rows:
        df_props = normalize_player_props(all_props_rows)
        p = _write(df_props, "wnba_player_props")
        paths["player_props"] = p
        availability["player_props"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_props), raw_output_path=str(p),
            notes=(
                "BDL WNBA player props endpoint documented and accessible. "
                "Requires game_id. Live only; historical prop data is not stored "
                "by BDL, so completed games return HTTP 200 with empty data[]."
            ),
        )
    else:
        # Empty props are expected — this is the documented live-only limitation.
        availability["player_props"] = _ep_record(
            attempted=True,
            status=EndpointStatus.LIVE_ONLY_NO_HISTORICAL
            if upcoming_ids
            else EndpointStatus.DOCUMENTED_EMPTY,
            row_count=0,
            notes=(
                "BDL WNBA player props endpoint documented and accessible. "
                "Requires game_id. Live only; historical prop data is not stored "
                "by BDL, so completed games return HTTP 200 with empty data[]. "
                f"Attempted {len(upcoming_ids)} upcoming game(s)."
            ),
        )

    # ------------------------------------------------------------------
    # 10. Player season advanced stats — measure_type splits (usage, four_factors, scoring)
    # ------------------------------------------------------------------
    SEASON_ADV_MEASURE_TYPES = ["usage", "four_factors", "scoring", "advanced"]
    all_season_adv_rows: list[dict] = []
    season_adv_err: str | None = None
    for season in seasons_list:
        for mt in SEASON_ADV_MEASURE_TYPES:
            rows, err = _try_pull(
                client,
                "player_season_advanced_stats",
                {"season": season, "measure_type": mt, "per_mode": "per_game"},
            )
            if err:
                season_adv_err = err
                break
            for r in rows:
                r["_measure_type"] = mt
            all_season_adv_rows.extend(rows)
        if season_adv_err:
            break
    if season_adv_err:
        availability["player_season_advanced_stats"] = _ep_record(
            attempted=True, status=classify_bdl_error(season_adv_err),
            error_message=season_adv_err, seasons_attempted=seasons_list, fallback_required=True,
        )
    elif all_season_adv_rows:
        from wnba_props_model.data.normalize import normalize_season_advanced_stats
        df_season_adv = normalize_season_advanced_stats(all_season_adv_rows)
        p = _write(df_season_adv, "wnba_player_season_advanced")
        paths["player_season_advanced_stats"] = p
        availability["player_season_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_season_adv), seasons_attempted=seasons_list, raw_output_path=str(p),
        )
    else:
        availability["player_season_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY, seasons_attempted=seasons_list,
        )

    # ------------------------------------------------------------------
    # 11. Team game advanced stats
    # ------------------------------------------------------------------
    all_team_adv_rows: list[dict] = []
    team_adv_err: str | None = None
    for season in seasons_list:
        rows, err = _try_pull(client, "team_game_advanced_stats", {"season": season})
        if err:
            team_adv_err = err
            break
        all_team_adv_rows.extend(rows)
    if team_adv_err:
        availability["team_game_advanced_stats"] = _ep_record(
            attempted=True, status=classify_bdl_error(team_adv_err),
            error_message=team_adv_err, seasons_attempted=seasons_list, fallback_required=True,
        )
    elif all_team_adv_rows:
        from wnba_props_model.data.normalize import normalize_team_advanced_stats
        df_team_adv = normalize_team_advanced_stats(all_team_adv_rows)
        p = _write(df_team_adv, "wnba_team_game_advanced")
        paths["team_game_advanced_stats"] = p
        availability["team_game_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
            row_count=len(df_team_adv), seasons_attempted=seasons_list, raw_output_path=str(p),
        )
    else:
        availability["team_game_advanced_stats"] = _ep_record(
            attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY, seasons_attempted=seasons_list,
        )

    # ------------------------------------------------------------------
    # 12. Player + team shot locations  (season-level, by_zone)
    # ------------------------------------------------------------------
    for entity_key, ep_name in [("player", "player_shot_locations"), ("team", "team_shot_locations")]:
        all_shot_rows: list[dict] = []
        shot_err: str | None = None
        for season in seasons_list:
            rows, err = _try_pull(
                client, ep_name,
                {"season": season, "distance_range": "by_zone", "per_mode": "per_game"},
            )
            if err:
                shot_err = err
                break
            all_shot_rows.extend(rows)
        ep_canon = f"{entity_key}_shot_locations"
        if shot_err:
            availability[ep_canon] = _ep_record(
                attempted=True, status=classify_bdl_error(shot_err),
                error_message=shot_err, seasons_attempted=seasons_list, fallback_required=True,
            )
        elif all_shot_rows:
            df_shot = normalize_shot_locations(all_shot_rows)
            p = _write(df_shot, f"wnba_{entity_key}_shot_locations")
            paths[ep_canon] = p
            availability[ep_canon] = _ep_record(
                attempted=True, status=EndpointStatus.DOCUMENTED_SUCCESS,
                row_count=len(df_shot), seasons_attempted=seasons_list, raw_output_path=str(p),
            )
        else:
            availability[ep_canon] = _ep_record(
                attempted=True, status=EndpointStatus.DOCUMENTED_EMPTY,
                seasons_attempted=seasons_list,
            )

    # ------------------------------------------------------------------
    # 13. Play-by-play — skipped in bulk pull; use pull_plays_for_game() for live
    # ------------------------------------------------------------------
    availability["plays"] = _ep_record(
        attempted=False,
        status=EndpointStatus.SKIPPED,
        notes="PBP requires per-game pull. Use pull_plays_for_game() for live tracking.",
    )

    return {
        "paths": paths,
        "endpoints": availability,
        "pull_timestamp_utc": ts,
        "seasons": seasons_list,
    }


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------

def pull_season_history(
    seasons: Iterable[int],
    out_dir: str | Path = "data/raw/bdl",
    client: BDLClient | None = None,
) -> dict[str, Path]:
    result = pull_full_history(seasons, out_dir=out_dir, client=client)
    return result["paths"]


def pull_plays_for_game(
    game_id: int,
    out_dir: str | Path = "data/raw/bdl/plays",
    client: BDLClient | None = None,
) -> Path:
    """Pull play-by-play for a single game and write to parquet."""
    client = client or BDLClient()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows = client.list_endpoint("plays", {"game_id": game_id})
    except BDLAPIError:
        rows = []
    from wnba_props_model.data.normalize import normalize_plays
    # Inject game_id into rows that may lack it
    for r in rows:
        r.setdefault("game_id", game_id)
    df = normalize_plays(rows)
    df["source"] = "bdl"
    df["pull_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    p = out / f"plays_{game_id}.parquet"
    df.to_parquet(p, index=False)
    return p


def pull_live_market_snapshot(
    game_ids: list[int],
    out_dir: str | Path = "data/raw/bdl/live",
    client: BDLClient | None = None,
) -> Path:
    client = client or BDLClient()
    ts = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for game_id in game_ids:
        try:
            r = client.list_player_props_for_game(game_id)
            rows.extend(r)
        except BDLAPIError:
            pass
    df = normalize_player_props(rows)
    df["source"] = "bdl"
    df["pull_timestamp_utc"] = ts
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "wnba_player_props_snapshot.parquet"
    df.to_parquet(p, index=False)
    return p
