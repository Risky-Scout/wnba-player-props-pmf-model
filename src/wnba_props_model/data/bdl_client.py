from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import requests


class BDLAPIError(RuntimeError):
    """Raised when the BALLDONTLIE API returns an unrecoverable error."""


@dataclass(frozen=True)
class Endpoint:
    sport: str
    path: str
    paginated: bool = True
    requires_goat: bool = False
    notes: str = ""


WNBA_ENDPOINTS: dict[str, Endpoint] = {
    "teams": Endpoint("wnba", "/wnba/v1/teams", paginated=False),
    "players": Endpoint("wnba", "/wnba/v1/players"),
    "players_active": Endpoint("wnba", "/wnba/v1/players/active"),
    "games": Endpoint("wnba", "/wnba/v1/games"),
    "player_stats": Endpoint("wnba", "/wnba/v1/player_stats", requires_goat=True),
    "team_stats": Endpoint("wnba", "/wnba/v1/team_stats", requires_goat=True),
    "player_season_stats": Endpoint("wnba", "/wnba/v1/player_season_stats", requires_goat=True),
    "team_season_stats": Endpoint("wnba", "/wnba/v1/team_season_stats", requires_goat=True),
    "player_game_advanced_stats": Endpoint("wnba", "/wnba/v1/player_game_advanced_stats", requires_goat=True),
    "team_game_advanced_stats": Endpoint("wnba", "/wnba/v1/team_game_advanced_stats", requires_goat=True),
    "player_season_advanced_stats": Endpoint("wnba", "/wnba/v1/player_season_advanced_stats", requires_goat=True),
    "team_season_advanced_stats": Endpoint("wnba", "/wnba/v1/team_season_advanced_stats", requires_goat=True),
    "player_shot_locations": Endpoint("wnba", "/wnba/v1/player_shot_locations", requires_goat=True),
    "team_shot_locations": Endpoint("wnba", "/wnba/v1/team_shot_locations", requires_goat=True),
    "standings": Endpoint("wnba", "/wnba/v1/standings"),
    "player_injuries": Endpoint("wnba", "/wnba/v1/player_injuries"),
    "odds": Endpoint("wnba", "/wnba/v1/odds", requires_goat=True),
    "player_props": Endpoint("wnba", "/wnba/v1/odds/player_props", paginated=False, requires_goat=True),
    "plays": Endpoint("wnba", "/wnba/v1/plays", paginated=False),
}

NBA_PARITY_ENDPOINTS: dict[str, Endpoint] = {
    "teams": Endpoint("nba", "/v1/teams"),
    "players": Endpoint("nba", "/v1/players"),
    "players_active": Endpoint("nba", "/v1/players/active"),
    "games": Endpoint("nba", "/v1/games"),
    "player_stats": Endpoint("nba", "/v1/stats", requires_goat=True),
    "advanced_stats": Endpoint("nba", "/nba/v1/stats/advanced", requires_goat=True),
    "box_scores": Endpoint("nba", "/v1/box_scores", requires_goat=True),
    "box_scores_live": Endpoint("nba", "/v1/box_scores/live", requires_goat=True),
    "lineups": Endpoint("nba", "/v1/lineups", requires_goat=True),
    "plays": Endpoint("nba", "/v1/plays", paginated=False, requires_goat=True),
    "player_injuries": Endpoint("nba", "/v1/player_injuries"),
    "standings": Endpoint("nba", "/v1/standings"),
    "odds": Endpoint("nba", "/v2/odds", requires_goat=True),
    "player_props": Endpoint("nba", "/v2/odds/player_props", paginated=False, requires_goat=True),
}


def _array_params(params: Mapping[str, Any] | None) -> list[tuple[str, Any]]:
    """Convert {'game_ids': [1,2]} into [('game_ids[]', 1), ('game_ids[]', 2)]."""
    if not params:
        return []
    out: list[tuple[str, Any]] = []
    for key, val in params.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            arr_key = key if key.endswith("[]") else f"{key}[]"
            for item in val:
                out.append((arr_key, item))
        else:
            out.append((key, val))
    return out


class BDLClient:
    """Small, explicit BALLDONTLIE REST client.

    The official SDK is fine for ad hoc work. This wrapper keeps endpoint contracts
    visible for auditability and gives us retry/pagination behavior appropriate for
    overnight model builds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.balldontlie.io",
        timeout: int = 30,
        max_retries: int = 3,
        sleep_seconds: float = 0.35,
    ) -> None:
        self.api_key = api_key or os.getenv("BDL_API_KEY")
        if not self.api_key:
            raise BDLAPIError("BDL_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        query = _array_params(params)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=query, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(self.sleep_seconds * (attempt + 2))
                    continue
                if resp.status_code >= 400:
                    raise BDLAPIError(f"{resp.status_code} from {url}: {resp.text[:500]}")
                return resp.json()
            except (requests.RequestException, BDLAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.sleep_seconds * (attempt + 1))
        raise BDLAPIError(f"failed GET {url}") from last_error

    def iter_endpoint(
        self,
        endpoint_name: str,
        params: Mapping[str, Any] | None = None,
        per_page: int = 100,
    ) -> Iterator[dict[str, Any]]:
        ep = WNBA_ENDPOINTS[endpoint_name]
        params = dict(params or {})
        if ep.paginated:
            params.setdefault("per_page", per_page)
        cursor = params.get("cursor")
        while True:
            if cursor is not None:
                params["cursor"] = cursor
            payload = self.get_json(ep.path, params=params)
            data = payload.get("data", [])
            if isinstance(data, dict):
                yield data
            else:
                yield from data
            meta = payload.get("meta", {}) or {}
            cursor = meta.get("next_cursor")
            if not ep.paginated or not cursor:
                break

    def list_endpoint(self, endpoint_name: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self.iter_endpoint(endpoint_name, params=params))
