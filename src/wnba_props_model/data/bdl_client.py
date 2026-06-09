"""Ball Don't Lie REST client for WNBA data.

Design rules:
  - Endpoint contracts are explicit and auditable.
  - Odds endpoint must always be called with dates[] or game_ids[].
  - Player props endpoint must always be called with game_id.
  - Cursor pagination is handled transparently.
  - HTTP errors are classified with EndpointStatus constants.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import requests


# ---------------------------------------------------------------------------
# Endpoint availability status vocabulary
# ---------------------------------------------------------------------------

class EndpointStatus:
    """Canonical status strings for endpoint availability audit records."""
    DOCUMENTED_SUCCESS = "documented_success"
    DOCUMENTED_EMPTY = "documented_empty"          # HTTP 200 + empty data[]
    DOCUMENTED_AUTH_FAILED = "documented_auth_failed"   # HTTP 401/403
    DOCUMENTED_BAD_REQUEST = "documented_bad_request"   # HTTP 400
    DOCUMENTED_UNAVAILABLE = "documented_unavailable"   # HTTP 404
    DOCUMENTED_QUERY_BUG = "documented_query_bug"       # HTTP 422 / param issue
    LIVE_ONLY_NO_HISTORICAL = "live_only_no_historical_storage"
    SKIPPED = "skipped"
    FAILED = "failed"                              # Unexpected failure


def classify_bdl_error(error_message: str) -> str:
    """Map a BDL error string to the correct EndpointStatus constant."""
    m = error_message
    if "401" in m or "403" in m:
        return EndpointStatus.DOCUMENTED_AUTH_FAILED
    if "400" in m:
        return EndpointStatus.DOCUMENTED_BAD_REQUEST
    if "404" in m:
        return EndpointStatus.DOCUMENTED_UNAVAILABLE
    if "422" in m:
        return EndpointStatus.DOCUMENTED_QUERY_BUG
    return EndpointStatus.FAILED


# ---------------------------------------------------------------------------
# Error type and Endpoint descriptor
# ---------------------------------------------------------------------------

class BDLAPIError(RuntimeError):
    """Raised when the BALLDONTLIE API returns an unrecoverable error."""


@dataclass(frozen=True)
class Endpoint:
    sport: str
    path: str
    paginated: bool = True
    requires_goat: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------

WNBA_ENDPOINTS: dict[str, Endpoint] = {
    "teams": Endpoint("wnba", "/wnba/v1/teams", paginated=False),
    "players": Endpoint("wnba", "/wnba/v1/players"),
    "players_active": Endpoint("wnba", "/wnba/v1/players/active"),
    "games": Endpoint("wnba", "/wnba/v1/games"),
    "player_stats": Endpoint("wnba", "/wnba/v1/player_stats", requires_goat=True),
    "team_stats": Endpoint("wnba", "/wnba/v1/team_stats", requires_goat=True),
    "player_season_stats": Endpoint("wnba", "/wnba/v1/player_season_stats", requires_goat=True),
    "team_season_stats": Endpoint("wnba", "/wnba/v1/team_season_stats", requires_goat=True),
    "player_game_advanced_stats": Endpoint(
        "wnba", "/wnba/v1/player_game_advanced_stats", requires_goat=True
    ),
    "team_game_advanced_stats": Endpoint(
        "wnba", "/wnba/v1/team_game_advanced_stats", requires_goat=True
    ),
    "player_season_advanced_stats": Endpoint(
        "wnba", "/wnba/v1/player_season_advanced_stats", requires_goat=True
    ),
    "team_season_advanced_stats": Endpoint(
        "wnba", "/wnba/v1/team_season_advanced_stats", requires_goat=True
    ),
    "player_shot_locations": Endpoint(
        "wnba", "/wnba/v1/player_shot_locations", requires_goat=True
    ),
    "team_shot_locations": Endpoint("wnba", "/wnba/v1/team_shot_locations", requires_goat=True),
    "standings": Endpoint("wnba", "/wnba/v1/standings"),
    "player_injuries": Endpoint("wnba", "/wnba/v1/player_injuries"),
    # Odds: requires dates[] or game_ids[] — use explicit methods below.
    "odds": Endpoint(
        "wnba", "/wnba/v1/odds", requires_goat=True,
        notes="Must always be called with dates[] or game_ids[]. Never call without filters.",
    ),
    # Player props: requires game_id — use list_player_props_for_game().
    "player_props": Endpoint(
        "wnba", "/wnba/v1/odds/player_props", paginated=False, requires_goat=True,
        notes="Must always be called with game_id. Live only; BDL does not store historical props.",
    ),
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


# ---------------------------------------------------------------------------
# Parameter serialization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BDLClient:
    """Ball Don't Lie REST client.

    Retry/pagination behaviour appropriate for overnight model builds.
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

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

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

    def list_endpoint(
        self,
        endpoint_name: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return list(self.iter_endpoint(endpoint_name, params=params))

    # ------------------------------------------------------------------
    # Explicit odds methods  (enforce required parameters)
    # ------------------------------------------------------------------

    def list_game_odds_by_date(
        self,
        date: str,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Pull game odds for a specific date (YYYY-MM-DD).

        Requires dates[] parameter. Never call /wnba/v1/odds without a filter.
        """
        return list(self.iter_endpoint("odds", {"dates": [date]}, per_page=per_page))

    def list_game_odds_by_game_ids(
        self,
        game_ids: list[int],
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Pull game odds for a list of game IDs.

        Requires game_ids[] parameter. Never call /wnba/v1/odds without a filter.
        Raises ValueError if game_ids is empty.
        """
        if not game_ids:
            raise ValueError(
                "game_ids must be non-empty. Never call /wnba/v1/odds without "
                "dates[] or game_ids[]."
            )
        return list(self.iter_endpoint("odds", {"game_ids": game_ids}, per_page=per_page))

    def list_player_props_for_game(
        self,
        game_id: int,
        vendors: list[str] | None = None,
        player_id: int | None = None,
        prop_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pull player props for a specific game.

        Requires game_id. Live only — BDL does not store historical player
        prop data; completed games return HTTP 200 with an empty data array.

        An empty response is NOT a failure; use EndpointStatus.LIVE_ONLY_NO_HISTORICAL.
        """
        params: dict[str, Any] = {"game_id": game_id}
        if vendors:
            params["vendors"] = vendors
        if player_id is not None:
            params["player_id"] = player_id
        if prop_type is not None:
            params["type"] = prop_type
        return list(self.iter_endpoint("player_props", params))
