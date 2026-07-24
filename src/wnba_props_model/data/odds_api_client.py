"""The Odds API v4 client for WNBA market data.

Provides player props, game odds (h2h/spreads/totals), live scores,
and historical closing lines for the WNBA PMF model.

API reference: https://the-odds-api.com/liveapi/guides/v4/
Market keys:  https://the-odds-api.com/sports-odds-data/betting-markets.html

Design:
- All requests log quota usage from X-Requests-Remaining / X-Requests-Used headers.
- Player props require per-event calls (not bulk /odds endpoint).
- Historical props cost 10× live rate; use sparingly (see QUOTA_BUDGET).
- Deep links are enabled via includeLinks=true&includeSids=true.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com"
SPORT_KEY = "basketball_wnba"

# ---------------------------------------------------------------------------
# Market key → internal stat mapping
# ---------------------------------------------------------------------------

ODDS_API_TO_STAT: dict[str, str] = {
    # Core individual props
    "player_points":                       "pts",
    "player_rebounds":                     "reb",
    "player_assists":                      "ast",
    "player_threes":                       "fg3m",
    "player_blocks":                       "blk",
    "player_steals":                       "stl",
    "player_turnovers":                    "turnover",
    "player_field_goals":                  "fgm",
    "player_frees_made":                   "ftm",
    # Combo props — use canonical pipeline names so build_edge_report.py join works
    "player_blocks_steals":                "stocks",
    "player_points_rebounds_assists":      "pts_reb_ast",
    "player_points_rebounds":              "pts_reb",
    "player_points_assists":               "pts_ast",
    "player_rebounds_assists":             "reb_ast",
    # Quarter props
    "player_points_q1":                    "pts_q1",
    "player_rebounds_q1":                  "reb_q1",
    "player_assists_q1":                   "ast_q1",
    # Niche / less common
    "player_frees_attempts":               "fta",
    # Alternate lines (map to same stat)
    "player_points_alternate":             "pts",
    "player_rebounds_alternate":           "reb",
    "player_blocks_alternate":             "blk",
    "player_steals_alternate":             "stl",
    "player_turnovers_alternate":          "turnover",
    "player_threes_alternate":             "fg3m",
    "player_points_assists_alternate":     "pts_ast",
    "player_points_rebounds_alternate":    "pts_reb",
    "player_rebounds_assists_alternate":   "reb_ast",
    "player_points_rebounds_assists_alternate": "pts_reb_ast",
}

# Core props requested for every game.
# Credit cost = n_markets × n_regions per event call.
# Current list: 20 markets × 1 region = 20 credits/event.
#
# Individual stats (9):  pts, reb, ast, fg3m, blk, stl, tov, fgm, ftm
# Combo stats (5):       stocks, pra, pts_reb, pts_ast, reb_ast
# Q1 quarter props (3):  pts_q1, reb_q1, ast_q1
# Alternate lines (3):   pts_alt, reb_alt, fg3m_alt
#
# NOTE: Each additional market costs 1 more credit per event call.
# Removing Q1 markets (3 saves 3 credits/event) is the easiest lever if quota is tight.
CORE_PROP_MARKETS = [
    # Individual stats
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_blocks", "player_steals",
    "player_turnovers", "player_field_goals", "player_frees_made",
    # Combo stats
    "player_blocks_steals",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
    # Q1 quarter props (3 credits/event — remove to save quota)
    "player_points_q1", "player_rebounds_q1", "player_assists_q1",
    # Alternate lines
    "player_points_alternate", "player_rebounds_alternate",
    "player_threes_alternate",
]

# Weekly quota budget (credits)
QUOTA_BUDGET = {
    "sports_list":      0,
    "bulk_odds":        252,
    "events_list":      0,
    "props_per_game":   1_680,
    "scores":           1_680,
    "historical_odds":  630,
    "historical_props": 1_400,
    "total_weekly":     7_322,
}


class OddsAPIError(RuntimeError):
    pass


class OddsAPIClient:
    """The Odds API v4 client with quota tracking and deep link support.

    Parameters
    ----------
    api_key : str | None
        Odds API key. Falls back to ODDS_API_KEY env var if not provided.
    region : str
        Odds region. 'us' (default) covers DraftKings, FanDuel, BetMGM, Caesars.
        Use 'us2' for additional US books.
    odds_format : str
        'american' (default) or 'decimal'.
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        region: str = "us",
        odds_format: str = "american",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise OddsAPIError(
                "No ODDS_API_KEY found. Set the ODDS_API_KEY env var or pass api_key=."
            )
        self.region = region
        self.odds_format = odds_format
        self.timeout = timeout
        self._requests_remaining: int | None = None
        self._requests_used: int | None = None
        self._session = requests.Session()

    # -----------------------------------------------------------------------
    # Internal HTTP layer
    # -----------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a GET request and log quota headers."""
        url = f"{BASE_URL}{path}"
        p: dict[str, Any] = {"apiKey": self.api_key, **(params or {})}
        for attempt in range(3):
            try:
                resp = self._session.get(url, params=p, timeout=self.timeout)
                # Parse quota headers
                if "X-Requests-Remaining" in resp.headers:
                    self._requests_remaining = int(resp.headers["X-Requests-Remaining"])
                if "X-Requests-Used" in resp.headers:
                    self._requests_used = int(resp.headers["X-Requests-Used"])

                if resp.status_code == 401:
                    raise OddsAPIError(f"ODDS_API_KEY invalid or expired (HTTP 401): {path}")
                if resp.status_code == 422:
                    raise OddsAPIError(f"Bad request params (HTTP 422): {path} — {resp.text[:200]}")
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    log.warning("[OddsAPI] Rate limited (HTTP 429) — waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                log.debug(
                    "[OddsAPI] GET %s → %d rows | quota remaining=%s used=%s",
                    path, len(resp.json()) if isinstance(resp.json(), list) else 1,
                    self._requests_remaining, self._requests_used,
                )
                return resp.json()
            except OddsAPIError:
                raise
            except requests.RequestException as exc:
                if attempt == 2:
                    raise OddsAPIError(f"Request failed after 3 attempts: {exc}") from exc
                time.sleep(2 ** attempt)
        return []

    @property
    def quota_remaining(self) -> int | None:
        return self._requests_remaining

    @property
    def quota_used(self) -> int | None:
        return self._requests_used

    # -----------------------------------------------------------------------
    # Sports discovery
    # -----------------------------------------------------------------------

    def list_sports(self, all_sports: bool = False) -> list[dict]:
        """GET /v4/sports/ — free, 0 credits."""
        params = {}
        if all_sports:
            params["all"] = "true"
        return self._get("/v4/sports/", params)

    def is_wnba_active(self) -> bool:
        """Return True if basketball_wnba is currently in season."""
        try:
            sports = self.list_sports(all_sports=False)
            return any(s.get("key") == SPORT_KEY and s.get("active") for s in sports)
        except OddsAPIError:
            return False

    # -----------------------------------------------------------------------
    # Game events (free)
    # -----------------------------------------------------------------------

    def list_events(
        self,
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
    ) -> list[dict]:
        """GET /v4/sports/basketball_wnba/events — 0 credits. Returns event IDs."""
        params: dict[str, Any] = {"dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        return self._get(f"/v4/sports/{SPORT_KEY}/events", params)

    def list_events_for_date(self, date_str: str) -> list[dict]:
        """Return all WNBA events on a given date (YYYY-MM-DD, Eastern).

        Window: 9 AM UTC on the target date through 4 AM UTC the NEXT day.
        This captures every possible tip-off in the Eastern timezone, including
        10 PM EDT games (= 2 AM UTC next day) that the old 23:59:59Z cutoff missed.
        """
        from datetime import datetime, timedelta  # noqa: PLC0415
        next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        from_dt = f"{date_str}T09:00:00Z"   # 5 AM ET = 9 AM UTC
        to_dt   = f"{next_day}T04:00:00Z"   # midnight ET = 4 AM UTC next day
        return self.list_events(commence_time_from=from_dt, commence_time_to=to_dt)

    # -----------------------------------------------------------------------
    # Bulk odds (h2h / spreads / totals)
    # -----------------------------------------------------------------------

    def get_bulk_odds(
        self,
        markets: list[str] | None = None,
        date_str: str | None = None,
    ) -> list[dict]:
        """GET /v4/sports/basketball_wnba/odds — 3 credits per call (3 mkts × 1 region).

        markets: list of market keys e.g. ['h2h', 'spreads', 'totals']
        """
        mkts = markets or ["h2h", "spreads", "totals"]
        params: dict[str, Any] = {
            "regions":    self.region,
            "markets":    ",".join(mkts),
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }
        return self._get(f"/v4/sports/{SPORT_KEY}/odds", params)

    # -----------------------------------------------------------------------
    # Player props per event (20 credits per call)
    # -----------------------------------------------------------------------

    def get_event_player_props(
        self,
        event_id: str,
        markets: list[str] | None = None,
        include_links: bool = True,
        include_sids: bool = True,
        bookmakers: list[str] | None = None,
    ) -> dict:
        """GET /v4/sports/basketball_wnba/events/{event_id}/odds

        Cost: n_markets × n_regions credits per call.
        Default 20 markets × 1 region = 20 credits.

        Parameters
        ----------
        event_id : str
            The Odds API event ID (from list_events).
        markets : list[str] | None
            Specific market keys to pull. Defaults to CORE_PROP_MARKETS.
        include_links : bool
            Add bookmaker deep links (event/market/outcome level).
        include_sids : bool
            Add source IDs for mobile deep links.
        bookmakers : list[str] | None
            Limit to specific bookmakers (e.g. ['draftkings', 'fanduel']).
            None = all available bookmakers in the region.
        """
        mkts = markets or CORE_PROP_MARKETS
        params: dict[str, Any] = {
            "regions":    self.region,
            "markets":    ",".join(mkts),
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }
        if include_links:
            params["includeLinks"] = "true"
        if include_sids:
            params["includeSids"] = "true"
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        return self._get(f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds", params)

    # -----------------------------------------------------------------------
    # Live scores
    # -----------------------------------------------------------------------

    def get_scores(self, days_from: int = 1, event_ids: list[str] | None = None) -> list[dict]:
        """GET /v4/sports/basketball_wnba/scores — 1 credit per call.

        Returns live and recently completed game scores.

        Parameters
        ----------
        days_from : int
            How many completed days back to include (1 = today's games).
        event_ids : list[str] | None
            Filter to specific event IDs.
        """
        params: dict[str, Any] = {"daysFrom": days_from, "dateFormat": "iso"}
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        return self._get(f"/v4/sports/{SPORT_KEY}/scores", params)

    # -----------------------------------------------------------------------
    # Historical odds (closing lines)
    # -----------------------------------------------------------------------

    def list_historical_events(self, date_str_iso: str,
                               commence_time_from: str | None = None,
                               commence_time_to: str | None = None) -> dict:
        """GET /v4/historical/sports/basketball_wnba/events?date=ISO

        Returns the historical events snapshot: {timestamp, previous_timestamp,
        next_timestamp, data: [ {id, commence_time, home_team, away_team}, ... ]}.
        Cost: historical surcharge (10× the free events endpoint). Use to obtain
        event IDs for PAST dates (the live events endpoint does not return them).

        commence_time_from / commence_time_to (ISO, e.g. '2026-06-16T00:00:00Z')
        restrict the returned events to those commencing inside that UTC window so
        coverage is computed against the correct per-game window rather than every
        event in an unfiltered daily snapshot.
        """
        params: dict[str, Any] = {"date": date_str_iso, "dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        return self._get(f"/v4/historical/sports/{SPORT_KEY}/events", params)

    def get_historical_event_odds(
        self,
        event_id: str,
        date_str: str,
        markets: list[str] | None = None,
    ) -> dict:
        """GET /v4/historical/sports/basketball_wnba/events/{id}/odds

        Cost: 10 × n_markets × n_regions credits (historical surcharge, 10× live rate).
        Use only for closing line archival (post_game_scoring.yml).

        Parameters
        ----------
        event_id : str
            The Odds API event ID.
        date_str : str
            ISO datetime string for the snapshot time (e.g. '2026-06-16T23:00:00Z').
            Use a time ~2 hours after game start for closing lines.
        markets : list[str] | None
            Defaults to CORE_PROP_MARKETS.
        """
        mkts = markets or CORE_PROP_MARKETS
        params: dict[str, Any] = {
            "date":       date_str,
            "regions":    self.region,
            "markets":    ",".join(mkts),
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
        }
        return self._get(
            f"/v4/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
            params,
        )

    # -----------------------------------------------------------------------
    # High-level helpers
    # -----------------------------------------------------------------------

    def get_all_props_for_date(
        self,
        date_str: str,
        markets: list[str] | None = None,
        include_links: bool = True,
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """Fetch player props for every WNBA event on a given date.

        Returns a flat list of normalized prop dicts (one per outcome).
        Each dict has keys: event_id, home_team, away_team, commence_time,
        bookmaker, market_key, player_name, stat, line, over_odds, under_odds,
        outcome_link, market_link, event_link.

        Cost: 20 credits × n_events (one API call per event).
        """
        events = self.list_events_for_date(date_str)
        if not events:
            log.info("[OddsAPI] No WNBA events found for %s", date_str)
            return []
        log.info("[OddsAPI] Found %d events for %s", len(events), date_str)

        rows: list[dict] = []
        for event in events:
            event_id = event.get("id", "")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence = event.get("commence_time", "")

            try:
                data = self.get_event_player_props(
                    event_id,
                    markets=markets,
                    include_links=include_links,
                    bookmakers=bookmakers,
                )
            except OddsAPIError as exc:
                log.warning("[OddsAPI] Props fetch failed for event %s: %s", event_id, exc)
                continue

            bookmakers_data = data.get("bookmakers", [])
            for book in bookmakers_data:
                bookmaker = book.get("key", "")
                book_link = book.get("link") or book.get("event_link")
                for market in book.get("markets", []):
                    market_key = market.get("key", "")
                    stat = ODDS_API_TO_STAT.get(market_key)
                    market_link = market.get("link")
                    last_update = market.get("last_update", "")
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        desc = outcome.get("description", "")  # player name
                        price = outcome.get("price")          # American odds
                        point = outcome.get("point")          # line value
                        side = outcome.get("name", "").lower()  # "Over" or "Under"
                        outcome_link = outcome.get("link")

                        rows.append({
                            "event_id": event_id,
                            "game_date": date_str,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence,
                            "bookmaker": bookmaker,
                            "market_key": market_key,
                            "stat": stat,
                            "player_name": desc,
                            "side": side,
                            "line": point,
                            "odds": price,
                            "last_update": last_update,
                            "event_link": book_link,
                            "market_link": market_link,
                            "outcome_link": outcome_link,
                        })

        log.info("[OddsAPI] Collected %d outcome rows from %d events", len(rows), len(events))
        return rows

    def get_closing_lines_for_date(
        self,
        date_str: str,
        close_time_utc: str | None = None,
    ) -> list[dict]:
        """Pull historical closing-line snapshots for all events on date_str.

        close_time_utc: ISO datetime of snapshot (e.g. '2026-06-16T23:00:00Z').
        Defaults to 11 PM UTC (7 PM ET + 4 hours — well into most games).
        Cost: 10 × 15 markets × n_events credits.
        """
        if close_time_utc is None:
            close_time_utc = f"{date_str}T23:00:00Z"

        # Event discovery MUST use the historical events endpoint: the live
        # /events endpoint returns nothing for past dates, which silently yields
        # zero closing-line rows (and breaks nightly CLV). Restrict to events
        # commencing in this game-date's UTC window.
        from datetime import datetime, timedelta  # noqa: PLC0415

        next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        ev_payload = self.list_historical_events(
            f"{date_str}T12:00:00Z",
            commence_time_from=f"{date_str}T00:00:00Z",
            commence_time_to=f"{next_day}T12:00:00Z",
        )
        events = (ev_payload or {}).get("data", []) or []
        if not events:
            return []

        rows: list[dict] = []
        for event in events:
            event_id = event.get("id", "")
            # The historical event-odds endpoint returns 404 for any snapshot at or
            # after tip-off (the event drops from the odds feed). Cap the requested
            # snapshot to 5 minutes before tip so a "closing" pull always lands on
            # the last real pre-tip quote instead of 404-ing to nothing.
            eff_snapshot = close_time_utc
            commence = event.get("commence_time")
            try:
                tip = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                req = datetime.fromisoformat(str(close_time_utc).replace("Z", "+00:00"))
                cap = tip - timedelta(minutes=5)
                if req > cap:
                    eff_snapshot = cap.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, ValueError):
                pass
            try:
                data = self.get_historical_event_odds(event_id, eff_snapshot)
            except OddsAPIError as exc:
                log.warning(
                    "[OddsAPI] Historical props failed for event %s: %s", event_id, exc
                )
                continue

            bookmakers_data = data.get("data", {}).get("bookmakers", [])
            for book in bookmakers_data:
                bookmaker = book.get("key", "")
                for market in book.get("markets", []):
                    market_key = market.get("key", "")
                    stat = ODDS_API_TO_STAT.get(market_key)
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "event_id": event_id,
                            "game_date": date_str,
                            "snapshot_time": eff_snapshot,
                            "home_team": event.get("home_team", ""),
                            "away_team": event.get("away_team", ""),
                            "bookmaker": bookmaker,
                            "market_key": market_key,
                            "stat": stat,
                            "player_name": outcome.get("description", ""),
                            "side": outcome.get("name", "").lower(),
                            "line": outcome.get("point"),
                            "odds": outcome.get("price"),
                        })
        return rows


# ---------------------------------------------------------------------------
# Normalization helpers (convert raw rows to pipeline schema)
# ---------------------------------------------------------------------------

def normalize_odds_api_props(rows: list[dict]) -> "pd.DataFrame":
    """Convert flat outcome rows from get_all_props_for_date into pipeline schema.

    Output columns match the schema expected by build_edge_report.py and
    normalize_player_props_snapshot():
      player_name, stat, line, over_odds, under_odds, bookmaker, market_key,
      event_id, game_date, event_link, market_link, outcome_link_over, outcome_link_under.
    """
    import pandas as _pd

    if not rows:
        return _pd.DataFrame(columns=[
            "player_name", "stat", "line", "over_odds", "under_odds",
            "bookmaker", "market_key", "event_id", "game_date",
            "event_link", "market_link", "outcome_link_over", "outcome_link_under",
            "last_update",
        ])

    df = _pd.DataFrame(rows)

    # Pivot Over/Under into one row per player-stat-line-bookmaker
    over_df  = df[df["side"].str.lower().str.startswith("over")].copy()
    under_df = df[df["side"].str.lower().str.startswith("under")].copy()

    key_cols = ["event_id", "game_date", "bookmaker", "market_key",
                "player_name", "line", "stat"]

    over_df  = over_df.rename(columns={"odds": "over_odds",  "outcome_link": "outcome_link_over"})
    under_df = under_df.rename(columns={"odds": "under_odds", "outcome_link": "outcome_link_under"})

    merged = over_df[key_cols + ["over_odds", "outcome_link_over", "event_link", "market_link",
                                  "home_team", "away_team", "commence_time", "last_update"]].merge(
        under_df[key_cols + ["under_odds", "outcome_link_under"]],
        on=key_cols,
        how="outer",
    )

    # Filter to stats the model handles
    merged = merged[merged["stat"].notna()].copy()
    return merged.reset_index(drop=True)


def get_bookmaker_deep_link(row: "dict | pd.Series") -> str | None:
    """Cascade through outcome → market → event link per blueprint spec.

    Priority:
    1. outcome_link_over (most specific — goes to betslip for over bet)
    2. market_link (goes to the market page)
    3. event_link (event-level fallback)
    """
    for field in ("outcome_link_over", "market_link", "event_link"):
        val = row.get(field) if isinstance(row, dict) else getattr(row, field, None)
        if val and str(val).startswith("http"):
            return str(val)
    return None
