"""Shared Odds-API payload -> atomic two-sided side-row extraction for Path B.

Kept in the package (not a script) so both the live scan and the offline fixture scan can
import it without fragile ``scripts`` path hacks.
"""
from __future__ import annotations

from datetime import datetime, timezone

# The seven single-stat markets Path B shops for dislocations.
BOARD_MARKETS = [
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_steals", "player_blocks", "player_turnovers",
]


def extract_side_rows(event_odds: dict, markets: list[str] | None = None) -> list[dict]:
    """Flatten one Odds-API event-odds payload into atomic per-side quote rows.

    Emits the exact schema ``scan_soft_book_edges`` consumes: collected_utc, event_id,
    commence_time, home_team, away_team, book, book_last_update, market_key, stat,
    player_name, side, line, american_odds.
    """
    from wnba_props_model.data.odds_api_client import ODDS_API_TO_STAT

    markets = markets or BOARD_MARKETS
    ts = datetime.now(timezone.utc).isoformat()
    sides: list[dict] = []
    for bm in event_odds.get("bookmakers", []) or []:
        for mk in bm.get("markets", []) or []:
            if mk.get("key") not in markets:
                continue
            for oc in mk.get("outcomes", []) or []:
                sides.append({
                    "collected_utc": ts,
                    "event_id": event_odds.get("id"),
                    "commence_time": event_odds.get("commence_time"),
                    "home_team": event_odds.get("home_team"),
                    "away_team": event_odds.get("away_team"),
                    "book": bm.get("key"),
                    "book_last_update": bm.get("last_update") or mk.get("last_update"),
                    "market_key": mk.get("key"),
                    "stat": ODDS_API_TO_STAT.get(mk.get("key")),
                    "player_name": oc.get("description") or oc.get("name"),
                    "side": str(oc.get("name", "")).lower(),
                    "line": oc.get("point"),
                    "american_odds": oc.get("price"),
                })
    return sides
