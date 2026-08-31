"""Deterministic, now-relative fixtures for the Path B end-to-end offline scan.

Structure is fixed; only the timestamps are computed relative to a supplied ``now`` so that
"future" commence times and "fresh"/"stale" quote ages stay meaningful whenever the fixture
runs (tests and CI). The fixture is crafted to exercise EVERY rejection path:

  * a genuine qualifying +EV soft-book dislocation (resolvable identity, 3 reference books),
  * an UNRESOLVED identity (name absent from the canonical roster),
  * a MISSING-opposite-side book (no-vig fail closed),
  * a STALE quote (older than the strict age gate),
  * a MALFORMED provider timestamp,
  * an ALTERNATE market at the same line as the standard market (must stay segregated),
  * a POST-TIP event (rejected wholesale).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def make_roster() -> list[dict]:
    """Canonical roster for exact identity resolution. Note: 'Bravo Center' is ABSENT."""
    return [
        {"player_name": "Alpha Guard", "player_id": 1001},
        {"player_name": "Charlie Wing", "player_id": 1003},
        {"player_name": "Delta Forward", "player_id": 1004},
    ]


def _outcomes(over_odds, under_odds, line):
    out = [{"name": "Over", "description": None, "price": over_odds, "point": line}]
    if under_odds is not None:
        out.append({"name": "Under", "description": None, "price": under_odds, "point": line})
    # description carries the player name; set by caller via _market
    return out


def _market(key, player, over_odds, under_odds, line):
    outs = []
    outs.append({"name": "Over", "description": player, "price": over_odds, "point": line})
    if under_odds is not None:
        outs.append({"name": "Under", "description": player, "price": under_odds, "point": line})
    return {"key": key, "outcomes": outs}


def make_events(now: datetime | None = None) -> list[dict]:
    """Return a list of Odds API-shaped event-odds payloads (now-relative timestamps)."""
    now = now or datetime.now(timezone.utc)
    future = _iso(now + timedelta(hours=6))
    past = _iso(now - timedelta(hours=6))
    fresh = _iso(now - timedelta(seconds=20))
    stale = _iso(now - timedelta(hours=3))
    malformed = "not-a-timestamp"

    def book(key, last_update, markets):
        return {"key": key, "last_update": last_update, "markets": markets}

    # ── Event 1: live slate with a qualifying dislocation + rejection cases ──────
    evt_good = {
        "id": "evt-good",
        "commence_time": future,
        "home_team": "Home City", "away_team": "Away Town",
        "bookmakers": [
            # Alpha Guard player_points @18.5 — 3 reference/sharp books ~ -110/-110 (fair ~0.5),
            # softbook offers a generous over +120 / under -140 => +EV over qualifies.
            book("pinnacle", fresh, [_market("player_points", "Alpha Guard", -110, -110, 18.5)]),
            book("betonlineag", fresh, [_market("player_points", "Alpha Guard", -110, -110, 18.5)]),
            book("draftkings", fresh, [_market("player_points", "Alpha Guard", -110, -110, 18.5)]),
            book("softbook", fresh, [_market("player_points", "Alpha Guard", 120, -140, 18.5)]),
            # Bravo Center — NOT in roster => unresolved identity rejection.
            book("pinnacle", fresh, [_market("player_rebounds", "Bravo Center", -110, -110, 7.5)]),
            book("draftkings", fresh, [_market("player_rebounds", "Bravo Center", 130, -150, 7.5)]),
            book("betonlineag", fresh, [_market("player_rebounds", "Bravo Center", -110, -110, 7.5)]),
            # Charlie Wing assists @4.5 — softbook posts ONLY the over (missing opposite side).
            book("pinnacle", fresh, [_market("player_assists", "Charlie Wing", -110, -110, 4.5)]),
            book("draftkings", fresh, [_market("player_assists", "Charlie Wing", -110, -110, 4.5)]),
            book("betonlineag", fresh, [_market("player_assists", "Charlie Wing", -105, -115, 4.5)]),
            book("softbook", fresh, [_market("player_assists", "Charlie Wing", 140, None, 4.5)]),
            # Delta Forward threes @1.5 — one book carries a STALE quote and one MALFORMED ts.
            book("pinnacle", fresh, [_market("player_threes", "Delta Forward", -110, -110, 1.5)]),
            book("draftkings", stale, [_market("player_threes", "Delta Forward", -110, -110, 1.5)]),
            book("betonlineag", malformed, [_market("player_threes", "Delta Forward", -110, -110, 1.5)]),
            book("softbook", fresh, [_market("player_threes", "Delta Forward", 115, -135, 1.5)]),
            # Alpha Guard player_points_ALTERNATE @18.5 — must stay segregated from standard.
            book("pinnacle", fresh, [_market("player_points_alternate", "Alpha Guard", 250, -320, 18.5)]),
            book("draftkings", fresh, [_market("player_points_alternate", "Alpha Guard", 260, -340, 18.5)]),
            book("betonlineag", fresh, [_market("player_points_alternate", "Alpha Guard", 255, -330, 18.5)]),
        ],
    }

    # ── Event 2: already tipped off — whole event rejected (post-tip). ───────────
    evt_post = {
        "id": "evt-post-tip",
        "commence_time": past,
        "home_team": "Old Home", "away_team": "Old Away",
        "bookmakers": [
            book("pinnacle", fresh, [_market("player_points", "Delta Forward", -110, -110, 12.5)]),
            book("draftkings", fresh, [_market("player_points", "Delta Forward", 150, -170, 12.5)]),
            book("betonlineag", fresh, [_market("player_points", "Delta Forward", -110, -110, 12.5)]),
        ],
    }
    return [evt_good, evt_post]
