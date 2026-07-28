#!/usr/bin/env python3
"""Multi-book, all-market forward collector for the soft-book +EV edge board (Path B).

Generalizes ``scripts/collect_forward_props.py`` to pull TWO-SIDED player-prop quotes for
pts / reb / ast / threes(fg3m) / stl / blk / turnover across ALL US books (region
``us,us2``) for a given date's WNBA events, then persists atomic per-side rows to a
timestamped snapshot under a gitignored data dir.

Emits the same atomic side schema as ``collect_forward_props.py`` (one row per book ×
market × player × side): collected_utc, event_id, commence_time, home_team, away_team,
book, book_last_update, market_key, stat, player_name, side, line, american_odds. These
columns are exactly what ``wnba_props_model.edge.soft_book_scan.scan_soft_book_edges``
consumes.

Fail-open: with no ODDS_API_KEY, no events, or no quotes, it writes a manifest and exits
0 (out-of-season / off-night no-op is clean, never a hard failure).

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/collect_soft_book_quotes.py          # today UTC
  PYTHONPATH=$(pwd)/src python3 scripts/collect_soft_book_quotes.py --date 2026-07-28
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.data.odds_api_client import (
    ODDS_API_TO_STAT,
    OddsAPIClient,
    OddsAPIError,
)

# All seven single-stat markets we shop for soft-book edges.
BOARD_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_steals",
    "player_blocks",
    "player_turnovers",
]


def extract_side_rows(event_odds: dict, markets: list[str]) -> tuple[list[dict], dict[str, int]]:
    """Flatten one event-odds payload into atomic per-side rows + per-market book counts."""
    ts = datetime.now(timezone.utc).isoformat()
    sides: list[dict] = []
    book_counts = {m: 0 for m in markets}
    for bm in event_odds.get("bookmakers", []) or []:
        offered = {mk.get("key") for mk in bm.get("markets", []) or []}
        for m in markets:
            if m in offered:
                book_counts[m] += 1
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
                    "book_last_update": bm.get("last_update"),
                    "market_key": mk.get("key"),
                    "stat": ODDS_API_TO_STAT.get(mk.get("key")),
                    "player_name": oc.get("description") or oc.get("name"),
                    "side": str(oc.get("name", "")).lower(),
                    "line": oc.get("point"),
                    "american_odds": oc.get("price"),
                })
    return sides, book_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--region", default="us,us2",
                    help="Odds API region(s); 'us,us2' maximizes book coverage.")
    ap.add_argument("--out-dir", default="data/snapshots/soft_book_quotes")
    ap.add_argument("--manifest-out",
                    default="artifacts/edge_board/COLLECTION_MANIFEST.json")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: dict = {
        "date": args.date,
        "region": args.region,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "markets": BOARD_MARKETS,
        "n_events": 0,
        "atomic_side_rows_written": 0,
        "quote_path": None,
        "max_books_per_market": {m: 0 for m in BOARD_MARKETS},
        "per_event": [],
        "status": "ok",
        "fail_open": True,
    }

    def _write_manifest() -> None:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, default=str))

    try:
        client = OddsAPIClient(region=args.region)
    except OddsAPIError as exc:
        manifest["status"] = "no_odds_api_key"
        manifest["error"] = str(exc)
        _write_manifest()
        print(f"[collect] no ODDS_API_KEY; wrote fail-open manifest to {args.manifest_out}")
        return

    try:
        events = client.list_events_for_date(args.date)
    except OddsAPIError as exc:
        manifest["status"] = "events_list_failed"
        manifest["error"] = str(exc)
        _write_manifest()
        print(f"[collect] list_events_for_date failed: {exc}")
        return

    manifest["n_events"] = len(events)
    if not events:
        manifest["status"] = "no_events"
        _write_manifest()
        print(f"[collect] no WNBA events for {args.date}; fail-open no-op.")
        return

    all_sides: list[dict] = []
    for ev in events:
        eid = ev.get("id")
        try:
            odds = client.get_event_player_props(eid, markets=BOARD_MARKETS)
        except OddsAPIError as exc:
            manifest["per_event"].append({"event_id": eid, "error": str(exc)[:200]})
            continue
        sides, counts = extract_side_rows(odds, BOARD_MARKETS)
        all_sides.extend(sides)
        for m in BOARD_MARKETS:
            manifest["max_books_per_market"][m] = max(
                manifest["max_books_per_market"][m], counts[m]
            )
        manifest["per_event"].append({
            "event_id": eid,
            "matchup": f"{ev.get('away_team')} @ {ev.get('home_team')}",
            "commence_time": ev.get("commence_time"),
            "book_counts": counts,
            "side_rows": len(sides),
        })

    if all_sides:
        part = Path(args.out_dir) / f"snapshot_date_utc={args.date}"
        part.mkdir(parents=True, exist_ok=True)
        quote_path = part / f"quotes_{stamp}.parquet"
        pd.DataFrame(all_sides).to_parquet(quote_path, index=False)
        manifest["atomic_side_rows_written"] = len(all_sides)
        manifest["quote_path"] = str(quote_path)
    else:
        manifest["status"] = "no_quotes"

    _write_manifest()
    print(json.dumps({k: manifest[k] for k in
                      ("date", "region", "n_events", "atomic_side_rows_written",
                       "max_books_per_market", "quote_path", "status")}, indent=2))


if __name__ == "__main__":
    main()
