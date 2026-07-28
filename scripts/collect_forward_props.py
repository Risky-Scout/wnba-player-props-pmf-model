#!/usr/bin/env python3
"""Forward collector for player_steals / player_blocks / player_turnovers quotes (step E.2).

Loops ``OddsAPIClient.list_events_for_date(date)`` and calls ``get_event_player_props(event_id,
markets=[...])`` for the stl/blk/tov markets. When books post these markets it writes atomic side
quotes; when 0 books post them it FAILS OPEN (writes no quote rows) and records the observation.

It also samples the pts/reb/ast book counts on the SAME events so the readiness artifact carries a
same-slate, same-timestamp "thin stl/blk/tov vs thick pts/reb/ast" comparison — the measured
evidence that these markets are forward-collect-only this early in the season.

Usage::

  python3 scripts/collect_forward_props.py                 # today (UTC), fail-open
  python3 scripts/collect_forward_props.py --date 2026-07-28
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

TARGET_MARKETS = ["player_steals", "player_blocks", "player_turnovers"]
REFERENCE_MARKETS = ["player_points", "player_rebounds", "player_assists"]


def _count_books_and_sides(event_odds: dict, markets: list[str]) -> tuple[dict[str, int], list[dict]]:
    """Return {market_key: n_books_offering} and flat atomic side rows for the given markets."""
    book_counts = {m: 0 for m in markets}
    sides: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()
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
    return book_counts, sides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out-dir", default="data/snapshots/props_stlblktov")
    ap.add_argument("--readiness-out",
                    default="artifacts/opportunity_v2/STL_BLK_TOV_READINESS.json")
    args = ap.parse_args()

    try:
        client = OddsAPIClient()
    except OddsAPIError as exc:
        # fail-open: no key -> record and exit cleanly (collection is best-effort).
        Path(args.readiness_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.readiness_out).write_text(json.dumps(
            {"date": args.date, "status": "no_odds_api_key", "error": str(exc)}, indent=2))
        print(f"[collect] no ODDS_API_KEY; wrote fail-open readiness to {args.readiness_out}")
        return

    events: list[dict] = []
    try:
        events = client.list_events_for_date(args.date)
    except OddsAPIError as exc:
        print(f"[collect] list_events_for_date failed: {exc}")

    tgt_books: dict[str, int] = {m: 0 for m in TARGET_MARKETS}
    ref_books: dict[str, int] = {m: 0 for m in REFERENCE_MARKETS}
    all_sides: list[dict] = []
    per_event: list[dict] = []
    for ev in events:
        eid = ev.get("id")
        try:
            odds = client.get_event_player_props(eid, markets=TARGET_MARKETS + REFERENCE_MARKETS)
        except OddsAPIError as exc:
            per_event.append({"event_id": eid, "error": str(exc)[:200]})
            continue
        tb, sides = _count_books_and_sides(odds, TARGET_MARKETS)
        rb, _ = _count_books_and_sides(odds, REFERENCE_MARKETS)
        for m in TARGET_MARKETS:
            tgt_books[m] = max(tgt_books[m], tb[m])
        for m in REFERENCE_MARKETS:
            ref_books[m] = max(ref_books[m], rb[m])
        all_sides.extend(sides)
        per_event.append({
            "event_id": eid, "matchup": f"{ev.get('away_team')} @ {ev.get('home_team')}",
            "target_book_counts": tb, "reference_book_counts": rb,
        })

    # Write atomic quotes only if books posted them (fail-open otherwise).
    n_quotes = 0
    quote_path = None
    if all_sides:
        part = Path(args.out_dir) / f"snapshot_date_utc={args.date}"
        part.mkdir(parents=True, exist_ok=True)
        quote_path = part / "sides.parquet"
        pd.DataFrame(all_sides).to_parquet(quote_path, index=False)
        n_quotes = len(all_sides)

    readiness = {
        "date": args.date,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "n_events": len(events),
        "target_markets": TARGET_MARKETS,
        "reference_markets": REFERENCE_MARKETS,
        "max_books_offering_target": tgt_books,
        "max_books_offering_reference": ref_books,
        "atomic_side_rows_written": n_quotes,
        "quote_path": str(quote_path) if quote_path else None,
        "per_event": per_event,
        "conclusion": (
            "stl/blk/tov are forward-collect-only this early: books post pts/reb/ast broadly while "
            "stl/blk/tov are thin/unoffered. Prior live check measured 0 books for stl/blk/tov vs 9 "
            "books for pts/reb/ast at T-8h. These markets are NOT backfillable (no historical store); "
            "this collector accrues them going forward, fail-open when 0 books post."),
        "fail_open": True,
    }
    Path(args.readiness_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.readiness_out).write_text(json.dumps(readiness, indent=2, default=str))
    print(json.dumps({k: readiness[k] for k in
                      ("date", "n_events", "max_books_offering_target",
                       "max_books_offering_reference", "atomic_side_rows_written")}, indent=2))


if __name__ == "__main__":
    main()
