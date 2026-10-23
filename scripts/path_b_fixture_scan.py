#!/usr/bin/env python3
"""Deterministic, OFFLINE Path B end-to-end scan for CI (no live odds required).

Builds the crafted fixture slate (``path_b_fixtures``), runs the full hardened dislocation
pipeline — identity resolution, atomic-line segregation, timestamp integrity, leave-one-out
consensus, no-vig fail-closed — and writes the same Path B audit artifacts as the live scan.
It exercises EVERY rejection path so the acceptance gate has a real, reproducible input in CI.

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/path_b_fixture_scan.py --out-dir artifacts/path_b_fixture
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.edge.path_b_audit import board_to_sample_csv, build_audit
from wnba_props_model.edge.path_b_collect import BOARD_MARKETS, extract_side_rows
from wnba_props_model.edge.path_b_fixtures import make_events, make_roster
from wnba_props_model.edge.prop_identity import build_name_index
from wnba_props_model.edge.soft_book_scan import (
    DEFAULT_EV_THRESHOLD,
    DEFAULT_MIN_CONSENSUS_BOOKS,
    REFERENCE_BOOKS,
    SHARP_BOOKS,
    scan_soft_book_edges,
)

MAX_QUOTE_AGE_SECONDS = 3600.0

# Include an alternate market so the fixture actually exercises atomic segregation of
# ``*_alternate`` keys (live collection stays lean on the 7 standard markets).
FIXTURE_MARKETS = BOARD_MARKETS + ["player_points_alternate"]


def run_fixture_scan(out_dir: Path, *, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    events = make_events(now)
    roster = make_roster()

    all_sides: list[dict] = []
    for ev in events:
        all_sides.extend(extract_side_rows(ev, FIXTURE_MARKETS))
    quotes = pd.DataFrame(all_sides)

    identity_index = build_name_index(roster)
    board, rejections = scan_soft_book_edges(
        quotes,
        ev_threshold=DEFAULT_EV_THRESHOLD,
        min_consensus_books=DEFAULT_MIN_CONSENSUS_BOOKS,
        sharp_books=SHARP_BOOKS,
        reference_books=REFERENCE_BOOKS,
        max_quote_age_seconds=MAX_QUOTE_AGE_SECONDS,
        identity_index=identity_index,
        require_identity=True,
        now=now,
        return_rejections=True,
    )

    books_observed = sorted({str(b) for b in quotes["book"].dropna().unique()})
    markets_observed = sorted({str(m) for m in quotes["market_key"].dropna().unique()})
    atomic_pairs = board[["event_id", "player_name", "market_key", "line"]].drop_duplicates().shape[0] \
        if len(board) else 0

    config = {
        "ev_threshold_pct": round(DEFAULT_EV_THRESHOLD * 100.0, 4),
        "min_consensus_books": DEFAULT_MIN_CONSENSUS_BOOKS,
        "max_quote_age_seconds": MAX_QUOTE_AGE_SECONDS,
        "reference_books": sorted(str(b) for b in REFERENCE_BOOKS),
        "region": "fixture",
        "no_vig_fail_closed": True,
        "identity_required": True,
        "mode": "fixture",
    }
    discovery = {
        "games_discovered": len({e["id"] for e in events}),
        "books_observed": books_observed,
        "n_books_observed": len(books_observed),
        "markets_observed": markets_observed,
        "atomic_pairs_created": int(atomic_pairs),
        "n_quote_rows": int(len(quotes)),
        "n_roster_players": len(roster),
    }
    credit_usage = {"mode": "fixture", "x_requests_remaining_before": None,
                    "x_requests_remaining_after": None, "consumed": 0,
                    "notes": ["offline fixture — no API credits consumed"]}
    # Fixture is offline: price-survival cannot be re-checked live. Disclose honestly.
    price_survival = {
        "rechecked": 0, "survived_30s": 0, "survived_60s": 0, "details": [],
        "note": "offline fixture — forward price-survival recheck requires live odds",
    }

    audit = build_audit(
        board, rejections,
        game_date=now.strftime("%Y-%m-%d"),
        config=config,
        discovery=discovery,
        credit_usage=credit_usage,
        price_survival=price_survival,
        diagnostic_ev_threshold=DEFAULT_EV_THRESHOLD,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "LIVE_SCAN_AUDIT.json").write_text(json.dumps(audit, indent=2, default=str))
    board_to_sample_csv(board, out_dir / "EDGE_BOARD_SAMPLE.csv")
    (out_dir / "CONSENSUS_CONSTRUCTION_AUDIT.json").write_text(json.dumps({
        "schema_version": audit["schema_version"], "game_date": audit["game_date"],
        "source_type": audit["source_type"], "generated_at": audit["generated_at"],
        "consensus": audit["consensus"],
        "method": {
            "leave_one_book_out": True, "self_excluded_from_own_consensus": True,
            "consensus_statistic": "median of per-book Shin no-vig P(over), self-excluded",
            "reference_books": config["reference_books"],
            "atomic_key": ["event_id", "player_name", "market_key", "line"],
            "alternate_markets_segregated": True,
        },
    }, indent=2, default=str))
    (out_dir / "QUOTE_LATENCY_AUDIT.json").write_text(json.dumps({
        "schema_version": audit["schema_version"], "game_date": audit["game_date"],
        "source_type": audit["source_type"], "generated_at": audit["generated_at"],
        "latency": audit["latency"], "price_survival": audit["price_survival"],
    }, indent=2, default=str))
    (out_dir / "API_CREDIT_USAGE.json").write_text(json.dumps(credit_usage, indent=2, default=str))

    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="artifacts/path_b_fixture")
    args = ap.parse_args()
    audit = run_fixture_scan(Path(args.out_dir))
    print(json.dumps({
        "mode": "fixture",
        "games_discovered": audit["discovery"]["games_discovered"],
        "atomic_pairs_created": audit["discovery"]["atomic_pairs_created"],
        "n_board_rows": audit["summary"]["n_board_rows"],
        "n_diagnostic_edges": audit["summary"]["n_diagnostic_edges"],
        "n_qualifying": audit["summary"]["n_qualifying"],
        "rejections": audit["rejections"],
        "out_dir": args.out_dir,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
