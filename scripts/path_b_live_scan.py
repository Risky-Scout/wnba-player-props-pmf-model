#!/usr/bin/env python3
"""Path B MARKET_DISLOCATION scan orchestrator — live or fixture, with full audit output.

Runs the hardened soft-book +EV dislocation scan end-to-end and writes the required Path B
audit artifacts. Works in two modes:

  * LIVE (default): pull two-sided player-prop quotes for today's WNBA slate from The Odds
    API (region us,us2), resolve identity against BDL's canonical roster, run the scan, and
    — when credits permit — RE-CHECK candidate prices at ~30s and ~60s to record price
    survival (EXECUTABLE_EV realism). Records API credit usage from the x-requests headers.

  * FIXTURE (``--events-json`` + ``--players-json``): fully offline / deterministic. Reads a
    saved list of event-odds payloads and a canonical roster, exercises the full pipeline
    incl. every rejection path, and writes the same artifacts. Used by CI (no live odds).

Artifacts written under ``--out-dir`` (default artifacts/path_b):
  LIVE_SCAN_AUDIT.json, EDGE_BOARD_SAMPLE.csv, CONSENSUS_CONSTRUCTION_AUDIT.json,
  QUOTE_LATENCY_AUDIT.json, API_CREDIT_USAGE.json

Nothing here is a model edge; nothing is claimed profitable/executable. Every row is
source_type=MARKET_DISLOCATION and actionable=False.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.edge.path_b_audit import build_audit, board_to_sample_csv
from wnba_props_model.edge.path_b_collect import BOARD_MARKETS, extract_side_rows
from wnba_props_model.edge.prop_identity import build_name_index
from wnba_props_model.edge.soft_book_scan import (
    DEFAULT_EV_THRESHOLD,
    DEFAULT_MIN_CONSENSUS_BOOKS,
    REFERENCE_BOOKS,
    SHARP_BOOKS,
    scan_soft_book_edges,
)


# --------------------------------------------------------------------------- #
# Live data acquisition
# --------------------------------------------------------------------------- #
def _live_collect(date_str: str, region: str):
    """Return (events_payloads, quotes_df, roster, credit_before, credit_after, notes)."""
    from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient
    from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError

    notes: list[str] = []
    try:
        client = OddsAPIClient(region=region)
    except OddsAPIError as exc:
        return [], pd.DataFrame(), [], None, None, [f"no_odds_api_key: {exc}"]

    try:
        events = client.list_events_for_date(date_str)
    except OddsAPIError as exc:
        return [], pd.DataFrame(), [], client.quota_remaining, client.quota_remaining, [
            f"events_list_failed: {exc}"]

    credit_before = client.quota_remaining
    payloads: list[dict] = []
    all_sides: list[dict] = []
    for ev in events:
        eid = ev.get("id")
        try:
            odds = client.get_event_player_props(eid, markets=BOARD_MARKETS)
        except OddsAPIError as exc:
            notes.append(f"event {eid} props failed: {str(exc)[:120]}")
            continue
        payloads.append(odds)
        all_sides.extend(extract_side_rows(odds, BOARD_MARKETS))
    credit_after = client.quota_remaining

    # Canonical roster for exact identity resolution.
    roster: list[dict] = []
    try:
        bdl = BDLClient()
        for p in bdl.list_endpoint("players_active"):
            fn = (p.get("first_name") or "").strip()
            ln = (p.get("last_name") or "").strip()
            name = f"{fn} {ln}".strip()
            if name and p.get("id") is not None:
                roster.append({"player_name": name, "player_id": p.get("id")})
    except BDLAPIError as exc:
        notes.append(f"bdl_roster_failed: {exc}")

    quotes = pd.DataFrame(all_sides) if all_sides else pd.DataFrame()
    return payloads, quotes, roster, credit_before, credit_after, notes, client


def _recheck_price_survival(client, board: pd.DataFrame, region: str,
                            recheck_events: int, waits=(30, 60)) -> dict:
    """Re-fetch candidate events after wait windows; record whether each candidate price
    survived (still present and equal-or-better for the bettor). Consumes credits."""
    details: list[dict] = []
    if client is None or not len(board):
        return {"rechecked": 0, "survived_30s": 0, "survived_60s": 0, "details": [],
                "note": "no candidates or no live client"}
    cand = board[board["qualified"]] if "qualified" in board.columns else board
    if not len(cand):
        cand = board.sort_values("theoretical_ev_frac", ascending=False).head(3)
    event_ids = list(dict.fromkeys(cand["event_id"].tolist()))[:recheck_events]
    if not event_ids:
        return {"rechecked": 0, "survived_30s": 0, "survived_60s": 0, "details": [],
                "note": "no candidate events"}

    from wnba_props_model.data.odds_api_client import OddsAPIError

    survived = {w: 0 for w in waits}
    prev_wait = 0
    for w in waits:
        time.sleep(max(0, w - prev_wait))
        prev_wait = w
        for eid in event_ids:
            try:
                odds = client.get_event_player_props(eid, markets=BOARD_MARKETS)
            except OddsAPIError as exc:
                details.append({"event_id": eid, "wait_s": w, "error": str(exc)[:120]})
                continue
            fresh = pd.DataFrame(extract_side_rows(odds, BOARD_MARKETS))
            ev_cands = cand[cand["event_id"] == eid]
            for _, r in ev_cands.iterrows():
                match = fresh[
                    (fresh["book"] == r["book"])
                    & (fresh["market_key"] == r["market_key"])
                    & (fresh["side"] == r["side"])
                    & (fresh["line"] == r["line"])
                ]
                still = None
                if len(match):
                    new_odds = float(match.iloc[0]["american_odds"])
                    orig = float(r["displayed_odds"])
                    # "survived" = price still offered and not worse for the bettor.
                    still = bool(new_odds == orig or _decimal_ge(new_odds, orig))
                    if still:
                        survived[w] += 1
                details.append({
                    "event_id": eid, "wait_s": w, "player_name": r["player_name"],
                    "book": r["book"], "market_key": r["market_key"], "side": r["side"],
                    "line": float(r["line"]), "original_odds": float(r["displayed_odds"]),
                    "found": bool(len(match)),
                    "new_odds": (float(match.iloc[0]["american_odds"]) if len(match) else None),
                    "survived": still,
                })
    return {
        "rechecked": len(event_ids),
        "survived_30s": survived.get(30, 0),
        "survived_60s": survived.get(60, 0),
        "waits_seconds": list(waits),
        "details": details,
        "note": "price-survival recheck of candidate prices; survived = still offered and "
                "not worse for the bettor",
    }


def _decimal_ge(a: float, b: float) -> bool:
    from wnba_props_model.edge.soft_book_scan import american_to_decimal_profit
    da, db = american_to_decimal_profit(a), american_to_decimal_profit(b)
    if da is None or db is None:
        return False
    return da >= db


# --------------------------------------------------------------------------- #
# Fixture mode
# --------------------------------------------------------------------------- #
def _fixture_collect(events_json: str, players_json: str):
    payloads = json.loads(Path(events_json).read_text())
    if isinstance(payloads, dict):
        payloads = payloads.get("events", [payloads])
    roster = json.loads(Path(players_json).read_text()) if players_json else []
    all_sides: list[dict] = []
    for odds in payloads:
        all_sides.extend(extract_side_rows(odds, BOARD_MARKETS))
    quotes = pd.DataFrame(all_sides) if all_sides else pd.DataFrame()
    return payloads, quotes, roster


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--region", default="us,us2")
    ap.add_argument("--out-dir", default="artifacts/path_b")
    ap.add_argument("--ev-threshold", type=float, default=DEFAULT_EV_THRESHOLD)
    ap.add_argument("--min-consensus-books", type=int, default=DEFAULT_MIN_CONSENSUS_BOOKS)
    ap.add_argument("--max-quote-age-seconds", type=float, default=None,
                    help="Strict live-age gate; quotes older than this are rejected.")
    ap.add_argument("--events-json", default=None,
                    help="FIXTURE mode: JSON file of event-odds payloads (offline).")
    ap.add_argument("--players-json", default=None,
                    help="FIXTURE mode: JSON roster [{player_name, player_id}, ...].")
    ap.add_argument("--recheck-events", type=int, default=1,
                    help="How many candidate events to re-check for price survival (live).")
    ap.add_argument("--no-recheck", action="store_true",
                    help="Skip the live price-survival recheck (save credits).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    client = None
    credit_before = credit_after = None

    if args.events_json:
        mode = "fixture"
        payloads, quotes, roster = _fixture_collect(args.events_json, args.players_json)
    else:
        mode = "live"
        result = _live_collect(args.date, args.region)
        if len(result) == 7:
            payloads, quotes, roster, credit_before, credit_after, notes, client = result
        else:
            payloads, quotes, roster, credit_before, credit_after, notes = result

    games_discovered = len({p.get("id") for p in payloads}) if payloads else 0
    books_observed = sorted({str(b) for b in quotes["book"].dropna().unique()}) if len(quotes) else []
    markets_observed = sorted({str(m) for m in quotes["market_key"].dropna().unique()}) if len(quotes) else []

    identity_index = build_name_index(roster) if roster else {}

    board, rejections = scan_soft_book_edges(
        quotes,
        ev_threshold=args.ev_threshold,
        min_consensus_books=args.min_consensus_books,
        sharp_books=SHARP_BOOKS,
        reference_books=REFERENCE_BOOKS,
        max_quote_age_seconds=args.max_quote_age_seconds,
        identity_index=identity_index,
        require_identity=True,
        return_rejections=True,
    )

    # atomic pairs = distinct (event, player, market_key, line) with a two-sided quote.
    atomic_pairs = 0
    if len(board):
        atomic_pairs = board[["event_id", "player_name", "market_key", "line"]].drop_duplicates().shape[0]

    # Execution realism (req 7): forward price-survival recheck (live only).
    price_survival = None
    if mode == "live" and client is not None and not args.no_recheck:
        try:
            price_survival = _recheck_price_survival(
                client, board, args.region, args.recheck_events)
            credit_after = client.quota_remaining
        except Exception as exc:  # never let recheck crash the audit
            notes.append(f"price_survival_recheck_error: {str(exc)[:160]}")

    credit_consumed = None
    if credit_before is not None and credit_after is not None:
        credit_consumed = int(credit_before) - int(credit_after)
    credit_usage = {
        "mode": mode,
        "x_requests_remaining_before": credit_before,
        "x_requests_remaining_after": credit_after,
        "consumed": credit_consumed,
        "notes": notes,
    }

    config = {
        "ev_threshold_pct": round(args.ev_threshold * 100.0, 4),
        "min_consensus_books": args.min_consensus_books,
        "max_quote_age_seconds": args.max_quote_age_seconds,
        "reference_books": sorted(str(b) for b in REFERENCE_BOOKS),
        "region": args.region,
        "no_vig_fail_closed": True,
        "identity_required": True,
        "mode": mode,
    }
    discovery = {
        "games_discovered": games_discovered,
        "books_observed": books_observed,
        "n_books_observed": len(books_observed),
        "markets_observed": markets_observed,
        "atomic_pairs_created": int(atomic_pairs),
        "n_quote_rows": int(len(quotes)),
        "n_roster_players": len(roster),
    }

    audit = build_audit(
        board, rejections,
        game_date=args.date,
        config=config,
        discovery=discovery,
        credit_usage=credit_usage,
        price_survival=price_survival,
        diagnostic_ev_threshold=args.ev_threshold,
    )

    (out_dir / "LIVE_SCAN_AUDIT.json").write_text(json.dumps(audit, indent=2, default=str))
    n_csv = board_to_sample_csv(board, out_dir / "EDGE_BOARD_SAMPLE.csv")
    (out_dir / "CONSENSUS_CONSTRUCTION_AUDIT.json").write_text(
        json.dumps({
            "schema_version": audit["schema_version"],
            "game_date": args.date,
            "source_type": audit["source_type"],
            "generated_at": audit["generated_at"],
            "consensus": audit["consensus"],
            "method": {
                "leave_one_book_out": True,
                "self_excluded_from_own_consensus": True,
                "consensus_statistic": "median of per-book Shin no-vig P(over), self-excluded",
                "reference_books": config["reference_books"],
                "atomic_key": ["event_id", "player_name", "market_key", "line"],
                "alternate_markets_segregated": True,
            },
        }, indent=2, default=str))
    (out_dir / "QUOTE_LATENCY_AUDIT.json").write_text(
        json.dumps({
            "schema_version": audit["schema_version"],
            "game_date": args.date,
            "source_type": audit["source_type"],
            "generated_at": audit["generated_at"],
            "latency": audit["latency"],
            "price_survival": audit["price_survival"],
        }, indent=2, default=str))
    (out_dir / "API_CREDIT_USAGE.json").write_text(
        json.dumps(credit_usage, indent=2, default=str))

    print(json.dumps({
        "mode": mode,
        "games_discovered": games_discovered,
        "n_books_observed": len(books_observed),
        "markets_observed": markets_observed,
        "atomic_pairs_created": int(atomic_pairs),
        "n_board_rows": int(len(board)),
        "n_diagnostic_edges": audit["summary"]["n_diagnostic_edges"],
        "n_qualifying": audit["summary"]["n_qualifying"],
        "rejections": audit["rejections"],
        "credit_consumed": credit_consumed,
        "edge_board_sample_rows": n_csv,
        "artifacts_dir": str(out_dir),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
