"""P1 — resumable historical WNBA player-prop backfill (offline evaluation only).

Pulls historical opening + closing line snapshots for the games present in the
OOF data (on/after 2023-05-03), canonicalizes provider event/player IDs to the
model's canonical IDs (exact, roster-constrained — never fuzzy), pairs Over/Under
at the same book/line/snapshot for no-vig probabilities, and writes quote-level +
consensus tables plus an unmatched audit and an API coverage/credit summary.

NEVER prints, persists, or logs the API key. Raw responses are cached to a local
cache dir (workflow artifact / external cache — not committed to Git). Cached
events/odds are not re-fetched (idempotent, resumable by date + event + snapshot).

Usage:
    python scripts/p1_historical_backfill.py \
        --oof data/oof/oof_player_stat_pmfs.parquet \
        --games data/processed/wnba_games.parquet \
        --roster data/processed/wnba_player_game_stats.parquet \
        --out-dir artifacts/p1 --cache-dir artifacts/p1_cache \
        [--pilot-dates 6]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError
from wnba_props_model.models.market import shin_no_vig_two_way_with_z
from wnba_props_model.evaluation import historical_market as hm

app = typer.Typer(add_completion=False)

MIN_DATE = "2023-05-03"  # Odds API historical props availability floor
BACKFILL_MARKETS = list(hm.MARKET_TO_STAT.keys())
DECISION_LEAD_HOURS = 12  # decision snapshot = tip - 12h (production morning publish)


def classify_coverage(
    n_events: int,
    matched_events: int,
    quotes_empty: bool,
    event_match_rate: float,
    min_event_match_rate: float,
    allow_empty: bool,
) -> tuple[str, str | None]:
    """Pure decision for the fail-closed identity/coverage gate.

    Returns (severity, table_kind):
      - 'fatal_no_events'      : provider returned no events at all.
      - 'fatal_stale'          : events exist but ZERO usable quotes -> stale table.
                                 table_kind is 'games' (no game_id resolved) or
                                 'roster' (games matched but no player_id resolved).
      - 'warn_empty_allowed'   : zero quotes but --allow-empty set.
      - 'warn_low'             : non-empty but event_match_rate below threshold.
      - 'ok'                   : healthy.
    """
    if n_events == 0:
        return ("fatal_no_events", None)
    if quotes_empty:
        if allow_empty:
            return ("warn_empty_allowed", None)
        return ("fatal_stale", "games" if matched_events == 0 else "roster")
    if event_match_rate < min_event_match_rate:
        return ("warn_low", None)
    return ("ok", None)


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_get(cache: Path, key: str):
    f = cache / f"{key}.json"
    return _read_json(f) if f.exists() else None


def _cache_put(cache: Path, key: str, obj) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{key}.json").write_text(json.dumps(obj))


def _parse_event_odds(payload: dict, event: dict, snapshot_time: str) -> list[dict]:
    """Flatten a historical event-odds payload into quote rows."""
    rows: list[dict] = []
    data = (payload or {}).get("data", {}) or {}
    for book in data.get("bookmakers", []):
        bkey = book.get("key", "")
        blast = book.get("last_update", "")
        for market in book.get("markets", []):
            mkey = market.get("key", "")
            stat = hm.MARKET_TO_STAT.get(mkey)
            if not stat:
                continue
            for oc in market.get("outcomes", []):
                rows.append({
                    "odds_event_id": event.get("id", ""),
                    "commence_time": event.get("commence_time", ""),
                    "home_team": event.get("home_team", ""),
                    "away_team": event.get("away_team", ""),
                    "book": bkey, "book_last_update": blast,
                    "market_key": mkey, "stat": stat,
                    "player_name": oc.get("description", ""),
                    "side": str(oc.get("name", "")).lower(),
                    "line": oc.get("point"),
                    "american_odds": oc.get("price"),
                    "snapshot_time": snapshot_time,
                })
    return rows


@app.command()
def main(
    oof: str = typer.Option(...),
    games: str = typer.Option("data/processed/wnba_games.parquet"),
    roster: str = typer.Option("data/processed/wnba_player_game_stats.parquet"),
    out_dir: str = typer.Option("artifacts/p1"),
    cache_dir: str = typer.Option("artifacts/p1_cache"),
    pilot_dates: int = typer.Option(0, help="If >0, limit to the first N eligible dates (pilot)."),
    preflight_only: bool = typer.Option(False, help="Run only the sanitized API preflight."),
    min_event_match_rate: float = typer.Option(
        0.5, "--min-event-match-rate",
        help="Fail-closed guard: fraction of provider events that MUST resolve to a "
             "canonical game. Below this, the games/roster tables are treated as stale."),
    allow_empty: bool = typer.Option(
        False, "--allow-empty",
        help="Bypass the stale-table safety gate and permit empty/low-coverage output."),
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    client = OddsAPIClient()  # reads ODDS_API_KEY from env; raises if missing

    oof_df = pd.read_parquet(oof)
    oof_df["_gd"] = pd.to_datetime(oof_df["game_date"]).dt.strftime("%Y-%m-%d")
    eligible_dates = sorted(d for d in oof_df["_gd"].unique() if d >= MIN_DATE)
    games_df = pd.read_parquet(games)
    if "game_date" in games_df.columns:
        games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.strftime("%Y-%m-%d")
    roster_df = pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()

    # ── Step 1: sanitized preflight ────────────────────────────────────────────
    pf = {"min_date": MIN_DATE, "eligible_dates": len(eligible_dates)}
    try:
        d0 = eligible_dates[0]
        ev_payload = client.list_historical_events(f"{d0}T12:00:00Z")
        events0 = (ev_payload or {}).get("data", []) or []
        pf.update({"preflight_date": d0, "http": "200",
                   "event_count": len(events0),
                   "snapshot_timestamp": (ev_payload or {}).get("timestamp"),
                   "requests_used": client.quota_used,
                   "requests_remaining": client.quota_remaining})
        if events0:
            eid = events0[0]["id"]
            odds0 = client.get_historical_event_odds(
                eid, f"{d0}T23:00:00Z", markets=BACKFILL_MARKETS)
            books = (odds0 or {}).get("data", {}).get("bookmakers", [])
            pf["preflight_bookmaker_count"] = len(books)
            pf["preflight_market_count"] = sum(len(b.get("markets", [])) for b in books)
        pf["requests_remaining"] = client.quota_remaining
    except OddsAPIError as exc:
        pf.update({"http": "error", "error": str(exc)[:200]})
        (out / "p1_coverage_summary.json").write_text(json.dumps(pf, indent=2))
        typer.echo(f"[P1][FATAL] Preflight failed (auth/entitlement): {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"[P1] Preflight OK: {json.dumps(pf)}")
    if preflight_only:
        (out / "p1_coverage_summary.json").write_text(json.dumps(pf, indent=2))
        return

    if pilot_dates and pilot_dates > 0:
        eligible_dates = eligible_dates[:pilot_dates]

    # ── Step 2: resumable backfill ─────────────────────────────────────────────
    all_quotes: list[dict] = []
    unmatched_events: list[dict] = []
    unmatched_players: dict[str, int] = {}
    seen_event_ids: set[str] = set()  # global dedupe by provider event id
    n_events = 0
    for gd in eligible_dates:
        gd_next = (datetime.fromisoformat(f"{gd}T00:00:00+00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
        # Filter events to those COMMENCING in this canonical game's UTC window so
        # coverage is measured per eligible game, not from an unfiltered daily snapshot.
        ctf, ctt = f"{gd}T00:00:00Z", f"{gd_next}T12:00:00Z"
        ev_key = f"events_{gd}_{ctf}_{ctt}"
        ev_payload = _cache_get(cache, ev_key)
        if ev_payload is None:
            try:
                ev_payload = client.list_historical_events(
                    f"{gd}T12:00:00Z", commence_time_from=ctf, commence_time_to=ctt)
                _cache_put(cache, ev_key, ev_payload)
            except OddsAPIError as exc:
                typer.echo(f"[P1][WARN] events fetch failed {gd}: {exc}", err=True)
                continue
        events = (ev_payload or {}).get("data", []) or []
        day_events = pd.DataFrame(games_df[games_df.get("game_date") == gd]) if "game_date" in games_df.columns else games_df
        for ev in events:
            if ev.get("id") in seen_event_ids:
                continue  # deduplicate globally by provider event id
            seen_event_ids.add(ev.get("id"))
            n_events += 1
            gid = hm.resolve_game_id(day_events, ev.get("home_team", ""), ev.get("away_team", ""), gd)
            if gid is None:
                unmatched_events.append({"game_date": gd, "odds_event_id": ev.get("id"),
                                         "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                                         "reason": "no_canonical_game_match"})
                continue
            commence = ev.get("commence_time", f"{gd}T23:00:00Z")
            try:
                tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            except Exception:
                tip = datetime.fromisoformat(f"{gd}T23:00:00+00:00")
            snapshots = {
                "open": (tip - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "decision": (tip - timedelta(hours=DECISION_LEAD_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "close": (tip - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            for _label, snap in snapshots.items():
                ok = f"odds_{ev.get('id')}_{_label}"
                payload = _cache_get(cache, ok)
                if payload is None:
                    try:
                        payload = client.get_historical_event_odds(
                            ev["id"], snap, markets=BACKFILL_MARKETS)
                        _cache_put(cache, ok, payload)
                    except OddsAPIError as exc:
                        typer.echo(f"[P1][WARN] odds fetch failed {ev.get('id')}/{_label}: {exc}", err=True)
                        continue
                actual_snap = (payload or {}).get("timestamp") or snap
                for row in _parse_event_odds(payload, ev, actual_snap):
                    pid, method = hm.resolve_player_id(row["player_name"], gid, roster_df)
                    if pid is None:
                        unmatched_players[row["player_name"]] = unmatched_players.get(row["player_name"], 0) + 1
                        continue
                    row.update({"game_id": gid, "player_id": pid, "game_date": gd,
                                "identity_method": method, "source": "odds_api_v4_historical",
                                "odds_format": "american", "snapshot_label": _label})
                    all_quotes.append(row)
        typer.echo(f"[P1] {gd}: events={len(events)} quotes_so_far={len(all_quotes)} "
                   f"remaining={client.quota_remaining}")

    # ── Identity / coverage gate (FAIL-CLOSED) ────────────────────────────────
    # Events found but resolution collapsing is the classic symptom of STALE
    # games/roster tables (wrong team-abbrev convention or game_id vintage). That
    # used to write empty outputs and exit 0 silently — costing hours. Now it
    # fails loudly with the exact match rate and where to look.
    matched_events_count = n_events - len(unmatched_events)
    event_match_rate = round(matched_events_count / n_events, 4) if n_events else 0.0

    # Always persist the unmatched-event audit so any failure is inspectable.
    audit_cols = ["game_date", "odds_event_id", "home_team", "away_team", "reason"]
    (pd.DataFrame(unmatched_events) if unmatched_events
     else pd.DataFrame(columns=audit_cols)).to_parquet(out / "p1_unmatched_audit.parquet", index=False)

    quotes = pd.DataFrame(all_quotes)

    def _write_coverage_summary(extra: dict) -> None:
        summary = {**pf, "eligible_dates_run": len(eligible_dates),
                   "unique_events_seen": n_events, "matched_events": int(matched_events_count),
                   "unmatched_events": len(unmatched_events), "event_match_rate": event_match_rate,
                   "distinct_unmatched_players": len(unmatched_players),
                   "top_unmatched_players": dict(sorted(unmatched_players.items(), key=lambda kv: -kv[1])[:15]),
                   "requests_used": client.quota_used, "requests_remaining": client.quota_remaining,
                   **extra}
        (out / "p1_coverage_summary.json").write_text(json.dumps(summary, indent=2))

    severity, table_kind = classify_coverage(
        n_events, matched_events_count, quotes.empty,
        event_match_rate, min_event_match_rate, allow_empty)

    if severity == "fatal_no_events":
        _write_coverage_summary({"quotes": 0, "note": "provider returned no historical events"})
        typer.echo("[P1][FATAL] No historical events returned for any requested date. "
                   "Check the dates and Odds API historical entitlement.", err=True)
        raise typer.Exit(1)

    # The catastrophic, hours-costing symptom is ZERO usable quotes despite events
    # existing — always a STALE/mismatched games (no game_id) or roster (no player_id)
    # table. Fail loudly and say exactly where to look. (A wide commence window +
    # cross-date dedup makes a fractional event-match threshold noisy, so we gate on
    # the unambiguous total-collapse signal, not a fraction.)
    if severity == "fatal_stale":
        top = dict(sorted(unmatched_players.items(), key=lambda kv: -kv[1])[:10])
        detail = "0 events resolved to a canonical game_id" if table_kind == "games" \
            else f"events matched but 0 player_ids resolved; top unmatched: {top}"
        _write_coverage_summary({"quotes": 0, "note": f"zero usable quotes — stale {table_kind} table"})
        typer.echo(
            f"[P1][FATAL] Found {n_events} events but produced 0 usable quotes. Your {table_kind} "
            f"table ({detail}) is STALE or on a different convention than the current code "
            f"({matched_events_count}/{n_events} events = {event_match_rate:.1%} matched). Refusing to "
            f"write empty outputs. Inspect {out}/p1_unmatched_audit.parquet, then regenerate canonical "
            "tables (scripts/build_canonical_tables.py) or fetch current ones (scripts/fetch_data.py). "
            "Override with --allow-empty only if the gap is expected.", err=True)
        raise typer.Exit(1)

    if severity == "warn_empty_allowed":
        _write_coverage_summary({"quotes": 0, "note": "events matched but no player quotes resolved"})
        pd.DataFrame().to_parquet(out / "p1_quotes.parquet")
        typer.echo("[P1][WARN] No quotes collected; --allow-empty set, wrote empty outputs.", err=True)
        return

    # Non-empty but low coverage: loud WARN (not fatal — partial staleness or normal
    # cross-date window artifacts shouldn't block a run that produced real quotes).
    if severity == "warn_low":
        typer.echo(
            f"[P1][WARN] Only {event_match_rate:.1%} of events ({matched_events_count}/{n_events}) resolved "
            f"to a canonical game. If you expected full coverage your tables may be partially stale — "
            f"inspect {out}/p1_unmatched_audit.parquet.", err=True)

    quotes = quotes[quotes["american_odds"].notna() & quotes["line"].notna()].copy()
    quotes["decimal_odds"] = quotes["american_odds"].map(hm.american_to_decimal)
    quotes["implied_probability"] = quotes["american_odds"].map(hm.american_to_implied)
    quotes.to_parquet(out / "p1_quotes.parquet", index=False)

    # Pair Over/Under (same book/line/snapshot) for no-vig, then re-attach the
    # canonical game_id/player_id (pairing keys on provider fields only).
    quotes_r = quotes.rename(columns={"odds_event_id": "event_id"})
    keycols = ["event_id", "book", "stat", "player_name", "line", "snapshot_time"]
    idmap = quotes_r[keycols + ["game_id", "player_id"]].drop_duplicates(subset=keycols)
    paired = hm.pair_over_under(quotes_r, shin_fn=shin_no_vig_two_way_with_z)
    paired = paired.merge(idmap, on=keycols, how="left")
    tagged = hm.select_open_close(paired)
    opening = tagged[tagged["is_opening"] == True].copy()  # noqa: E712
    closing = tagged[tagged["is_closing"] == True].copy()  # noqa: E712
    opening_consensus = hm.build_consensus(opening.assign(is_closing=True))
    closing_consensus = hm.build_consensus(closing)
    opening_consensus.to_parquet(out / "p1_opening_consensus.parquet", index=False)
    closing_consensus.to_parquet(out / "p1_closing_consensus.parquet", index=False)

    matched_events = quotes["game_id"].nunique()
    summary = {
        **pf,
        "eligible_dates_run": len(eligible_dates),
        "unique_events_seen": n_events,
        "matched_events": int(matched_events_count),
        "unmatched_events": len(unmatched_events),
        "event_match_rate": event_match_rate,
        "quotes": int(len(quotes)),
        "paired_rows": int(len(paired)),
        "closing_consensus_rows": int(len(closing_consensus)),
        "matched_games": int(matched_events),
        "distinct_unmatched_players": len(unmatched_players),
        "top_unmatched_players": dict(sorted(unmatched_players.items(), key=lambda kv: -kv[1])[:15]),
        "requests_used": client.quota_used,
        "requests_remaining": client.quota_remaining,
    }
    (out / "p1_coverage_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(f"[P1] Backfill complete: {json.dumps({k: summary[k] for k in ['quotes','closing_consensus_rows','matched_games','requests_remaining']})}")


if __name__ == "__main__":
    app()
