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
    n_events = 0
    for gd in eligible_dates:
        ev_key = f"events_{gd}"
        ev_payload = _cache_get(cache, ev_key)
        if ev_payload is None:
            try:
                ev_payload = client.list_historical_events(f"{gd}T12:00:00Z")
                _cache_put(cache, ev_key, ev_payload)
            except OddsAPIError as exc:
                typer.echo(f"[P1][WARN] events fetch failed {gd}: {exc}", err=True)
                continue
        events = (ev_payload or {}).get("data", []) or []
        day_events = pd.DataFrame(games_df[games_df.get("game_date") == gd]) if "game_date" in games_df.columns else games_df
        for ev in events:
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
                                "identity_method": method, "source": "odds_api_v4_historical"})
                    all_quotes.append(row)
        typer.echo(f"[P1] {gd}: events={len(events)} quotes_so_far={len(all_quotes)} "
                   f"remaining={client.quota_remaining}")

    quotes = pd.DataFrame(all_quotes)
    if quotes.empty:
        summary = {**pf, "unique_events": n_events, "quotes": 0,
                   "note": "no quotes matched — provider coverage or identity gap"}
        (out / "p1_coverage_summary.json").write_text(json.dumps(summary, indent=2))
        (out / "p1_quotes.parquet").write_text("") if False else pd.DataFrame().to_parquet(out / "p1_quotes.parquet")
        typer.echo("[P1] No quotes collected; wrote empty outputs + summary.")
        return

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

    audit = pd.DataFrame(unmatched_events)
    audit.to_parquet(out / "p1_unmatched_audit.parquet", index=False) if not audit.empty else \
        pd.DataFrame(columns=["game_date", "odds_event_id", "home_team", "away_team", "reason"]).to_parquet(out / "p1_unmatched_audit.parquet", index=False)

    matched_events = quotes["game_id"].nunique()
    summary = {
        **pf,
        "eligible_dates_run": len(eligible_dates),
        "unique_events_seen": n_events,
        "unmatched_events": len(unmatched_events),
        "event_match_rate": round(1 - len(unmatched_events) / max(n_events, 1), 4),
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
