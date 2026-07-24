"""W0.7 - collect atomic (same-book, point-in-time) quotes into the append-only store.

Pulls decision-time (tip - lead) and closing (tip - 5m) per-book player-prop quotes from
The Odds API historical endpoints and APPENDS them to data/atomic_quotes/atomic_quotes.parquet.
Books are never averaged. Settlement/outcome is joined from canonical player-game stats when
available; unresolved games are recorded with exact_quote_status=BLOCKED_EXACT_QUOTES.

Run this EARLY and often so no future proof observation is lost. It is additive/idempotent
(dedup by quote_id), so re-running never corrupts previously captured evidence.

Usage:
    python3 scripts/collect_atomic_quotes.py --game-dates 2026-07-20 2026-07-21 \\
        --store data/atomic_quotes/atomic_quotes.parquet
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.atomic_quotes import (  # noqa: E402
    ATOMIC_QUOTE_COLUMNS,
    BLOCKED_EXACT_QUOTES,
    EXACT,
    append_atomic_quotes,
    atomic_quote_id,
)
from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError  # noqa: E402
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402

app = typer.Typer(add_completion=False)
DECISION_LEAD_HOURS = 12


def _snapshots(commence: str, gd: str) -> dict[str, str]:
    try:
        tip = datetime.fromisoformat(commence.replace("Z", "+00:00"))
    except Exception:
        tip = datetime.fromisoformat(f"{gd}T23:00:00+00:00")
    return {
        "decision": (tip - timedelta(hours=DECISION_LEAD_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "closing": (tip - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, tip.strftime("%Y-%m-%dT%H:%M:%SZ")


@app.command()
def collect(
    game_dates: list[str] = typer.Option(..., "--game-dates"),
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    roster: str = typer.Option("data/processed/wnba_player_game_stats.parquet", "--roster"),
    games: str = typer.Option("data/processed/wnba_games.parquet", "--games"),
) -> None:
    client = OddsAPIClient()
    games_df = pd.read_parquet(games) if Path(games).exists() else pd.DataFrame()
    if not games_df.empty and "game_date" in games_df.columns:
        games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.strftime("%Y-%m-%d")
    roster_df = (pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()
                 if Path(roster).exists() else pd.DataFrame())
    markets = list(hm.MARKET_TO_STAT.keys())
    now = datetime.now(timezone.utc).isoformat()

    rows: list[dict] = []
    for gd in game_dates:
        gd_next = (datetime.fromisoformat(f"{gd}T00:00:00+00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            ev = client.list_historical_events(f"{gd}T12:00:00Z",
                                                commence_time_from=f"{gd}T00:00:00Z",
                                                commence_time_to=f"{gd_next}T12:00:00Z")
        except OddsAPIError as exc:
            typer.echo(f"[atomic] events fetch failed {gd}: {exc}", err=True); continue
        day_games = games_df[games_df.get("game_date") == gd] if "game_date" in games_df.columns else games_df
        for e in (ev or {}).get("data", []) or []:
            eid = e.get("id", "")
            gid = hm.resolve_game_id(day_games, e.get("home_team", ""), e.get("away_team", ""), gd) \
                if not day_games.empty else None
            snaps, tip = _snapshots(e.get("commence_time", ""), gd)
            for label, snap in snaps.items():
                try:
                    odds = client.get_historical_event_odds(eid, snap, markets=markets)
                except OddsAPIError:
                    continue
                for book in (odds or {}).get("data", {}).get("bookmakers", []):
                    bkey = book.get("key", "")
                    for m in book.get("markets", []):
                        stat = hm.MARKET_TO_STAT.get(m.get("key", ""))
                        if not stat:
                            continue
                        for oc in m.get("outcomes", []):
                            name = oc.get("description", "")
                            pid, _ = (hm.resolve_player_id(name, gid, roster_df)
                                      if (gid is not None and not roster_df.empty) else (None, "unmatched"))
                            side = str(oc.get("name", "")).lower()
                            line = oc.get("point")
                            status = EXACT if (gid is not None and pid is not None) else BLOCKED_EXACT_QUOTES
                            rows.append({
                                "quote_id": atomic_quote_id(bkey, eid, pid or name, stat, line, side, snap),
                                "sportsbook": bkey, "event_id": eid, "game_id": gid, "player_id": pid,
                                "player_name": name, "prop": stat, "line": line, "side": side,
                                "american_odds": oc.get("price"), "snapshot_label": label,
                                "snapshot_time": snap, "decision_timestamp": snaps["decision"],
                                "scheduled_tip_utc": tip, "prediction_timestamp": None,
                                "model_prob_over_final": None, "probability_lineage_version": None,
                                "model_hash": None, "calibrator_hash": None,
                                "feature_schema_hash": None, "quote_policy_hash": None,
                                "settlement_status": "pending", "actual_outcome": None,
                                "exact_quote_status": status, "source": "odds_api_v4_historical",
                            })
        typer.echo(f"[atomic] {gd}: rows_so_far={len(rows)} quota_remaining={client.quota_remaining}")

    if not rows:
        typer.echo("[atomic] no quotes collected.", err=True); raise typer.Exit(1)
    new_df = pd.DataFrame(rows)[ATOMIC_QUOTE_COLUMNS]
    summary = append_atomic_quotes(store, new_df)
    n_blocked = int((new_df["exact_quote_status"] == BLOCKED_EXACT_QUOTES).sum())
    typer.echo(f"[atomic] appended={summary['added']} (of {len(new_df)}) total={summary['total']} "
               f"blocked_exact_quotes={n_blocked} -> {store}")


if __name__ == "__main__":
    app()
