#!/usr/bin/env python3
"""Bulk historical play-by-play ingestion for WNBA (owner directive step A).

Enumerates completed *real* regular-season games for the requested season(s) and pulls their
full play-by-play from BDL ``/wnba/v1/plays`` (cursor-paginated), normalizes the plays, and writes
a snapshot parquet partitioned by ``game_date`` plus the raw JSON payload.

Design guarantees:
  * Idempotent / resumable: a game is skipped when a non-empty normalized parquet already exists
    (unless ``--force``). Progress is safe to interrupt and re-run.
  * Rate-limit safe: the shared :class:`BDLClient` already retries 429s with backoff; a small inter
    game sleep is added.
  * All-Star / exhibition games are skipped. A "real" game is one whose home and visitor team
    abbreviations are BOTH in the canonical WNBA team set (from the teams table / games table).

Layout::

  data/snapshots/pbp/game_date=YYYY-MM-DD/plays_<game_id>.parquet   # normalized
  data/raw/bdl/plays/raw/plays_<game_id>.json                       # raw payload
  data/snapshots/pbp/_INGEST_MANIFEST.json                          # run manifest

Usage::

  python3 scripts/pull_pbp_history.py --seasons 2026
  python3 scripts/pull_pbp_history.py --seasons 2026 --games data/processed/wnba_games.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.data.bdl_client import BDLAPIError, BDLClient
from wnba_props_model.data.normalize import normalize_plays

# Games known to be exhibitions / non-regulation that must never enter modeling.
# game_id 24955 is the 2026 All-Star game (Team Spoon vs Team Coop) — see owner directive.
KNOWN_EXHIBITION_GAME_IDS = {24955}
EXHIBITION_TEAM_TOKENS = ("team ", "all-star", "all star", "rising", "usa ", "world")


def _canonical_team_abbrs(games: pd.DataFrame | None) -> set[str]:
    """Canonical WNBA team abbreviations, best-effort from the teams table then games table."""
    abbrs: set[str] = set()
    teams_path = Path("data/processed/wnba_teams.parquet")
    if teams_path.exists():
        t = pd.read_parquet(teams_path)
        col = next((c for c in ("team_abbreviation", "abbreviation") if c in t.columns), None)
        if col:
            abbrs |= {str(x).upper() for x in t[col].dropna().unique()}
    if games is not None:
        for c in ("home_team_abbreviation", "visitor_team_abbreviation"):
            if c in games.columns:
                abbrs |= {str(x).upper() for x in games[c].dropna().unique()}
    return {a for a in abbrs if a and a != "NAN"}


def _is_real_game(row: dict, canon_abbrs: set[str]) -> bool:
    gid = int(row.get("game_id"))
    if gid in KNOWN_EXHIBITION_GAME_IDS:
        return False
    home = str(row.get("home_team_abbreviation") or "").strip()
    away = str(row.get("visitor_team_abbreviation") or "").strip()
    home_full = str(row.get("home_team_name") or "").lower()
    away_full = str(row.get("visitor_team_name") or "").lower()
    for tok in EXHIBITION_TEAM_TOKENS:
        if tok in home_full or tok in away_full:
            return False
    if canon_abbrs:
        return home.upper() in canon_abbrs and away.upper() in canon_abbrs
    # No canonical set available: fall back to requiring short real abbreviations.
    return bool(home) and bool(away) and len(home) <= 4 and len(away) <= 4


def _completed_games(client: BDLClient, seasons: list[int],
                     games_path: str | None) -> pd.DataFrame:
    """Return completed real games: [game_id, game_date, home/visitor abbr]. Prefers local games table."""
    if games_path and Path(games_path).exists():
        g = pd.read_parquet(games_path)
        g = g[g["season"].isin(seasons)].copy()
        status_col = "status_normalized" if "status_normalized" in g.columns else "status"
        # completed games: BDL status "post" (or normalized "final"/"completed").
        done = g[status_col].astype(str).str.lower().isin({"post", "final", "completed", "closed"})
        played = g["is_played_game"] if "is_played_game" in g.columns else True
        g = g[done & played].copy()
        g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return g[["game_id", "game_date", "home_team_abbreviation",
                  "visitor_team_abbreviation"]].dropna(subset=["game_id"]).reset_index(drop=True)

    # Live enumeration via the API.
    rows: list[dict] = []
    for season in seasons:
        for gm in client.iter_endpoint("games", {"seasons": [season]}):
            if str(gm.get("status")).lower() != "post":
                continue
            rows.append({
                "game_id": gm.get("id"),
                "game_date": str(gm.get("date"))[:10],
                "home_team_abbreviation": (gm.get("home_team") or {}).get("abbreviation"),
                "visitor_team_abbreviation": (gm.get("visitor_team") or {}).get("abbreviation"),
                "home_team_name": (gm.get("home_team") or {}).get("full_name"),
                "visitor_team_name": (gm.get("visitor_team") or {}).get("full_name"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2026])
    ap.add_argument("--games", default="data/processed/wnba_games.parquet",
                    help="local games table (preferred); falls back to API enumeration if absent")
    ap.add_argument("--out-dir", default="data/snapshots/pbp")
    ap.add_argument("--raw-dir", default="data/raw/bdl/plays/raw")
    ap.add_argument("--sleep", type=float, default=0.15, help="inter-game courtesy sleep seconds")
    ap.add_argument("--limit", type=int, default=0, help="cap games (0 = all); for smoke tests")
    ap.add_argument("--force", action="store_true", help="re-pull even if snapshot exists")
    args = ap.parse_args()

    client = BDLClient()
    out_root = Path(args.out_dir)
    raw_root = Path(args.raw_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(args.games) if Path(args.games).exists() else None
    canon = _canonical_team_abbrs(games)
    completed = _completed_games(client, args.seasons, args.games)
    completed = completed[[_is_real_game(r, canon) for r in completed.to_dict("records")]]
    completed = completed.sort_values("game_date").reset_index(drop=True)
    if args.limit:
        completed = completed.head(args.limit)

    n_games = 0
    n_plays = 0
    n_skipped_existing = 0
    n_empty = 0
    errors: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for rec in completed.to_dict("records"):
        gid = int(rec["game_id"])
        gdate = rec["game_date"]
        part_dir = out_root / f"game_date={gdate}"
        part_dir.mkdir(parents=True, exist_ok=True)
        norm_path = part_dir / f"plays_{gid}.parquet"
        if norm_path.exists() and not args.force:
            try:
                existing = pd.read_parquet(norm_path)
                if len(existing) > 0:
                    n_skipped_existing += 1
                    n_plays += len(existing)
                    continue
            except Exception:
                pass  # corrupt/empty -> re-pull
        try:
            rows = client.list_endpoint("plays", {"game_id": gid})
        except BDLAPIError as exc:
            errors.append({"game_id": gid, "error": str(exc)[:300]})
            continue
        for r in rows:
            r.setdefault("game_id", gid)
        (raw_root / f"plays_{gid}.json").write_text(json.dumps(rows))
        df = normalize_plays(rows)
        if df.empty:
            n_empty += 1
            continue
        df["game_date"] = gdate
        df["source"] = "bdl"
        df["pull_timestamp_utc"] = ts
        df.to_parquet(norm_path, index=False)
        n_games += 1
        n_plays += len(df)
        if args.sleep:
            time.sleep(args.sleep)
        if (n_games % 25) == 0:
            print(f"[pbp] pulled {n_games} games, {n_plays} plays so far", flush=True)

    manifest = {
        "pull_timestamp_utc": ts,
        "seasons": args.seasons,
        "games_enumerated_completed_real": int(len(completed)),
        "games_pulled_this_run": n_games,
        "games_reused_existing": n_skipped_existing,
        "games_empty_pbp": n_empty,
        "total_games_with_pbp": n_games + n_skipped_existing,
        "total_plays": int(n_plays),
        "errors": errors,
        "known_exhibition_game_ids_skipped": sorted(KNOWN_EXHIBITION_GAME_IDS),
    }
    (out_root / "_INGEST_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "errors"}, indent=2))
    if errors:
        print(f"[pbp] {len(errors)} game(s) errored; see manifest")


if __name__ == "__main__":
    main()
