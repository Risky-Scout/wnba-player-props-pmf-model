#!/usr/bin/env python3
"""LOCAL-ONLY WNBA tracking + hustle puller (run on your home machine, not in CI).

stats.nba.com / stats.wnba.com block datacenter IPs but answer residential ones, so this
must run on your machine. It pulls, per WNBA game (2021-2026, regular season + playoffs):

  * player tracking (BoxScorePlayerTrackV3): touches, passes, secondary assists, speed,
    distance, rebound chances, contested/uncontested FG -> playmaking/usage signal.
  * hustle (HustleStatsBoxScore): deflections, contested 2s/3s, loose balls, charges,
    screen assists, box-outs -> defensive-intensity / rebounding signal.

Outputs two parquet files you upload back:
  wnba_tracking_2021_2026.parquet
  wnba_hustle_2021_2026.parquet

Setup:
  pip install nba_api pandas pyarrow
  python pull_wnba_tracking_local.py

It is resumable: re-running skips games already saved. If it errors on a game it logs and
continues. Expect it to take a while (be patient; it self-throttles to avoid rate limits).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

WNBA_LEAGUE_ID = "10"
SEASONS = ["2021", "2022", "2023", "2024", "2025", "2026"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SLEEP = 0.8            # courtesy delay between calls
MAX_RETRIES = 3
TIMEOUT = 60
OUT_TRACK = "wnba_tracking_2021_2026.parquet"
OUT_HUSTLE = "wnba_hustle_2021_2026.parquet"
CKPT_EVERY = 25


def _game_ids() -> list[str]:
    from nba_api.stats.endpoints import leaguegamelog
    ids: set[str] = set()
    for season in SEASONS:
        for st in SEASON_TYPES:
            for attempt in range(MAX_RETRIES):
                try:
                    df = leaguegamelog.LeagueGameLog(
                        league_id=WNBA_LEAGUE_ID, season=season,
                        season_type_all_star=st, timeout=TIMEOUT).get_data_frames()[0]
                    ids.update(df["GAME_ID"].astype(str).unique().tolist())
                    print(f"  [{season} {st}] {df['GAME_ID'].nunique()} games")
                    time.sleep(SLEEP)
                    break
                except Exception as exc:
                    print(f"  [WARN] {season} {st} attempt {attempt + 1}: {exc}")
                    time.sleep(SLEEP * (attempt + 2))
    return sorted(ids)


def _person_frame(frames):
    """Pick the player-level frame (has a person/player id column and the most columns)."""
    cand = [f for f in frames if any(c in f.columns for c in ("personId", "PLAYER_ID"))]
    return max(cand, key=lambda f: f.shape[1]) if cand else None


def _pull_one(game_id: str):
    from nba_api.stats.endpoints import boxscoreplayertrackv3, hustlestatsboxscore
    track = hustle = None
    for attempt in range(MAX_RETRIES):
        try:
            t = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id, timeout=TIMEOUT)
            track = _person_frame(t.get_data_frames())
            if track is not None:
                track = track.copy(); track["GAME_ID"] = game_id
            break
        except Exception as exc:
            print(f"    [track WARN] {game_id} try {attempt + 1}: {exc}")
            time.sleep(SLEEP * (attempt + 2))
    time.sleep(SLEEP)
    for attempt in range(MAX_RETRIES):
        try:
            h = hustlestatsboxscore.HustleStatsBoxScore(game_id=game_id, timeout=TIMEOUT)
            # player-level hustle frame has deflections + a person/player id column
            frames = [f for f in h.get_data_frames()
                      if any("DEFLECT" in str(c).upper() for c in f.columns)
                      and any(c in f.columns for c in ("PLAYER_ID", "personId"))]
            hustle = max(frames, key=lambda f: f.shape[1]) if frames else None
            if hustle is not None:
                hustle = hustle.copy(); hustle["GAME_ID"] = game_id
            break
        except Exception as exc:
            print(f"    [hustle WARN] {game_id} try {attempt + 1}: {exc}")
            time.sleep(SLEEP * (attempt + 2))
    return track, hustle


def _done_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        return set(pd.read_parquet(p, columns=["GAME_ID"])["GAME_ID"].astype(str).unique())
    except Exception:
        return set()


def main() -> None:
    print("[1/2] Collecting WNBA game IDs 2021-2026 ...")
    game_ids = _game_ids()
    print(f"  total unique games: {len(game_ids)}")

    done = _done_ids(OUT_TRACK) & _done_ids(OUT_HUSTLE)
    todo = [g for g in game_ids if g not in done]
    print(f"[2/2] Pulling {len(todo)} games ({len(done)} already done) ...")

    track_rows: list[pd.DataFrame] = []
    hustle_rows: list[pd.DataFrame] = []

    def _dedup_cols(df):
        pid = "PLAYER_ID" if "PLAYER_ID" in df.columns else ("personId" if "personId" in df.columns else None)
        return [c for c in ("GAME_ID", pid) if c]

    def _flush():
        if track_rows:
            new = pd.concat(track_rows, ignore_index=True)
            if Path(OUT_TRACK).exists():
                new = pd.concat([pd.read_parquet(OUT_TRACK), new], ignore_index=True)
            new.drop_duplicates(subset=_dedup_cols(new)).to_parquet(OUT_TRACK, index=False)
            track_rows.clear()
        if hustle_rows:
            new = pd.concat(hustle_rows, ignore_index=True)
            if Path(OUT_HUSTLE).exists():
                new = pd.concat([pd.read_parquet(OUT_HUSTLE), new], ignore_index=True)
            new.drop_duplicates(subset=_dedup_cols(new)).to_parquet(OUT_HUSTLE, index=False)
            hustle_rows.clear()

    for i, gid in enumerate(todo, 1):
        track, hustle = _pull_one(gid)
        if track is not None:
            track_rows.append(track)
        if hustle is not None:
            hustle_rows.append(hustle)
        time.sleep(SLEEP)
        if i % CKPT_EVERY == 0:
            _flush()
            print(f"  ... {i}/{len(todo)} games (checkpoint saved)")
    _flush()

    for label, path in [("tracking", OUT_TRACK), ("hustle", OUT_HUSTLE)]:
        if Path(path).exists():
            df = pd.read_parquet(path)
            print(f"[done] {label}: {len(df)} player-game rows across "
                  f"{df['GAME_ID'].nunique()} games -> {path}")
            print(f"       columns: {list(df.columns)}")
        else:
            print(f"[done] {label}: NO DATA written to {path} (all pulls failed?)")
    print("\nUpload both parquet files back to the chat.")


if __name__ == "__main__":
    main()
