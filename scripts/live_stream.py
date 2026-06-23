#!/usr/bin/env python3
"""Live In-Play streaming engine for WNBA player props.

Enhancement 6: Live Bayesian Engine orchestration script.

Architecture
------------
1. Load today's pre-game PMFs (from daily pipeline artifacts)
2. Load any active game's current market lines from BDL
3. Initialize LiveEngine with pre-game projections as Gamma-Poisson priors
4. Poll BDL play-by-play API in a loop
5. Process each PBP event → update Bayesian posteriors
6. Output live P(over) per prop to artifacts/live_predictions_<game_id>.json

Usage
-----
    python scripts/live_stream.py [--game-id GAME_ID] [--interval 30] [--dry-run]

    --game-id  : Specific game ID to track. If omitted, auto-detects today's live games.
    --interval : Polling interval in seconds (default: 30).
    --dry-run  : Load engine, print one snapshot, then exit.

GitHub Actions: this script is called by .github/workflows/live_pipeline.yml
which fires on a schedule during WNBA game windows (typically 7pm–11pm ET).

All heavy computation (model training, OOF) runs in pre-game pipeline.
This script is lightweight O(K) per event.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.models.live_engine import LiveEngine, build_pregame_ratings_from_pmfs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("live_stream")


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
LIVE_DIR = ARTIFACTS_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_pregame_pmfs(game_date: str | None = None) -> pd.DataFrame:
    """Load today's pre-game PMF predictions from artifacts."""
    pmf_path = ARTIFACTS_DIR / "pmfs_long.parquet"
    if not pmf_path.exists():
        # Try CSV fallback
        pmf_path = ARTIFACTS_DIR / "pmfs_long.csv"
    if not pmf_path.exists():
        logger.error("No pre-game PMF file found in %s", ARTIFACTS_DIR)
        return pd.DataFrame()
    logger.info("Loading pre-game PMFs from %s", pmf_path)
    df = pd.read_parquet(pmf_path) if pmf_path.suffix == ".parquet" else pd.read_csv(pmf_path)
    if game_date and "game_date" in df.columns:
        df = df[df["game_date"] == game_date]
    return df


def load_feature_wide() -> pd.DataFrame:
    """Load wide feature table for player metadata (position, minutes, team)."""
    wide_path = ARTIFACTS_DIR / "wide_features.parquet"
    if not wide_path.exists():
        wide_path = ARTIFACTS_DIR / "wide_features.csv"
    if not wide_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(wide_path) if wide_path.suffix == ".parquet" else pd.read_csv(wide_path)


def load_active_games(bdl_client=None) -> list[dict]:
    """Load today's active WNBA games from BDL API or cached schedule."""
    try:
        if bdl_client is not None:
            games = bdl_client.get_games(
                dates=[datetime.now(timezone.utc).strftime("%Y-%m-%d")]
            )
            return [g for g in games if g.get("status") == "in_progress"]
    except Exception as exc:
        logger.warning("Could not load live games from BDL: %s", exc)

    # Fallback to schedule artifact
    sched_path = ARTIFACTS_DIR / "schedule.json"
    if sched_path.exists():
        with open(sched_path) as f:
            games = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return [g for g in games if g.get("date", "") == today]
    return []


def fetch_pbp_events(game_id: int, since_event: int = 0, bdl_client=None) -> list[dict]:
    """Fetch new play-by-play events for a game since a given event index.

    Returns list of normalised PBP event dicts with keys:
    {event_type, period, clock, home_score, away_score,
     primary_player_id, secondary_player_id, stat_credits, game_id}
    """
    if bdl_client is None:
        return []

    try:
        raw_events = bdl_client.get_play_by_play(game_id=game_id, since=since_event)
    except Exception as exc:
        logger.warning("PBP fetch failed for game %s: %s", game_id, exc)
        return []

    normalised = []
    for ev in raw_events:
        normalised.append({
            "game_id": game_id,
            "event_type": _map_event_type(ev),
            "period": ev.get("period", 1),
            "clock": ev.get("clock") or ev.get("time") or "",
            "home_score": ev.get("home_score") or ev.get("home_team_score", 0),
            "away_score": ev.get("away_score") or ev.get("away_team_score", 0),
            "primary_player_id":   ev.get("player1_id") or ev.get("primary_player_id"),
            "secondary_player_id": ev.get("player2_id") or ev.get("secondary_player_id"),
            "stat_credits": _extract_stat_credits(ev),
        })
    return normalised


def _map_event_type(ev: dict) -> str:
    """Normalise BDL event type string."""
    raw = str(ev.get("event_type", "") or ev.get("type", "")).lower()
    if "sub" in raw:
        return "substitution"
    if "foul" in raw:
        return "foul"
    if "score" in raw or "2pt" in raw or "3pt" in raw or "pts" in raw:
        return "score"
    if "rebound" in raw or "reb" in raw:
        return "rebound"
    if "assist" in raw or "ast" in raw:
        return "assist"
    if "turnover" in raw or "tov" in raw:
        return "turnover"
    return raw or "unknown"


def _extract_stat_credits(ev: dict) -> dict[int, dict[str, int]]:
    """Extract {player_id: {stat: count}} from a PBP event."""
    credits: dict[int, dict[str, int]] = {}
    pid = ev.get("player1_id") or ev.get("primary_player_id")
    if not pid:
        return credits
    pid = int(pid)
    ev_type = _map_event_type(ev)

    # Points
    pts = ev.get("points") or ev.get("pts")
    if pts and int(pts) > 0:
        credits.setdefault(pid, {})["pts"] = int(pts)

    # Map event type to stat
    if ev_type == "rebound":
        credits.setdefault(pid, {})["reb"] = 1
    elif ev_type == "assist":
        pid2 = ev.get("player2_id") or ev.get("secondary_player_id")
        if pid2:
            credits.setdefault(int(pid2), {})["ast"] = 1
    elif ev_type == "turnover":
        credits.setdefault(pid, {})["turnover"] = 1
    elif ev_type == "steal":
        credits.setdefault(pid, {})["stl"] = 1
    elif ev_type == "block":
        pid2 = ev.get("player2_id") or ev.get("secondary_player_id")
        if pid2:
            credits.setdefault(int(pid2), {})["blk"] = 1

    return credits


def load_live_market_lines(game_id: int, bdl_client=None) -> list[tuple[int, str, float]]:
    """Load current market lines for a game as [(player_id, stat, line), ...]."""
    # Fall back to pre-game lines from artifacts
    lines_path = ARTIFACTS_DIR / "market_comparison.parquet"
    if not lines_path.exists():
        lines_path = ARTIFACTS_DIR / "market_comparison.csv"
    if not lines_path.exists():
        return []

    df = pd.read_parquet(lines_path) if lines_path.suffix == ".parquet" else pd.read_csv(lines_path)
    if "game_id" in df.columns:
        df = df[df["game_id"] == game_id]
    if df.empty or "line" not in df.columns:
        return []

    return [(int(r["player_id"]), str(r["stat"]), float(r["line"]))
            for _, r in df.iterrows() if pd.notna(r.get("player_id"))]


# ---------------------------------------------------------------------------
# Core streaming loop
# ---------------------------------------------------------------------------

def stream_game(
    game_id: int,
    pmfs_long: pd.DataFrame,
    feature_wide: pd.DataFrame,
    props: list[tuple[int, str, float]],
    interval: int = 30,
    dry_run: bool = False,
    bdl_client=None,
) -> None:
    """Main streaming loop for a single game."""
    game_info = {"game_id": game_id, "home_team_id": None, "away_team_id": None}

    # Build pre-game ratings dict for LiveEngine
    pregame_ratings = build_pregame_ratings_from_pmfs(pmfs_long, feature_wide)
    if not pregame_ratings:
        logger.error("No pre-game ratings found for game %s", game_id)
        return

    engine = LiveEngine(pregame_ratings)
    engine.initialize_game(game_info)

    logger.info("LiveEngine initialised with %d player priors for game %s",
                len(engine.player_states), game_id)

    seen_events = 0
    output_path = LIVE_DIR / f"live_{game_id}.json"

    while True:
        new_events = fetch_pbp_events(game_id, since_event=seen_events, bdl_client=bdl_client)

        for ev in new_events:
            engine.process_event(ev)
            seen_events += 1

        # Compute live probabilities for all props
        live_probs = engine.get_all_live_probabilities(props)

        # Game total
        game_total_proj, game_total_p_over = engine.compute_game_total_live(market_line=160.5)

        output = {
            "game_id": game_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_minutes": engine.game_state.elapsed_minutes,
            "period": engine.game_state.period,
            "home_score": engine.game_state.home_score,
            "away_score": engine.game_state.away_score,
            "game_total_projection": round(game_total_proj, 2),
            "game_total_p_over": round(game_total_p_over, 4),
            "props": live_probs,
            "n_pbp_events_processed": seen_events,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            "Game %s | Period %d | %s min elapsed | %d props updated | "
            "score %d-%d | game total proj %.1f",
            game_id, engine.game_state.period,
            round(engine.game_state.elapsed_minutes, 1),
            len(live_probs),
            engine.game_state.home_score, engine.game_state.away_score,
            game_total_proj,
        )

        if dry_run:
            logger.info("Dry-run mode — printing snapshot and exiting")
            print(json.dumps(output, indent=2))
            return

        # Check if game is over
        if engine.game_state.elapsed_minutes >= 40.0 and not new_events:
            logger.info("Game %s appears complete (40+ minutes, no new events)", game_id)
            break

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WNBA Live Bayesian Prop Engine")
    parser.add_argument("--game-id", type=int, default=None, help="BDL game ID to stream")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Single snapshot then exit")
    parser.add_argument("--game-date", default=None, help="YYYY-MM-DD date for PMF lookup")
    args = parser.parse_args()

    game_date = args.game_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load pre-game data
    pmfs_long     = load_pregame_pmfs(game_date)
    feature_wide  = load_feature_wide()

    if pmfs_long.empty:
        logger.error("Cannot start live engine: no pre-game PMFs available. "
                     "Run run_daily_pipeline.py first.")
        sys.exit(1)

    # Optionally init BDL client
    try:
        from wnba_props_model.data.bdl_client import BDLClient  # noqa: PLC0415
        bdl_client = BDLClient()
    except Exception:
        bdl_client = None
        logger.warning("BDL client unavailable — PBP events will be empty (simulation mode)")

    # Determine which game(s) to stream
    if args.game_id:
        game_ids = [args.game_id]
    else:
        active = load_active_games(bdl_client)
        game_ids = [int(g["id"]) for g in active if g.get("id")]
        if not game_ids:
            logger.info("No active games detected for %s — exiting", game_date)
            sys.exit(0)

    for gid in game_ids:
        props = load_live_market_lines(gid, bdl_client)
        if not props:
            # Generate synthetic props from PMF file for testing
            if not pmfs_long.empty:
                props = [
                    (int(r["player_id"]), str(r["stat"]),
                     float(r.get("pmf_mean", r.get("mean", 5.0))))
                    for _, r in pmfs_long.head(30).iterrows()
                    if pd.notna(r.get("player_id"))
                ]
        logger.info("Streaming game %s with %d props", gid, len(props))
        stream_game(
            game_id=gid,
            pmfs_long=pmfs_long,
            feature_wide=feature_wide,
            props=props,
            interval=args.interval,
            dry_run=args.dry_run,
            bdl_client=bdl_client,
        )


if __name__ == "__main__":
    main()
