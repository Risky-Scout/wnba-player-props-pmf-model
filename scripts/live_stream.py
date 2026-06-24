#!/usr/bin/env python3
"""Live In-Play streaming engine for WNBA player props.

Enhancement 6: Live Bayesian Engine orchestration script.

Architecture
------------
1. Load today's pre-game PMFs (from daily pipeline artifacts)
2. Discover active WNBA games from BDL API
3. Initialize GammaPoissonLiveEngine with pre-game projections as priors
4. Poll BDL play-by-play API via list_endpoint('plays', ...)
5. Process each PBP event via PBPParser → update Bayesian posteriors
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

from wnba_props_model.live.pbp_parser import PBPParser  # noqa: PLC0415
from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine  # noqa: PLC0415
from wnba_props_model.live.live_edge import LiveEdgeCalculator  # noqa: PLC0415

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
    # Try deliveries/today/ first (new canonical output path)
    for candidate in [
        ARTIFACTS_DIR.parent / "deliveries" / "today" / "full_pmfs_wide.parquet",
        ARTIFACTS_DIR / "pmfs_long.parquet",
        ARTIFACTS_DIR / "pmfs_long.csv",
    ]:
        if candidate.exists():
            logger.info("Loading pre-game PMFs from %s", candidate)
            df = pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
            if game_date and "game_date" in df.columns:
                df = df[df["game_date"] == game_date]
            return df
    logger.error("No pre-game PMF file found in %s", ARTIFACTS_DIR)
    return pd.DataFrame()


def load_active_games(bdl_client=None) -> list[dict]:
    """Load today's active WNBA games from BDL API or cached schedule."""
    if bdl_client is not None:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # BDL fix: use list_endpoint('games', ...) NOT get_games()
            games = bdl_client.list_endpoint("games", {"dates": [today]})
            live_statuses = {"in_progress", "live", "halftime", "end_of_period", "scheduled", "pregame"}
            return [g for g in games if str(g.get("status", "")).lower() in live_statuses]
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


def fetch_pbp_plays(game_id: int, bdl_client=None) -> list[dict]:
    """Fetch ALL play-by-play events for a game from BDL.

    Returns raw BDL play dicts; deduplication is handled by PBPParser.
    BDL fix: use list_endpoint('plays', {'game_id': game_id}) NOT get_play_by_play().
    """
    if bdl_client is None:
        return []
    try:
        return bdl_client.list_endpoint("plays", {"game_id": game_id})
    except Exception as exc:
        logger.warning("PBP fetch failed for game %s: %s", game_id, exc)
        return []


def load_live_market_lines(game_id: int, bdl_client=None) -> pd.DataFrame:
    """Load current market lines as a DataFrame with columns
    [player_id, prop_type, line_value, over_odds, under_odds]."""
    if bdl_client is not None:
        try:
            from wnba_props_model.data.bdl_client import BDLClient  # noqa: PLC0415
            if hasattr(bdl_client, "list_player_props_for_game"):
                raw = bdl_client.list_player_props_for_game(game_id=game_id)
                if raw:
                    return pd.DataFrame(raw)
        except Exception as exc:
            logger.debug("Props fetch failed: %s", exc)

    # Fallback to pre-game artifact
    for candidate in [
        ARTIFACTS_DIR / "market_comparison.parquet",
        ARTIFACTS_DIR / "market_comparison.csv",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate) if candidate.suffix == ".parquet" else pd.read_csv(candidate)
            if "game_id" in df.columns:
                df = df[df["game_id"] == game_id]
            return df
    return pd.DataFrame()


def _build_projections_dict(pmfs_df: pd.DataFrame) -> dict[int, dict]:
    """Convert PMF DataFrame (wide OR long format) to {player_id: {stat: {mean, line}, projected_minutes}}.

    Long format: player_id | stat | mean | (minutes_mean)  — canonical output from predict_today.py
    Wide format: player_id | pts_mean | reb_mean | ...  — legacy fallback
    """
    projections: dict[int, dict] = {}

    if "stat" in pmfs_df.columns and "mean" in pmfs_df.columns:
        # Long format (canonical from predict_today.py / write_delivery)
        min_col = next((c for c in ["minutes_mean", "projected_minutes"] if c in pmfs_df.columns), None)
        for _, row in pmfs_df.iterrows():
            pid = int(row.get("player_id", 0))
            if pid == 0:
                continue
            stat = str(row["stat"])
            mean_val = float(row["mean"]) if pd.notna(row.get("mean")) else 0.0
            projections.setdefault(pid, {})[stat] = {"mean": mean_val, "line": mean_val}
            if "projected_minutes" not in projections[pid] and min_col:
                mv = row.get(min_col)
                projections[pid]["projected_minutes"] = float(mv) if pd.notna(mv) else 28.0
        for pid in projections:
            projections[pid].setdefault("projected_minutes", 28.0)
    else:
        # Wide format fallback
        stat_cols = [c for c in pmfs_df.columns if c.startswith("mean_")]
        for _, row in pmfs_df.iterrows():
            pid = int(row.get("player_id", 0))
            if pid == 0:
                continue
            player_proj: dict = {}
            for col in stat_cols:
                stat = col.replace("mean_", "")
                mean_val = float(row[col]) if pd.notna(row[col]) else 0.0
                line_col = f"line_{stat}"
                line_val = float(row[line_col]) if line_col in row.index and pd.notna(row[line_col]) else mean_val
                player_proj[stat] = {"mean": mean_val, "line": line_val}
            for min_col in ("projected_minutes", "mean_min", "minutes"):
                if min_col in row.index and pd.notna(row[min_col]):
                    player_proj["projected_minutes"] = float(row[min_col])
                    break
            player_proj.setdefault("projected_minutes", 28.0)
            if player_proj:
                projections[pid] = player_proj
    return projections


# ---------------------------------------------------------------------------
# Core streaming loop
# ---------------------------------------------------------------------------

def _build_roster_lookup_from_bdl(game_id: int, bdl_client) -> dict[str, dict]:
    """Build name→{player_id, team_id, team_side} lookup from BDL players endpoint."""
    if bdl_client is None:
        return {}
    try:
        # Get game details to find home/away team IDs
        games = bdl_client.list_endpoint("games", params={"ids": [game_id]})
        if not games:
            return {}
        game = games[0]
        home_tid = (game.get("home_team") or {}).get("id")
        away_tid = (game.get("visitor_team") or {}).get("id")
        team_ids = [t for t in [home_tid, away_tid] if t]
        if not team_ids:
            return {}
        players = bdl_client.list_endpoint("players", params={"team_ids": team_ids, "per_page": 100})
        lookup: dict[str, dict] = {}
        for p in players:
            fn, ln = p.get("first_name", ""), p.get("last_name", "")
            if not fn or not ln:
                continue
            # BDL PBP uses abbreviated format "F. Lastname" — store both forms
            abbrev = f"{fn[0]}. {ln}"
            full = f"{fn} {ln}"
            team_id = (p.get("team") or {}).get("id")
            side = "home" if team_id == home_tid else "away"
            info = {"player_id": int(p["id"]), "team_id": team_id, "team_side": side}
            lookup[abbrev] = info
            lookup[full] = info
        return lookup
    except Exception as exc:
        logger.warning("Roster lookup failed for game %s: %s", game_id, exc)
        return {}


def stream_game(
    game_id: int,
    pre_game_projections: dict[int, dict],
    live_props_df: pd.DataFrame,
    interval: int = 30,
    dry_run: bool = False,
    bdl_client=None,
    min_edge: float = 0.04,
) -> None:
    """Main streaming loop for a single game using new live modules."""
    engine = GammaPoissonLiveEngine()
    parser = PBPParser()
    edge_calc = LiveEdgeCalculator(min_edge=min_edge)

    # Build roster lookup from BDL so PBP name→player_id resolution works
    roster_lookup = _build_roster_lookup_from_bdl(game_id, bdl_client)
    logger.info("Roster lookup for game %s: %d names loaded", game_id, len(roster_lookup))

    seen_play_ids: set[int] = set()
    output_path = LIVE_DIR / f"live_{game_id}.json"

    logger.info("LiveStream started for game %s with %d player projections",
                game_id, len(pre_game_projections))

    while True:
        # Fetch all plays and process only new ones
        all_plays = fetch_pbp_plays(game_id, bdl_client)
        new_plays = [p for p in all_plays
                     if (p.get("order") or p.get("id") or 0) not in seen_play_ids]

        if new_plays:
            player_states, game_state = parser.process_plays(new_plays, roster_lookup)
            for p in new_plays:
                seen_play_ids.add(int(p.get("order") or p.get("id") or 0))

        # Bayesian update using pre-game projections + observed stats
        elapsed = parser.elapsed_minutes()
        results = engine.batch_compute(
            pre_game_projections,
            parser.player_states,
            elapsed_minutes=elapsed,
        )

        # Compute edges against market lines
        live_edges = edge_calc.compute_live_edges(results, live_props_df) if not live_props_df.empty else []
        bettable = [e for e in live_edges if e["bettable"]]
        gs = parser.game_state

        output = {
            "game_id": game_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_minutes": round(elapsed, 2),
            "period": gs.get("period", 0),
            "home_score": gs.get("home_score", 0),
            "away_score": gs.get("away_score", 0),
            "n_players_tracked": len(results),
            "n_props_with_edge": len(live_edges),
            "n_bettable_edges": len(bettable),
            "bettable_edges": bettable[:20],
            "n_pbp_plays_processed": len(seen_play_ids),
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            "Game %s | Period %d | %.1f min elapsed | %d players | %d bettable edges",
            game_id, gs.get("period", 0), elapsed,
            len(results), len(bettable),
        )

        if dry_run:
            logger.info("Dry-run mode — printing snapshot and exiting")
            print(json.dumps(output, indent=2))
            return

        # Check if game is over (40+ min elapsed + no new plays last poll)
        if elapsed >= 40.0 and not new_plays:
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
    parser.add_argument("--min-edge", type=float, default=0.04, help="Min edge pp to flag bettable")
    args = parser.parse_args()

    game_date = args.game_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load pre-game data
    pmfs_df = load_pregame_pmfs(game_date)
    if pmfs_df.empty:
        logger.error("Cannot start live engine: no pre-game PMFs available. "
                     "Run run_daily_pipeline.py first.")
        sys.exit(1)

    pre_game_projections = _build_projections_dict(pmfs_df)
    logger.info("Loaded pre-game projections for %d players", len(pre_game_projections))

    # Init BDL client
    bdl_client = None
    try:
        from wnba_props_model.data.bdl_client import BDLClient  # noqa: PLC0415
        bdl_client = BDLClient()
    except Exception:
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
        live_props_df = load_live_market_lines(gid, bdl_client)
        logger.info("Streaming game %s with %d market props", gid, len(live_props_df))
        stream_game(
            game_id=gid,
            pre_game_projections=pre_game_projections,
            live_props_df=live_props_df,
            interval=args.interval,
            dry_run=args.dry_run,
            bdl_client=bdl_client,
            min_edge=args.min_edge,
        )


if __name__ == "__main__":
    main()
