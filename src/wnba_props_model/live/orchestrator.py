"""Live game orchestrator for WNBA player props.

Orchestrates live game tracking: polls PBP, updates posteriors, computes edges.

Polling frequency:
  - Pre-game: every 60 seconds (check for game start)
  - Q1-Q3: every 15 seconds (balanced reactivity vs rate limits)
  - Q4 / <5 min: every 10 seconds (higher frequency for critical game scripts)
  - Halftime: every 60 seconds

Rate limit budget: BDL allows ~600 requests/minute.
For 2 live games: 8 polls/min per game = 16 total — well within budget.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine
from wnba_props_model.live.live_edge import LiveEdgeCalculator
from wnba_props_model.live.pbp_parser import PBPParser, clock_to_minutes

log = logging.getLogger(__name__)


class LiveGameOrchestrator:
    """Orchestrates live game tracking: polls PBP, updates posteriors, computes edges.

    Usage:
        orch = LiveGameOrchestrator(bdl_client, ...)
        for _ in range(n_polls):
            edges, game_state = orch.run_game(game_id, pre_game_projections, roster_lookup)
            # Output edges to delivery system
            time.sleep(orch.next_poll_interval(game_state))
    """

    def __init__(
        self,
        bdl_client,
        bayesian_engine: Optional[GammaPoissonLiveEngine] = None,
        edge_calculator: Optional[LiveEdgeCalculator] = None,
        props_poll_interval: int = 30,
        pbp_poll_interval: int = 15,
        out_dir: Optional[str | Path] = None,
    ) -> None:
        self.client = bdl_client
        self.engine = bayesian_engine or GammaPoissonLiveEngine()
        self.edge_calc = edge_calculator or LiveEdgeCalculator(min_edge=0.04)
        self.props_interval = props_poll_interval
        self.pbp_interval = pbp_poll_interval
        self.out_dir = Path(out_dir) if out_dir else None
        # Track last processed play order per game
        self.last_play_order: dict[int, int] = {}
        # Parser per game
        self._parsers: dict[int, PBPParser] = {}

    def run_game(
        self,
        game_id: int,
        pre_game_projections: dict,
        roster_lookup: dict[str, dict],
    ) -> tuple[list[dict], dict]:
        """Run one live-tracking cycle for a single game.

        Args:
            game_id: BDL game ID
            pre_game_projections: {player_id: {stat: {mean, line}, projected_minutes}}
            roster_lookup: {player_name_abbrev: {player_id, team_id, team_side}}

        Returns:
            (live_edges, game_state) — edges sorted by |edge| descending
        """
        # 1. Fetch latest PBP
        plays = self._fetch_plays(game_id)

        # 2. Parse only new plays since last poll
        last_order = self.last_play_order.get(game_id, 0)
        new_plays = [p for p in plays if int(p.get("order", 0)) > last_order]

        if game_id not in self._parsers:
            self._parsers[game_id] = PBPParser()

        parser = self._parsers[game_id]
        if new_plays:
            parser.process_plays(new_plays, roster_lookup)
            max_order = max(int(p.get("order", 0)) for p in new_plays)
            self.last_play_order[game_id] = max_order

        player_states = parser.player_states
        game_state = parser.game_state

        # 3. Compute elapsed minutes
        elapsed = parser.elapsed_minutes()

        # 4. Update posterior PMFs for each player-stat
        live_predictions = self.engine.batch_compute(
            pre_game_projections, player_states, elapsed
        )

        # 5. Fetch live props
        live_props = self._fetch_live_props(game_id)

        # 6. Compute edges
        edges = self.edge_calc.compute_live_edges(live_predictions, live_props)

        # 7. Optionally write output
        if self.out_dir is not None:
            self._write_edges(game_id, edges, game_state)

        log.info(
            "LiveGameOrchestrator: game=%d elapsed=%.1fmin edges=%d bettable=%d",
            game_id, elapsed, len(edges),
            sum(1 for e in edges if e["bettable"]),
        )
        return edges, game_state

    def next_poll_interval(self, game_state: dict) -> int:
        """Determine polling interval based on current game state."""
        period = game_state.get("period", 1)
        clock_str = game_state.get("clock", "10:00")
        elapsed = clock_to_minutes(period, clock_str)

        # Halftime (between Q2 and Q3)
        if period == 2 and clock_str in ("0:00", "00:00"):
            return 60  # Low frequency during halftime

        # Late Q4 / under 5 minutes
        if period >= 4:
            try:
                parts = clock_str.split(":")
                mins_left = int(parts[0])
                if mins_left < 5:
                    return 10
            except Exception:
                pass
            return 15

        # Q1-Q3 normal play
        if period <= 3:
            return 15

        return self.pbp_interval

    def _fetch_plays(self, game_id: int) -> list[dict]:
        """Fetch all plays from /wnba/v1/plays?game_id=X."""
        try:
            resp = self.client.get_json("/wnba/v1/plays", {"game_id": game_id})
            return resp.get("data", [])
        except Exception as exc:
            log.warning("LiveGameOrchestrator: failed to fetch plays for game %d: %s", game_id, exc)
            return []

    def _fetch_live_props(self, game_id: int) -> pd.DataFrame:
        """Fetch live player props from /wnba/v1/odds/player_props?game_id=X."""
        try:
            resp = self.client.get_json("/wnba/v1/odds/player_props", {"game_id": game_id})
            data = resp.get("data", [])
            if not data:
                return pd.DataFrame()
            # Normalize nested market dict
            rows = []
            for row in data:
                market = row.get("market") or {}
                rows.append({
                    "prop_id": row.get("id"),
                    "game_id": row.get("game_id"),
                    "player_id": row.get("player_id"),
                    "vendor": row.get("vendor"),
                    "prop_type": row.get("prop_type"),
                    "line_value": row.get("line_value"),
                    "over_odds": market.get("over_odds") or row.get("over_odds"),
                    "under_odds": market.get("under_odds") or row.get("under_odds"),
                    "updated_at": row.get("updated_at"),
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            log.warning("LiveGameOrchestrator: failed to fetch props for game %d: %s", game_id, exc)
            return pd.DataFrame()

    def _write_edges(self, game_id: int, edges: list[dict], game_state: dict) -> None:
        """Write live edges to output directory."""
        if not self.out_dir:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self.out_dir / f"live_edges_{game_id}_{ts}.json"
        import json  # noqa: PLC0415
        payload = {
            "game_id": game_id,
            "timestamp_utc": ts,
            "game_state": game_state,
            "n_edges": len(edges),
            "n_bettable": sum(1 for e in edges if e.get("bettable")),
            "edges": edges[:50],  # Cap output to top-50 edges
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))


def build_roster_lookup(
    players_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> dict[str, dict]:
    """Build a roster lookup dict from a players DataFrame.

    Args:
        players_df: DataFrame with columns player_id, player_name, team_id
        home_team_id: BDL team ID for the home team
        away_team_id: BDL team ID for the away team

    Returns:
        {abbreviated_name: {player_id, team_id, team_side}}
    """
    lookup: dict[str, dict] = {}
    for _, row in players_df.iterrows():
        name = str(row.get("player_name", ""))
        pid = row.get("player_id")
        tid = row.get("team_id")
        if not name or pid is None:
            continue
        side = "home" if tid == home_team_id else "away" if tid == away_team_id else "unknown"
        # Create abbreviated version (e.g. "Aja Wilson" → "A. Wilson")
        parts = name.split()
        if len(parts) >= 2:
            abbrev = f"{parts[0][0]}. {' '.join(parts[1:])}"
        else:
            abbrev = name
        info = {"player_id": pid, "team_id": tid, "team_side": side}
        lookup[abbrev] = info
        lookup[name] = info  # also map full name
    return lookup
