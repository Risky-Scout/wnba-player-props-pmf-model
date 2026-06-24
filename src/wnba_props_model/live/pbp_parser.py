"""BDL play-by-play parser for live WNBA game tracking.

Parses /wnba/v1/plays?game_id=X event text into per-player running stat totals.
Handles the BDL play text format which looks like:
  "A. Wilson made 2-point layup"
  "A. Wilson made 3-point jumper. K. Collier assist."
  "A. Wilson defensive rebound"
  "A. Wilson steal"
  "A. Wilson turnover"
  "A. Wilson made free throw 1 of 2"
  "A. Wilson fouled (draws foul)"

Reference: Yeh et al. (2020). Evaluating Real-Time Probabilistic Forecasts with
Application to National Basketball Association Outcome Prediction.
https://figshare.com/articles/journal_contribution/...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LivePlayerState:
    """Running stat totals for a player during a live game."""

    player_id: int
    player_name: str
    team_id: Optional[int]
    team_side: str  # "home" or "away"

    # Counting stats
    pts: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    turnover: int = 0
    fg3m: int = 0
    fgm: int = 0
    fga: int = 0
    ftm: int = 0
    fta: int = 0
    minutes_played: float = 0.0
    fouls: int = 0
    ejected: bool = False

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team_id": self.team_id,
            "team_side": self.team_side,
            "pts": self.pts,
            "reb": self.reb,
            "ast": self.ast,
            "stl": self.stl,
            "blk": self.blk,
            "turnover": self.turnover,
            "fg3m": self.fg3m,
            "fgm": self.fgm,
            "fga": self.fga,
            "ftm": self.ftm,
            "fta": self.fta,
            "minutes_played": self.minutes_played,
            "fouls": self.fouls,
            "ejected": self.ejected,
        }


class PBPParser:
    """Parse BDL play-by-play text into live player stat updates.

    BDL play.text format examples:
      "A. Wilson made 2-point layup"
      "A. Wilson made 3-point jumper"
      "A. Wilson missed 3-point jumper"
      "A. Wilson defensive rebound"
      "A. Wilson assist to K. Collier"
      "A. Wilson steal"
      "A. Wilson block"
      "A. Wilson turnover"
      "A. Wilson made free throw 1 of 2"
      "A. Wilson fouled (draws foul)"

    Also tracks game state (score, period, clock) for game-script effects.
    """

    # Regex patterns for play text parsing
    SCORING_PATTERNS = [
        (r"(\w[\w.']*\s+\w+)\s+made\s+3-point", "fg3m"),
        (r"(\w[\w.']*\s+\w+)\s+made\s+2-point", "fg2m"),
        (r"(\w[\w.']*\s+\w+)\s+made\s+free\s+throw", "ft"),
        (r"(\w[\w.']*\s+\w+)\s+missed\s+3-point", "fg3miss"),
        (r"(\w[\w.']*\s+\w+)\s+missed\s+2-point", "fg2miss"),
        (r"(\w[\w.']*\s+\w+)\s+missed\s+free\s+throw", "ftmiss"),
    ]
    REBOUND_PATTERN = r"(\w[\w.']*\s+\w+)\s+(offensive|defensive)\s+rebound"
    ASSIST_PATTERN = r"(\w[\w.']*\s+\w+)\s+assist"
    STEAL_PATTERN = r"(\w[\w.']*\s+\w+)\s+steal"
    BLOCK_PATTERN = r"(\w[\w.']*\s+\w+)\s+block"
    TURNOVER_PATTERN = r"(\w[\w.']*\s+\w+)\s+turnover"
    FOUL_PATTERN = r"(\w[\w.']*\s+\w+)\s+foul"
    EJECTION_PATTERN = r"(\w[\w.']*\s+\w+)\s+ejected"

    def __init__(self) -> None:
        self.player_states: dict[int, LivePlayerState] = {}
        self.game_state: dict = {
            "home_score": 0,
            "away_score": 0,
            "period": 1,
            "clock": "10:00",
            "possession_team_id": None,
            "n_plays_processed": 0,
        }

    def reset(self) -> None:
        """Reset parser state for a new game."""
        self.player_states.clear()
        self.game_state = {
            "home_score": 0,
            "away_score": 0,
            "period": 1,
            "clock": "10:00",
            "possession_team_id": None,
            "n_plays_processed": 0,
        }

    def process_plays(
        self,
        plays_data: list[dict],
        roster_lookup: dict[str, dict],
    ) -> tuple[dict[int, LivePlayerState], dict]:
        """Process a batch of plays from /wnba/v1/plays.

        Args:
            plays_data: list of play dicts from BDL API
            roster_lookup: dict mapping player_name_abbrev → {player_id, team_id, team_side}
                e.g. {"A. Wilson": {"player_id": 123, "team_id": 5, "team_side": "home"}}

        Returns:
            (player_states, game_state) tuple — both updated in place
        """
        for play in plays_data:
            # Update game state
            if play.get("home_score") is not None:
                self.game_state["home_score"] = play["home_score"]
            away_val = play.get("away_score") if play.get("away_score") is not None else play.get("visitor_score")
            if away_val is not None:
                self.game_state["away_score"] = away_val
            if play.get("period") is not None:
                self.game_state["period"] = play["period"]
            if play.get("clock") is not None:
                self.game_state["clock"] = play["clock"]
            if play.get("team") and isinstance(play["team"], dict):
                self.game_state["possession_team_id"] = play["team"].get("id")

            text = play.get("text") or play.get("description") or ""
            if text:
                self._parse_play_text(text, play.get("team") or {}, roster_lookup)
            self.game_state["n_plays_processed"] += 1

        return self.player_states, self.game_state

    def _parse_play_text(
        self,
        text: str,
        team_info: dict,
        roster_lookup: dict[str, dict],
    ) -> None:
        """Extract player actions from play text."""
        team_id = team_info.get("id") if team_info else None

        # Scoring plays (process first since they're most common)
        for pattern, stat_type in self.SCORING_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                player_name = m.group(1).strip()
                info = self._lookup_player(player_name, roster_lookup)
                if info is None:
                    continue
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                ps = self.player_states[pid]
                if stat_type == "fg3m":
                    ps.fg3m += 1; ps.fgm += 1; ps.fga += 1; ps.pts += 3
                elif stat_type == "fg2m":
                    ps.fgm += 1; ps.fga += 1; ps.pts += 2
                elif stat_type == "ft":
                    ps.ftm += 1; ps.fta += 1; ps.pts += 1
                elif stat_type in ("fg3miss", "fg2miss"):
                    ps.fga += 1
                elif stat_type == "ftmiss":
                    ps.fta += 1
                return

        # Rebounds
        m = re.search(self.REBOUND_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].reb += 1
            return

        # Assists
        m = re.search(self.ASSIST_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].ast += 1
            return

        # Steals
        m = re.search(self.STEAL_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].stl += 1
            return

        # Blocks
        m = re.search(self.BLOCK_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].blk += 1
            return

        # Turnovers
        m = re.search(self.TURNOVER_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].turnover += 1
            return

        # Fouls
        m = re.search(self.FOUL_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].fouls += 1
            return

        # Ejections
        m = re.search(self.EJECTION_PATTERN, text, re.IGNORECASE)
        if m:
            player_name = m.group(1).strip()
            info = self._lookup_player(player_name, roster_lookup)
            if info:
                pid = info["player_id"]
                self._ensure_player(pid, player_name, team_id, info)
                self.player_states[pid].ejected = True

    def _lookup_player(
        self,
        player_name: str,
        roster_lookup: dict[str, dict],
    ) -> Optional[dict]:
        """Look up player info from roster, trying fuzzy match if needed."""
        # Exact match first
        if player_name in roster_lookup:
            return roster_lookup[player_name]
        # Try abbreviation match: "A. Wilson" matches "A. Wilson", "Aja Wilson", etc.
        name_lower = player_name.lower()
        for key, info in roster_lookup.items():
            if key.lower() == name_lower:
                return info
        return None

    def _ensure_player(
        self,
        player_id: int,
        player_name: str,
        team_id: Optional[int],
        info: dict,
    ) -> None:
        if player_id not in self.player_states:
            self.player_states[player_id] = LivePlayerState(
                player_id=player_id,
                player_name=player_name,
                team_id=team_id or info.get("team_id"),
                team_side=info.get("team_side", "unknown"),
            )

    def elapsed_minutes(self) -> float:
        """Compute total elapsed game minutes from current game state."""
        return clock_to_minutes(
            self.game_state.get("period", 1),
            self.game_state.get("clock", "10:00"),
        )

    def score_margin(self) -> int:
        """Home score minus away score."""
        return self.game_state["home_score"] - self.game_state["away_score"]


def clock_to_minutes(period: int, clock_str: str) -> float:
    """Convert period + clock string to elapsed game minutes.

    WNBA: 10-minute quarters. Clock counts down from 10:00.
    Regulation = 4 quarters × 10 min = 40 min.
    """
    try:
        parts = clock_str.split(":")
        mins_remaining = int(parts[0]) + int(parts[1]) / 60
        quarter_minutes = 10.0 - mins_remaining
        total = (period - 1) * 10.0 + quarter_minutes
        return max(total, 0.0)
    except Exception:
        return 0.0
