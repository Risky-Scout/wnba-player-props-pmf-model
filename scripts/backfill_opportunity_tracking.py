#!/usr/bin/env python3
"""Build Tier-0 team/player opportunity 'tracking' tables from the box score (directive section 8).

HONEST scope: the box score yields only Tier-0 team environment (possessions/attempts/misses/
turnovers) and player minutes/started flags. True tracking columns (touches, potential assists,
rebound chances, catch-and-shoot 3PA, ...) are left NULL and data_tier=0 -- never fabricated from
postgame box outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.contracts import DATA_TIER_BOX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--team-out", default="data/processed/wnba_team_game_opportunity_tracking.parquet")
    ap.add_argument("--player-out", default="data/processed/wnba_player_game_opportunity_tracking.parquet")
    args = ap.parse_args()
    now = pd.Timestamp(datetime.now(timezone.utc))

    box = pd.read_parquet(args.box)
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce")

    team = (box.groupby(["game_id", "team_id", "opponent_team_id", "game_date"], as_index=False)
            .agg(fga=("fga", "sum"), fg3a=("fg3a", "sum"), fta=("fta", "sum"),
                 oreb=("oreb", "sum"), turnovers=("turnover", "sum")))
    team["fg2a"] = team["fga"] - team["fg3a"]
    team["possessions"] = team["fga"] + 0.44 * team["fta"] - team["oreb"] + team["turnovers"]
    # Tier-0: makes-derived misses require FGM which the box lacks -> leave null (honest).
    for c in ("fg_misses", "fg2_misses", "fg3_misses", "live_ball_turnovers", "rim_attempts",
              "potential_assists", "rebound_chances", "touches", "passes"):
        team[c] = np.nan
    team["source"] = "box_tier0"
    team["source_available_at_utc"] = now
    team["data_tier"] = DATA_TIER_BOX

    player = box[["game_id", "game_date", "player_id", "team_id", "opponent_team_id",
                  "minutes", "started_proxy"]].copy()
    player = player.rename(columns={"minutes": "actual_minutes", "started_proxy": "actual_started"})
    for c in ("touches", "passes_made", "passes_received", "potential_assists",
              "time_of_possession_seconds", "drives", "paint_touches", "frontcourt_touches",
              "rebound_chances", "oreb_chances", "dreb_chances", "contested_rebound_chances",
              "uncontested_rebound_chances", "catch_shoot_3pa", "pullup_3pa", "wide_open_3pa",
              "open_3pa", "contested_3pa", "rim_attempts_defended", "shot_contests",
              "defensive_matchup_possessions", "primary_ballhandler_matchup_possessions"):
        player[c] = np.nan
    player["source"] = "box_tier0"
    player["source_available_at_utc"] = now
    player["data_tier"] = DATA_TIER_BOX

    Path(args.team_out).parent.mkdir(parents=True, exist_ok=True)
    team.to_parquet(args.team_out, index=False)
    player.to_parquet(args.player_out, index=False)
    print(f"wrote {args.team_out} rows={len(team)} (Tier-0 possessions); "
          f"{args.player_out} rows={len(player)} (tracking cols NULL by design)")


if __name__ == "__main__":
    main()
