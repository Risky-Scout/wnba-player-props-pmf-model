#!/usr/bin/env python3
"""Compute per-stat league means from current-season data and save as JSON.

Usage:
    python scripts/update_league_priors.py \
        --stats-path data/processed/wnba_player_game_stats.parquet \
        --season 2026 \
        --out artifacts/models/league_priors.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

_DIRECT_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def compute_league_priors(stats_path: str, season: int) -> dict:
    df = pd.read_parquet(stats_path)
    if "season" in df.columns:
        df = df[df["season"] == season]
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    priors: dict = {}
    for stat in _DIRECT_STATS:
        col = f"actual_{stat}"
        if col in df.columns and df[col].notna().sum() > 10:
            priors[stat] = round(float(df[col].mean()), 3)
        elif stat in df.columns and df[stat].notna().sum() > 10:
            priors[stat] = round(float(df[stat].mean()), 3)
    return priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", default="artifacts/models/league_priors.json")
    args = parser.parse_args()

    priors = compute_league_priors(args.stats_path, args.season)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"League priors for {args.season}: {priors}")


if __name__ == "__main__":
    main()
