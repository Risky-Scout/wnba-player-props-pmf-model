#!/usr/bin/env python3
"""Derive per-(game_id, player_id, game_date) STL/BLK/TOV outcome labels from parsed PBP (step E.1).

These are the label side for future steals/blocks/turnovers models. They are ready NOW even though
the market side is not yet posted this early in the season (see collect_forward_props.py /
STL_BLK_TOV_READINESS.json). Labels are validated by the same parser reconciliation that showed
stl/blk/tov exact-match >= 0.999 vs the official box (PBP_PARSER_VALIDATION.json).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data/processed/wnba_pbp_player_game_counts.parquet")
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--out", default="data/processed/wnba_stlblktov_labels.parquet")
    args = ap.parse_args()

    parsed = pd.read_parquet(args.parsed)
    box = pd.read_parquet(args.box)[["game_id", "player_id", "game_date", "did_play"]]
    for c in ("game_id", "player_id"):
        parsed[c] = pd.to_numeric(parsed[c], errors="coerce")
        box[c] = pd.to_numeric(box[c], errors="coerce")
    cols = ["game_id", "player_id", "stl", "blk", "tov"]
    labels = parsed[[c for c in cols if c in parsed.columns]].copy()
    labels = labels.merge(box, on=["game_id", "player_id"], how="left")
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce")
    labels = labels.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out, index=False)
    print(f"wrote {args.out} rows={len(labels)} games={labels['game_id'].nunique()}")
    print(labels[["stl", "blk", "tov"]].describe().to_string())


if __name__ == "__main__":
    main()
