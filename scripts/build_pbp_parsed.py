#!/usr/bin/env python3
"""Parse the PBP snapshot into per-(game_id, player_id) event counts and validate vs the box.

Steps B (parser) + B-validation of the owner directive. Reads the play snapshot written by
``pull_pbp_history.py``, resolves player names via the per-game box roster, writes the parsed
per-player-per-game count table, and writes ``PBP_PARSER_VALIDATION.json`` reconciling parsed
pts/reb/ast/stl/blk/tov/fg3m against the official BDL box for real games.

Usage::

  python3 scripts/build_pbp_parsed.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from wnba_props_model.data.pbp_parse import (
    parse_plays_to_player_game,
    reconcile_against_box,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp-dir", default="data/snapshots/pbp")
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--out", default="data/processed/wnba_pbp_player_game_counts.parquet")
    ap.add_argument("--validation-out",
                    default="artifacts/opportunity_v2/PBP_PARSER_VALIDATION.json")
    ap.add_argument("--min-validation-games", type=int, default=20)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.pbp_dir}/game_date=*/plays_*.parquet"))
    if not files:
        raise SystemExit(f"no PBP snapshot files under {args.pbp_dir}")
    plays = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    box = pd.read_parquet(args.box)

    parsed, stats = parse_plays_to_player_game(plays, box)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(args.out, index=False)

    report = reconcile_against_box(parsed, box)
    attr_rate = (stats.n_attributable / stats.n_plays) if stats.n_plays else 0.0
    unmatched_rate = (stats.n_unmatched_actor / stats.n_plays) if stats.n_plays else 0.0
    validation = {
        "games_parsed": int(parsed["game_id"].nunique()) if not parsed.empty else 0,
        "player_game_rows": int(len(parsed)),
        "total_plays_parsed": int(stats.n_plays),
        "attributable_play_rate": round(attr_rate, 4),
        "unmatched_actor_play_rate": round(unmatched_rate, 4),
        "unmatched_actor_examples": stats.unmatched_examples,
        "reconciliation_rows": report["n_rows"],
        "reconciliation_games": report["n_games"],
        "min_validation_games_required": args.min_validation_games,
        "meets_min_validation_games": report["n_games"] >= args.min_validation_games,
        "per_stat_reconciliation": report["per_stat"],
    }
    Path(args.validation_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.validation_out).write_text(json.dumps(validation, indent=2, default=float))

    print(json.dumps({
        "games_parsed": validation["games_parsed"],
        "player_game_rows": validation["player_game_rows"],
        "attributable_play_rate": validation["attributable_play_rate"],
        "unmatched_actor_play_rate": validation["unmatched_actor_play_rate"],
        "recon_rows": validation["reconciliation_rows"],
        "per_stat": {k: {"exact": round(v["exact_match_rate"], 3),
                         "within1": round(v["within_1_rate"], 3),
                         "mae": round(v["mean_abs_error"], 4)}
                     for k, v in report["per_stat"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
