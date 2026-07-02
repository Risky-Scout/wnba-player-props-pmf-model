#!/usr/bin/env python3
"""Compute position-stratified Pearson correlations for combo stats from OOF data.

Used to improve combo stat (pts+reb, pts+ast, etc.) covariance estimates
by leveraging position-stratified empirical correlations rather than
using a single global correlation.

Usage:
    python scripts/compute_combo_correlations_by_pos.py \
        --oof-path data/oof/oof_player_stat_pmfs.parquet \
        --out artifacts/models/stage4_baseline/combo_correlations_by_pos.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_COMBO_PAIRS = [
    ("pts", "reb"),
    ("pts", "ast"),
    ("pts", "fg3m"),
    ("reb", "ast"),
    ("ast", "stl"),
    ("pts", "stl"),
    ("blk", "reb"),
]

_POSITION_COL_CANDIDATES = ["position", "pos", "player_position"]


def get_position_col(df: pd.DataFrame) -> str | None:
    for col in _POSITION_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def compute_correlations_by_pos(oof: pd.DataFrame) -> dict:
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True]  # noqa: E712

    pos_col = get_position_col(oof)
    results: dict = {}

    # Pivot to wide format: one row per (player_id, game_id) with stat columns
    if "actual_outcome" not in oof.columns or "stat" not in oof.columns:
        print("WARNING: OOF missing actual_outcome or stat column — skipping combo correlations")
        return results

    pivot = oof.pivot_table(
        index=["player_id", "game_id"],
        columns="stat",
        values="actual_outcome",
        aggfunc="first",
    )
    pivot.columns = [str(c) for c in pivot.columns]

    # Attach position
    if pos_col is not None:
        pos_map = (
            oof[["player_id", pos_col]]
            .drop_duplicates("player_id")
            .set_index("player_id")[pos_col]
        )
        pivot = pivot.join(pos_map, on="player_id")
        positions = pivot[pos_col].unique()
    else:
        positions = ["all"]
        pivot["position"] = "all"
        pos_col = "position"

    for stat_a, stat_b in _COMBO_PAIRS:
        if stat_a not in pivot.columns or stat_b not in pivot.columns:
            continue
        pair_key = f"{stat_a}+{stat_b}"
        results[pair_key] = {}

        pair_data = pivot[[stat_a, stat_b, pos_col]].dropna()
        if len(pair_data) < 30:
            continue

        # Global correlation
        r_global = float(np.corrcoef(pair_data[stat_a], pair_data[stat_b])[0, 1])
        results[pair_key]["all"] = round(r_global, 4)

        # Per-position correlations
        for pos in positions:
            if pos == "all":
                continue
            pos_rows = pair_data[pair_data[pos_col] == pos]
            if len(pos_rows) < 20:
                continue
            r_pos = float(np.corrcoef(pos_rows[stat_a], pos_rows[stat_b])[0, 1])
            results[pair_key][str(pos)] = round(r_pos, 4)

        print(f"  {pair_key}: global_r={r_global:.3f}, positions={list(results[pair_key].keys())}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-path", required=True)
    parser.add_argument("--out", default="artifacts/models/stage4_baseline/combo_correlations_by_pos.json")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof_path)
    correlations = compute_correlations_by_pos(oof)

    if not correlations:
        print("WARNING: No correlations computed. Check OOF data structure.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"Combo correlations saved to {out_path}")


if __name__ == "__main__":
    main()
