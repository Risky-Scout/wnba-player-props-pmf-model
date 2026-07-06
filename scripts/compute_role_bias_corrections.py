#!/usr/bin/env python3
"""
Compute empirical role-stratified bias corrections from 2026 actual game stats
vs. PMF predictions. Saves bias_corrections_by_role.json.

Usage:
    python scripts/compute_role_bias_corrections.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

STATS_PARQUET = "data/processed/wnba_player_game_stats.parquet"
DELIVERIES_DIR = "deliveries/next_game"
FALLBACK_DELIVERIES_DIR = "deliveries/tonight"
BIAS_CORRECTIONS_PATH = "artifacts/models/calibration/bias_corrections.json"
OUTPUT_PATH = "artifacts/models/calibration/bias_corrections_by_role.json"

BASE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
MIN_GAMES = 5
MIN_GROUP_SIZE = 3
CLAMP_LOW = 0.70
CLAMP_HIGH = 1.50


def find_latest_predictions_parquet():
    """Find the most recently modified predictions parquet with required columns."""
    for search_dir in [DELIVERIES_DIR, FALLBACK_DELIVERIES_DIR]:
        files = sorted(
            glob.glob(os.path.join(search_dir, "*.parquet")),
            key=os.path.getmtime,
            reverse=True,
        )
        for f in files:
            try:
                df = pd.read_parquet(f, columns=["pmf_mean", "role_bucket", "stat", "player_name"])
                print(f"[compute_role_bias] Using predictions parquet: {f}")
                return f
            except Exception:
                continue
    return None


def load_global_bias_corrections():
    if not os.path.exists(BIAS_CORRECTIONS_PATH):
        return {}
    with open(BIAS_CORRECTIONS_PATH) as fh:
        return json.load(fh)


def main():
    # --- Load 2026 game stats ---
    if not os.path.exists(STATS_PARQUET):
        print(f"ERROR: Stats parquet not found: {STATS_PARQUET}", file=sys.stderr)
        sys.exit(1)

    stats_df = pd.read_parquet(STATS_PARQUET)

    # Filter to 2026 season, played games only
    if "game_date" not in stats_df.columns:
        print("ERROR: game_date column missing from stats parquet", file=sys.stderr)
        sys.exit(1)

    stats_df["game_date"] = pd.to_datetime(stats_df["game_date"])
    stats_2026 = stats_df[
        (stats_df["game_date"] >= "2026-01-01")
        & (stats_df.get("did_play", pd.Series(True, index=stats_df.index)).fillna(True))
        & (stats_df.get("non_playing_flag", pd.Series(False, index=stats_df.index)).fillna(False) == False)
    ].copy()

    print(f"[compute_role_bias] 2026 game rows: {len(stats_2026)}")

    # --- Load predictions parquet ---
    pred_path = find_latest_predictions_parquet()
    if pred_path is None:
        print(
            f"ERROR: No predictions parquet found in {DELIVERIES_DIR} or {FALLBACK_DELIVERIES_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        pred_df = pd.read_parquet(pred_path)
    except Exception as e:
        print(f"ERROR: Could not read predictions parquet: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"pmf_mean", "role_bucket", "stat", "player_name"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        print(f"ERROR: Predictions parquet missing columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Filter to base stats only
    pred_df = pred_df[pred_df["stat"].isin(BASE_STATS)].copy()
    print(f"[compute_role_bias] Prediction rows (base stats): {len(pred_df)}")

    # --- Compute 2026 season averages per player/stat ---
    # Melt stats_2026 to long format
    stat_cols = [c for c in BASE_STATS if c in stats_2026.columns]
    id_cols = ["player_name", "game_date"]
    stats_long = stats_2026[id_cols + stat_cols].melt(
        id_vars=id_cols, value_vars=stat_cols, var_name="stat", value_name="actual_value"
    )
    stats_long = stats_long.dropna(subset=["actual_value"])

    # Count games per player (use pts as proxy for game count)
    game_counts = (
        stats_2026.groupby("player_name")["game_date"]
        .nunique()
        .reset_index(name="n_games")
    )
    eligible_players = game_counts[game_counts["n_games"] >= MIN_GAMES]["player_name"]
    print(f"[compute_role_bias] Players with >= {MIN_GAMES} games: {len(eligible_players)}")

    stats_long = stats_long[stats_long["player_name"].isin(eligible_players)]

    player_stat_avg = (
        stats_long.groupby(["player_name", "stat"])["actual_value"]
        .mean()
        .reset_index(name="actual_avg")
    )

    # --- Merge with predictions to get pmf_mean and role_bucket ---
    # Use latest prediction row per player/stat
    pred_latest = (
        pred_df.sort_values("player_name")
        .groupby(["player_name", "stat"])[["pmf_mean", "role_bucket"]]
        .last()
        .reset_index()
    )

    merged = player_stat_avg.merge(pred_latest, on=["player_name", "stat"], how="inner")
    merged = merged[merged["pmf_mean"] > 0].copy()
    merged["ratio"] = merged["actual_avg"] / merged["pmf_mean"]

    print(f"[compute_role_bias] Merged player/stat pairs: {len(merged)}")

    # --- Load global fallback corrections ---
    global_corrections = load_global_bias_corrections()

    # --- Group by (role_bucket, stat), compute median ratio ---
    results = {}
    summary_rows = []

    for (role_bucket, stat), grp in merged.groupby(["role_bucket", "stat"]):
        n = len(grp)
        if n < MIN_GROUP_SIZE:
            fallback = global_corrections.get(stat, 1.0)
            correction = fallback
            source = f"global_fallback (n={n})"
        else:
            median_ratio = float(np.median(grp["ratio"].values))
            correction = float(np.clip(median_ratio, CLAMP_LOW, CLAMP_HIGH))
            source = f"empirical (n={n})"

        key = f"{role_bucket}|{stat}"
        results[key] = round(correction, 4)
        summary_rows.append(
            {
                "role_bucket": role_bucket,
                "stat": stat,
                "n": n,
                "correction": correction,
                "source": source,
            }
        )

    # --- Save output ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "_comment": (
            "Role-stratified bias corrections: median(actual_avg/pmf_mean) per (role_bucket, stat). "
            "Groups with <3 data points fall back to global bias_corrections.json."
        ),
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corrections": results,
    }
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[compute_role_bias] Saved {len(results)} corrections to {OUTPUT_PATH}")

    # --- Print summary table ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["stat", "role_bucket"])
        print("\n=== Role Bias Corrections Summary ===")
        print(summary_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
