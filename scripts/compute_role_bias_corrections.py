#!/usr/bin/env python3
"""
Compute empirical role-stratified bias corrections from OOF predictions
vs. actual game stats. Saves bias_corrections_by_role.json.

Joins on player_id (not player_name) and loads OOF data from
data/oof/oof_player_stat_pmfs.parquet (not delivery parquets).

Outputs net_mult = empirical_ratio / global_correction, with a 30-game
minimum per (role, stat) group before trusting the correction, and enforces
per-role monotonicity: starter >= core >= rotation >= bench >= fringe.

Usage:
    python scripts/compute_role_bias_corrections.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

OOF_PATH = "data/oof/oof_player_stat_pmfs.parquet"
STATS_PARQUET = "data/processed/wnba_player_game_stats.parquet"
BIAS_CORRECTIONS_PATH = "artifacts/models/calibration/bias_corrections.json"
OUTPUT_PATH = "artifacts/models/calibration/bias_corrections_by_role.json"

BASE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
MIN_GAMES = 5
MIN_GROUP_SIZE = 30   # minimum games per (role, stat) before trusting correction
CLAMP_LOW = 0.70
CLAMP_HIGH = 1.50

# Monotonicity order: each role's correction must be >= the role below it.
ROLE_ORDER = ["starter", "core", "rotation", "bench", "fringe"]


def load_global_bias_corrections():
    if not os.path.exists(BIAS_CORRECTIONS_PATH):
        return {}
    with open(BIAS_CORRECTIONS_PATH) as fh:
        return json.load(fh)


def enforce_monotonicity(corrections: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Enforce starter >= core >= rotation >= bench >= fringe per stat.

    Iterates from top role downward and clips each role's correction
    to be <= the correction of the role above it.
    """
    result = {role: dict(vals) for role, vals in corrections.items()}
    for stat in BASE_STATS:
        prev_val = None
        for role in ROLE_ORDER:
            if role not in result or stat not in result[role]:
                continue
            val = result[role][stat]
            if prev_val is not None and val > prev_val:
                result[role][stat] = prev_val
            prev_val = result[role][stat]
    return result


def main():
    # --- Load OOF predictions ---
    if not os.path.exists(OOF_PATH):
        print(f"ERROR: OOF parquet not found: {OOF_PATH}", file=sys.stderr)
        print("[compute_role_bias] Falling back to stats-only baseline", file=sys.stderr)
        sys.exit(1)

    oof_df = pd.read_parquet(OOF_PATH)
    required_oof = {"player_id", "stat", "pmf_mean", "actual_outcome"}
    missing_oof = required_oof - set(oof_df.columns)
    if missing_oof:
        print(f"ERROR: OOF parquet missing columns: {missing_oof}", file=sys.stderr)
        sys.exit(1)

    # Filter to base stats and prop-eligible rows
    oof_df = oof_df[oof_df["stat"].isin(BASE_STATS)].copy()
    if "calibration_eligible" in oof_df.columns:
        oof_df = oof_df[oof_df["calibration_eligible"] == True].copy()  # noqa: E712
    if "did_play" in oof_df.columns:
        oof_df = oof_df[oof_df["did_play"] == True].copy()  # noqa: E712
    if "actual_minutes" in oof_df.columns:
        oof_df = oof_df[oof_df["actual_minutes"].fillna(0) >= 10].copy()

    # Attach role_bucket if not present (derive from minutes_mean)
    if "role_bucket" not in oof_df.columns:
        if "minutes_mean" in oof_df.columns:
            try:
                from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket
                oof_df = add_ex_ante_role_bucket(oof_df, minutes_col="minutes_mean")
            except Exception as exc:
                print(f"[compute_role_bias] Could not derive role_bucket: {exc}", file=sys.stderr)
                oof_df["role_bucket"] = "rotation"
        else:
            oof_df["role_bucket"] = "rotation"

    oof_df = oof_df.dropna(subset=["pmf_mean", "actual_outcome"])
    oof_df = oof_df[oof_df["pmf_mean"] > 0].copy()

    print(f"[compute_role_bias] OOF rows (base stats, prop-eligible): {len(oof_df)}")
    print(f"[compute_role_bias] Roles present: {sorted(oof_df['role_bucket'].dropna().unique().tolist())}")

    # --- Load 2026 game stats for actual game-level averages ---
    if os.path.exists(STATS_PARQUET):
        stats_df = pd.read_parquet(STATS_PARQUET)
        if "game_date" in stats_df.columns and "player_id" in stats_df.columns:
            stats_df["game_date"] = pd.to_datetime(stats_df["game_date"])
            stats_2026 = stats_df[
                (stats_df["game_date"] >= "2026-01-01")
                & (stats_df.get("did_play", pd.Series(True, index=stats_df.index)).fillna(True))
                & (stats_df.get("non_playing_flag", pd.Series(False, index=stats_df.index)).fillna(False) == False)
            ].copy()
            game_counts = (
                stats_2026.groupby("player_id")["game_date"]
                .nunique()
                .reset_index(name="n_games")
            )
            eligible_players = set(game_counts[game_counts["n_games"] >= MIN_GAMES]["player_id"])
            print(f"[compute_role_bias] 2026 game rows: {len(stats_2026)}, players with >={MIN_GAMES} games: {len(eligible_players)}")
        else:
            eligible_players = set()
            print("[compute_role_bias] Stats parquet missing game_date or player_id — using all OOF players")
    else:
        eligible_players = set()
        print(f"[compute_role_bias] Stats parquet not found ({STATS_PARQUET}) — using all OOF players")

    if eligible_players:
        oof_df = oof_df[oof_df["player_id"].isin(eligible_players)]
        print(f"[compute_role_bias] OOF rows after player eligibility filter: {len(oof_df)}")

    # --- Load global fallback corrections ---
    global_corrections = load_global_bias_corrections()

    # --- Group by (player_id, role_bucket, stat), compute per-player actual/pmf ratio ---
    # Use OOF rows directly: actual_outcome / pmf_mean per row, then aggregate by role/stat
    oof_df["ratio"] = oof_df["actual_outcome"] / oof_df["pmf_mean"]

    # --- Group by (role_bucket, stat), compute median ratio ---
    results_flat: dict[str, float] = {}
    results_nested: dict[str, dict[str, float]] = {}
    summary_rows = []

    grouped = oof_df.groupby(["role_bucket", "stat"])
    for (role_bucket, stat), grp in grouped:
        n = len(grp)
        global_corr = float(global_corrections.get(stat, 1.0))

        if n < MIN_GROUP_SIZE:
            # Fall back to 1.0 net_mult (no role adjustment) when insufficient data
            net_mult = 1.0
            source = f"fallback_insufficient_data (n={n}<{MIN_GROUP_SIZE})"
        else:
            empirical_ratio = float(np.median(grp["ratio"].values))
            empirical_ratio = float(np.clip(empirical_ratio, CLAMP_LOW, CLAMP_HIGH))
            # Net multiplier: role correction relative to global correction
            net_mult = (empirical_ratio / global_corr) if global_corr > 0.0 else empirical_ratio
            net_mult = float(np.clip(net_mult, CLAMP_LOW, CLAMP_HIGH))
            source = f"empirical (n={n}, empirical_ratio={empirical_ratio:.4f}, global_corr={global_corr:.4f})"

        flat_key = f"{role_bucket}|{stat}"
        results_flat[flat_key] = round(net_mult, 4)
        if role_bucket not in results_nested:
            results_nested[role_bucket] = {}
        results_nested[role_bucket][stat] = round(net_mult, 4)

        summary_rows.append({
            "role_bucket": role_bucket,
            "stat": stat,
            "n": n,
            "net_mult": net_mult,
            "source": source,
        })

    # --- Enforce monotonicity: starter >= core >= rotation >= bench >= fringe ---
    results_nested = enforce_monotonicity(results_nested)
    # Rebuild flat from monotonicity-enforced nested
    for role_bucket, stat_vals in results_nested.items():
        for stat, val in stat_vals.items():
            results_flat[f"{role_bucket}|{stat}"] = round(val, 4)

    # --- Save output ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {
        "_comment": (
            "Role-stratified bias corrections: net_mult = empirical_ratio / global_correction "
            "per (role_bucket, stat), joined on player_id from OOF data. "
            f"Groups with <{MIN_GROUP_SIZE} rows fall back to net_mult=1.0. "
            "Monotonicity enforced: starter >= core >= rotation >= bench >= fringe."
        ),
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corrections": results_flat,
    }
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[compute_role_bias] Saved {len(results_flat)} corrections to {OUTPUT_PATH}")

    # --- Print summary table ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["stat", "role_bucket"])
        print("\n=== Role Bias Corrections Summary (net_mult = empirical/global) ===")
        print(summary_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
