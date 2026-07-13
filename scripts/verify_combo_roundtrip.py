"""Round-trip serialization verification for combo PMFs.

Item 6: Loads the latest full_pmfs_wide.parquet and verifies that JSON
serialization/deserialization preserves pmf_mean and model_prob_over within 1e-8.

Usage:
    python scripts/verify_combo_roundtrip.py
    python scripts/verify_combo_roundtrip.py --delivery-dir deliveries/tonight
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMBO_STATS = {"pts_reb", "pts_ast", "pts_reb_ast", "reb_ast", "stocks"}
MEAN_GATE = 1e-8
POVER_GATE = 1e-8


def run_verification(delivery_dir: Path) -> int:
    pmf_path = delivery_dir / "full_pmfs_wide.parquet"
    edge_path = delivery_dir / "publishable_edges.parquet"

    if not pmf_path.exists():
        print(f"ERROR: full_pmfs_wide.parquet not found at {pmf_path}")
        print("Run the pipeline first to generate deliveries.")
        return 1

    pmfs = pd.read_parquet(pmf_path)
    combo_pmfs = pmfs[pmfs["stat"].isin(COMBO_STATS)].copy()

    if combo_pmfs.empty:
        print("WARNING: No combo PMF rows found in full_pmfs_wide.parquet")
        return 0

    print(f"Checking {len(combo_pmfs)} combo PMF rows across stats: "
          f"{sorted(combo_pmfs['stat'].unique().tolist())}")

    # Load edges for P(over) cross-check
    edges = pd.DataFrame()
    if edge_path.exists():
        try:
            edges = pd.read_parquet(edge_path)
        except Exception as e:
            print(f"WARNING: Could not load publishable_edges.parquet: {e}")

    max_mean_err = 0.0
    max_pover_err = 0.0
    mean_err_by_stat: dict[str, float] = {}
    pover_err_by_stat: dict[str, float] = {}
    errors: list[str] = []

    for _, row in combo_pmfs.iterrows():
        stat = str(row["stat"])
        try:
            d = json.loads(row["pmf_json"])
        except Exception as e:
            errors.append(f"JSON parse error for {row.get('player_id')} {stat}: {e}")
            continue

        ks = np.array([int(k) for k in d.keys()])
        vs = np.array(list(d.values()), dtype=float)
        s = vs.sum()
        if s < 1e-15:
            errors.append(f"Zero-sum PMF for {row.get('player_id')} {stat}")
            continue
        vs /= s

        # Mean round-trip error
        mean_rt = float(ks @ vs)
        mean_stored = float(row.get("pmf_mean", np.nan))
        if not np.isnan(mean_stored):
            err = abs(mean_rt - mean_stored)
            if err > max_mean_err:
                max_mean_err = err
            mean_err_by_stat[stat] = max(mean_err_by_stat.get(stat, 0.0), err)

        # P(over) round-trip error using matching market edges
        if not edges.empty:
            pid = row.get("player_id")
            gid = row.get("game_id")
            match = edges[
                (edges["player_id"] == pid) &
                (edges["game_id"] == gid) &
                (edges["stat"] == stat)
            ]
            for _, e_row in match.iterrows():
                line = float(e_row.get("line", np.nan))
                p_over_stored = float(e_row.get("model_prob_over", np.nan))
                if np.isnan(line) or np.isnan(p_over_stored):
                    continue
                p_over_rt = float(vs[ks > line].sum())
                pover_err = abs(p_over_rt - p_over_stored)
                if pover_err > max_pover_err:
                    max_pover_err = pover_err
                pover_err_by_stat[stat] = max(pover_err_by_stat.get(stat, 0.0), pover_err)

    print()
    print("=== Round-trip validation results ===")
    print(f"Total combo rows checked: {len(combo_pmfs)}")
    print(f"JSON parse errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  {e}")
    print()
    print("Max round-trip errors by stat (mean):")
    for s in sorted(mean_err_by_stat.keys()):
        flag = "FAIL" if mean_err_by_stat[s] > MEAN_GATE else "PASS"
        print(f"  {s:20s}: {mean_err_by_stat[s]:.2e}  [{flag}]")
    if pover_err_by_stat:
        print("\nMax round-trip errors by stat (P(over) at market line):")
        for s in sorted(pover_err_by_stat.keys()):
            flag = "FAIL" if pover_err_by_stat[s] > POVER_GATE else "PASS"
            print(f"  {s:20s}: {pover_err_by_stat[s]:.2e}  [{flag}]")
    else:
        print("\n(No P(over) cross-check: publishable_edges.parquet not found or no matches)")

    print()
    print(f"Max round-trip mean error:    {max_mean_err:.2e}")
    print(f"Max round-trip P(over) error: {max_pover_err:.2e}")
    print(f"Gate: mean   <= {MEAN_GATE:.0e}? "
          f"{'PASS' if max_mean_err <= MEAN_GATE else 'FAIL'}")
    print(f"Gate: p_over <= {POVER_GATE:.0e}? "
          f"{'PASS' if max_pover_err <= POVER_GATE or not pover_err_by_stat else 'FAIL'}")

    failed = (max_mean_err > MEAN_GATE) or (pover_err_by_stat and max_pover_err > POVER_GATE)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delivery-dir",
        default="deliveries/tonight",
        help="Directory containing full_pmfs_wide.parquet and publishable_edges.parquet",
    )
    args = parser.parse_args()
    delivery_dir = Path(args.delivery_dir)
    sys.exit(run_verification(delivery_dir))


if __name__ == "__main__":
    main()
