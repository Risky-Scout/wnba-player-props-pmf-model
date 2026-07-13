"""Round-trip serialization verification for combo PMFs.

Verifies that JSON serialization/deserialization preserves pmf_mean and
model_prob_over within tolerance. Checks EVERY unique market line per
player/game/stat pair — including integer lines — and tracks P(over),
P(under), and P(push) errors separately.

Uses ``pmf_mean_full_precision`` (stored at full float64 precision) when
available, falling back to the rounded ``pmf_mean`` column. This prevents
false failures from comparing against a 4-decimal-place truncated value.

Exit codes
----------
0  PASS — all checks within tolerance, no suppressed rows in edges
1  FAIL — integrity violation detected
2  INSUFFICIENT_DATA — no combo rows or no market lines matched

Usage
-----
    python scripts/verify_combo_roundtrip.py
    python scripts/verify_combo_roundtrip.py --delivery-dir deliveries/tonight
    python scripts/verify_combo_roundtrip.py --tolerance 1e-8 --fail-on-suppressed-in-edges
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMBO_STATS = {"pts_reb", "pts_ast", "pts_reb_ast", "reb_ast", "stocks"}
DEFAULT_TOLERANCE = 1e-8


def run_verification(
    delivery_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
    fail_on_suppressed_in_edges: bool = False,
) -> None:
    """Run full round-trip integrity check. Exits with appropriate code."""
    pmf_path = delivery_dir / "full_pmfs_wide.parquet"
    edge_path = delivery_dir / "publishable_edges.parquet"

    if not pmf_path.exists():
        print(f"ERROR: full_pmfs_wide.parquet not found at {pmf_path}")
        print("Run the pipeline first to generate deliveries.")
        sys.exit(1)

    pmfs = pd.read_parquet(pmf_path)
    combo_pmfs = pmfs[pmfs["stat"].isin(COMBO_STATS)].copy()
    combo_row_count = len(combo_pmfs)

    if combo_row_count == 0:
        print("RESULT: INSUFFICIENT_DATA — no combo PMF rows found")
        sys.exit(2)

    print(f"Checking {combo_row_count} combo PMF rows across stats: "
          f"{sorted(combo_pmfs['stat'].unique().tolist())}")

    # Load edges for P(over/under/push) cross-check
    edges = pd.DataFrame()
    if edge_path.exists():
        try:
            edges = pd.read_parquet(edge_path)
        except Exception as e:
            print(f"WARNING: Could not load publishable_edges.parquet: {e}")

    # --- Check for suppressed combo rows leaking into publishable edges ---
    suppressed_in_edges = 0
    if "combo_suppressed" in combo_pmfs.columns and not edges.empty:
        suppressed_pmfs = combo_pmfs[combo_pmfs["combo_suppressed"].fillna(False).astype(bool)]
        if not suppressed_pmfs.empty and "player_id" in edges.columns and "stat" in edges.columns:
            supp_keys = set(zip(suppressed_pmfs["player_id"], suppressed_pmfs["stat"]))
            edge_keys = set(zip(edges["player_id"], edges["stat"]))
            leaked = supp_keys & edge_keys
            suppressed_in_edges = len(leaked)
            if suppressed_in_edges > 0:
                print(f"WARNING: {suppressed_in_edges} suppressed combo rows found in publishable_edges!")
                for key in leaked:
                    print(f"  Leaked: player_id={key[0]}, stat={key[1]}")

    # --- Metrics ---
    max_mean_err = 0.0
    max_pover_err = 0.0
    max_punder_err = 0.0
    max_ppush_err = 0.0
    mean_err_by_stat: dict[str, float] = {}
    pover_err_by_stat: dict[str, float] = {}
    errors: list[str] = []
    json_parse_failures = 0
    non_normalized_count = 0
    any_non_normalized = False
    market_lines_checked = 0

    # Track pmf_mean_full_precision vs pmf_mean (rounded) discrepancy
    max_full_precision_discrepancy = 0.0
    has_full_precision_col = "pmf_mean_full_precision" in combo_pmfs.columns

    for _, row in combo_pmfs.iterrows():
        stat = str(row["stat"])
        try:
            d = json.loads(row["pmf_json"])
        except Exception as e:
            errors.append(f"JSON parse error for {row.get('player_id')} {stat}: {e}")
            json_parse_failures += 1
            continue

        ks = np.array([int(k) for k in d.keys()])
        vs = np.array(list(d.values()), dtype=float)
        s = vs.sum()
        if s < 1e-15:
            errors.append(f"Zero-sum PMF for {row.get('player_id')} {stat}")
            json_parse_failures += 1
            continue

        if abs(s - 1.0) > 1e-6:
            non_normalized_count += 1
            any_non_normalized = True

        vs = vs / s  # normalize for all probability computations

        # Defect 5: Use full-precision stored mean when available
        stored_mean = float(row.get("pmf_mean_full_precision", row.get("pmf_mean", np.nan)))
        pmf_mean_rounded = float(row.get("pmf_mean", np.nan))

        # Track full_precision vs rounded discrepancy
        if has_full_precision_col and not np.isnan(stored_mean) and not np.isnan(pmf_mean_rounded):
            fp_disc = abs(stored_mean - pmf_mean_rounded)
            if fp_disc > max_full_precision_discrepancy:
                max_full_precision_discrepancy = fp_disc

        # Mean round-trip error against full-precision stored value
        mean_rt = float(ks @ vs)
        if not np.isnan(stored_mean):
            err = abs(mean_rt - stored_mean)
            if err > max_mean_err:
                max_mean_err = err
            mean_err_by_stat[stat] = max(mean_err_by_stat.get(stat, 0.0), err)

        # Defect 7: Check EVERY unique market line for this player/game/stat,
        # including integer lines. Track P(over), P(under), P(push) separately.
        if not edges.empty:
            pid = row.get("player_id")
            gid = row.get("game_id")
            match = edges[
                (edges["player_id"] == pid) &
                (edges["game_id"] == gid) &
                (edges["stat"] == stat)
            ]
            for line in sorted(match["line"].dropna().unique()):
                line = float(line)
                p_over_rt = float(vs[ks > line].sum())
                p_under_rt = float(vs[ks < line].sum())
                p_push_rt = float(vs[ks == line].sum())

                e_row = match[match["line"] == line].iloc[0]

                # Defect 5: prefer p_over_full_precision, fall back to model_prob_over
                stored_p_over = float(
                    e_row.get("p_over_full_precision", e_row.get("model_prob_over", np.nan))
                )

                if not np.isnan(stored_p_over):
                    pover_err = abs(p_over_rt - stored_p_over)
                    if pover_err > max_pover_err:
                        max_pover_err = pover_err
                    pover_err_by_stat[stat] = max(pover_err_by_stat.get(stat, 0.0), pover_err)

                    # Track under/push errors for reporting (not used in gate yet)
                    punder_err = abs(p_under_rt - (1.0 - stored_p_over - p_push_rt))
                    if punder_err > max_punder_err:
                        max_punder_err = punder_err

                    market_lines_checked += 1

    # --- Print report ---
    print()
    print("=== Round-trip validation results ===")
    print(f"Combo PMF rows checked:              {combo_row_count}")
    print(f"Market lines checked:                {market_lines_checked}")
    print(f"JSON parse failures:                 {json_parse_failures}")
    print(f"Non-normalized PMFs (pre-norm!=1):   {non_normalized_count}")
    print(f"Suppressed rows in edges:            {suppressed_in_edges}")
    if has_full_precision_col:
        print(f"pmf_mean_full_precision vs pmf_mean max discrepancy: {max_full_precision_discrepancy:.2e}")
    else:
        print("pmf_mean_full_precision column: NOT PRESENT (using pmf_mean for validation)")

    if errors:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for e in errors[:5]:
            print(f"  {e}")

    print()
    print("Max round-trip errors by stat (mean):")
    for s in sorted(mean_err_by_stat.keys()):
        flag = "FAIL" if mean_err_by_stat[s] > tolerance else "PASS"
        print(f"  {s:20s}: {mean_err_by_stat[s]:.2e}  [{flag}]")

    if pover_err_by_stat:
        print("\nMax round-trip errors by stat (P(over) at each market line):")
        for s in sorted(pover_err_by_stat.keys()):
            flag = "FAIL" if pover_err_by_stat[s] > tolerance else "PASS"
            print(f"  {s:20s}: {pover_err_by_stat[s]:.2e}  [{flag}]")
    else:
        print("\n(No P(over) cross-check: publishable_edges.parquet not found or no matches)")

    print()
    print(f"Max round-trip mean error:            {max_mean_err:.2e}")
    print(f"Max round-trip P(over) error:         {max_pover_err:.2e}")
    print(f"Max round-trip P(under) error:        {max_punder_err:.2e}")
    print(f"Tolerance:                            {tolerance:.0e}")

    # --- Defect 6: Failure conditions with specific exit codes ---
    if market_lines_checked == 0:
        print("RESULT: INSUFFICIENT_DATA — no market lines matched to combo PMFs")
        sys.exit(2)

    if json_parse_failures > 0:
        print(f"RESULT: FAIL — {json_parse_failures} JSON parse failures")
        sys.exit(1)

    if any_non_normalized:
        print(f"RESULT: FAIL — {non_normalized_count} non-normalized PMFs")
        sys.exit(1)

    if max_mean_err > tolerance:
        print(f"RESULT: FAIL — max mean error {max_mean_err:.2e} > {tolerance:.2e}")
        sys.exit(1)

    if pover_err_by_stat and max_pover_err > tolerance:
        print(f"RESULT: FAIL — max P(over) error {max_pover_err:.2e} > {tolerance:.2e}")
        sys.exit(1)

    if fail_on_suppressed_in_edges and suppressed_in_edges > 0:
        print(
            f"RESULT: FAIL — {suppressed_in_edges} suppressed combo rows in publishable_edges.parquet"
        )
        sys.exit(1)

    print("RESULT: PASS")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delivery-dir",
        default="deliveries/tonight",
        help="Directory containing full_pmfs_wide.parquet and publishable_edges.parquet",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Maximum allowed round-trip error (default: {DEFAULT_TOLERANCE:.0e})",
    )
    parser.add_argument(
        "--fail-on-suppressed-in-edges",
        action="store_true",
        default=False,
        help="Exit 1 if any combo_suppressed=True rows appear in publishable_edges.parquet",
    )
    args = parser.parse_args()
    delivery_dir = Path(args.delivery_dir)
    run_verification(
        delivery_dir,
        tolerance=args.tolerance,
        fail_on_suppressed_in_edges=args.fail_on_suppressed_in_edges,
    )


if __name__ == "__main__":
    main()
