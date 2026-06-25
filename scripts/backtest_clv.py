"""Compute CLV (Closing Line Value) for historical model projections (Blueprint R6.2).

CLV = did the model's edge direction match the direction the line moved at close?
If the model said OVER with edge >= 4% and the closing line moved UP, that is a
CLV hit — the model "beat the close."

Usage:
    python scripts/backtest_clv.py \\
        --comparisons-dir data/delivery/ \\
        --out artifacts/audits/clv_backtest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_clv(comp: pd.DataFrame, threshold: float = 0.04) -> dict:
    """Compute CLV hit rate from a market comparison DataFrame.

    A prop "beats the close" when:
      - Model had edge_over >= threshold AND closing line moved up (OVER direction), OR
      - Model had edge_under >= threshold AND closing line moved down (UNDER direction).

    Line movement is measured as (closing_line - opening_line); positive = line moved up.
    """
    if comp.empty or "edge_over" not in comp.columns:
        return {"clv_hit_rate": float("nan"), "n_props": 0, "n_significant_edges": 0,
                "n_clv_hits": 0, "over_edges": 0, "under_edges": 0}

    model_direction = np.sign(comp["edge_over"])

    if "line_delta" in comp.columns:
        line_move = np.sign(comp["line_delta"].fillna(0))
    else:
        line_move = np.zeros(len(comp))

    significant = comp["edge_over"].abs() >= threshold
    clv_hit = (model_direction == line_move) & significant & (line_move != 0)

    n_significant = int(significant.sum())
    n_hits = int(clv_hit.sum())
    hit_rate = n_hits / n_significant if n_significant > 0 else float("nan")

    return {
        "clv_hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else float("nan"),
        "n_props": len(comp),
        "n_significant_edges": n_significant,
        "n_clv_hits": n_hits,
        "over_edges": int((comp["edge_over"] > threshold).sum()),
        "under_edges": int((comp["edge_over"] < -threshold).sum()),
    }


def main(comparisons_dir: str = "data/delivery/", out_path: str = "artifacts/audits/clv_backtest.json") -> None:
    comp_files = sorted(Path(comparisons_dir).rglob("market_comparison.parquet"))
    if not comp_files:
        print("No market_comparison.parquet files found — skipping CLV backtest")
        return

    all_results: list[dict] = []
    for f in comp_files:
        try:
            comp = pd.read_parquet(f)
            result = compute_clv(comp)
            result["file"] = str(f)
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not process {f}: {exc}")

    total_significant = sum(r["n_significant_edges"] for r in all_results)
    total_hits = sum(r["n_clv_hits"] for r in all_results)
    agg_rate = total_hits / total_significant if total_significant > 0 else float("nan")

    summary = {
        "aggregate_clv_hit_rate": round(agg_rate, 4) if not np.isnan(agg_rate) else None,
        "target_clv_hit_rate": 0.75,
        "meets_target": (agg_rate >= 0.75) if not np.isnan(agg_rate) else None,
        "total_significant_edges": total_significant,
        "total_clv_hits": total_hits,
        "n_files": len(all_results),
        "per_file": all_results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))

    if np.isnan(agg_rate):
        print("CLV backtest: no line-delta data available — cannot compute hit rate")
        print("  (line_delta column is populated by pull_closing_lines.py)")
    elif agg_rate >= 0.75:
        print(f"CLV backtest: {agg_rate:.1%} hit rate ({total_hits}/{total_significant}) — PASS ✓")
    else:
        print(f"CLV backtest: {agg_rate:.1%} hit rate ({total_hits}/{total_significant}) — FAIL (need ≥75%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute CLV hit rate from historical market comparisons")
    parser.add_argument("--comparisons-dir", default="data/delivery/", help="Directory tree to search for market_comparison.parquet files")
    parser.add_argument("--out", default="artifacts/audits/clv_backtest.json", help="Output path for JSON summary")
    args = parser.parse_args()
    main(args.comparisons_dir, args.out)
