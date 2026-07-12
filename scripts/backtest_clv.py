"""Compute model-edge-vs-open agreement rate from historical projections.

IMPORTANT NAMING NOTE
---------------------
The metric previously called "clv_hit_rate" was NOT closing-line value (CLV).
It measured whether the model's edge direction matched the direction of line
movement between opening and closing quote. This is renamed to:

  model_edge_vs_open_agreement_rate

True CLV requires:
  1. An archived closing quote (pulled before tip-off, from same vendor).
  2. The closing no-vig probability at the entry line.
  3. A valid entry-time quote.

Without archived closing quotes, true CLV cannot be computed.
Use scripts/generate_clv_report.py + src/wnba_props_model/evaluation/clv.py
to compute true CLV when closing snapshots are available.

Usage:
    python scripts/backtest_clv.py \\
        --comparisons-dir data/delivery/ \\
        --out artifacts/audits/model_edge_agreement_backtest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_model_edge_agreement(comp: pd.DataFrame, threshold: float = 0.04) -> dict:
    """Compute model-edge-vs-open agreement rate.

    This metric measures: when the model projected a significant edge in one
    direction, did the opening-to-closing line movement agree?

    THIS IS NOT CLV. True CLV requires a closing quote on the same line,
    compared against the actual entry-time price.

    A prop shows agreement when:
      - Model had edge_over >= threshold AND line moved up (OVER direction), OR
      - Model had edge_under >= threshold AND line moved down (UNDER direction).

    Line movement is (closing_line - opening_line); positive = line moved up.
    """
    if comp.empty or "edge_over" not in comp.columns:
        return {
            "model_edge_vs_open_agreement_rate": float("nan"),
            "metric_note": "NOT_CLV: this is model_edge_direction vs opening_line_movement",
            "n_props": 0,
            "n_significant_edges": 0,
            "n_agreement_hits": 0,
            "over_edges": 0,
            "under_edges": 0,
        }

    model_direction = np.sign(comp["edge_over"])

    if "line_delta" in comp.columns:
        line_move = np.sign(comp["line_delta"].fillna(0))
    else:
        line_move = np.zeros(len(comp))

    significant = comp["edge_over"].abs() >= threshold
    agreement_hit = (model_direction == line_move) & significant & (line_move != 0)

    n_significant = int(significant.sum())
    n_hits = int(agreement_hit.sum())
    hit_rate = n_hits / n_significant if n_significant > 0 else float("nan")

    return {
        "model_edge_vs_open_agreement_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else float("nan"),
        "metric_note": "NOT_CLV: direction agreement between model edge and opening→closing line movement",
        "n_props": len(comp),
        "n_significant_edges": n_significant,
        "n_agreement_hits": n_hits,
        "over_edges": int((comp["edge_over"] > threshold).sum()),
        "under_edges": int((comp["edge_over"] < -threshold).sum()),
    }


def main(comparisons_dir: str = "data/delivery/", out_path: str = "artifacts/audits/model_edge_agreement_backtest.json") -> None:
    comp_files = sorted(Path(comparisons_dir).rglob("market_comparison.parquet"))
    if not comp_files:
        print("No market_comparison.parquet files found — skipping model-edge agreement backtest")
        return

    all_results: list[dict] = []
    for f in comp_files:
        try:
            comp = pd.read_parquet(f)
            result = compute_model_edge_agreement(comp)
            result["file"] = str(f)
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not process {f}: {exc}")

    total_significant = sum(r["n_significant_edges"] for r in all_results)
    total_hits = sum(r["n_agreement_hits"] for r in all_results)
    agg_rate = total_hits / total_significant if total_significant > 0 else float("nan")

    summary = {
        "metric": "model_edge_vs_open_agreement_rate",
        "metric_note": (
            "NOT CLV. This measures whether the model's significant-edge direction matched "
            "the direction of opening-to-closing line movement. True CLV requires archived "
            "closing quotes and entry-time prices. See evaluation/clv.py for true CLV."
        ),
        "aggregate_model_edge_agreement_rate": round(agg_rate, 4) if not np.isnan(agg_rate) else None,
        "total_significant_edges": total_significant,
        "total_agreement_hits": total_hits,
        "n_files": len(all_results),
        "true_clv_available": False,
        "true_clv_note": "Requires archived closing quotes from pull_closing_lines.py + append-only ledger",
        "per_file": all_results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))

    if np.isnan(agg_rate):
        print("Model-edge agreement backtest: no line-delta data available — cannot compute rate")
        print("  (line_delta populated by pull_closing_lines.py)")
    else:
        print(f"Model-edge agreement rate: {agg_rate:.1%} ({total_hits}/{total_significant})")
        print("  NOTE: This is NOT CLV. See artifacts/audits/model_edge_agreement_backtest.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute model-edge-vs-open agreement rate. "
            "NOTE: This metric was formerly (incorrectly) called CLV. "
            "It measures direction agreement, not true closing-line value."
        )
    )
    parser.add_argument("--comparisons-dir", default="data/delivery/",
                        help="Directory tree to search for market_comparison.parquet files")
    parser.add_argument("--out", default="artifacts/audits/model_edge_agreement_backtest.json",
                        help="Output path for JSON summary")
    args = parser.parse_args()
    main(args.comparisons_dir, args.out)
