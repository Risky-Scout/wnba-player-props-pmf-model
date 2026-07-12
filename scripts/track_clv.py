"""Track true Closing Line Value (CLV) from archived quote snapshots.

This script computes true CLV for historical model predictions, using
an append-only quote ledger. CLV is only available where closing snapshots
were archived before game tip-off.

IMPORTANT: The metric computed here is TRUE CLV (closing quote vs entry quote).
This is different from model_edge_at_entry or directional line movement agreement.

See docs/clv_methodology.md for full definitions.

Usage:
    python scripts/track_clv.py \\
        --ledger-path data/processed/quote_ledger.parquet \\
        --predictions-dir deliveries/ \\
        --out artifacts/audits/clv_tracking.json

Requirements:
    - An archived quote ledger built by pull_closing_lines.py
    - Historical predictions with entry-time market quotes
    - Scheduled game start times for temporal validation

Without these inputs, the script reports INSUFFICIENT_DATA for all metrics.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.evaluation.clv import (
    NOT_AVAILABLE,
    CLVResult,
    compute_clv_for_bet,
    select_closing_quote,
)
from wnba_props_model.pipeline.safety import american_to_no_vig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def track_clv(
    ledger_path: str | None,
    predictions_dir: str,
    out_path: str = "artifacts/audits/clv_tracking.json",
) -> dict:
    """Compute true CLV from archived snapshots.

    Returns summary dict (also written to out_path).
    """
    summary: dict = {
        "metric": "true_clv",
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "True CLV requires archived closing quotes from pull_closing_lines.py. "
            "Without these, all results are NOT_AVAILABLE."
        ),
        "true_clv_available": False,
        "same_line_price_clv_results": [],
        "line_clv_results": [],
        "n_bets_analyzed": 0,
        "n_same_line_clv_available": 0,
        "n_line_clv_available": 0,
        "mean_same_line_price_clv": None,
        "mean_line_clv_over_bets": None,
        "mean_model_edge_at_entry": None,
        "status": "INSUFFICIENT_DATA",
    }

    # Load ledger
    ledger = None
    if ledger_path and Path(ledger_path).exists():
        try:
            ledger = pd.read_parquet(ledger_path)
            logger.info("Loaded quote ledger: %d rows from %s", len(ledger), ledger_path)
            summary["true_clv_available"] = True
        except Exception as exc:
            logger.warning("Could not load ledger: %s", exc)

    if ledger is None or ledger.empty:
        logger.info("No quote ledger available — CLV cannot be computed")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str))
        return summary

    # Load historical predictions
    pred_files = sorted(Path(predictions_dir).rglob("market_comparison.parquet"))
    if not pred_files:
        logger.info("No market_comparison.parquet files found in %s", predictions_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str))
        return summary

    all_preds = []
    for f in pred_files:
        try:
            df = pd.read_parquet(f)
            all_preds.append(df)
        except Exception as exc:
            logger.warning("Could not read %s: %s", f, exc)

    if not all_preds:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str))
        return summary

    preds = pd.concat(all_preds, ignore_index=True)
    logger.info("Loaded %d prediction rows from %d files", len(preds), len(pred_files))

    results: list[CLVResult] = []

    required_cols = {"player_id", "game_id", "stat", "vendor", "line",
                     "over_odds", "under_odds", "scheduled_start_utc"}
    if not required_cols.issubset(preds.columns):
        missing = required_cols - set(preds.columns)
        logger.warning("Prediction table missing required columns for CLV: %s", missing)
        summary["status"] = "INSUFFICIENT_DATA"
        summary["missing_columns"] = list(missing)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str))
        return summary

    for _, row in preds.iterrows():
        entry_row = row.to_dict()

        # Determine side from edge
        model_p_over = float(row.get("model_p_over_calibrated", row.get("p_over", float("nan"))))
        side = "over" if float(row.get("edge_over", 0.0)) >= 0 else "under"

        # Look up closing quote
        closing_row = select_closing_quote(
            ledger,
            game_id=row.get("game_id"),
            player_id=row.get("player_id"),
            stat=row.get("stat", ""),
            market_type=row.get("market_type", "player_prop"),
            vendor=row.get("vendor", ""),
            scheduled_start_utc=str(row.get("scheduled_start_utc", "")),
        )

        clv_result = compute_clv_for_bet(
            entry_row,
            closing_row.to_dict() if closing_row is not None else None,
            model_p_over,
            side,
            scheduled_start_utc=str(row.get("scheduled_start_utc", "")),
        )
        results.append(clv_result)

    # Aggregate
    n_bets = len(results)
    same_line_clvs = [r.same_line_price_clv for r in results if isinstance(r.same_line_price_clv, float)]
    line_clvs = [r.line_clv for r in results if isinstance(r.line_clv, float)]
    model_edges = [r.model_edge_at_entry for r in results if not np.isnan(r.model_edge_at_entry)]

    summary.update({
        "true_clv_available": True,
        "n_bets_analyzed": n_bets,
        "n_same_line_clv_available": len(same_line_clvs),
        "n_line_clv_available": len(line_clvs),
        "mean_same_line_price_clv": float(np.mean(same_line_clvs)) if same_line_clvs else None,
        "mean_line_clv": float(np.mean(line_clvs)) if line_clvs else None,
        "mean_model_edge_at_entry": float(np.mean(model_edges)) if model_edges else None,
        "clv_note": (
            "mean_model_edge_at_entry is NOT CLV — it is model probability minus entry-time market price. "
            "mean_same_line_price_clv is true price CLV where lines matched."
        ),
        "status": "AVAILABLE" if same_line_clvs else "INSUFFICIENT_DATA",
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "CLV tracking complete: %d bets, %d with same-line price CLV, mean_clv=%.4f",
        n_bets,
        len(same_line_clvs),
        float(np.mean(same_line_clvs)) if same_line_clvs else float("nan"),
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track true CLV from archived quote snapshots")
    parser.add_argument("--ledger-path", default=None, help="Path to append-only quote ledger parquet")
    parser.add_argument("--predictions-dir", default="deliveries/", help="Directory to search for market_comparison.parquet files")
    parser.add_argument("--out", default="artifacts/audits/clv_tracking.json", help="Output JSON path")
    args = parser.parse_args()

    result = track_clv(args.ledger_path, args.predictions_dir, args.out)
    print(f"CLV status: {result['status']}")
    if result.get("mean_same_line_price_clv") is not None:
        print(f"Mean same-line price CLV: {result['mean_same_line_price_clv']:.4f}")
    else:
        print("True CLV: NOT_AVAILABLE (requires archived closing quotes)")
