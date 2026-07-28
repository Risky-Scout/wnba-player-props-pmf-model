#!/usr/bin/env python3
"""Run the soft-book / MARKET_DISLOCATION CLV backtest and persist the artifacts.

Replays ``scan_soft_book_edges`` at the decision snapshot over every historical slate in the
quotes parquet, computes Closing Line Value (price CLV vs the closing consensus, and
same-book CLV) for each flagged +EV candidate, aggregates by market / book / EV bucket /
market×EV-bucket with a date-cluster bootstrap 95% CI, and writes:

  * ``artifacts/path_b/CLV_BACKTEST.json``          — full methodology + per-segment tables +
                                                       overall verdict + honest limitations.
  * ``artifacts/path_b/CLV_VALIDATION_TABLE.json``   — the compact table the edge board and
                                                       acceptance gate consume to set actionable.

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/run_clv_backtest.py \
      --quotes artifacts/p1/p1_quotes.parquet --out-dir artifacts/path_b
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from wnba_props_model.edge.clv_backtest import (
    DEFAULT_BOOTSTRAP_ITERS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MIN_SEGMENT_N,
    EV_BUCKETS,
    MIN_DATE_CLUSTERS,
    PRIMARY_METRIC,
    load_quotes,
    run_backtest,
)
from wnba_props_model.edge.soft_book_scan import (
    DEFAULT_EV_THRESHOLD,
    DEFAULT_MIN_CONSENSUS_BOOKS,
)


def _methodology(args) -> dict:
    return {
        "goal": (
            "Determine whether soft-book / MARKET_DISLOCATION edges are actionable by measuring "
            "Closing Line Value (CLV) — the gold-standard proxy that a betting strategy is +EV."
        ),
        "replay": (
            "scan_soft_book_edges is replayed unchanged at the DECISION snapshot per game_date "
            "(falling back to OPEN for props with no decision snapshot). Same rigor as the "
            "acceptance gate: exact identity match, atomic line matching (standard never mixed "
            "with *_alternate), leave-one-book-out no-vig consensus (candidate book excluded), "
            "Shin de-vig of the same book's over+under (fail closed if a side is missing), and a "
            "min-consensus-books quality floor. Live-freshness gates (commence staleness / quote "
            "age) are disabled because this is a historical replay."
        ),
        "price_clv": (
            "closing no-vig CONSENSUS P(side) minus the candidate book's own decision no-vig "
            "P(side). Positive => the market closed at a higher fair probability for our side "
            "than the price we took => we beat the close."
        ),
        "same_book_clv": (
            "the SAME book's closing no-vig P(side) minus its decision no-vig P(side). Fails "
            "closed (excluded) when the book has no two-sided closing price for the prop."
        ),
        "units": "CLV reported in probability terms and in percentage points ('cents' = prob * 100).",
        "ev_buckets": [f"{lo:.3f}<=EV<{hi}" for lo, hi, _ in EV_BUCKETS],
        "bootstrap": (
            f"date-cluster bootstrap of the mean: {args.bootstrap_iters} resamples of the "
            f"game_date clusters with replacement (seed {args.bootstrap_seed}); 95% CI from the "
            f"2.5/97.5 percentiles. A segment needs >= {MIN_DATE_CLUSTERS} date clusters."
        ),
        "actionability_rule": (
            "Fail closed: a segment (market, or market x EV-bucket) is actionable iff mean "
            f"price_clv > 0 AND its bootstrap 95% CI excludes 0 AND N >= {args.min_segment_n}. "
            "A board row is actionable iff its market x EV-bucket OR its market segment qualifies."
        ),
        "primary_metric": PRIMARY_METRIC,
    }


def _overall_verdict(segments_price: dict, validation_table: dict, min_segment_n: int) -> dict:
    ov = segments_price["overall"]
    actionable = validation_table["actionable_segments"]
    return {
        "overall_price_clv_mean": ov["mean"],
        "overall_price_clv_mean_cents": ov.get("mean_cents"),
        "overall_price_clv_median": ov["median"],
        "overall_pct_beat_close": ov["pct_beat_close"],
        "overall_ci_low": ov["ci_low"],
        "overall_ci_high": ov["ci_high"],
        "overall_ci_low_cents": ov.get("ci_low_cents"),
        "overall_ci_high_cents": ov.get("ci_high_cents"),
        "overall_n": ov["n"],
        "overall_n_dates": ov["n_dates"],
        "overall_significant": ov["significant"],
        "overall_meets_min_n": bool(ov["n"] >= min_segment_n),
        "actionable_segments": actionable,
        "any_actionable": len(actionable) > 0,
        "verdict": _verdict_text(ov, actionable, min_segment_n),
    }


def _verdict_text(ov: dict, actionable: list, min_segment_n: int) -> str:
    if actionable:
        return (
            f"POSITIVE & ACTIONABLE: {len(actionable)} segment(s) show positive mean price CLV "
            f"with a 95% CI excluding 0 and N >= {min_segment_n}: {', '.join(actionable)}."
        )
    if ov["significant"] and ov["n"] < min_segment_n:
        return (
            f"POSITIVE OVERALL BUT UNDERPOWERED PER SEGMENT: the overall strategy shows a "
            f"significant positive mean price CLV of {ov['mean_cents']:+.2f} cents "
            f"(95% CI [{ov['ci_low_cents']:+.2f}, {ov['ci_high_cents']:+.2f}] cents, "
            f"{ov['pct_beat_close']:.0f}% beat close) across N={ov['n']} flagged bets on "
            f"{ov['n_dates']} dates, but NO market segment reaches the fail-closed bar of "
            f"N >= {min_segment_n}. Under the fail-closed rule NO rows are actionable; the "
            f"signal is real but needs more slates to validate per-segment."
        )
    if ov["significant"]:
        return (
            f"POSITIVE OVERALL: mean price CLV {ov['mean_cents']:+.2f} cents (CI excludes 0), "
            f"but no per-segment qualifier — NO rows actionable (fail closed)."
        )
    return (
        "NOT SIGNIFICANT: overall price CLV is not positive with a CI excluding 0. "
        "NO rows are actionable (fail closed)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quotes", default="artifacts/p1/p1_quotes.parquet")
    ap.add_argument("--out-dir", default="artifacts/path_b")
    ap.add_argument("--ev-threshold", type=float, default=DEFAULT_EV_THRESHOLD)
    ap.add_argument("--min-consensus-books", type=int, default=DEFAULT_MIN_CONSENSUS_BOOKS)
    ap.add_argument("--min-segment-n", type=int, default=DEFAULT_MIN_SEGMENT_N,
                    help="Minimum realistic sample per segment for actionability (fail closed).")
    ap.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--clv-rows-out", default=None,
                    help="Optional path to also write the per-candidate CLV rows as CSV.")
    args = ap.parse_args()

    quotes = load_quotes(args.quotes)
    result = run_backtest(
        quotes,
        ev_threshold=args.ev_threshold,
        min_consensus_books=args.min_consensus_books,
        min_segment_n=args.min_segment_n,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    limitations = [
        f"Only {len(result['coverage']['books'])} books in the panel "
        f"({', '.join(result['coverage']['books'])}) — the no-vig consensus is thin and its "
        "leave-one-out subset is thinner; a single mispriced book moves it more than in a "
        "10+ book market.",
        f"Only {result['coverage']['n_game_dates_total']} game dates of history — the "
        "date-cluster bootstrap has few clusters, so per-segment CIs are wide and per-market "
        "samples are small.",
        "The closing snapshot is the last collected price, a proxy for the true settle-time "
        "close; a book that pulled a market before tip has no closing row (excluded, not "
        "imputed).",
        "Consensus treats all books as equally informative (median of all two-sided books). "
        "Sharp books are annotated but not up-weighted, so 'beating the close' is beating a "
        "median-of-all close, not a Pinnacle close.",
        "CLV is a proxy for +EV, not realized profit: it ignores bet availability at the quoted "
        "price, limits, and vig actually paid on settlement.",
    ]

    report = {
        "schema_version": "clv_backtest_v1",
        "source_type": "MARKET_DISLOCATION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quotes_path": args.quotes,
        "config": {
            "ev_threshold": args.ev_threshold,
            "min_consensus_books": args.min_consensus_books,
            "min_segment_n": args.min_segment_n,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "methodology": _methodology(args),
        "coverage": result["coverage"],
        "verdict": _overall_verdict(
            result["segments_price_clv"], result["validation_table"], args.min_segment_n
        ),
        "segments_price_clv": result["segments_price_clv"],
        "segments_same_book_clv": result["segments_same_book_clv"],
        "limitations": limitations,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backtest_path = out_dir / "CLV_BACKTEST.json"
    table_path = out_dir / "CLV_VALIDATION_TABLE.json"
    backtest_path.write_text(json.dumps(report, indent=2, default=str))
    table_path.write_text(json.dumps(result["validation_table"], indent=2, default=str))

    if args.clv_rows_out:
        rows_path = Path(args.clv_rows_out)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        result["clv_rows"].to_csv(rows_path, index=False)

    v = report["verdict"]
    print(json.dumps({
        "backtest": str(backtest_path),
        "validation_table": str(table_path),
        "n_flagged_candidates": result["coverage"]["n_flagged_candidates"],
        "n_with_close_consensus": result["coverage"]["n_with_close_consensus"],
        "overall_price_clv_mean_cents": v["overall_price_clv_mean_cents"],
        "overall_ci_cents": [v["overall_ci_low_cents"], v["overall_ci_high_cents"]],
        "overall_pct_beat_close": v["overall_pct_beat_close"],
        "overall_significant": v["overall_significant"],
        "actionable_segments": v["actionable_segments"],
    }, indent=2))
    print("\nVERDICT:", v["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
