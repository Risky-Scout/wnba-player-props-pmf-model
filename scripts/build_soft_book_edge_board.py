#!/usr/bin/env python3
"""Build the soft-book +EV edge board (Path B, Definition B) from collected quotes.

Reads an atomic two-sided quote snapshot (from ``collect_soft_book_quotes.py``), runs
``scan_soft_book_edges``, and writes TWO artifacts:

  1. ``artifacts/edge_board/SOFT_BOOK_EDGE_<date>.json`` — the authoritative tidy board
     (player, team, stat, line, side, book, offered odds, fair_p, EV%, consensus_n_books,
     best-available book/odds), plus method + summary blocks.
  2. ``tools/odds-scanner/predictions/WNBA/Soft-Book-Edge/{latest,<date>}.json`` — the SAME
     board reshaped into the odds-scanner render schema (games -> players ->
     stat_projections.calibrated_p_over) so the static board renders it.

Fail-open: a missing snapshot / empty board writes honest zero-edge artifacts and exits 0.

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/build_soft_book_edge_board.py \
      --quotes data/snapshots/soft_book_quotes/snapshot_date_utc=2026-07-28/quotes_*.parquet
  # or auto-discover the latest snapshot for a date:
  PYTHONPATH=$(pwd)/src python3 scripts/build_soft_book_edge_board.py --date 2026-07-28
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.edge.clv_backtest import apply_validation_table_to_board
from wnba_props_model.edge.soft_book_scan import (
    DEFAULT_EV_THRESHOLD,
    DEFAULT_MIN_CONSENSUS_BOOKS,
    SHARP_BOOKS,
    scan_soft_book_edges,
)

BOARD_VERSION = "soft_book_edge_v1"
DEFAULT_CLV_VALIDATION_TABLE = "artifacts/path_b/CLV_VALIDATION_TABLE.json"

# internal stat key -> odds-scanner frontend stat key
_STAT_TO_FRONTEND = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "threes",
    "stl": "steals",
    "blk": "blocks",
    "turnover": "turnovers",
}

_TIDY_COLS = [
    "player_name", "team", "stat", "line", "side", "book", "offered_odds",
    "fair_p", "ev_pct", "consensus_n_books", "consensus_p_over", "is_sharp_book",
    "sharp_consensus_p_over", "best_book", "best_odds", "event_id",
    "home_team", "away_team", "commence_time",
    "actionable", "actionable_reason", "clv_segment", "clv_mean", "clv_ci_low",
]

_METHOD_BLOCK = {
    "definition": (
        "Definition B: soft book vs sharp consensus. Find individual books whose posted "
        "price is better for the bettor than the no-vig consensus of the wider book set. "
        "No model and no information edge is used — pure market-vs-market line shopping."
    ),
    "devig": "Per-book two-sided de-vig via Shin's method (shin_no_vig_two_way).",
    "consensus": (
        "Consensus fair P(over) per (event, player, stat, line) = MEDIAN of the per-book "
        "no-vig P(over) across all books posting a two-sided price. The scored book is "
        "EXCLUDED from its own consensus (no self-reference)."
    ),
    "ev_formula": "EV = fair_p * decimal_profit - (1 - fair_p); positive EV beats the fair line.",
    "sharp_books_note": (
        "Known-sharper books (pinnacle, betonlineag, lowvig) are ANNOTATED (is_sharp_book, "
        "sharp_consensus_p_over) but the qualifying EV is computed against the median-of-all "
        "consensus, which is robust to a single outlier without hard-coding sharpness."
    ),
    "guards": (
        "Require >= min_consensus_books after self-exclusion, valid |American odds| >= 100, "
        "both sides present for the scored book, and drop stale (commence_time in the past)."
    ),
    "render_note": (
        "In the odds-scanner render JSON, calibrated_p_over.edge_vs_market carries the SIGNED "
        "EV (positive for over plays -> green OVER badge, negative for under plays -> red UNDER "
        "badge); the |value| is the EV. p_over is always the no-vig consensus P(over). The card "
        "meta (position field) shows 'book odds · EV +X%' with the EV as a positive magnitude."
    ),
}


def _load_validation_table(path: str) -> dict | None:
    """Load the persisted CLV validation table (fail closed: missing/unreadable => None,
    which makes every row actionable=false with reason ``no_validation_table``)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_quotes(args) -> pd.DataFrame | None:
    paths: list[str] = []
    if args.quotes:
        paths = sorted(glob.glob(args.quotes))
    else:
        pattern = str(Path(args.snapshot_dir) / f"snapshot_date_utc={args.date}" / "quotes_*.parquet")
        paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    # Use the most recent snapshot file (last by sorted timestamped name).
    return pd.read_parquet(paths[-1])


def _to_render_json(board_q: pd.DataFrame, date: str, generated: str, summary: dict) -> dict:
    games: dict[str, dict] = {}
    for _, r in board_q.iterrows():
        eid = str(r["event_id"])
        if eid not in games:
            home, away = r.get("home_team") or "", r.get("away_team") or ""
            games[eid] = {
                "game_id": eid,
                "odds_api_event_id": eid,
                "commence_time": r.get("commence_time"),
                "home_team": {"name": home, "abbreviation": home},
                "away_team": {"name": away, "abbreviation": away},
                "players": [],
            }
        ev_frac = float(r["ev_frac"]) if "ev_frac" in r else float(r["ev_pct"]) / 100.0
        ev_pct = float(r["ev_pct"])
        side = str(r["side"])
        line = float(r["line"])
        offered = int(r["offered_odds"])
        fair_side = float(r["fair_p"])
        # Signed edge drives OVER (green) / UNDER (red) in the shared card widget.
        signed_edge = ev_frac if side == "over" else -ev_frac
        # MANDATE: this is a MARKET_DISLOCATION diagnostic, not a model edge. NO stake / NO
        # Kelly is emitted during the validation period — kelly_fraction stays null.
        fe_stat = _STAT_TO_FRONTEND.get(str(r["stat"]), str(r["stat"]))
        odds_str = f"+{offered}" if offered > 0 else str(offered)
        games[eid]["players"].append({
            "player_id": f"{eid}:{r['player_name']}:{r['stat']}:{line}:{side}:{r['book']}",
            "player_name": r["player_name"],
            "team_name": r.get("team") or "",
            "position": f"{r['book']} {odds_str} · EV +{ev_pct:.1f}%",
            "is_starter": True,
            "injury_status": None,
            "projected_minutes": {"mean": None},
            "dnp_risk": 0,
            "stat_projections": {
                fe_stat: {
                    "mean": line,
                    "median": line,
                    "pmf": {},
                    "conformal_90_ci": None,
                    "calibrated_p_over": {
                        "market_line": line,
                        "p_over": round(float(r["consensus_p_over"]), 4),
                        "edge_vs_market": round(signed_edge, 6),
                        # No Kelly / no stake during the validation period (MARKET_DISLOCATION).
                        "kelly_fraction": None,
                    },
                    "soft_book": {
                        "book": r["book"],
                        "side": side,
                        "offered_odds": offered,
                        "ev_pct": round(ev_pct, 3),
                        "theoretical_ev_pct": round(ev_pct, 3),
                        "executable_ev_pct": None,
                        "source_type": "MARKET_DISLOCATION",
                        "actionable": bool(r.get("actionable", False)),
                        "actionable_reason": r.get("actionable_reason"),
                        "clv_segment": r.get("clv_segment"),
                        "clv_mean": (None if pd.isna(r.get("clv_mean")) else r.get("clv_mean")),
                        "clv_ci_low": (None if pd.isna(r.get("clv_ci_low")) else r.get("clv_ci_low")),
                        "fair_p_side": round(fair_side, 4),
                        "consensus_p_over": round(float(r["consensus_p_over"]), 4),
                        "consensus_n_books": int(r["consensus_n_books"]),
                        "is_sharp_book": bool(r["is_sharp_book"]),
                        "sharp_consensus_p_over": (
                            None if pd.isna(r.get("sharp_consensus_p_over"))
                            else round(float(r["sharp_consensus_p_over"]), 4)
                        ),
                        "best_book": r.get("best_book"),
                        "best_odds": (None if pd.isna(r.get("best_odds")) else int(r["best_odds"])),
                    },
                }
            },
        })
    return {
        "schema_version": BOARD_VERSION,
        "generated_at": generated,
        "game_date": date,
        "board_type": "soft_book_ev",
        "model_version": f"{BOARD_VERSION} (Definition B: market-vs-market, no PMF model)",
        "ev_threshold_pct": round(summary["ev_threshold_pct"], 3),
        "method": _METHOD_BLOCK,
        "summary": summary,
        "games": list(games.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--quotes", default=None,
                    help="Glob/path to snapshot parquet(s). If omitted, auto-discover by --date.")
    ap.add_argument("--snapshot-dir", default="data/snapshots/soft_book_quotes")
    ap.add_argument("--ev-threshold", type=float, default=DEFAULT_EV_THRESHOLD)
    ap.add_argument("--min-consensus-books", type=int, default=DEFAULT_MIN_CONSENSUS_BOOKS)
    ap.add_argument("--artifact-dir", default="artifacts/edge_board")
    ap.add_argument("--clv-validation-table", default=DEFAULT_CLV_VALIDATION_TABLE,
                    help="Path to the persisted CLV validation table that drives actionability "
                         "(fail closed: if missing, every row stays actionable=false).")
    ap.add_argument("--render-dir",
                    default="tools/odds-scanner/predictions/WNBA/Soft-Book-Edge")
    ap.add_argument("--keep-stale", action="store_true",
                    help="Do not drop events whose commence_time is already past.")
    args = ap.parse_args()

    generated = datetime.now(timezone.utc).isoformat()
    quotes = _resolve_quotes(args)

    summary: dict = {
        "date": args.date,
        "generated_utc": generated,
        "ev_threshold_pct": args.ev_threshold * 100.0,
        "min_consensus_books": args.min_consensus_books,
        "n_events": 0,
        "n_books_seen": 0,
        "books_seen": [],
        "n_quote_rows": 0,
        "n_scored_rows": 0,
        "n_qualifying": 0,
        "n_books_per_stat": {},
        "status": "ok",
    }

    board = pd.DataFrame()
    if quotes is None or len(quotes) == 0:
        summary["status"] = "no_quotes"
    else:
        summary["n_quote_rows"] = len(quotes)
        summary["n_events"] = int(quotes["event_id"].nunique())
        books = sorted({str(b) for b in quotes["book"].dropna().unique()})
        summary["n_books_seen"] = len(books)
        summary["books_seen"] = books
        summary["n_books_per_stat"] = {
            str(s): int(g.groupby("book").ngroups)
            for s, g in quotes[quotes["stat"].notna()].groupby("stat")
        }
        board = scan_soft_book_edges(
            quotes,
            ev_threshold=args.ev_threshold,
            min_consensus_books=args.min_consensus_books,
            sharp_books=SHARP_BOOKS,
            drop_stale=not args.keep_stale,
        )
        summary["n_scored_rows"] = len(board)

    # Actionability is driven ONLY by the persisted, backtest-derived CLV validation table
    # (fail closed). Missing table => every row stays actionable=false.
    clv_table = _load_validation_table(args.clv_validation_table)
    summary["clv_validation_table"] = args.clv_validation_table
    summary["clv_validation_table_loaded"] = clv_table is not None
    summary["clv_actionable_segments"] = (
        list(clv_table.get("actionable_segments", [])) if clv_table else []
    )
    if len(board):
        board = apply_validation_table_to_board(board, clv_table)
    summary["n_actionable"] = (
        int(board["actionable"].sum()) if len(board) and "actionable" in board else 0
    )

    board_q = board[board["qualified"]].copy() if len(board) else board
    summary["n_qualifying"] = len(board_q)

    # Near-miss transparency: top plays by EV regardless of threshold.
    top_all = []
    if len(board):
        top_all = (
            board.sort_values("ev_frac", ascending=False)
            .head(20)[_TIDY_COLS + ["ev_frac", "qualified"]]
            .to_dict(orient="records")
        )

    tidy_board = board_q[_TIDY_COLS].to_dict(orient="records") if len(board_q) else []

    artifact = {
        "schema_version": BOARD_VERSION,
        "generated_at": generated,
        "game_date": args.date,
        "source_type": "MARKET_DISLOCATION",
        "actionable_default": False,
        "disclaimer": (
            "MARKET DISLOCATION diagnostics only — NOT a model edge, NOT claimed "
            "profitable/executable. No stake/Kelly emitted during the validation period."
        ),
        "method": _METHOD_BLOCK,
        "summary": summary,
        "board": tidy_board,             # qualifying +EV plays, sorted by EV desc
        "top_by_ev_all": top_all,        # top 20 by EV incl. below-threshold near-misses
    }

    art_dir = Path(args.artifact_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / f"SOFT_BOOK_EDGE_{args.date}.json"
    art_path.write_text(json.dumps(artifact, indent=2, default=str))

    render = _to_render_json(board_q, args.date, generated, summary) if len(board_q) else {
        "schema_version": BOARD_VERSION,
        "generated_at": generated,
        "game_date": args.date,
        "board_type": "soft_book_ev",
        "model_version": f"{BOARD_VERSION} (Definition B: market-vs-market, no PMF model)",
        "ev_threshold_pct": round(args.ev_threshold * 100.0, 3),
        "method": _METHOD_BLOCK,
        "summary": summary,
        "games": [],
    }
    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "latest.json").write_text(json.dumps(render, indent=2, default=str))
    (render_dir / f"{args.date}.json").write_text(json.dumps(render, indent=2, default=str))

    print(json.dumps({
        "date": args.date,
        "status": summary["status"],
        "n_events": summary["n_events"],
        "n_books_seen": summary["n_books_seen"],
        "n_books_per_stat": summary["n_books_per_stat"],
        "n_scored_rows": summary["n_scored_rows"],
        "n_qualifying": summary["n_qualifying"],
        "n_actionable": summary.get("n_actionable", 0),
        "clv_actionable_segments": summary.get("clv_actionable_segments", []),
        "artifact": str(art_path),
        "render": str(render_dir / "latest.json"),
    }, indent=2))

    if tidy_board:
        print("\nTop qualifying +EV plays:")
        for r in tidy_board[:5]:
            print(f"  {r['player_name']:22s} {r['stat']:4s} {r['side']:5s} "
                  f"{r['line']:>5} @ {r['book']:12s} {int(r['offered_odds']):+d}  "
                  f"EV {r['ev_pct']:+.2f}%  fair_p={r['fair_p']:.3f}  "
                  f"n_books={r['consensus_n_books']}  best={r['best_book']}({r['best_odds']})")


if __name__ == "__main__":
    main()
