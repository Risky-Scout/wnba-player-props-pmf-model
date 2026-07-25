"""P10 - exact-quote proof-readiness audit (per prop).

Reports whether exact, same-book, decision-time Over/Under quote PAIRS with delivered
model_prob_over_final and known settlement exist in sufficient quantity to build G0-v2 and run
the untouched proof. This is the honest check that the mission is NOT merely compute-bound: if
the atomic quote store is empty, G0-v2/proof are data-blocked regardless of the OOF baseline.

Reads (when present):
  data/atomic_quotes/pairs/**/pairs.parquet   (A6 validated quote pairs)
  data/atomic_quotes/atomic_quotes.parquet    (legacy flat atomic store)
Emits artifacts/market_feature_proof/EXACT_QUOTE_READINESS.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
REQUIRED_ROWS = 300          # frozen proof floor (per prop)
REQUIRED_DATES = 30          # frozen distinct-date floor (per prop)
OUT = REPO / "artifacts" / "market_feature_proof" / "EXACT_QUOTE_READINESS.json"

READY = "READY_FOR_G0_V2"
B_QUOTES = "BLOCKED_MISSING_ATOMIC_QUOTES"
B_PROBS = "BLOCKED_MISSING_DELIVERED_PROBABILITIES"
B_SETTLE = "BLOCKED_MISSING_SETTLEMENT"
B_ID = "BLOCKED_IDENTITY"
B_DATES = "BLOCKED_INSUFFICIENT_DATES"
FWD = "FORWARD_PROOF_ONLY"


def _load_pairs() -> pd.DataFrame:
    pairs_root = REPO / "data" / "atomic_quotes" / "pairs"
    flat = REPO / "data" / "atomic_quotes" / "atomic_quotes.parquet"
    frames = []
    if pairs_root.exists():
        frames += [pd.read_parquet(p) for p in pairs_root.rglob("pairs.parquet")]
    if flat.exists():
        frames.append(pd.read_parquet(flat))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prop_report(df: pd.DataFrame, prop: str) -> dict:
    g = df[df.get("prop", pd.Series(dtype=str)).astype(str) == prop] if len(df) else df
    def _c(mask):
        return int(mask.sum()) if len(g) else 0
    exact = g[g.get("quote_pair_status", pd.Series(dtype=str)) == "EXACT_PAIR"] if len(g) else g
    has_prob = exact[exact.get("model_prob_over_final").notna()] if ("model_prob_over_final" in exact.columns and len(exact)) else exact.iloc[0:0]
    settled = has_prob[has_prob.get("settlement_status").isin(["OVER_WIN", "UNDER_WIN"])] \
        if ("settlement_status" in has_prob.columns and len(has_prob)) else has_prob.iloc[0:0]
    n_rows = int(len(settled))
    n_dates = int(settled["game_date"].nunique()) if ("game_date" in settled.columns and len(settled)) else 0

    if len(g) == 0:
        status = B_QUOTES
    elif len(exact) == 0:
        status = B_QUOTES
    elif len(has_prob) == 0:
        status = B_PROBS
    elif len(settled) == 0:
        status = B_SETTLE
    elif n_dates < REQUIRED_DATES or n_rows < REQUIRED_ROWS:
        status = B_DATES
    else:
        status = READY
    return {
        "raw_sides": int(len(g)),
        "exact_pairs": int(len(exact)),
        "with_delivered_prob": int(len(has_prob)),
        "settled_binary_eligible_non_push": n_rows,
        "unique_dates": n_dates,
        "required_rows": REQUIRED_ROWS, "required_dates": REQUIRED_DATES,
        "row_shortfall": max(0, REQUIRED_ROWS - n_rows),
        "date_shortfall": max(0, REQUIRED_DATES - n_dates),
        "status": status,
    }


def main() -> int:
    df = _load_pairs()
    per = {p: _prop_report(df, p) for p in PROPS}
    any_ready = any(v["status"] == READY for v in per.values())
    report = {
        "version": "exact-quote-readiness-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "atomic_quote_store_present": bool(len(df)),
        "total_rows_in_store": int(len(df)),
        "per_prop": per,
        "any_prop_ready_for_g0v2": any_ready,
        "mission_is_compute_only": bool(any_ready),
        "note": ("No exact player-prop atomic quote store exists yet; game-level odds are NOT "
                 "exact same-book player-prop pairs. G0-v2 and proof are DATA-BLOCKED on atomic "
                 "decision-time quote collection (collect_atomic_quotes.py + settlement), not "
                 "merely compute." if not any_ready else "At least one prop has sufficient exact "
                 "quote coverage for G0-v2."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[quote-readiness] store_present={report['atomic_quote_store_present']} "
          f"any_ready={any_ready}")
    for p, v in per.items():
        print(f"  {p:9s}: {v['status']}  exact_pairs={v['exact_pairs']} "
              f"settled={v['settled_binary_eligible_non_push']} dates={v['unique_dates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
