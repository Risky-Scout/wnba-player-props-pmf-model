"""Deliverables 4 & 5: preserve the delivered model_prob_over_final values and verify the
exact same-book quote pairing, identities, timestamps and lineage of the canonical scored
rows (PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet).

Compute-only over committed artifacts. Writes:
  * DELIVERED_PROBABILITY_PRESERVATION.json  - per-prop delivered-probability counts + hashes
  * EXACT_PAIR_LINEAGE_VERIFICATION.json      - same-book / identity / timestamp / lineage checks
  * DELIVERED_MODEL_PROB_OVER_FINAL.parquet   - preserved copy of the delivered probabilities
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
G0 = REPO / "artifacts/market_feature_proof/G0_v2"
SCORED = G0 / "PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet"
FROZEN_MODEL_HASH = "a851cccdd00e28cc82dd5bd1a7edf94bdef75a18ee81fed09a5850d2bf92ac06"
FROZEN_FEATURE_HASH = "302de341643008330520bc9c76c6b397f9ba24b80bd011faf038366ad6a95357"
ALL_SEVEN = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _ts_ok(s: pd.Series) -> bool:
    try:
        pd.to_datetime(s, utc=True, errors="raise")
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    df = pd.read_parquet(SCORED)
    now = datetime.now(timezone.utc).isoformat()

    # ---- Deliverable 4: preserve delivered probabilities -------------------------------
    delivered_cols = ["game_date", "game_id", "player_id", "prop", "sportsbook",
                      "quote_pair_id", "model_prob_over_final", "market_prob_over_no_vig",
                      "outcome_over", "settlement_status", "model_hash", "feature_hash",
                      "calibration_hash"]
    delivered = df[delivered_cols].copy()
    out_parq = G0 / "DELIVERED_MODEL_PROB_OVER_FINAL.parquet"
    delivered.to_parquet(out_parq, index=False)
    delivered_sha = hashlib.sha256(out_parq.read_bytes()).hexdigest()

    per_prop_delivered = {}
    for p in ALL_SEVEN:
        s = df[df["prop"] == p]["model_prob_over_final"] if len(df[df["prop"] == p]) else pd.Series(dtype=float)
        per_prop_delivered[p] = {
            "delivered_prob_rows": int(s.notna().sum()),
            "in_unit_interval": int(((s >= 0) & (s <= 1)).sum()),
            "status": "OK" if len(s) else "NO_EXACT_QUOTES",
        }
    preservation = {
        "version": "delivered-probability-preservation-v1",
        "generated_at_utc": now,
        "probability_column": "model_prob_over_final",
        "total_delivered_rows": int(df["model_prob_over_final"].notna().sum()),
        "per_prop": per_prop_delivered,
        "model_hash": FROZEN_MODEL_HASH,
        "feature_hash": FROZEN_FEATURE_HASH,
        "canonical_scored_row_sha256": hashlib.sha256(SCORED.read_bytes()).hexdigest(),
        "preserved_artifact": str(out_parq.relative_to(REPO)),
        "preserved_artifact_sha256": delivered_sha,
    }
    (G0 / "DELIVERED_PROBABILITY_PRESERVATION.json").write_text(json.dumps(preservation, indent=2) + "\n")

    # ---- Deliverable 5: exact same-book pairing / identity / timestamp / lineage --------
    checks = {}
    n = len(df)
    checks["same_book_pairing"] = {
        "sportsbook_non_null": int(df["sportsbook"].notna().sum()) == n,
        "over_and_under_present": int((df["over_odds"].notna() & df["under_odds"].notna()).sum()) == n,
        "quote_pair_id_unique": int(df["quote_pair_id"].nunique()) == n,
        "note": ("each row is ONE deterministic same-book quote (over+under from a single "
                 "sportsbook) under book-quote-priority-v1; books are never averaged"),
    }
    checks["identities"] = {
        "game_id_non_null": int(df["game_id"].notna().sum()) == n,
        "player_id_non_null": int(df["player_id"].notna().sum()) == n,
        "unique_key_no_dups": int(df.duplicated(["game_id", "player_id", "prop"]).sum()) == 0,
    }
    checks["timestamps"] = {
        "pair_timestamp_non_null": int(df["pair_timestamp"].notna().sum()) == n,
        "decision_timestamp_non_null": int(df["decision_timestamp"].notna().sum()) == n,
        "pair_timestamp_parseable": _ts_ok(df["pair_timestamp"]),
        "decision_timestamp_parseable": _ts_ok(df["decision_timestamp"]),
    }
    checks["lineage"] = {
        "single_model_hash": int(df["model_hash"].nunique()) == 1,
        "single_feature_hash": int(df["feature_hash"].nunique()) == 1,
        "single_calibration_hash": int(df["calibration_hash"].nunique()) == 1,
        "model_hash_matches_frozen": str(df["model_hash"].dropna().iloc[0]) == FROZEN_MODEL_HASH,
        "feature_hash_matches_frozen": str(df["feature_hash"].dropna().iloc[0]) == FROZEN_FEATURE_HASH,
    }
    all_pass = all(bool(v) for grp in checks.values() for k, v in grp.items() if isinstance(v, bool))
    verification = {
        "version": "exact-pair-lineage-verification-v1",
        "generated_at_utc": now,
        "rows_verified": n,
        "checks": checks,
        "all_checks_pass": bool(all_pass),
    }
    (G0 / "EXACT_PAIR_LINEAGE_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")

    print("[deliverable 4] delivered model_prob_over_final rows by prop:")
    for p, r in per_prop_delivered.items():
        print(f"   {p:9s} {r['delivered_prob_rows']:5d}  {r['status']}")
    print(f"[deliverable 4] preserved -> {out_parq.relative_to(REPO)} (sha {delivered_sha[:12]}…)")
    print(f"[deliverable 5] exact-pair/identity/timestamp/lineage all_checks_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
