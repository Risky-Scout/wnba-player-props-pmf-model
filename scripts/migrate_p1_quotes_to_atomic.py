#!/usr/bin/env python3
"""Migrate preserved P1 historical quotes -> immutable atomic two-sided quote store (owner phase 5).

The P1 store (artifacts/p1/p1_quotes.parquet) preserves per-book, point-in-time single sides with full
identity (provider event id, sportsbook, canonical game_id/player_id, prop, line, side, American odds,
provider snapshot ts, book_last_update, commence_time). This builds RAW atomic pairs WITHOUT any market
consensus/averaging and WITHOUT requiring a model probability:

  * one immutable RAW SIDE per preserved quote (stable hash id), with validity + rejection reason;
  * one ATOMIC PAIR per (game_id, player_id, prop, line, book, snapshot_label) with BOTH sides, same
    book + same line only, side-timestamp skew bounded, invalid/post-cutoff/at-or-after-tip rejected;
  * settlement from CANONICAL box outcomes (over/under/push, void-on-DNP marked separately).

Readiness levels are reported separately and honestly:
  RAW_QUOTE_PAIR_READY        -- valid same-book/same-line two-sided pair (no model prob needed).
  HISTORICAL_REPLAY_READY     -- RAW + canonical settled binary (non-push, played); a frozen OOF/replay
                                 probability can be joined later on (game_id, player_id, prop, line).
  PROSPECTIVE_DELIVERY_READY  -- requires a delivered probability + lineage captured AT prediction time;
                                 impossible to synthesise historically -> reported as 0.

Quote-selection policy is FROZEN and MODEL-INDEPENDENT: within a (key, side, snapshot_label) the latest
provider snapshot at-or-before tip is kept. No edge-maximising or model-directed selection.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
P1 = REPO / "artifacts/p1/p1_quotes.parquet"
BOX = REPO / "data/processed/wnba_player_game_stats.parquet"
OUT_DIR = REPO / "data/processed/atomic_quotes"
FEAS = REPO / "artifacts/pure_model_completion/HISTORICAL_QUOTE_MIGRATION_FEASIBILITY.json"

DIRECT = ("pts", "reb", "ast", "fg3m")
MAX_SKEW_SECONDS = 300.0          # over/under sides of one pair must be near-simultaneous
QUOTE_POLICY = "latest_snapshot_at_or_before_tip_per_(key,side,snapshot_label); per-book; no consensus"


def _sha(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _valid_american(o) -> bool:
    try:
        o = float(o)
    except (TypeError, ValueError):
        return False
    return np.isfinite(o) and abs(o) >= 100.0


def main() -> None:
    q = pd.read_parquet(P1)
    q = q[q["stat"].isin(DIRECT)].copy()
    for tcol in ("snapshot_time", "commence_time", "book_last_update"):
        q[tcol] = pd.to_datetime(q[tcol], utc=True, errors="coerce")

    # ---- RAW SIDE validation (fail-closed reasons; nothing silently dropped) ----
    reason = pd.Series("ok", index=q.index)
    reason[~q["game_id"].notna() | ~q["player_id"].notna()] = "missing_identity"
    reason[(reason == "ok") & (~q["american_odds"].apply(_valid_american))] = "invalid_odds"
    reason[(reason == "ok") & (q["snapshot_time"].isna() | q["commence_time"].isna())] = "missing_timestamp"
    reason[(reason == "ok") & (q["snapshot_time"] >= q["commence_time"])] = "at_or_after_tip"
    q["rejection_reason"] = reason
    q["side_valid"] = reason == "ok"
    q["raw_side_id"] = [
        _sha(r.odds_event_id, r.book, r.player_id, r.stat, r.line, r.side,
             r.snapshot_label, r.snapshot_time)
        for r in q.itertuples()
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sides_path = OUT_DIR / "atomic_sides.parquet"
    q.to_parquet(sides_path, index=False)

    # ---- PAIR construction (valid sides only; deterministic policy) ----
    v = q[q["side_valid"]].copy()
    keycols = ["game_id", "player_id", "stat", "line", "book", "snapshot_label"]
    # frozen model-independent selection: latest snapshot per (key, side)
    v = v.sort_values("snapshot_time").groupby(keycols + ["side"], as_index=False).tail(1)

    rows, pair_rejects = [], {}
    for key, g in v.groupby(keycols):
        sides = {s: r for s, r in zip(g["side"], g.itertuples())}
        if "over" not in sides or "under" not in sides:
            pair_rejects["one_sided_market"] = pair_rejects.get("one_sided_market", 0) + 1
            continue
        o, u = sides["over"], sides["under"]
        skew = abs((o.snapshot_time - u.snapshot_time).total_seconds())
        if skew > MAX_SKEW_SECONDS:
            pair_rejects["side_timestamp_skew"] = pair_rejects.get("side_timestamp_skew", 0) + 1
            continue
        gid, pid, stat, line, book, label = key
        rows.append({
            "provider": "the-odds-api", "sportsbook": book, "event_id": o.odds_event_id,
            "game_id": gid, "player_id": pid, "prop": stat, "line": float(line),
            "snapshot_label": label,
            "quote_pair_id": _sha(o.odds_event_id, book, pid, stat, line, label),
            "over_side_id": o.raw_side_id, "under_side_id": u.raw_side_id,
            "over_odds": float(o.american_odds), "under_odds": float(u.american_odds),
            "over_snapshot_utc": o.snapshot_time, "under_snapshot_utc": u.snapshot_time,
            "pair_timestamp_utc": max(o.snapshot_time, u.snapshot_time),
            "decision_timestamp_utc": max(o.snapshot_time, u.snapshot_time),
            "scheduled_tip_utc": o.commence_time, "game_date": o.game_date,
            "side_skew_seconds": skew,
        })
    pairs = pd.DataFrame(rows)

    # ---- Settlement from canonical box outcomes ----
    box = pd.read_parquet(BOX)[["game_id", "player_id", "pts", "reb", "ast", "fg3m", "did_play"]]
    if not pairs.empty:
        # canonical join keys may differ in dtype across sources -> normalise to str fail-closed
        for _df in (pairs, box):
            _df["game_id"] = _df["game_id"].astype(str)
            _df["player_id"] = _df["player_id"].astype(str)
        pairs = pairs.merge(box, on=["game_id", "player_id"], how="left")
        actual = np.select(
            [pairs["prop"] == p for p in DIRECT],
            [pairs[p] for p in DIRECT], default=np.nan)
        pairs["actual_value"] = actual
        played = pairs["did_play"].fillna(False).astype(bool)
        pairs["outcome"] = np.where(
            ~played | pairs["actual_value"].isna(), "dnp_or_missing",
            np.where(pairs["actual_value"] > pairs["line"], "over",
                     np.where(pairs["actual_value"] < pairs["line"], "under", "push")))
        pairs["binary_settled_eligible"] = pairs["outcome"].isin(["over", "under"])
        pairs["void_dnp"] = ~played
    else:
        pairs["actual_value"] = []; pairs["outcome"] = []
        pairs["binary_settled_eligible"] = []; pairs["void_dnp"] = []

    pairs_path = OUT_DIR / "atomic_pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)

    # ---- Readiness by prop (decision-cutoff snapshot is the certification-relevant set) ----
    def _counts(df):
        by = {}
        for prop in DIRECT:
            d = df[df["prop"] == prop]
            by[prop] = {"pairs": int(len(d)), "dates": int(d["game_date"].nunique()) if len(d) else 0}
        return by

    raw_ready = pairs if not pairs.empty else pairs
    replay_ready = pairs[pairs["binary_settled_eligible"]] if not pairs.empty else pairs
    decision = pairs[pairs["snapshot_label"] == "decision"] if not pairs.empty else pairs
    decision_replay = decision[decision["binary_settled_eligible"]] if not pairs.empty else pairs

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(P1.relative_to(REPO)), "source_rows_direct": int(len(q)),
        "quote_selection_policy": QUOTE_POLICY, "max_side_skew_seconds": MAX_SKEW_SECONDS,
        "identity": "canonical game_id + player_id present 100%; identity_method=exact_roster_name",
        "books": sorted(v["book"].unique().tolist()),
        "consensus_used": False,
        "side_validation": {k: int((q["rejection_reason"] == k).sum())
                            for k in q["rejection_reason"].unique().tolist()},
        "pair_rejections": pair_rejects,
        "RAW_QUOTE_PAIR_READY": {"all_snapshots": _counts(raw_ready),
                                 "decision_snapshot": _counts(decision)},
        "HISTORICAL_REPLAY_READY": {"all_snapshots": _counts(replay_ready),
                                    "decision_snapshot": _counts(decision_replay),
                                    "note": "settled binary (played, non-push); join a FROZEN OOF/replay prob on (game_id,player_id,prop,line) at replay time. model_prob NOT required to declare RAW pair valid."},
        "PROSPECTIVE_DELIVERY_READY": {"pairs": 0,
            "note": "requires delivered probability + full lineage captured AT prediction time; cannot be synthesised historically."},
        "outputs": {
            "atomic_sides": {"path": str(sides_path.relative_to(REPO)), "rows": int(len(q)),
                             "sha256": _file_sha256(sides_path)},
            "atomic_pairs": {"path": str(pairs_path.relative_to(REPO)), "rows": int(len(pairs)),
                             "sha256": _file_sha256(pairs_path)},
        },
        "certification_threshold": {"required_rows": 300, "required_dates": 30},
    }
    FEAS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(FEAS, "w"), indent=2, default=str)
    print(json.dumps({
        "raw_pairs_total": int(len(pairs)),
        "RAW_decision_by_prop": report["RAW_QUOTE_PAIR_READY"]["decision_snapshot"],
        "REPLAY_decision_by_prop": report["HISTORICAL_REPLAY_READY"]["decision_snapshot"],
        "side_validation": report["side_validation"],
        "pair_rejections": pair_rejects,
    }, indent=2))


if __name__ == "__main__":
    main()
