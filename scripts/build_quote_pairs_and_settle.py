"""Build validated quote pairs, settle from BDL outcomes, and emit readiness (Sections I/J/K/L).

Inputs:
  data/atomic_quotes/atomic_quotes.parquet          (raw single-side quotes)
  data/processed/wnba_player_game_stats.parquet     (BDL realized outcomes)
  artifacts/audits/EVENT_ID_MAPPING_AUDIT.csv        (odds_event_id <-> bdl_game_id)

Outputs:
  data/atomic_quotes/quote_pairs.parquet
  data/atomic_quotes/settled_quote_pairs.parquet
  artifacts/audits/EXACT_QUOTE_READINESS.json

Settlement uses BDL player_stats only (direct props from the box column; combos from the
deterministic sum columns). Evidence is labeled RETROSPECTIVE_OOF: no prediction_timestamp or
model hashes are invented (prospective proof requires a pre-game captured prediction).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.atomic_quotes import to_raw_side_snapshots  # noqa: E402
from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs  # noqa: E402
from wnba_props_model.data.settlement import (  # noqa: E402
    OVER_WIN,
    PUSH,
    UNDER_WIN,
    SPORTSBOOK_SETTLEMENT_RULES,
    settle_one,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
STORE_DIR = REPO / "data" / "atomic_quotes"
PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
         "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]
REQUIRED_ROWS, REQUIRED_DATES = 300, 30
_RULE = SPORTSBOOK_SETTLEMENT_RULES["wnba_player_prop_standard_v1"]


def _build_pairs(atomic: pd.DataFrame) -> pd.DataFrame:
    # The adapter carries the correct ROLE cutoff per row (decision->tip-12h, closing->tip-5m).
    # We NEVER null the cutoff; each role is paired against its own cutoff.
    raw = to_raw_side_snapshots(atomic)
    if raw.empty:
        return pd.DataFrame()
    if "snapshot_label" not in raw.columns:
        raw["snapshot_label"] = "decision"
    frames = [build_quote_pairs(sub, snapshot_label=str(label))
              for label, sub in raw.groupby("snapshot_label")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _settle(pairs: pd.DataFrame, box: pd.DataFrame, evmap: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    e2g = {str(r["odds_event_id"]): r["bdl_game_id"] for _, r in evmap.iterrows()
           if pd.notna(r.get("bdl_game_id"))}
    box = box.copy()
    box["player_id"] = box["player_id"].astype(str)
    box["game_id"] = box["game_id"].astype(str)
    box_idx = box.set_index(["game_id", "player_id"])
    settled_ts = datetime.now(timezone.utc).isoformat()

    out_rows = []
    for _, r in pairs.iterrows():
        rec = r.to_dict()
        rec["evidence_type"] = "RETROSPECTIVE_OOF"     # never fabricate prospective evidence
        rec["settlement_source"] = "bdl_player_stats"
        rec["settled_timestamp"] = settled_ts
        rec["actual_outcome"] = None
        rec["settlement_status"] = "UNRESOLVED"
        rec["outcome_field"] = rec["prop"]
        if r["quote_pair_status"] != EXACT_PAIR:
            out_rows.append(rec); continue
        gid = e2g.get(str(r["event_id"]))
        pid = str(r["player_id"])
        if gid is None or (str(gid), pid) not in box_idx.index:
            out_rows.append(rec); continue
        brow = box_idx.loc[(str(gid), pid)]
        if isinstance(brow, pd.DataFrame):
            brow = brow.iloc[0]
        prop = r["prop"]
        actual = brow.get(prop)
        appeared = bool(brow.get("did_play", True))
        rec["bdl_game_id"] = gid
        s = settle_one(rule=_RULE, line=r["line"], appeared=appeared, actual_outcome=actual)
        rec["actual_outcome"] = s["actual_outcome"]
        rec["settlement_status"] = s["settlement_status"]
        rec["binary_score_eligible"] = s["binary_score_eligible"]
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _readiness(atomic: pd.DataFrame, pairs: pd.DataFrame, settled: pd.DataFrame) -> dict:
    per = {}
    for prop in PROPS:
        a = atomic[atomic["prop"] == prop]
        p = pairs[pairs["prop"] == prop] if len(pairs) else pairs
        st = settled[settled["prop"] == prop] if len(settled) else settled
        status_counts = p["quote_pair_status"].value_counts().to_dict() if len(p) else {}
        exact = st[st.get("quote_pair_status") == EXACT_PAIR] if len(st) else st
        settled_rows = exact[exact["settlement_status"].isin([OVER_WIN, UNDER_WIN, PUSH])] if len(exact) else exact
        non_push = exact[exact["settlement_status"].isin([OVER_WIN, UNDER_WIN])] if len(exact) else exact
        # cross-line: (event,book,player) offering >1 distinct line
        cross_line = 0
        if len(a):
            grp = a.groupby(["event_id", "sportsbook", "player_id"])["line"].nunique()
            cross_line = int((grp > 1).sum())
        dates = pd.to_datetime(non_push["pair_timestamp"], errors="coerce").dt.date if len(non_push) else pd.Series([], dtype=object)
        per[prop] = {
            "raw_side_rows": int(len(a)),
            "exact_same_book_pairs": int((p["quote_pair_status"] == EXACT_PAIR).sum()) if len(p) else 0,
            "one_sided_rejects": int(status_counts.get("ONE_SIDED", 0)),
            "cross_line_events": cross_line,
            "identity_rejects": int(status_counts.get("AMBIGUOUS_PLAYER", 0) + status_counts.get("AMBIGUOUS_GAME", 0)),
            "post_cutoff_rejects": int(status_counts.get("AFTER_DECISION_CUTOFF", 0) + status_counts.get("AT_OR_AFTER_TIP", 0)),
            "settled_rows": int(len(settled_rows)),
            "pushes": int((exact["settlement_status"] == PUSH).sum()) if len(exact) else 0,
            "settled_non_push_rows": int(len(non_push)),
            "unique_games": int(non_push["event_id"].nunique()) if len(non_push) else 0,
            "unique_game_dates": int(pd.Series(dates).nunique()) if len(non_push) else 0,
            "sportsbook_coverage": sorted(a["sportsbook"].dropna().unique().tolist()) if len(a) else [],
            "first_quote_date": str(pd.to_datetime(a["snapshot_time"], errors="coerce").min()) if len(a) else None,
            "last_quote_date": str(pd.to_datetime(a["snapshot_time"], errors="coerce").max()) if len(a) else None,
            "pair_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        }
    retro = {prop: {"meets_min": bool(per[prop]["settled_non_push_rows"] >= REQUIRED_ROWS
                                       and per[prop]["unique_game_dates"] >= REQUIRED_DATES),
                    "settled_non_push": per[prop]["settled_non_push_rows"],
                    "dates": per[prop]["unique_game_dates"]} for prop in PROPS}
    return {
        "version": "exact-quote-readiness-v2-historical",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_rows": REQUIRED_ROWS, "required_dates": REQUIRED_DATES,
        "atomic_rows_total": int(len(atomic)),
        "per_prop": per,
        "historical_retrospective_readiness": retro,
        "prospective_readiness": {prop: {"meets_min": False, "reason":
            "requires a pre-game captured, hashed prediction; not synthesizable historically"}
            for prop in PROPS},
        "note": "Historical Odds API snapshots + BDL outcomes support a RETROSPECTIVE OOF "
                "comparison. Prospective proof requires live pre-game prediction capture.",
    }


@app.command()
def main(
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    box: str = typer.Option("data/processed/wnba_player_game_stats.parquet", "--box"),
    evmap: str = typer.Option("artifacts/audits/EVENT_ID_MAPPING_AUDIT.csv", "--evmap"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    atomic = pd.read_parquet(store)
    box_df = pd.read_parquet(box)
    evmap_df = pd.read_csv(evmap) if Path(evmap).exists() else pd.DataFrame(columns=["odds_event_id", "bdl_game_id"])

    pairs = _build_pairs(atomic)
    pairs.to_parquet(STORE_DIR / "quote_pairs.parquet", index=False)
    settled = _settle(pairs, box_df, evmap_df)
    if len(settled):
        settled.to_parquet(STORE_DIR / "settled_quote_pairs.parquet", index=False)

    readiness = _readiness(atomic, pairs, settled)
    (AUD / "EXACT_QUOTE_READINESS.json").write_text(json.dumps(readiness, indent=2, default=str))

    typer.echo("================ READINESS (historical retrospective) ================")
    for prop in PROPS:
        d = readiness["per_prop"][prop]
        typer.echo(f"  {prop:12s} raw={d['raw_side_rows']:5d} exact_pairs={d['exact_same_book_pairs']:5d} "
                   f"settled_non_push={d['settled_non_push_rows']:5d} dates={d['unique_game_dates']:3d} "
                   f"books={len(d['sportsbook_coverage'])}")
    ready = [p for p in PROPS if readiness["historical_retrospective_readiness"][p]["meets_min"]]
    typer.echo(f"  RETROSPECTIVE-READY (>=300 non-push & >=30 dates): {ready or 'none'}")
    typer.echo(f"  wrote {STORE_DIR}/quote_pairs.parquet, settled_quote_pairs.parquet, "
               f"{AUD}/EXACT_QUOTE_READINESS.json")


if __name__ == "__main__":
    app()
