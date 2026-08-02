"""Stage 8: exact Over/Under pairs + BDL settlement (offline, no API).

Builds EXACT same-book/same-line/same-role pairs from ELIGIBLE side rows only, keeps decision
and closing datasets SEPARATE, settles from BDL player game outcomes (direct + deterministic
combination sums), and emits readiness per prop x role.

Outputs:
  data/atomic_quotes/quote_pairs.parquet          (all EXACT pairs, with snapshot_role)
  data/atomic_quotes/decision_pairs.parquet       (model-vs-market + executable backtest)
  data/atomic_quotes/closing_pairs.parquet        (CLV / market-movement only)
  data/atomic_quotes/settled_quote_pairs.parquet  (decision pairs settled from BDL)
  artifacts/audits/EXACT_QUOTE_READINESS.json      (per prop x role)
  artifacts/audits/SETTLEMENT_LINEAGE_AUDIT.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.atomic_quotes import to_raw_side_snapshots  # noqa: E402
from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs  # noqa: E402
from wnba_props_model.evaluation.historical_market import american_to_implied  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
AQ = REPO / "data" / "atomic_quotes"
PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
         "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]
REQUIRED_ROWS, REQUIRED_DATES = 300, 30


def _build_pairs(atomic: pd.DataFrame) -> pd.DataFrame:
    raw = to_raw_side_snapshots(atomic)   # ELIGIBLE rows only, role cutoff carried per row
    if raw.empty:
        return pd.DataFrame()
    frames = [build_quote_pairs(sub, snapshot_label=str(label))
              for label, sub in raw.groupby("snapshot_label")]
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pairs = pairs[pairs["quote_pair_status"] == EXACT_PAIR].copy()
    if len(pairs):
        pairs["snapshot_role"] = pairs["snapshot_label"]   # build_quote_pairs names it snapshot_label
    return pairs


def _settle_vectorized(pairs: pd.DataFrame, ev2g: dict, box: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    p = pairs.copy()
    p["bdl_game_id"] = p["event_id"].map(lambda e: ev2g.get(str(e)))
    p["player_id"] = p["player_id"].astype(str)
    box = box.copy()
    box["player_id"] = box["player_id"].astype(str)
    box["game_id"] = box["game_id"].astype(str)
    # long outcome table: one row per (game, player, prop)
    keep = ["game_id", "player_id", "did_play"] + [c for c in PROPS if c in box.columns]
    b = box[keep].drop_duplicates(["game_id", "player_id"])
    out = []
    for prop, g in p.groupby("prop"):
        m = g.merge(b[["game_id", "player_id", "did_play", prop]].rename(columns={prop: "actual_outcome"}),
                    left_on=["bdl_game_id", "player_id"], right_on=["game_id", "player_id"], how="left")
        line = m["line"].astype(float)
        actual = pd.to_numeric(m["actual_outcome"], errors="coerce")
        did_play = m["did_play"]
        status = np.full(len(m), "UNRESOLVED", dtype=object)
        has_actual = actual.notna()
        # confirmed DNP -> VOID
        status = np.where(did_play.eq(False), "VOID", status)
        settled = has_actual & did_play.fillna(False).astype(bool)
        is_int = line.apply(lambda x: float(x).is_integer())
        status = np.where(settled & is_int & (actual == line), "PUSH", status)
        status = np.where(settled & (actual > line) & ~((is_int) & (actual == line)), "OVER_WIN", status)
        status = np.where(settled & (actual < line) & ~((is_int) & (actual == line)), "UNDER_WIN", status)
        m["settlement_status"] = status
        m["settlement_source"] = "bdl_player_stats"
        m["outcome_field"] = prop
        m["evidence_type"] = "RETROSPECTIVE_OOF"
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _readiness(atomic: pd.DataFrame, pairs: pd.DataFrame, settled: pd.DataFrame) -> dict:
    per = {}
    for prop in PROPS:
        per[prop] = {}
        for role in ("decision", "closing"):
            a = atomic[(atomic["prop"] == prop) & (atomic["snapshot_role"] == role)]
            ael = a[a["eligibility_status"] == "ELIGIBLE"]
            pr = pairs[(pairs["prop"] == prop) & (pairs["snapshot_role"] == role)] if len(pairs) else pairs
            n_pairs = int(len(pr))
            settled_rows = pushes = voids = unresolved = non_push = 0
            uniq_games = uniq_dates = 0
            if role == "decision" and len(settled):
                s = settled[(settled["prop"] == prop)]
                settled_rows = int(s["settlement_status"].isin(["OVER_WIN", "UNDER_WIN", "PUSH"]).sum())
                pushes = int((s["settlement_status"] == "PUSH").sum())
                voids = int((s["settlement_status"] == "VOID").sum())
                unresolved = int((s["settlement_status"] == "UNRESOLVED").sum())
                nonp = s[s["settlement_status"].isin(["OVER_WIN", "UNDER_WIN"])]
                non_push = int(len(nonp))
                uniq_games = int(nonp["bdl_game_id"].nunique()) if len(nonp) else 0
                dts = pd.to_datetime(nonp["scheduled_tip_utc"], errors="coerce").dt.date if len(nonp) else pd.Series([], dtype=object)
                uniq_dates = int(pd.Series(dts).nunique()) if len(nonp) else 0
            per[prop][role] = {
                "raw_side_rows": int(len(a)), "eligible_side_rows": int(len(ael)),
                "exact_pairs": n_pairs,
                "unique_players": int(pr["player_id"].nunique()) if n_pairs else 0,
                "unique_events": int(pr["event_id"].nunique()) if n_pairs else 0,
                "sportsbooks": sorted(pr["sportsbook"].dropna().unique().tolist()) if n_pairs else [],
                "settled_pairs": settled_rows, "pushes": pushes, "voids": voids,
                "unresolved": unresolved, "settled_non_push_pairs": non_push,
                "unique_games": uniq_games, "unique_game_dates": uniq_dates,
                "meets_market_evidence_min": bool(role == "decision" and non_push >= REQUIRED_ROWS and uniq_dates >= REQUIRED_DATES),
            }
    return {
        "artifact": "EXACT_QUOTE_READINESS", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_settled_non_push_decision_pairs": REQUIRED_ROWS, "required_unique_dates": REQUIRED_DATES,
        "per_prop_per_role": per,
        "market_evidence_ready_props": [p for p in PROPS if per[p]["decision"]["meets_market_evidence_min"]],
        "note": "Decision pairs support model-vs-market + executable backtest; closing pairs are "
                "CLV only. A market-comparison claim requires >=300 settled non-push DECISION pairs "
                "and >=30 unique game dates, else MARKET_EVIDENCE_INSUFFICIENT.",
    }


@app.command()
def main(
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    box: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--box"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    atomic = pd.read_parquet(store)
    box_df = pd.read_parquet(box)
    ev2g = (atomic[atomic["game_id"].notna()].dropna(subset=["event_id"])
            .drop_duplicates("event_id").set_index("event_id")["game_id"].astype(str).to_dict())

    pairs = _build_pairs(atomic)
    if len(pairs):
        pairs.to_parquet(AQ / "quote_pairs.parquet", index=False)
        dec = pairs[pairs["snapshot_role"] == "decision"].copy()
        clo = pairs[pairs["snapshot_role"] == "closing"].copy()
        dec.to_parquet(AQ / "decision_pairs.parquet", index=False)
        clo.to_parquet(AQ / "closing_pairs.parquet", index=False)
        settled = _settle_vectorized(dec, ev2g, box_df)   # settle DECISION pairs
        if len(settled):
            settled.to_parquet(AQ / "settled_quote_pairs.parquet", index=False)
    else:
        settled = pd.DataFrame()

    readiness = _readiness(atomic, pairs, settled)
    (AUD / "EXACT_QUOTE_READINESS.json").write_text(json.dumps(readiness, indent=2, default=str))

    # settlement lineage audit
    lineage = {"artifact": "SETTLEMENT_LINEAGE_AUDIT", "generated_at_utc": readiness["generated_at_utc"],
               "settlement_source": "bdl_player_stats (data/recovered_v2)",
               "decision_pairs": int(len(pairs[pairs["snapshot_role"] == "decision"])) if len(pairs) else 0,
               "closing_pairs": int(len(pairs[pairs["snapshot_role"] == "closing"])) if len(pairs) else 0,
               "settled_status_counts": ({str(k): int(v) for k, v in settled["settlement_status"].value_counts().to_dict().items()} if len(settled) else {})}
    (AUD / "SETTLEMENT_LINEAGE_AUDIT.json").write_text(json.dumps(lineage, indent=2, default=str))

    typer.echo("================ EXACT PAIRS + SETTLEMENT ================")
    typer.echo(f"  EXACT pairs: decision={lineage['decision_pairs']} closing={lineage['closing_pairs']}")
    typer.echo(f"  settled decision status: {lineage['settled_status_counts']}")
    typer.echo("  per-prop DECISION settled non-push / dates:")
    for p in PROPS:
        d = readiness["per_prop_per_role"][p]["decision"]
        flag = "READY" if d["meets_market_evidence_min"] else "insufficient"
        typer.echo(f"    {p:12s} pairs={d['exact_pairs']:5d} settled_non_push={d['settled_non_push_pairs']:5d} "
                   f"dates={d['unique_game_dates']:3d}  {flag}")
    typer.echo(f"  market-evidence-ready props: {readiness['market_evidence_ready_props']}")


if __name__ == "__main__":
    app()
