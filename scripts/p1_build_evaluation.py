"""P1 — historical evaluation & forced verdict (offline; leakage-free; REPAIRED).

Grades the production recommendation policy against REAL, EXACT sportsbook quotes
(never synthetic consensus prices):

  * decision snapshot  -> the executable quote available at production lead time is
    used to SELECT the recommendation and to price ROI (exact book/line/side/price);
  * closing snapshot   -> latest complete pre-tip quote, used ONLY for line/price CLV
    and closing-market calibration comparison;
  * fold-safe calibration -> per-stat isotonic P(over) calibration fit on strictly
    earlier games and applied walk-forward (production-equivalent, no lookahead);
  * a single SHARED selector (wnba_props_model.pipeline.recommendation) reproduces
    production selection exactly.

Reports three separated comparisons — raw fold-safe OOF, fold-safe calibrated
production-equivalent, and the sportsbook market — and enforces the 15 accounting
invariants before writing any verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402
from wnba_props_model.models.market import shin_no_vig_two_way_with_z  # noqa: E402

app = typer.Typer(add_completion=False)
KEYS = ["game_id", "player_id", "stat"]


def _pmf_array(pmf_json) -> np.ndarray:
    obj = json.loads(pmf_json) if isinstance(pmf_json, str) else pmf_json
    if obj is None:
        return np.array([])
    if isinstance(obj, dict):
        n = max(int(k) for k in obj.keys()) + 1
        arr = np.zeros(n)
        for k, v in obj.items():
            arr[int(k)] = float(v)
        return arr
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 2:
        n = int(arr[:, 0].max()) + 1
        out = np.zeros(n)
        for val, prob in arr:
            out[int(val)] = float(prob)
        return out
    return arr


def _res_to_dict(g: hm.GradeResult) -> dict:
    d = g.__dict__.copy()
    d["roi_ci95"] = list(g.roi_ci95)
    d.pop("extra", None)
    return d


def _write_inconclusive(out: Path, reason: str, extra: dict | None = None) -> None:
    res = {"verdict": "INCONCLUSIVE", "reason": reason}
    if extra:
        res.update(extra)
    (out / "p1_results.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "p1_validation_report.md").write_text(
        f"# P1 Historical Validation (repaired)\n\nVERDICT: **INCONCLUSIVE** — {reason}\n")
    typer.echo(f"[P1][EVAL] INCONCLUSIVE — {reason}")


@app.command()
def main(
    oof: str = typer.Option(...),
    quotes: str = typer.Option(..., help="Raw quote-level history (p1_quotes.parquet)."),
    out_dir: str = typer.Option("artifacts/p1"),
    edge_threshold: float = typer.Option(0.02, help="Min |edge| to publish (production floor)."),
    decision_lead_hours: float = typer.Option(12.0, help="Decision snapshot = tip - lead."),
    min_market_prob: float = typer.Option(0.05),
    max_shin_z: float = typer.Option(0.15),
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(oof)
    q = pd.read_parquet(quotes)
    if "event_id" not in q.columns and "odds_event_id" in q.columns:
        q["event_id"] = q["odds_event_id"]
    if "odds_event_id" not in q.columns and "event_id" in q.columns:
        q["odds_event_id"] = q["event_id"]

    if q.empty:
        return _write_inconclusive(out, "no historical quotes obtained (coverage gap)")

    # ---- strict price format audit (root cause #3) ----
    audit = hm.price_audit(q)
    (out / "p1_rejected_price_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    q = q[pd.to_numeric(q["american_odds"], errors="coerce").apply(hm._valid_american)].copy()
    if q.empty:
        return _write_inconclusive(out, "all quotes failed strict American-odds validation",
                                   {"price_audit": {k: audit[k] for k in ("n_quotes", "n_rejected")}})

    # ---- OOF integrity (root cause: duplicate keys must FAIL, not silently drop) ----
    for k in KEYS:
        oof[k] = oof[k].astype("string")
    hm.assert_no_lookahead(oof)
    oof_valid = oof.dropna(subset=["actual_outcome", "pmf_json"])
    if oof_valid.duplicated(subset=KEYS).any():
        raise typer.Exit("[P1][FATAL] duplicate OOF keys (game_id,player_id,stat) — refusing to drop_duplicates.")
    oof_u = oof_valid.copy()
    pmf_by_key = {(str(r.game_id), str(r.player_id), str(r.stat)): _pmf_array(r.pmf_json)
                  for r in oof_u.itertuples()}
    actual_by_key = {(str(r.game_id), str(r.player_id), str(r.stat)): float(r.actual_outcome)
                     for r in oof_u.itertuples()}
    gamedate_by_key = {(str(r.game_id), str(r.player_id), str(r.stat)): r.game_date
                       for r in oof_u.itertuples()}
    eligible_games = set(oof_u["game_id"].astype(str).unique())

    # ---- dedupe exact-identity quote rows (same event/book/stat/player/line/side/
    # snapshot). Identical rows are benign provider repeats; conflicting prices at one
    # identity keep the last observed. ----
    idkey = ["event_id", "book", "stat", "player_name", "line", "side", "snapshot_time"]
    n_q_before = len(q)
    q = q.drop_duplicates(subset=idkey, keep="last").copy()
    n_dup_quotes = n_q_before - len(q)

    # ---- pair, then decision & closing snapshots (book-grouped) ----
    paired = hm.pair_over_under(q, shin_fn=shin_no_vig_two_way_with_z)
    key = ["event_id", "book", "stat", "player_name", "line", "snapshot_time"]
    idmap = q[key + ["game_id", "player_id"]].drop_duplicates(subset=key)
    n_paired = len(paired)
    paired = paired.merge(idmap, on=key, how="left")
    if len(paired) != n_paired:
        raise typer.Exit("[P1][FATAL] id-map merge expanded rows — cardinality violation.")

    decision = hm.select_decision_snapshot(paired, lead_hours=decision_lead_hours)
    tagged = hm.select_open_close(paired)
    closing = tagged[tagged["is_closing"]].copy()
    opening = tagged[tagged["is_opening"]].copy()
    if not decision.empty:
        decision.to_parquet(out / "p1_decision_quotes.parquet", index=False)
    if not closing.empty:
        closing.to_parquet(out / "p1_closing_quotes.parquet", index=False)
    if not opening.empty:
        opening.to_parquet(out / "p1_opening_quotes.parquet", index=False)

    if decision.empty:
        return _write_inconclusive(out, "no quotes available at the decision snapshot")

    # ---- attach model P(over): raw + fold-safe calibrated (root cause #8) ----
    dec = decision.copy()
    dec["raw_p_over"] = [
        hm.p_over_conditional(pmf_by_key[(str(g), str(p), str(s))], float(ln))
        if (str(g), str(p), str(s)) in pmf_by_key else np.nan
        for g, p, s, ln in zip(dec["game_id"], dec["player_id"], dec["stat"], dec["line"])]
    dec = dec[dec["raw_p_over"].notna()].copy()
    if dec.empty:
        return _write_inconclusive(out, "no decision quotes matched fold-safe OOF PMFs")
    dec["actual"] = [actual_by_key.get((str(g), str(p), str(s)))
                     for g, p, s in zip(dec["game_id"], dec["player_id"], dec["stat"])]
    dec["game_date"] = [gamedate_by_key.get((str(g), str(p), str(s)))
                        for g, p, s in zip(dec["game_id"], dec["player_id"], dec["stat"])]
    # over_outcome for calibration fitting (push excluded)
    def _over_outcome(a, ln):
        if a is None or (float(ln).is_integer() and float(a) == float(ln)):
            return np.nan
        return 1.0 if float(a) > float(ln) else 0.0
    dec["over_outcome"] = [_over_outcome(a, ln) for a, ln in zip(dec["actual"], dec["line"])]
    dec["cal_p_over"] = hm.fold_safe_calibrated_prob_over(
        dec, prob_col="raw_p_over", outcome_col="over_outcome",
        stat_col="stat", date_col="game_date")

    publishable = set(dec["stat"].astype(str).unique())

    def _run(prob_col: str, label: str):
        recs = hm.build_executable_recs(
            dec, no_vig_fn=shin_no_vig_two_way_with_z, publishable_stats=publishable,
            edge_threshold=edge_threshold, min_market_prob=min_market_prob,
            max_shin_z=max_shin_z, prob_over_col=prob_col)
        recs = hm.settle_recs(recs, actual_by_key)
        if recs.empty:
            return recs, {}
        recs = recs.merge(
            pd.DataFrame([{"game_id": k[0], "player_id": k[1], "stat": k[2], "game_date": v}
                          for k, v in gamedate_by_key.items()]),
            on=KEYS, how="left")
        recs = hm.same_book_clv(recs, closing)
        recs.attrs["total_profit"] = float(recs[recs["won"].notna()]["profit"].sum())
        staked = float(recs["won"].notna().sum())
        recs.attrs["roi"] = recs.attrs["total_profit"] / staked if staked else float("nan")
        hm.assert_accounting_invariants(recs, q)  # 15 blocking invariants
        recs.to_parquet(out / f"p1_selected_bets_{label}.parquet", index=False)
        return recs, {}

    raw_recs, _ = _run("raw_p_over", "raw")
    cal_recs, _ = _run("cal_p_over", "calibrated")

    if cal_recs.empty:
        return _write_inconclusive(out, "no calibrated recommendations met the edge threshold",
                                   {"n_eligible_games": len(eligible_games)})

    # ---- coverage: denominator = eligible canonical OOF games (root cause #5) ----
    matched_games = set(dec["game_id"].astype(str).unique())
    coverage = {
        "eligible_canonical_games": len(eligible_games),
        "matched_with_usable_props": len(matched_games),
        "matched_no_supported_props": len(eligible_games - matched_games),
        "coverage_rate": len(matched_games) / max(len(eligible_games), 1),
    }
    (out / "p1_coverage_summary.json").write_text(json.dumps(coverage, indent=2))

    def _summarize(recs: pd.DataFrame, tag: str) -> dict:
        unders = recs[recs["side"] == "under"]; overs = recs[recs["side"] == "over"]
        n_under_games = int(unders["game_id"].nunique()) if not unders.empty else 0
        under_res = hm.grade(unders); over_res = hm.grade(overs); all_res = hm.grade(recs)
        verdict, reason = hm.verdict_with_reason(under_res, n_game_clusters=n_under_games)
        per_stat = {str(s): _res_to_dict(hm.grade(g)) for s, g in recs.groupby("stat")}
        clv = {
            "line_clv_mean": float(recs["line_clv"].dropna().mean()) if "line_clv" in recs else None,
            "price_clv_mean": float(recs["price_clv"].dropna().mean()) if "price_clv" in recs else None,
            "same_book_close_available": int(recs.get("clv_available", pd.Series(dtype=bool)).sum()),
        }
        return {
            "tag": tag, "verdict": verdict, "verdict_reason": reason,
            "n_recommendations": int(len(recs)), "under_pct": all_res.under_pct,
            "n_distinct_games": int(recs["game_id"].nunique()),
            "n_distinct_under_games": n_under_games,
            "all": _res_to_dict(all_res), "under": _res_to_dict(under_res),
            "over": _res_to_dict(over_res), "per_stat": per_stat, "clv": clv,
        }

    raw_summary = _summarize(raw_recs, "raw_fold_safe_oof") if not raw_recs.empty else {"tag": "raw_fold_safe_oof", "n_recommendations": 0}
    cal_summary = _summarize(cal_recs, "fold_safe_calibrated_production_equivalent")

    results = {
        "primary": "fold_safe_calibrated_production_equivalent",
        "verdict": cal_summary["verdict"],
        "verdict_reason": cal_summary["verdict_reason"],
        "edge_threshold": edge_threshold,
        "decision_lead_hours": decision_lead_hours,
        "coverage": coverage,
        "price_audit": {k: audit[k] for k in ("n_quotes", "n_rejected", "n_inside_100",
                                              "price_min", "price_max", "price_pctiles")},
        "raw_fold_safe": raw_summary,
        "fold_safe_calibrated": cal_summary,
        "accounting_invariants": "all 15 passed",
    }
    (out / "p1_results.json").write_text(json.dumps(results, indent=2, default=str))

    pd.DataFrame([{**{"stat": s}, **v} for s, v in cal_summary["per_stat"].items()]).to_csv(
        out / "p1_per_stat_summary.csv", index=False)

    u = cal_summary["under"]; o = cal_summary["over"]
    lines = [
        "# P1 Historical Validation — WNBA Under Lean (REPAIRED, executable prices)", "",
        f"**PRIMARY VERDICT (fold-safe calibrated, production-equivalent): {cal_summary['verdict']}**", "",
        f"_Reason: {cal_summary['verdict_reason']}_", "",
        "## Coverage (denominator = eligible canonical OOF games)",
        f"- eligible canonical games: {coverage['eligible_canonical_games']}",
        f"- matched with usable props: {coverage['matched_with_usable_props']} "
        f"(rate {coverage['coverage_rate']:.1%})", "",
        "## Price integrity",
        f"- quotes graded: {audit['n_quotes'] - audit['n_rejected']} | rejected (invalid American): "
        f"{audit['n_rejected']} | inside (-100,100): {audit['n_inside_100']}",
        f"- price range: {audit['price_min']:.0f} .. {audit['price_max']:.0f}",
        "- ROI is priced from EXACT executable quotes (no synthetic consensus prices).", "",
        "## Fold-safe calibrated (production-equivalent)",
        f"- recommendations: {cal_summary['n_recommendations']} (Under {cal_summary['under_pct']:.1f}%) "
        f"across {cal_summary['n_distinct_games']} games",
        f"- UNDER: n={u['n']} W/L/P={u['wins']}/{u['losses']}/{u['pushes']} hit={u['hit_rate']:.3f} "
        f"med_am={u['median_american_odds']:+.0f} avg_dec={u['avg_decimal_odds']:.3f} "
        f"ROI={u['roi']:+.4f} (95% CI {u['roi_ci95'][0]:+.4f}..{u['roi_ci95'][1]:+.4f})",
        f"- OVER: n={o['n']} hit={o['hit_rate']:.3f} ROI={o['roi']:+.4f} "
        f"(95% CI {o['roi_ci95'][0]:+.4f}..{o['roi_ci95'][1]:+.4f})",
        f"- Brier model/market: {u['brier']:.4f}/{u['market_brier']:.4f} | "
        f"logloss model-market: {u['model_minus_market_logloss']:+.4f}",
        f"- CLV (same-book close available for {cal_summary['clv']['same_book_close_available']} recs): "
        f"line {cal_summary['clv']['line_clv_mean']}, price {cal_summary['clv']['price_clv_mean']}", "",
        "## Raw fold-safe OOF (reported separately, NOT combined)",
        f"- recommendations: {raw_summary.get('n_recommendations', 0)}",
    ]
    if raw_summary.get("n_recommendations"):
        ru = raw_summary["under"]
        lines.append(f"- UNDER: n={ru['n']} hit={ru['hit_rate']:.3f} ROI={ru['roi']:+.4f} "
                     f"(95% CI {ru['roi_ci95'][0]:+.4f}..{ru['roi_ci95'][1]:+.4f})")
    (out / "p1_validation_report.md").write_text("\n".join(lines))
    typer.echo(f"[P1][EVAL] PRIMARY VERDICT={cal_summary['verdict']} | "
               f"cal_recs={cal_summary['n_recommendations']} under_ROI={u['roi']:+.4f} "
               f"CI={u['roi_ci95']} | coverage={coverage['coverage_rate']:.1%}")


if __name__ == "__main__":
    app()
