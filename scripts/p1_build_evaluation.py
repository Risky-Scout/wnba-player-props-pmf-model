"""P1 — historical evaluation & forced verdict (offline; leakage-free).

Joins the historical CLOSING consensus (real sportsbook lines) to the model's
fold-safe raw OOF predictions, reproduces the recommendation policy (no-vig edge,
side selection, eligibility), grades Over/Under against realized outcomes at the
real offered price (ROI, Brier, log-loss, game-date-clustered CIs), and issues a
forced SUPPORTED / NOT SUPPORTED / INCONCLUSIVE verdict on the Under lean.

Primary result uses the RAW fold-safe OOF PMF (walk-forward, no lookahead). The
calibrated OOF is NOT used as primary (its calibration saw all folds).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import historical_market as hm

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
    if arr.ndim == 2:  # [[val,prob],...]
        n = int(arr[:, 0].max()) + 1
        out = np.zeros(n)
        for val, prob in arr:
            out[int(val)] = float(prob)
        return out
    return arr


def _grade_df(df: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    """Reproduce the recommendation policy and settle each pick (no lookahead in
    selection: side/edge/eligibility use only model + market, never the outcome)."""
    rows = []
    for _, r in df.iterrows():
        try:
            pmf = _pmf_array(r["pmf_json"])
        except Exception:
            continue
        line = float(r["line"])
        m_over = hm.p_over_conditional(pmf, line)
        if not np.isfinite(m_over):
            continue
        k_over = float(r["market_prob_over_no_vig"])
        edge = m_over - k_over
        side = "over" if edge > 0 else "under"
        if abs(edge) < min_edge:
            continue  # not a recommendation
        actual = float(r["actual_outcome"])
        is_push = float(line).is_integer() and actual == line
        if side == "over":
            won = np.nan if is_push else (actual > line)
            price = r.get("over_odds")
            m_side, k_side = m_over, k_over
        else:
            won = np.nan if is_push else (actual < line)
            price = r.get("under_odds")
            m_side, k_side = 1.0 - m_over, 1.0 - k_over
        if price is None or (isinstance(price, float) and not np.isfinite(price)):
            continue
        rows.append({
            "game_id": r["game_id"], "player_id": r["player_id"], "stat": r["stat"],
            "game_date": r.get("game_date"), "season": str(r.get("game_date", ""))[:4],
            "line": line, "side": side, "edge": edge,
            "price_american": float(price),
            "won": won,
            "model_prob_side": m_side, "market_prob_side": k_side,
            "model_prob_over": m_over, "market_prob_over": k_over,
            "actual": actual,
        })
    return pd.DataFrame(rows)


def _res_to_dict(g: hm.GradeResult) -> dict:
    d = g.__dict__.copy()
    d["roi_ci95"] = list(g.roi_ci95)
    d.pop("extra", None)
    return d


@app.command()
def main(
    oof: str = typer.Option(...),
    closing_consensus: str = typer.Option(...),
    out_dir: str = typer.Option("artifacts/p1"),
    min_edge: float = typer.Option(0.02, help="Min |edge| to count as a recommendation (2pp = production floor)."),
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(oof)
    cc = pd.read_parquet(closing_consensus)

    verdict_stub = {"verdict": "INCONCLUSIVE", "min_edge": min_edge}
    if cc.empty:
        verdict_stub["reason"] = "no historical closing consensus rows (coverage gap)"
        (out / "p1_results.json").write_text(json.dumps(verdict_stub, indent=2))
        (out / "p1_validation_report.md").write_text(
            "# P1 Validation\n\nVERDICT: **INCONCLUSIVE** — no historical closing lines were "
            "obtained for the OOF games (provider coverage gap). Under lean cannot be graded.\n")
        typer.echo("[P1][EVAL] No closing consensus — INCONCLUSIVE.")
        return

    for k in KEYS:
        oof[k] = oof[k].astype("string")
        cc[k] = cc[k].astype("string")

    # Leakage guard: fold training must precede the evaluated game.
    hm.assert_no_lookahead(oof)

    # OOF must be unique per key for a clean 1:1 join.
    oof_u = oof.dropna(subset=["actual_outcome", "pmf_json"]).drop_duplicates(subset=KEYS)
    if cc.duplicated(subset=KEYS).any():
        raise typer.Exit("[P1][FATAL] closing consensus not unique per (game_id,player_id,stat).")
    n_before = len(cc)
    ev = cc.merge(oof_u[KEYS + ["pmf_json", "actual_outcome", "game_date"]], on=KEYS, how="inner")
    if len(ev) > n_before:
        raise typer.Exit("[P1][FATAL] many-to-many join expansion detected.")
    ev.to_parquet(out / "p1_oof_plus_closing_eval.parquet", index=False)

    graded = _grade_df(ev, min_edge)
    n_consensus = int(len(ev))
    if graded.empty:
        res = {"verdict": "INCONCLUSIVE", "reason": "no recommendations met edge threshold",
               "n_consensus_observations": n_consensus, "min_edge": min_edge}
        (out / "p1_results.json").write_text(json.dumps(res, indent=2))
        (out / "p1_validation_report.md").write_text(
            f"# P1 Validation\n\nVERDICT: **INCONCLUSIVE** — {n_consensus} consensus observations "
            f"joined, but no recommendations met the {min_edge:.0%} edge threshold.\n")
        typer.echo("[P1][EVAL] No recommendations — INCONCLUSIVE.")
        return

    unders = graded[graded["side"] == "under"]
    overs = graded[graded["side"] == "over"]
    all_res = hm.grade(graded)
    under_res = hm.grade(unders)
    over_res = hm.grade(overs)
    verdict = hm.forced_verdict(under_res)

    per_stat = {}
    for stat, g in graded.groupby("stat"):
        per_stat[str(stat)] = _res_to_dict(hm.grade(g))

    results = {
        "verdict": verdict,
        "min_edge": min_edge,
        "n_consensus_observations": n_consensus,
        "n_recommendations": int(len(graded)),
        "under_pct": all_res.under_pct,
        "all": _res_to_dict(all_res),
        "under": _res_to_dict(under_res),
        "over": _res_to_dict(over_res),
        "per_stat": per_stat,
        "seasons": sorted(graded["season"].dropna().unique().tolist()),
        "primary_prediction_source": "raw_fold_safe_oof_pmf",
    }
    (out / "p1_results.json").write_text(json.dumps(results, indent=2, default=str))

    # per-stat CSV
    pd.DataFrame([{**{"stat": s}, **v} for s, v in per_stat.items()]).to_csv(
        out / "p1_per_stat_summary.csv", index=False)

    # human-readable report
    u = under_res
    lines = [
        "# P1 Historical Validation — WNBA Under Lean", "",
        f"**VERDICT: {verdict}**", "",
        f"- Consensus observations graded: {n_consensus}",
        f"- Recommendations (|edge| >= {min_edge:.0%}): {len(graded)}",
        f"- Under share of recommendations: {all_res.under_pct:.1f}%",
        "",
        "## Under recommendations",
        f"- n={u.n}, wins={u.wins}, losses={u.losses}, pushes={u.pushes}",
        f"- hit rate: {u.hit_rate:.3f} | avg odds: {u.avg_american_odds:+.0f} | "
        f"break-even: {u.breakeven_rate:.3f}",
        f"- ROI: {u.roi:+.4f} (95% CI {u.roi_ci95[0]:+.4f} .. {u.roi_ci95[1]:+.4f})",
        f"- Brier model/market: {u.brier:.4f} / {u.market_brier:.4f} | "
        f"logloss model-market: {u.model_minus_market_logloss:+.4f}",
        "",
        "## Over recommendations",
        f"- n={over_res.n}, hit rate: {over_res.hit_rate:.3f}, ROI: {over_res.roi:+.4f} "
        f"(95% CI {over_res.roi_ci95[0]:+.4f} .. {over_res.roi_ci95[1]:+.4f})",
        "",
        "Primary result uses fold-safe raw OOF PMFs (no lookahead). ROI is price-adjusted; "
        "hit rate alone is not used for the verdict.",
    ]
    (out / "p1_validation_report.md").write_text("\n".join(lines))
    typer.echo(f"[P1][EVAL] VERDICT={verdict} | recs={len(graded)} under%={all_res.under_pct:.1f} "
               f"under_ROI={u.roi:+.4f} CI={u.roi_ci95}")


if __name__ == "__main__":
    app()
