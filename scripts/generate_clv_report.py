"""Weekly CLV and calibration performance report.

Reads cumulative scored predictions from data/clv_tracking/results.parquet
and outputs:
  - artifacts/audits/clv_report_{date}.json  (machine-readable)
  - artifacts/audits/clv_summary_{date}.md   (human-readable)

Metrics reported per stat:
  - CLV (Closing Line Value): mean model_prob - market_prob, weighted by |edge|
  - Ignorance Score vs. market Ignorance Score (PenaltyBlog primary metric)
  - NLL per stat
  - RPS per stat
  - ECE-style calibration check
  - Hit rate vs. expected hit rate

Usage:
    python scripts/generate_clv_report.py \\
        --results data/clv_tracking/results.parquet \\
        --lookback-days 30 \\
        --out-dir artifacts/audits
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _date_cluster_ci(df: pd.DataFrame, value_col: str, date_col: str,
                     n_boot: int = 2000, seed: int = 20260722) -> tuple[float, float]:
    """95% CI for the mean of ``value_col`` using a paired date-block bootstrap.

    Games on the same date share market/league conditions, so a plain row bootstrap
    understates uncertainty. Resample whole game-dates with replacement instead.
    Returns (nan, nan) when there are fewer than 2 date clusters.
    """
    if date_col not in df.columns or df.empty:
        return (float("nan"), float("nan"))
    codes, labels = pd.factorize(df[date_col], sort=True)
    k = len(labels)
    if k < 2:
        return (float("nan"), float("nan"))
    vals = df[value_col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        counts = rng.multinomial(k, np.full(k, 1.0 / k))
        w = counts[codes].astype(float)
        means[b] = float(np.sum(vals * w) / max(np.sum(w), 1e-9))
    return (round(float(np.percentile(means, 2.5)), 5),
            round(float(np.percentile(means, 97.5)), 5))


def _calibration_curve(
    model_probs: np.ndarray,
    hit_results: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Compute binned calibration curve (predicted prob vs. empirical hit rate).

    PenaltyBlog's 250M-bet study showed this is the fastest way to spot which
    stat categories are over- or under-confident.

    Returns list of dicts with keys: bin_center, mean_predicted, empirical_hit_rate, n.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (model_probs >= lo) & (model_probs < hi)
        if i == n_bins - 1:
            mask = (model_probs >= lo) & (model_probs <= hi)
        if mask.sum() == 0:
            continue
        curve.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "bin_center": round(float((lo + hi) / 2), 3),
            "mean_predicted": round(float(model_probs[mask].mean()), 4),
            "empirical_hit_rate": round(float(hit_results[mask].mean()), 4),
            "n": int(mask.sum()),
        })
    return curve


def _kelly_roi_stats(df: pd.DataFrame) -> dict:
    """Compute flat-bet ROI and Kelly-weighted ROI from scored predictions.

    Requires: model_prob_over_final, hit_result, market_prob_over_no_vig columns.
    Kelly ROI uses the quarter-Kelly stake fraction for each bet.
    """
    result: dict = {}
    required = {"model_prob_over_final", "hit_result", "market_prob_over_no_vig"}
    if not required.issubset(df.columns) or df.empty:
        return result

    valid = df[["model_prob_over_final", "hit_result", "market_prob_over_no_vig"]].dropna()
    if valid.empty:
        return result

    # Flat-bet ROI: assume we bet $1 on every model-favored edge
    # Payoff: market implied odds = 1/market_prob_over_no_vig - 1
    model_bets_over = valid["model_prob_over_final"] > valid["market_prob_over_no_vig"]
    p_win    = np.where(model_bets_over, valid["model_prob_over_final"], 1 - valid["model_prob_over_final"])
    p_mkt    = np.where(model_bets_over, valid["market_prob_over_no_vig"], 1 - valid["market_prob_over_no_vig"])
    hit      = np.where(model_bets_over, valid["hit_result"], 1 - valid["hit_result"])
    b        = np.where(p_mkt > 0, 1.0 / p_mkt - 1.0, 0.0)  # net odds per $1

    flat_pnl = hit * b - (1 - hit)
    result["flat_bet_roi"] = round(float(np.mean(flat_pnl)), 4)
    result["flat_bet_roi_pct"] = f"{float(np.mean(flat_pnl)):.1%}"

    # Quarter-Kelly ROI
    k = 0.25
    kelly_stakes = np.clip((b * p_win - (1 - p_win)) / np.where(b > 0, b, 1.0) * k, 0, 0.25)
    kelly_pnl    = kelly_stakes * (hit * b - (1 - hit))
    total_staked = kelly_stakes.sum()
    result["kelly_quarter_roi"]     = round(float(kelly_pnl.sum() / max(total_staked, 1e-9)), 4)
    result["kelly_quarter_roi_pct"] = f"{float(kelly_pnl.sum() / max(total_staked, 1e-9)):.1%}"
    result["n_bets"]                = int(len(valid))
    result["n_model_over_bets"]     = int(model_bets_over.sum())

    return result


def _per_book_margin(df: pd.DataFrame) -> list[dict]:
    """Compute average book margin (overround) and mean Shin z per vendor.

    Shin z < 0.03 indicates a soft market (few informed bettors) — target these.
    """
    if "vendor" not in df.columns:
        return []
    result = []
    for vendor, g in df.groupby("vendor"):
        rec: dict = {"vendor": str(vendor), "n": len(g)}
        # Overround = (1/over_implied + 1/under_implied) - 1
        if "over_odds" in g.columns and "under_odds" in g.columns:
            from wnba_props_model.models.market import american_to_prob  # noqa: PLC0415
            p_o = g["over_odds"].map(american_to_prob).dropna()
            p_u = g["under_odds"].map(american_to_prob).dropna()
            if len(p_o) > 0 and len(p_u) > 0:
                overround = ((p_o + p_u) - 1.0).mean()
                rec["mean_overround"] = round(float(overround), 4)
                rec["mean_margin_pct"] = f"{float(overround):.1%}"
        if "shin_z" in g.columns and g["shin_z"].notna().any():
            rec["mean_shin_z"] = round(float(g["shin_z"].dropna().mean()), 4)
        result.append(rec)
    result.sort(key=lambda r: r.get("mean_shin_z", 1.0))  # softest markets first
    return result


def _stat_report(df: pd.DataFrame) -> list[dict]:
    rows = []
    for stat, g in df.groupby("stat"):
        rec: dict = {"stat": stat, "n": len(g)}

        # PMF scores
        if "pmf_nll" in g and g["pmf_nll"].notna().any():
            rec["mean_nll"] = float(g["pmf_nll"].mean())
        if "pmf_rps" in g and g["pmf_rps"].notna().any():
            rec["mean_rps"] = float(g["pmf_rps"].mean())

        # Binary over/under market comparison
        mkt_valid = g["market_prob_over_no_vig"].notna() if "market_prob_over_no_vig" in g else pd.Series(False, index=g.index)
        if mkt_valid.any():
            gv = g[mkt_valid]
            rec["n_market_lines"] = int(mkt_valid.sum())

            if "model_ignorance_score" in gv and gv["model_ignorance_score"].notna().any():
                rec["mean_model_ignorance_score"] = float(gv["model_ignorance_score"].mean())
            if "market_ignorance_score" in gv and gv["market_ignorance_score"].notna().any():
                rec["mean_market_ignorance_score"] = float(gv["market_ignorance_score"].mean())
            if "model_bin_logloss" in gv and "market_bin_logloss" in gv:
                rec["logloss_delta"] = float((gv["model_bin_logloss"] - gv["market_bin_logloss"]).mean())
            # Model edge vs close (model-vs-market prob diff; NOT CLV; outcome-independent)
            if "model_close_edge" in gv and gv["model_close_edge"].notna().any():
                rec["mean_model_close_edge"] = float(gv["model_close_edge"].mean())
                rec["positive_model_close_edge_pct"] = float((gv["model_close_edge"] > 0).mean())
            # True (economic) CLV — market movement, outcome-independent
            if "price_clv" in gv and gv["price_clv"].notna().any():
                rec["mean_price_clv"] = float(gv["price_clv"].mean())
                rec["positive_price_clv_pct"] = float((gv["price_clv"] > 0).mean())
            if "line_clv" in gv and gv["line_clv"].notna().any():
                rec["mean_line_clv"] = float(gv["line_clv"].mean())

            if "hit_result" in gv and "model_prob_over_final" in gv:
                rec["empirical_hit_rate"] = float(gv["hit_result"].mean())
                rec["mean_model_prob"] = float(gv["model_prob_over_final"].mean())
                rec["mean_market_prob"] = float(gv["market_prob_over_no_vig"].mean())

                # Calibration curve (10-bin)
                hit_arr   = gv["hit_result"].dropna().values.astype(float)
                prob_arr  = gv.loc[gv["hit_result"].notna(), "model_prob_over_final"].values.astype(float)
                if len(hit_arr) >= 20:
                    rec["calibration_curve"] = _calibration_curve(prob_arr, hit_arr)

        rows.append(rec)
    return rows


@app.command()
def main(
    results: str = typer.Option("data/clv_tracking/results.parquet", help="Cumulative scored results parquet."),
    lookback_days: int = typer.Option(30, help="Number of past days to include in report."),
    out_dir: str = typer.Option("artifacts/audits", help="Output directory for report files."),
    report_date: str | None = typer.Option(None, help="Report reference date (YYYY-MM-DD, default today)."),
) -> None:
    """Generate weekly CLV and calibration report."""
    today_str = report_date or date.today().isoformat()
    today = date.fromisoformat(today_str)
    cutoff = (today - timedelta(days=lookback_days)).isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = Path(results)
    if not results_path.exists():
        typer.echo(f"[WARN] No results file at {results_path} — nothing to report")
        return

    df = pd.read_parquet(results_path)
    if df.empty:
        typer.echo("[WARN] Results file is empty")
        return

    df["game_date"] = df["game_date"].astype(str)
    recent = df[df["game_date"] >= cutoff].copy()
    typer.echo(f"Loaded {len(recent):,} rows from {cutoff} to {today_str} ({lookback_days}d window)")

    stat_rows = _stat_report(recent)

    # Overall CLV and ROI
    roi_stats   = _kelly_roi_stats(recent)
    book_margins = _per_book_margin(recent)

    overall: dict = {
        "report_date": today_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "n_total_rows": len(recent),
        "n_game_dates": recent["game_date"].nunique() if "game_date" in recent else 0,
        "stats": stat_rows,
        "roi": roi_stats,
        "per_book_margins": book_margins,
    }

    # Overall model-close-edge summary (model-vs-market prob diff; NOT CLV)
    if "model_close_edge" in recent.columns and recent["model_close_edge"].notna().any():
        overall["overall_mean_model_close_edge"] = float(recent["model_close_edge"].mean())
        overall["overall_positive_model_close_edge_pct"] = float((recent["model_close_edge"] > 0).mean())

    # Economic CLV (market movement, outcome-independent). Signed: can be negative.
    if "price_clv" in recent.columns and recent["price_clv"].notna().any():
        _clv = recent.dropna(subset=["price_clv"])
        overall["overall_mean_price_clv"] = float(_clv["price_clv"].mean())
        overall["overall_positive_price_clv_pct"] = float((_clv["price_clv"] > 0).mean())
        overall["overall_n_price_clv"] = int(len(_clv))
        # 95% date-clustered bootstrap CI (games on a date are not independent).
        lo, hi = _date_cluster_ci(_clv, "price_clv", "game_date")
        overall["price_clv_ci95_low"] = lo
        overall["price_clv_ci95_high"] = hi
        # Honest gate: positive CLV is only claimable when the clustered lower bound
        # excludes zero. Otherwise the number is not distinguishable from break-even.
        overall["positive_clv_established"] = bool(lo == lo and lo > 0.0)

    if "logloss_delta" in [r.get("logloss_delta") for r in stat_rows if "logloss_delta" in r]:
        deltas = [r["logloss_delta"] for r in stat_rows if "logloss_delta" in r]
        overall["mean_logloss_delta_vs_market"] = float(np.mean(deltas))
        overall["model_beats_market"] = bool(np.mean(deltas) < 0)

    # JSON report
    json_path = out / f"clv_report_{today_str}.json"
    json_path.write_text(json.dumps(overall, indent=2))
    typer.echo(f"Wrote JSON report → {json_path}")

    # Markdown summary
    lines = [
        f"# CLV & Calibration Report — {today_str}",
        f"*{lookback_days}-day window: {cutoff} → {today_str}*",
        f"*{len(recent):,} scored predictions across {overall.get('n_game_dates',0)} game dates*",
        "",
        "## Overall",
        f"- Mean CLV: {overall.get('overall_mean_clv', 'N/A'):.4f}" if "overall_mean_clv" in overall else "- CLV: N/A (no market data)",
        f"- Positive CLV %: {overall.get('overall_positive_clv_pct', 0):.1%}" if "overall_mean_clv" in overall else "",
        f"- Model beats market (log loss): {overall.get('model_beats_market', 'N/A')}",
        "",
        "## Per-Stat Summary",
        "",
        "| Stat | N | NLL | RPS | LogLoss Δ | Mean CLV | Hit Rate | Model Prob |",
        "|------|---|-----|-----|-----------|----------|----------|------------|",
    ]
    for r in stat_rows:
        lines.append(
            f"| {r['stat']} | {r['n']} "
            f"| {r.get('mean_nll', float('nan')):.3f} "
            f"| {r.get('mean_rps', float('nan')):.4f} "
            f"| {r.get('logloss_delta', float('nan')):.4f} "
            f"| {r.get('mean_clv', float('nan')):.4f} "
            f"| {r.get('empirical_hit_rate', float('nan')):.3f} "
            f"| {r.get('mean_model_prob', float('nan')):.3f} |"
        )

    # ROI section
    if roi_stats:
        lines += [
            "",
            "## ROI & Kelly Performance",
            "",
            f"- Flat-bet ROI: {roi_stats.get('flat_bet_roi_pct', 'N/A')}",
            f"- Quarter-Kelly ROI: {roi_stats.get('kelly_quarter_roi_pct', 'N/A')}",
            f"- Total bets: {roi_stats.get('n_bets', 'N/A')} "
            f"(OVER: {roi_stats.get('n_model_over_bets', '?')})",
        ]
        if "overall_mean_model_close_edge" in overall:
            lines.append(f"- Mean model edge vs close (NOT CLV): {overall['overall_mean_model_close_edge']:+.4f}")
        if "overall_mean_price_clv" in overall:
            lines.append(f"- Mean price CLV (market movement): {overall['overall_mean_price_clv']:+.4f}")

    # Per-book margin section
    if book_margins:
        lines += [
            "",
            "## Per-Book Margin Analysis (30d, softest first)",
            "",
            "| Book | N | Margin | Shin z |",
            "|------|---|--------|--------|",
        ]
        for b in book_margins:
            lines.append(
                f"| {b['vendor']} | {b['n']} "
                f"| {b.get('mean_margin_pct', 'N/A')} "
                f"| {b.get('mean_shin_z', 'N/A')} |"
            )
        lines.append("*Low Shin z = fewer informed bettors = softer market = higher confidence in model edge.*")

    lines += [
        "",
        "---",
        "*Ignorance Score (log loss in bits) is the primary binary metric per PenaltyBlog methodology.*",
        "*CLV = (model_prob - market_prob) × (2 × hit_result - 1). Positive CLV = model edge was correct.*",
        "*True CLV uses closing-line props (most efficient price) rather than open-line snapshot.*",
    ]
    md_path = out / f"clv_summary_{today_str}.md"
    md_path.write_text("\n".join(lines))
    typer.echo(f"Wrote Markdown report → {md_path}")

    # Print quick summary
    if "overall_mean_clv" in overall:
        typer.echo(f"\nMean CLV ({lookback_days}d): {overall['overall_mean_clv']:+.4f}")
    if "model_beats_market" in overall:
        status = "YES" if overall["model_beats_market"] else "NO"
        typer.echo(f"Model beats market (log loss): {status}")


if __name__ == "__main__":
    app()
