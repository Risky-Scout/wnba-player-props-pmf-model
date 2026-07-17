"""Generate a historical review package for model evaluation.

Produces a self-contained Markdown report summarising:
  - Model architecture and methodology
  - OOF walk-forward calibration metrics (if available)
  - Per-stat calibration curve summaries
  - Market comparison summary (if available)
  - Known limitations and data lag notes
  - Sample prediction rows

Output:
    artifacts/historical_review/review_{date}.md

Usage:
    python scripts/build_historical_review.py
    python scripts/build_historical_review.py --report-date 2026-06-18 --lookback-days 30
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

_OOF_PATH = "data/oof/oof_player_stat_pmfs.parquet"
_OOF_CAL_PATH = "data/oof/oof_player_stat_pmfs_calibrated.parquet"
_RESULTS_PATH = "data/clv_tracking/results.parquet"
_EDGES_PATH = "deliveries/next_game/publishable_edges.parquet"
_MARKET_COMP_PATH = "deliveries/next_game/market_comparison.parquet"


def _pending(msg: str = "Awaiting first weekly calibration run") -> str:
    return f"*{msg}*"


def _fmt_float(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _oof_summary_table(oof_path: str) -> str:
    """Return Markdown table of per-stat OOF metrics or pending message."""
    p = Path(oof_path)
    if not p.exists():
        return _pending()

    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        return _pending(f"Could not load OOF data: {exc}")

    if df.empty:
        return _pending("OOF file is empty — re-run weekly calibration")

    # Normalize columns
    if "outcome" not in df.columns and "actual_outcome" in df.columns:
        df["outcome"] = df["actual_outcome"]
    if "calibration_eligible" in df.columns:
        df = df[df["calibration_eligible"] == True].copy()  # noqa: E712

    from wnba_props_model.models.simulation import json_to_pmf
    from wnba_props_model.evaluation.diagnostics import calibration_report

    if "pmf" not in df.columns and "pmf_json" in df.columns:
        df["pmf"] = df["pmf_json"].map(json_to_pmf)
    if "role_bucket" not in df.columns:
        df["role_bucket"] = "all"

    try:
        rep = calibration_report(df)
    except Exception as exc:
        return _pending(f"calibration_report failed: {exc}")

    # Aggregate to global per-stat
    agg = (
        rep.groupby("stat")
        .agg(
            n=("n", "sum"),
            ece=("ece", "mean"),
            pit_ks=("pit_ks", "mean"),
            mean_error=("mean_error", "mean"),
        )
        .reset_index()
        .sort_values("stat")
    )

    lines = [
        "| Stat | N | ECE | PIT KS | Mean Error |",
        "|------|---|-----|--------|------------|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['stat']} | {int(row['n']):,} | "
            f"{_fmt_float(row['ece'], 4)} | "
            f"{_fmt_float(row['pit_ks'], 4)} | "
            f"{_fmt_float(row['mean_error'], 4)} |"
        )
    return "\n".join(lines)


def _calibration_curve_summary(oof_raw_path: str, oof_cal_path: str) -> str:
    """Compare ECE before vs. after isotonic calibration."""
    rp, cp = Path(oof_raw_path), Path(oof_cal_path)
    if not rp.exists():
        return _pending()

    try:
        from wnba_props_model.models.simulation import json_to_pmf
        from wnba_props_model.evaluation.diagnostics import calibration_report

        def _load(path: Path) -> pd.DataFrame:
            df = pd.read_parquet(path).copy()
            if "outcome" not in df.columns and "actual_outcome" in df.columns:
                df["outcome"] = df["actual_outcome"]
            if "calibration_eligible" in df.columns:
                df = df[df["calibration_eligible"] == True].copy()  # noqa: E712
            if "pmf" not in df.columns and "pmf_json" in df.columns:
                df["pmf"] = df["pmf_json"].map(json_to_pmf)
            if "role_bucket" not in df.columns:
                df["role_bucket"] = "all"
            return df

        raw_rep = calibration_report(_load(rp)).groupby("stat")["ece"].mean()
        if cp.exists():
            cal_rep = calibration_report(_load(cp)).groupby("stat")["ece"].mean()
        else:
            cal_rep = None

        lines = [
            "| Stat | Raw ECE | Calibrated ECE | Improvement |",
            "|------|---------|----------------|-------------|",
        ]
        for stat in sorted(raw_rep.index):
            raw_v = raw_rep.get(stat, np.nan)
            cal_v = cal_rep.get(stat, np.nan) if cal_rep is not None else np.nan
            imp = (raw_v - cal_v) if (not np.isnan(raw_v) and not np.isnan(cal_v)) else np.nan
            lines.append(
                f"| {stat} | {_fmt_float(raw_v, 4)} | "
                f"{_fmt_float(cal_v, 4) if cal_rep is not None else '—'} | "
                f"{_fmt_float(imp, 4)} |"
            )
        return "\n".join(lines)
    except Exception as exc:
        return _pending(f"Could not compute calibration curves: {exc}")


def _market_summary(results_path: str, edges_path: str, lookback_days: int) -> str:
    """Summarise market edge and CLV performance."""
    rp = Path(results_path)
    ep = Path(edges_path)

    lines = []

    if rp.exists():
        try:
            res = pd.read_parquet(rp)
            if "game_date" in res.columns:
                cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
                res = res[res["game_date"].astype(str) >= cutoff]

            n_total = len(res)
            if n_total == 0:
                lines.append("*No scored predictions in the lookback window yet.*")
            else:
                lines.append(f"**Scored predictions (last {lookback_days} days):** {n_total:,}")
                if "model_edge_open" in res.columns:
                    e_valid = res["model_edge_open"].dropna()
                    if len(e_valid):
                        lines.append(f"**Model edge vs open (mean, NOT CLV):** {e_valid.mean():+.4f} ({len(e_valid):,} bets)")
                if "model_close_edge" in res.columns:
                    mce = res["model_close_edge"].dropna()
                    if len(mce):
                        lines.append(f"**Model edge vs close (mean, NOT CLV):** {mce.mean():+.4f} ({len(mce):,} bets)")
                if "price_clv" in res.columns:
                    pc = res["price_clv"].dropna()
                    if len(pc):
                        lines.append(f"**Price CLV vs closing (mean, outcome-independent):** {pc.mean():+.4f} ({len(pc):,} bets)")
                if "model_bin_logloss" in res.columns and "market_bin_logloss" in res.columns:
                    valid = res.dropna(subset=["model_bin_logloss", "market_bin_logloss"])
                    if len(valid):
                        ll_delta = (valid["model_bin_logloss"] - valid["market_bin_logloss"]).mean()
                        lines.append(f"**Log-loss delta vs market (mean):** {ll_delta:+.5f} (negative = model beats market)")
        except Exception as exc:
            lines.append(f"*Could not load results: {exc}*")
    else:
        lines.append(_pending("No scored predictions yet. CLV tracking starts after first live prediction day."))

    if ep.exists():
        try:
            edges = pd.read_parquet(ep)
            if not edges.empty:
                lines.append(f"\n**Publishable edges in latest slate:** {len(edges):,}")
                if "stat" in edges.columns:
                    by_stat = edges.groupby("stat")["edge_over"].agg(["count", "mean"])
                    lines.append("\n| Stat | # Edges | Mean Edge |")
                    lines.append("|------|---------|-----------|")
                    for stat, row in by_stat.iterrows():
                        lines.append(f"| {stat} | {int(row['count'])} | {row['mean']:+.3f} |")
        except Exception:
            pass

    return "\n".join(lines) if lines else _pending()


def _sample_predictions(edges_path: str, n: int = 5) -> str:
    """Show N sample predictions in human-readable format."""
    ep = Path(edges_path)
    if not ep.exists():
        return _pending("No edge report yet — run the daily pipeline first.")

    try:
        df = pd.read_parquet(ep)
        if df.empty:
            return _pending("Edge report is empty.")

        cols = [c for c in (
            "player_name", "stat", "line", "model_prob_over",
            "market_prob_over_no_vig", "edge_over", "pmf_mean",
        ) if c in df.columns]

        sample = df[cols].head(n)
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in sample.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    except Exception as exc:
        return _pending(f"Could not load sample predictions: {exc}")


@app.command()
def main(
    report_date: str | None = typer.Option(None, help="ISO date for report title (default: today)."),
    lookback_days: int = typer.Option(30, help="CLV lookback window in days."),
    oof_path: str = typer.Option(_OOF_PATH),
    oof_cal_path: str = typer.Option(_OOF_CAL_PATH),
    results_path: str = typer.Option(_RESULTS_PATH),
    edges_path: str = typer.Option(_EDGES_PATH),
    out_dir: str = typer.Option("artifacts/historical_review"),
) -> None:
    """Build a self-contained Markdown historical review report."""
    report_date = report_date or date.today().isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    typer.echo("Building historical review report...")

    oof_table = _oof_summary_table(oof_path)
    cal_curves = _calibration_curve_summary(oof_path, oof_cal_path)
    market_section = _market_summary(results_path, edges_path, lookback_days)
    samples = _sample_predictions(edges_path)

    report = f"""# WNBA Player Props PMF Model — Historical Review
**Report Date:** {report_date}
**Lookback Window:** {lookback_days} days

---

## 1. Model Overview

### What It Predicts
Full probability mass functions (PMFs) for WNBA player prop markets across
**7 direct stats** and **5 combination stats**:

- **Direct stats:** points (pts), rebounds (reb), assists (ast), 3-pointers made (fg3m),
  steals (stl), blocks (blk), turnovers (turnover)
- **Combo stats:** stocks (stl+blk), pts+ast, pts+reb, reb+ast, pts+reb+ast

### Methodology
- **Base model:** HistGradientBoosting regressor/classifier (HGB), one per stat.
  Minutes prediction is separated from per-stat prediction (two-stage model).
- **Hurdle model:** Sparse stats (blk) use a zero-inflated hurdle model.
- **Bayesian shrinkage:** Small-sample players are shrunk toward hierarchical
  Gamma-Poisson priors; shrinkage strength is learned per-stat from data.
- **Calibration:** PenaltyBlog-style isotonic regression calibrators fitted on
  strict chronological out-of-fold (OOF) predictions. One calibrator per
  stat × role-bucket (starter/core/bench/fringe/inactive_risk).
- **Combo PMFs:** Discrete convolution of base stat PMFs; Gaussian copula
  correction for empirical correlations (e.g. pts/ast r≈0.45).
- **Market comparison:** Shin's no-vig method for extracting true probabilities
  from BDL over/under American odds. Edge = model prob − Shin prob.

### Pipeline
```
Daily (9 AM ET):  BDL ingest → Features → HGB inference → Calibration
                  → PMFs → Edge report → D+1 delivery
Weekly (Mon):     Full OOF walk-forward → Fit calibrators → Gate check
Nightly (2 AM):   Pull actuals → Score predictions → CLV tracking
```

---

## 2. OOF Walk-Forward Calibration Metrics

*Computed on calibration-eligible model_oof rows (excludes prior_only and DNP rows).
ECE = Expected Calibration Error; PIT KS = Probability Integral Transform Kolmogorov-Smirnov.
Gate thresholds: ECE < 0.03, PIT KS < 0.075, |mean_error| < 0.15.*

{oof_table}

---

## 3. Calibration Curves (Before vs. After Isotonic Calibration)

{cal_curves}

---

## 4. Market Performance (Last {lookback_days} Days)

{market_section}

---

## 5. Known Limitations

| Limitation | Details |
|-----------|---------|
| BDL API latency | Player props and injury data may lag 15–60 min from official announcements |
| Injury reaction speed | Injury news from unofficial sources is not auto-ingested; use `apply_injury_news.py` for manual updates |
| Small sample sizes | Early-season projections rely more heavily on priors; accuracy improves as season progresses |
| Combo props | Calibrated on convolved OOF data; slightly wider calibration intervals than direct stats |
| No play-by-play features | Current feature set uses game-level box scores; play-by-play shot quality / on-off splits not included |
| New players (rookies) | Use league-average Gamma priors until ≥10 games played |
| Minutes model | Minutes prediction is the largest source of PMF variance; injury/rest news dominates |

---

## 6. Sample Predictions (Latest Slate)

{samples}

---

## 7. Forward Testing Guide

After running for at least 7 days, check:

```bash
# View CLV report
python scripts/generate_clv_report.py \\
  --results data/clv_tracking/results.parquet \\
  --lookback-days 7 \\
  --out-dir artifacts/audits

# Check calibration drift
python scripts/check_calibration_drift.py \\
  --scored-predictions data/clv_tracking/drift_window.parquet

# Market superiority gate (informational)
python scripts/verify_gates.py market \\
  data/clv_tracking/results.parquet --min-rows 50
```

Key metrics to track:
- `mean_true_clv > 0` per stat (model is finding real edge vs closing line)
- `logloss_delta < 0` per stat (model beats market log-loss)
- ECE drift stays below 0.06 (calibrators are still accurate)

---

*Generated by `scripts/build_historical_review.py` on {report_date}.*
"""

    out_path = out / f"review_{report_date}.md"
    out_path.write_text(report, encoding="utf-8")
    typer.echo(f"Historical review report → {out_path}")


if __name__ == "__main__":
    app()
