"""P2 Phase 2+3 — chronological split + forecasting-model validation on the
untouched holdout. Evaluates the raw OOF PMF per stat and applies the per-stat
forecasting launch gate. Suppresses stats that fail; a strong aggregate cannot
hide an individual failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import forecasting as fc  # noqa: E402
from wnba_props_model.evaluation.pmf_recalibration import fold_safe_pmf_recalibration  # noqa: E402

app = typer.Typer(add_completion=False)


def _eval_all(frame, pmf_col):
    if pmf_col != "pmf_json":
        g = frame.drop(columns=["pmf_json"]).rename(columns={pmf_col: "pmf_json"})
    else:
        g = frame
    return {str(s): {k: v for k, v in fc.evaluate_stat(sg).__dict__.items()}
            for s, sg in g.groupby("stat")}


@app.command()
def main(
    oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
    holdout_start: str = typer.Option("2026-07-01", help="First date of the untouched holdout (inclusive)."),
    out_dir: str = typer.Option("artifacts/p2"),
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    # only rows where the player actually played (forecasting a stat for a DNP is undefined)
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712

    cut = pd.Timestamp(holdout_start)
    dev = df[df["game_date"] < cut]
    hold = df[df["game_date"] >= cut]

    boundaries = {
        "dev_start": str(dev["game_date"].min().date()) if len(dev) else None,
        "dev_end": str(dev["game_date"].max().date()) if len(dev) else None,
        "dev_rows": int(len(dev)), "dev_games": int(dev["game_id"].nunique()),
        "holdout_start": str(hold["game_date"].min().date()) if len(hold) else None,
        "holdout_end": str(hold["game_date"].max().date()) if len(hold) else None,
        "holdout_rows": int(len(hold)), "holdout_games": int(hold["game_id"].nunique()),
        "holdout_cutoff": holdout_start,
    }
    (out / "p2_split_boundaries.json").write_text(json.dumps(boundaries, indent=2))

    # Fold-safe calibrated PMFs (Challenger C): fit on strictly-earlier folds only.
    df_cal = df.copy()
    df_cal["pmf_json_cal"] = fold_safe_pmf_recalibration(df_cal)
    hold_cal = df_cal[df_cal["game_date"] >= cut]

    results = _eval_all(hold, "pmf_json")               # raw
    results_cal = _eval_all(hold_cal, "pmf_json_cal")   # fold-safe calibrated

    # Launch gate uses the CALIBRATED (production-equivalent) model.
    passed = [s for s, r in results_cal.items() if r["passed"]]
    suppressed = [s for s, r in results_cal.items() if not r["passed"]]
    summary = {
        "split": boundaries,
        "primary_model": "fold_safe_calibrated",
        "stats_passed": sorted(passed),
        "stats_suppressed": sorted(suppressed),
        "suppression_reasons": {s: results_cal[s]["reasons"] for s in suppressed},
        "per_stat_raw": results,
        "per_stat_calibrated": results_cal,
    }
    (out / "p2_forecasting_diagnostics.json").write_text(json.dumps(summary, indent=2, default=str))

    def _table(res, title):
        out_lines = [f"## {title}", "",
                     "| stat | n | dates | bias | RMSE | CRPS | logS | PIT-KS-p | 50%cov | 80%cov | 90%cov | PASS |",
                     "|------|---|-------|------|------|------|------|----------|--------|--------|--------|------|"]
        for s in sorted(res):
            r = res[s]
            def _c(level):
                c = r["coverage"].get(level, {})
                emp = c.get("empirical", "?"); bad = c.get("fail", False)
                return f"{emp}{'✗' if bad else '✓'}"
            out_lines.append(
                f"| {s} | {r['n']} | {r.get('n_dates','?')} | {r['bias']:+.2f} | {r['rmse']:.2f} | "
                f"{r['crps']:.3f} | {r.get('log_score', float('nan')):.3f} | {r.get('pit_ks_p', float('nan')):.4f} | "
                f"{_c('0.5')} | {_c('0.8')} | {_c('0.9')} | {'YES' if r['passed'] else 'NO'} |")
        return out_lines

    lines = ["# P2 Forecasting Diagnostics (untouched holdout)", "",
             f"Holdout: {boundaries['holdout_start']} → {boundaries['holdout_end']} "
             f"({boundaries['holdout_rows']} rows, {boundaries['holdout_games']} games)",
             f"Development: {boundaries['dev_start']} → {boundaries['dev_end']} "
             f"({boundaries['dev_rows']} rows)", "",
             "Primary launch gate uses the fold-safe calibrated model.", ""]
    lines += _table(results, "Raw OOF PMF")
    lines += [""] + _table(results_cal, "Fold-safe calibrated PMF (PRIMARY)")
    lines += ["", f"**Passed (launchable forecast):** {', '.join(sorted(passed)) or 'none'}",
              f"**Suppressed:** {', '.join(sorted(suppressed)) or 'none'}"]
    for s in suppressed:
        lines.append(f"- `{s}`: {'; '.join(results_cal[s]['reasons'])}")
    (out / "p2_forecasting_report.md").write_text("\n".join(lines))
    typer.echo(f"[P2][FORECAST] holdout {boundaries['holdout_start']}..{boundaries['holdout_end']} "
               f"| passed={sorted(passed)} suppressed={sorted(suppressed)}")


if __name__ == "__main__":
    app()
