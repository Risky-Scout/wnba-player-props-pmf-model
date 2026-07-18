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

app = typer.Typer(add_completion=False)


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

    results = {}
    for stat, g in hold.groupby("stat"):
        res = fc.evaluate_stat(g)
        results[str(stat)] = {k: v for k, v in res.__dict__.items()}

    passed = [s for s, r in results.items() if r["passed"]]
    suppressed = [s for s, r in results.items() if not r["passed"]]
    summary = {
        "split": boundaries,
        "stats_passed": sorted(passed),
        "stats_suppressed": sorted(suppressed),
        "suppression_reasons": {s: results[s]["reasons"] for s in suppressed},
        "per_stat": results,
    }
    (out / "p2_forecasting_diagnostics.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = ["# P2 Forecasting Diagnostics (untouched holdout)", "",
             f"Holdout: {boundaries['holdout_start']} → {boundaries['holdout_end']} "
             f"({boundaries['holdout_rows']} rows, {boundaries['holdout_games']} games)",
             f"Development: {boundaries['dev_start']} → {boundaries['dev_end']} "
             f"({boundaries['dev_rows']} rows)", "",
             "| stat | n | bias | MAE | RMSE | CRPS | PIT-ECE | 80%cov | 90%cov | calib-ECE | PASS |",
             "|------|---|------|-----|------|------|---------|--------|--------|-----------|------|"]
    for s in sorted(results):
        r = results[s]; c80 = r["coverage"].get("0.8", {}); c90 = r["coverage"].get("0.9", {})
        lines.append(
            f"| {s} | {r['n']} | {r['bias']:+.2f} | {r['mae']:.2f} | {r['rmse']:.2f} | "
            f"{r['crps']:.3f} | {r['pit_ece']:.3f} | "
            f"{c80.get('empirical','?')}({'ok' if c80.get('compatible') else 'X'}) | "
            f"{c90.get('empirical','?')}({'ok' if c90.get('compatible') else 'X'}) | "
            f"{r['calib_ece']:.3f} | {'YES' if r['passed'] else 'NO'} |")
    lines += ["", f"**Passed (launchable forecast):** {', '.join(sorted(passed)) or 'none'}",
              f"**Suppressed:** {', '.join(sorted(suppressed)) or 'none'}"]
    for s in suppressed:
        lines.append(f"- `{s}`: {'; '.join(results[s]['reasons'])}")
    (out / "p2_forecasting_report.md").write_text("\n".join(lines))
    typer.echo(f"[P2][FORECAST] holdout {boundaries['holdout_start']}..{boundaries['holdout_end']} "
               f"| passed={sorted(passed)} suppressed={sorted(suppressed)}")


if __name__ == "__main__":
    app()
