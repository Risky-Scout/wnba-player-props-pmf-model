"""P3 final correction — STRICTLY PREQUENTIAL five-block forecast gate.

For each of five chronological 5-date holdout blocks, choose among {raw, location,
location-and-scale} using ONLY dates before that block (by CRPS on pre-block rows),
freeze that choice, and score the block out-of-sample. Concatenate the five block
ledgers and run the forecast gate ONCE on that strictly-prequential ledger.

NO candidate is ever selected by scoring it on the complete 25-date holdout.
Emits the concatenated ledger, its hash, per-block choices, and the per-stat registry.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import forecasting as fc  # noqa: E402
from wnba_props_model.evaluation.pmf_recalibration import fold_safe_pmf_recalibration  # noqa: E402

app = typer.Typer(add_completion=False)
VARIANT_COL = {"raw": "pmf_raw", "location": "pmf_loc", "location_and_scale": "pmf_ls"}


def _mean_crps(frame: pd.DataFrame, col: str) -> float:
    vals = []
    for pj, y in zip(frame[col], frame["actual_outcome"]):
        p = fc.pmf_to_array(pj)
        if p.size:
            vals.append(fc.crps_discrete(p, int(y)))
    return float(np.mean(vals)) if vals else float("inf")


def _baseline(devcal_actuals, holdout_actuals):
    ms = int(max(np.max(holdout_actuals), np.max(devcal_actuals))) + 5
    b = np.zeros(ms + 1)
    for a in devcal_actuals:
        ai = int(round(a))
        if 0 <= ai <= ms:
            b[ai] += 1
    b = b / b.sum() if b.sum() else b
    return {"crps": float(np.mean([fc.crps_discrete(b, int(y)) for y in holdout_actuals])),
            "log_score": float(np.mean([fc.log_score(b, int(y)) for y in holdout_actuals])),
            "matched_width_80": float(fc.matched_mass_width(b, 0.8))}


def prequential_ledger(df: pd.DataFrame, stat: str, blocks, devcal_max_date):
    """Return (ledger_df, per_block_choices) using strictly pre-block selection."""
    s = df[df["stat"] == stat]
    rows = []; choices = []
    for k, block in enumerate(blocks):
        block_start = min(block)
        before = s[s["game_date"] < block_start]
        blk = s[s["game_date"].isin(set(block))]
        if len(before) < 50:
            variant = "raw"
        else:
            crps = {v: _mean_crps(before, c) for v, c in VARIANT_COL.items()}
            variant = min(crps, key=crps.get)
        choices.append({"block": k, "block_start": str(pd.Timestamp(block_start).date()),
                        "n_before": int(len(before)), "n_block": int(len(blk)), "variant": variant})
        col = VARIANT_COL[variant]
        b2 = blk.copy(); b2["pmf_json"] = b2[col]
        rows.append(b2)
    ledger = pd.concat(rows, ignore_index=True) if rows else s.iloc[0:0].copy()
    return ledger, choices


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         out_dir: str = typer.Option("artifacts/p3"),
         registry_out: str = typer.Option("config/stat_registry.json"),
         holdout_dates: int = typer.Option(25)) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712

    df["pmf_raw"] = df["pmf_json"]
    df["pmf_loc"] = fold_safe_pmf_recalibration(df, use_dispersion=False)
    df["pmf_ls"] = fold_safe_pmf_recalibration(df, use_dispersion=True)

    dates = np.sort(df["game_date"].unique())
    hold = dates[-holdout_dates:]
    blocks = [list(b) for b in np.array_split(hold, 5)]
    devcal = df[df["game_date"] < min(hold)]
    devcal_max = max(devcal["game_date"]) if len(devcal) else None

    registry = {}; report = {}; all_ledger_parts = []
    for stat in sorted(df["stat"].unique()):
        ledger, choices = prequential_ledger(df, stat, blocks, devcal_max)
        if ledger.empty:
            continue
        base = _baseline(devcal[devcal["stat"] == stat]["actual_outcome"].values,
                         ledger["actual_outcome"].values)
        r = fc.evaluate_stat(ledger, baseline=base)   # SINGLE gate run on prequential ledger
        registry[stat] = {
            "forecast_allowed": bool(r.forecast_allowed),
            "market_comparison_allowed": bool(r.market_comparison_allowed),
            "betting_recommendation_allowed": False,
            "n": r.n, "n_dates": r.n_dates, "crps": round(r.crps, 4),
            "crps_vs_baseline": round(r.crps_vs_baseline, 4), "pit_ks_p": round(r.pit_ks_p, 4),
            "block_choices": choices,
            "suppression_reason": "" if r.forecast_allowed else "; ".join(r.reasons),
        }
        report[stat] = registry[stat]
        all_ledger_parts.append(ledger[["game_id", "player_id", "stat", "pmf_json", "actual_outcome"]])

    full_ledger = pd.concat(all_ledger_parts, ignore_index=True)
    full_ledger.to_parquet(out / "p3_prequential_ledger.parquet", index=False)
    ledger_hash = hashlib.sha256(
        pd.util.hash_pandas_object(full_ledger, index=False).values.tobytes()).hexdigest()[:16]

    certified = sorted([s for s, e in registry.items() if e["forecast_allowed"]])
    (Path(registry_out)).write_text(json.dumps(registry, indent=2))
    (out / "p3_prequential_result.json").write_text(json.dumps(
        {"certified": certified, "ledger_hash": ledger_hash,
         "blocks": [[str(pd.Timestamp(d).date()) for d in b] for b in blocks],
         "per_stat": report}, indent=2, default=str))
    status = "LIVE_VALIDATED_FORECAST_ONLY" if certified else "BLOCKED_MODEL"
    lines = ["# P3 strictly-prequential five-block gate", "",
             f"ledger_hash: `{ledger_hash}`  | certified: {certified or 'none'}  | status: {status}", "",
             "| stat | n | dates | CRPS | dCRPS(base) | PIT-KS-p | pass | block variants |",
             "|------|---|-------|------|-------------|----------|------|----------------|"]
    for stat in sorted(report):
        e = report[stat]
        variants = ",".join(c["variant"][:3] for c in e["block_choices"])
        lines.append(f"| {stat} | {e['n']} | {e['n_dates']} | {e['crps']:.3f} | {e['crps_vs_baseline']:+.3f} | "
                     f"{e['pit_ks_p']:.4f} | {e['forecast_allowed']} | {variants} |")
    (out / "p3_prequential_report.md").write_text("\n".join(lines))
    typer.echo(f"[P3][prequential] status={status} certified={certified} ledger_hash={ledger_hash}")


if __name__ == "__main__":
    app()
