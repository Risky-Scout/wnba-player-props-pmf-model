"""P3 - validate pts_reb and pts_reb_ast combos, strictly prequential.

Builds combos from the EXACT registered/validated production component PMFs (pts, reb, ast) -
applying the champion package's per-component calibration (pts/ast hierarchical; reb
hierarchical + scale 0.9) via the same forecast_publication.apply_market used in production.
Components are never retrained or recalibrated here.

Primary candidate: component-based correlated combo (build_combo_pmfs with pre-block-estimated
dependence, frozen per block). Bounded fallback (only if the correlated PMF fails the gate):
hierarchical empirical combo-residual PMF centered on the sum of validated component
expectations, conditioned on role x minutes bucket, shrunk to the pooled combo residual, with a
dispersion-scale grid selected on pre-block dates only.
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
from wnba_props_model.evaluation import distribution_calibration as dc  # noqa: E402
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf  # noqa: E402
from wnba_props_model.models.simulation import build_combo_pmfs, estimate_oof_correlations  # noqa: E402
from wnba_props_model.pipeline.forecast_publication import apply_market  # noqa: E402

app = typer.Typer(add_completion=False)
COMBO_KEY = {"pts_reb": "pr", "pts_reb_ast": "pra"}
COMBO_PARTS = {"pts_reb": ["pts", "reb"], "pts_reb_ast": ["pts", "reb", "ast"]}
SCALE_GRID = (1.0, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8)
UNDER_TOL, OVER_TOL, PIT_ENV = 0.05, 0.07, 0.10


def _minbucket(m: float) -> str:
    for lo, hi, lab in [(-1, 10, "m0"), (10, 20, "m1"), (20, 28, "m2"), (28, 200, "m3")]:
        if lo < m <= hi:
            return lab
    return "m3"


def _load_components(oof: str, calib_path: str):
    """Return per (game_id, player_id) validated component calibrated PMFs + meta + actuals."""
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    calib = json.loads(Path(calib_path).read_text())["markets"]
    rows = {}
    for _, r in df[df["stat"].isin(["pts", "reb", "ast"])].iterrows():
        stat = str(r["stat"]); spec = calib.get(stat)
        if spec is None:
            continue
        pmf = apply_market(fc.pmf_to_array(r["pmf_json"]), float(r["pmf_mean"]),
                           str(r.get("role_bucket", "")), float(r.get("minutes_mean", 0.0)), spec)
        key = (r["game_id"], r["player_id"])
        d = rows.setdefault(key, {"game_date": r["game_date"], "role_bucket": str(r.get("role_bucket", "")),
                                  "minutes_mean": float(r.get("minutes_mean", 0.0)),
                                  "player_name": r.get("player_name"), "pmf": {}, "actual": {}, "mean": {}})
        d["pmf"][stat] = pmf
        d["actual"][stat] = float(r["actual_outcome"])
        d["mean"][stat] = float((np.arange(len(pmf)) * pmf).sum())
    return df, rows


def _calib_penalty(pmfs, actuals, gdate):
    y = np.asarray(actuals, dtype=int); pen = 0.0
    for cl in (0.5, 0.8, 0.9):
        res = []
        for p, yy in zip(pmfs, y):
            lo, hi = fc.central_interval(p, cl)
            res.append((1.0 if lo <= yy <= hi else 0.0) - float(p[lo:hi + 1].sum()))
        pen += abs(float(np.mean(res)))
    seeds = [f"{g}|{i}" for i, g in enumerate(gdate)]
    u = np.array([fc.randomized_pit(p, int(yy), k) for p, yy, k in zip(pmfs, y, seeds)])
    u = u[~np.isnan(u)]
    _, cd, _ = fc.clustered_pit_deviation(u, np.asarray(gdate))
    return pen + cd


def _combo_ledger_correlated(combo, rows, pre_oof_by_block, blocks, base_df):
    """Build the strictly-prequential correlated-combo ledger for one combo market."""
    parts = COMBO_PARTS[combo]; key = COMBO_KEY[combo]
    ledger, block_corr = [], []
    for k, block in enumerate(blocks):
        pre = pre_oof_by_block[k]
        corr = estimate_oof_correlations(pre) if len(pre) else {}
        frozen = {kk: float(vv) for kk, vv in corr.items()}          # freeze dependence
        block_corr.append({"block": k, "correlations": frozen})
        bset = set(block)
        for (gid, pid), d in rows.items():
            if d["game_date"] not in bset:
                continue
            if not all(p in d["pmf"] for p in parts):
                continue
            built = build_combo_pmfs({p: d["pmf"][p] for p in parts}, correlations=frozen)
            pmf = built.get(key)
            if pmf is None:
                continue
            pmf = np.asarray(pmf, float)
            actual = int(round(sum(d["actual"][p] for p in parts)))
            ledger.append({"game_id": gid, "player_id": pid, "stat": combo,
                           "game_date": d["game_date"], "role_bucket": d["role_bucket"],
                           "minutes_mean": d["minutes_mean"], "actual_outcome": actual,
                           "pmf_json": json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(pmf) if v > 1e-9}),
                           "pmf_mean": float((np.arange(len(pmf)) * pmf).sum())})
    return pd.DataFrame(ledger), block_corr


def _combo_ledger_fallback(combo, rows, blocks, holdout):
    """Bounded fallback: hierarchical empirical combo-residual PMF, pre-block scale-selected."""
    parts = COMBO_PARTS[combo]
    recs = []
    for (gid, pid), d in rows.items():
        if not all(p in d["pmf"] for p in parts):
            continue
        recs.append({"game_id": gid, "player_id": pid, "game_date": d["game_date"],
                     "role_bucket": d["role_bucket"], "_minbucket": _minbucket(d["minutes_mean"]),
                     "_point": float(sum(d["mean"][p] for p in parts)),
                     "actual_outcome": int(round(sum(d["actual"][p] for p in parts)))})
    s = pd.DataFrame(recs)
    ms = int(s["actual_outcome"].max()) + 8
    ledger, choices = [], []
    for k, block in enumerate(blocks):
        bstart = min(block)
        before = s[s["game_date"] < bstart]; blk = s[s["game_date"].isin(set(block))]
        cells = dc.fit_residual_hist(before)

        def _mk(rows_df, scale):
            return [recalibrate_pmf(dc.hierarchical_empirical_pmf(
                float(r["_point"]), f"{r['role_bucket']}|{r['_minbucket']}", cells, ms), 0.0, scale)
                for _, r in rows_df.iterrows()]
        best_sc, best_pen = 1.0, float("inf")
        for sc in SCALE_GRID:
            pen = _calib_penalty(_mk(before, sc), before["actual_outcome"].values,
                                 before["game_date"].astype(str).values)
            if pen < best_pen:
                best_pen, best_sc = pen, sc
        choices.append({"block": k, "scale": best_sc})
        for (_, r), pmf in zip(blk.iterrows(), _mk(blk, best_sc)):
            ledger.append({"game_id": r["game_id"], "player_id": r["player_id"], "stat": combo,
                           "game_date": r["game_date"], "role_bucket": r["role_bucket"],
                           "minutes_mean": 0.0, "actual_outcome": int(r["actual_outcome"]),
                           "pmf_json": json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(pmf) if v > 1e-9}),
                           "pmf_mean": float((np.arange(len(pmf)) * pmf).sum())})
    return pd.DataFrame(ledger), choices


def _gate(ledger, base_df, holdout, ms_pad=8):
    ms = int(ledger["actual_outcome"].max()) + ms_pad
    emp = dc.empirical_pmf(base_df["actual_outcome"].values, ms)
    base = {"crps": float(np.mean([fc.crps_discrete(emp, int(y)) for y in ledger["actual_outcome"]])),
            "log_score": float(np.mean([fc.log_score(emp, int(y)) for y in ledger["actual_outcome"]])),
            "matched_width_80": float(fc.matched_mass_width(emp, 0.8))}
    return fc.evaluate_stat(ledger, baseline=base)


def _result(combo, method, r, extra):
    out = {"market": combo, "method": method, "forecast_allowed": bool(r.forecast_allowed),
           "n": r.n, "n_dates": r.n_dates, "crps": round(r.crps, 4),
           "crps_vs_baseline": round(r.crps_vs_baseline, 4), "log_vs_baseline": round(r.log_vs_baseline, 4),
           "pit_clustered_dev": round(r.pit_clustered_dev, 4), "coverage": r.coverage,
           "bias": round(r.bias, 4), "sharpness_ratio": round(r.sharpness_ratio, 4),
           "reasons": r.reasons}
    out.update(extra)
    return out


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         calib: str = typer.Option("config/certified_forecast_calibration.json"),
         out_dir: str = typer.Option("artifacts/p3"), holdout_dates: int = typer.Option(25)) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df, rows = _load_components(oof, calib)
    dates = np.sort(df["game_date"].unique()); hold = dates[-holdout_dates:]
    blocks = [list(b) for b in np.array_split(hold, 5)]
    holdset = set(hold)
    pre_by_block = [df[df["game_date"] < min(block)] for block in blocks]

    results = {}
    for combo in ("pts_reb", "pts_reb_ast"):
        parts = COMBO_PARTS[combo]
        base_df = pd.DataFrame({"actual_outcome": [
            int(round(sum(d["actual"][p] for p in parts)))
            for d in rows.values() if all(p in d["actual"] for p in parts) and d["game_date"] not in holdset]})
        led, block_corr = _combo_ledger_correlated(combo, rows, pre_by_block, blocks, df)
        led_hash = hashlib.sha256(pd.util.hash_pandas_object(
            led[["game_id", "player_id", "stat", "pmf_json", "actual_outcome"]], index=False).values.tobytes()).hexdigest()[:16]
        r = _gate(led, base_df, holdset)
        res = _result(combo, "correlated_component_combo", r,
                      {"ledger_hash": led_hash, "block_correlations": block_corr})
        if not r.forecast_allowed:
            typer.echo(f"[{combo}] correlated FAILED -> evaluating bounded fallback. reasons={r.reasons}")
            fled, fchoices = _combo_ledger_fallback(combo, rows, blocks, holdset)
            fr = _gate(fled, base_df, holdset)
            fhash = hashlib.sha256(pd.util.hash_pandas_object(
                fled[["game_id", "player_id", "stat", "pmf_json", "actual_outcome"]], index=False).values.tobytes()).hexdigest()[:16]
            if fr.forecast_allowed:
                res = _result(combo, "hierarchical_combo_residual_fallback", fr,
                              {"ledger_hash": fhash, "scale_choices": fchoices})
        results[combo] = res
        typer.echo(f"[{combo}] method={res['method']} forecast_allowed={res['forecast_allowed']} "
                   f"n={res['n']} dCRPS={res['crps_vs_baseline']:+.3f} clustered_PIT={res['pit_clustered_dev']:.3f} "
                   f"ledger_hash={res['ledger_hash']}")
        if res["reasons"]:
            typer.echo(f"[{combo}] reasons: {res['reasons']}")
    (out / "p3_combo_repair_result.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    app()
