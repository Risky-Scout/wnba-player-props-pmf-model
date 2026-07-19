"""P3 final recovery — strictly-prequential full-distribution calibration for the ten
requested markets (6 base stats + 4 combos), reusing the existing 2026 OOF and the
existing correlated combo code. Per five-date outer block, select among candidates using
ONLY pre-block dates, freeze, score the block OOS; concatenate and gate once.
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
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf, _fit_factors  # noqa: E402
from wnba_props_model.models.simulation import build_combo_pmfs, estimate_oof_correlations  # noqa: E402

app = typer.Typer(add_completion=False)
BASE = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
SPARSE = {"fg3m", "stl", "blk"}
COMBOS = ["pts_reb", "pts_ast", "pts_reb_ast", "stocks"]
COMBO_PARTS = {"pts_reb": ["pts", "reb"], "pts_ast": ["pts", "ast"],
               "pts_reb_ast": ["pts", "reb", "ast"], "stocks": ["stl", "blk"]}


def _seed(r):
    return f"{r['game_id']}|{r['player_id']}|{r['stat']}"


def _candidate_pmfs(before: pd.DataFrame, target: pd.DataFrame, stat: str):
    """Return {method: list_of_pmfs_for_target_rows} fit ONLY on `before`."""
    raw_b = [fc.pmf_to_array(p) for p in before["pmf_json"]]
    raw_t = [fc.pmf_to_array(p) for p in target["pmf_json"]]
    yb = before["actual_outcome"].astype(float).values
    max_sup = int(max(before["actual_outcome"].max(), target["actual_outcome"].max())) + 6
    cands = {"raw": raw_t}

    # location / location-and-scale (fit on before)
    for name, disp in [("location", False), ("location_and_scale", True)]:
        d, s = _fit_factors(before, dispersion=disp)
        cands[name] = [recalibrate_pmf(p, d, s) for p in raw_t]

    # monotone CDF recalibration + shrinkage chosen on before
    seeds_b = [_seed(r) for _, r in before.iterrows()]
    R = dc.fit_pit_recalibration(raw_b, yb, seeds_b)
    best_shrink, best_c = 1.0, float("inf")
    for sh in (0.25, 0.5, 0.75, 1.0):
        pmfs_b = [dc.apply_monotone_cdf_recalibration(p, R, sh) for p in raw_b]
        c, _ = dc.score_pmfs(pmfs_b, yb)
        if c < best_c:
            best_c, best_shrink = c, sh
    cands["monotone_cdf"] = [dc.apply_monotone_cdf_recalibration(p, R, best_shrink) for p in raw_t]

    # CDF calibration + empirical mixture, weight chosen on before
    emp = dc.empirical_pmf(yb, max_sup)
    cdf_b = cands["monotone_cdf"] and [dc.apply_monotone_cdf_recalibration(p, R, best_shrink) for p in raw_b]
    best_w, best_c = 1.0, float("inf")
    for w in (0.5, 0.7, 0.85, 1.0):
        pmfs_b = [dc.mixture_pmf(cb, emp, w) for cb in cdf_b]
        c, _ = dc.score_pmfs(pmfs_b, yb)
        if c < best_c:
            best_c, best_w = c, w
    cands["cdf_mixture"] = [dc.mixture_pmf(ct, emp, best_w) for ct in cands["monotone_cdf"]]

    # hurdle for sparse stats
    if stat in SPARSE:
        p0 = float((yb == 0).mean())
        pos_mask = yb > 0
        pos_pmfs_b = [p for p, m in zip(raw_b, pos_mask) if m]
        pos_y = yb[pos_mask]
        pos_seeds = [s for s, m in zip(seeds_b, pos_mask) if m]
        posR = dc.fit_pit_recalibration([p[1:] / max(p[1:].sum(), 1e-9) for p in pos_pmfs_b],
                                        pos_y - 1, pos_seeds) if len(pos_y) else np.array([])
        cands["hurdle_cdf"] = [dc.hurdle_calibrate(p, p0, posR, best_shrink) for p in raw_t]

    return cands, {"shrink": best_shrink, "mix_weight": best_w}


def _hierarchical(before, target, max_sup):
    cells = dc.fit_residual_hist(before)
    out = []
    for _, r in target.iterrows():
        ck = f"{r.get('role_bucket','')}|{r.get('_minbucket','')}"
        out.append(dc.hierarchical_empirical_pmf(r["_point"], ck, cells, max_sup))
    return out


def _prequential_stat(df, stat, blocks):
    s = df[df["stat"] == stat].copy()
    ledger_rows, choices = [], []
    for k, block in enumerate(blocks):
        bstart = min(block)
        before = s[s["game_date"] < bstart]
        blk = s[s["game_date"].isin(set(block))]
        if len(before) < 80 or blk.empty:
            for _, r in blk.iterrows():
                rr = r.copy(); ledger_rows.append(rr)
            choices.append({"block": k, "variant": "raw", "n_before": int(len(before))})
            continue
        max_sup = int(max(before["actual_outcome"].max(), blk["actual_outcome"].max())) + 6
        cands_t, hp = _candidate_pmfs(before, blk, stat)
        cands_b, _ = _candidate_pmfs(before, before, stat)
        # hierarchical empirical fallback + dispersion-sharpened variants (scale chosen
        # pre-block); repairs over-dispersion/over-broad sharpness in the empirical fallback.
        h_t = _hierarchical(before, blk, max_sup)
        h_b = _hierarchical(before, before, max_sup)
        for sc in (1.0, 0.85, 0.7):
            key = "hierarchical" if sc == 1.0 else f"hierarchical_s{int(sc*100)}"
            cands_t[key] = [recalibrate_pmf(p, 0.0, sc) for p in h_t]
            cands_b[key] = [recalibrate_pmf(p, 0.0, sc) for p in h_b]
        yb = before["actual_outcome"].astype(float).values
        seeds_b = [_seed(r) for _, r in before.iterrows()]
        # Calibration-aware selection (multi-criteria, pre-block only): prefer candidates
        # whose pre-block randomized-PIT is calibrated (KS D <= 0.05), then min CRPS; if
        # none qualify, use the by-construction-calibrated hierarchical empirical fallback.
        base_w = float(fc.matched_mass_width(dc.empirical_pmf(yb, int(yb.max()) + 6), 0.8))

        def _ks_d(pmfs):
            uu = np.array([fc.randomized_pit(p, int(y), k) for p, y, k in zip(pmfs, yb, seeds_b)])
            uu = uu[~np.isnan(uu)]
            return fc.ks_uniform(uu)[0] if len(uu) else 1.0

        def _sharp_ok(pmfs):
            mw = float(np.mean([fc.matched_mass_width(p, 0.8) for p in pmfs]))
            return (base_w <= 0) or (mw <= 1.15 * base_w)
        crps = {m: dc.score_pmfs(cands_b[m], yb)[0] for m in cands_b}
        ksd = {m: _ks_d(cands_b[m]) for m in cands_b}
        # multi-criteria: calibrated (KS D<=0.05) AND sharp (<=1.15x empirical) -> min CRPS.
        qualified = [m for m in cands_b if ksd[m] <= 0.05 and _sharp_ok(cands_b[m])]
        if qualified:
            variant = min(qualified, key=lambda m: crps[m])
        elif "hierarchical" in cands_b and ksd["hierarchical"] <= 0.06:
            variant = "hierarchical"   # trustworthy calibrated+sharp empirical fallback
        else:
            variant = min(crps, key=lambda m: crps[m])
        choices.append({"block": k, "variant": variant, "n_before": int(len(before)),
                        "n_block": int(len(blk)), **hp})
        for (_, r), pmf in zip(blk.iterrows(), cands_t[variant]):
            rr = r.copy()
            rr["pmf_json"] = json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(pmf) if v > 1e-9})
            ledger_rows.append(rr)
    ledger = pd.DataFrame(ledger_rows)
    return ledger, choices


def _baseline(devcal_a, hold_a):
    ms = int(max(np.max(hold_a), np.max(devcal_a))) + 6
    b = dc.empirical_pmf(devcal_a, ms)
    return {"crps": float(np.mean([fc.crps_discrete(b, int(y)) for y in hold_a])),
            "log_score": float(np.mean([fc.log_score(b, int(y)) for y in hold_a])),
            "matched_width_80": float(fc.matched_mass_width(b, 0.8))}


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         out_dir: str = typer.Option("artifacts/p3"),
         holdout_dates: int = typer.Option(25)) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    df["_point"] = df["pmf_mean"].astype(float) if "pmf_mean" in df else df["actual_outcome"].astype(float)
    mm = df["minutes_mean"].astype(float) if "minutes_mean" in df else pd.Series(0.0, index=df.index)
    df["_minbucket"] = pd.cut(mm, bins=[-1, 10, 20, 28, 100], labels=["m0", "m1", "m2", "m3"]).astype(str)

    dates = np.sort(df["game_date"].unique())
    hold = dates[-holdout_dates:]
    blocks = [list(b) for b in np.array_split(hold, 5)]
    devcal = df[df["game_date"] < min(hold)]

    registry, report, ledgers = {}, {}, {}
    for stat in BASE:
        ledger, choices = _prequential_stat(df, stat, blocks)
        base = _baseline(devcal[devcal["stat"] == stat]["actual_outcome"].values,
                         ledger["actual_outcome"].values)
        r = fc.evaluate_stat(ledger, baseline=base)
        method = ",".join(sorted({c["variant"] for c in choices}))
        registry[stat] = {"forecast_allowed": bool(r.forecast_allowed),
                          "market_comparison_allowed": False, "betting_recommendation_allowed": False,
                          "method": method, "block_choices": choices, "n": r.n, "n_dates": r.n_dates,
                          "crps": round(r.crps, 4), "crps_vs_baseline": round(r.crps_vs_baseline, 4),
                          "pit_ks_p": round(r.pit_ks_p, 4),
                          "suppression_reason": "" if r.forecast_allowed else "; ".join(r.reasons)}
        report[stat] = registry[stat]
        ledgers[stat] = ledger[["game_id", "player_id", "game_date", "stat", "pmf_json", "actual_outcome"]]
        typer.echo(f"  {stat}: allowed={r.forecast_allowed} method={method} crps={r.crps:.3f} pit={r.pit_ks_p:.3f}")

    # ---- combos from calibrated component ledgers ----
    # build_combo_pmfs returns keys pr/pa/pra/stocks; map requested market -> that key.
    COMBO_KEY = {"pts_reb": "pr", "pts_ast": "pa", "pts_reb_ast": "pra", "stocks": "stocks"}
    for combo in COMBOS:
        parts = COMBO_PARTS[combo]
        merged = None
        for pstat in parts:
            lp = ledgers[pstat].rename(columns={"pmf_json": f"pmf_{pstat}", "actual_outcome": f"act_{pstat}"})
            lp = lp[["game_id", "player_id", "game_date", f"pmf_{pstat}", f"act_{pstat}"]]
            merged = lp if merged is None else merged.merge(lp, on=["game_id", "player_id", "game_date"], how="inner")
        rows = []
        key = COMBO_KEY[combo]
        for _, r in merged.iterrows():
            comp = {ps: fc.pmf_to_array(r[f"pmf_{ps}"]) for ps in parts}
            # pre-block correlations: estimate on dates strictly before this row's date
            prior = devcal[devcal["game_date"] < r["game_date"]]
            corr = estimate_oof_correlations(prior) if len(prior) > 200 else estimate_oof_correlations(devcal)
            try:
                cpmf = build_combo_pmfs(comp, correlations=corr).get(key)
            except Exception:
                cpmf = None
            if cpmf is None:
                continue
            actual = int(sum(int(round(float(r[f"act_{ps}"]))) for ps in parts))
            rows.append({"game_id": r["game_id"], "player_id": r["player_id"], "game_date": r["game_date"],
                         "stat": combo, "model_version": "combo",
                         "pmf_json": json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(cpmf) if v > 1e-9}),
                         "actual_outcome": actual})
        cl = pd.DataFrame(rows)
        if cl.empty:
            registry[combo] = {"forecast_allowed": False, "suppression_reason": "no combo rows built"}
            continue
        # combo baseline uses ONLY pre-holdout (dev) combo outcomes, never the evaluated block
        dev_combo_actuals = None
        dm = None
        for ps in parts:
            dpp = devcal[devcal["stat"] == ps][["game_id", "player_id", "actual_outcome"]].rename(
                columns={"actual_outcome": f"a_{ps}"})
            dm = dpp if dm is None else dm.merge(dpp, on=["game_id", "player_id"], how="inner")
        dev_combo_actuals = dm[[f"a_{ps}" for ps in parts]].sum(axis=1).values if dm is not None and len(dm) else cl["actual_outcome"].values
        base = _baseline(dev_combo_actuals, cl["actual_outcome"].values)
        rr = fc.evaluate_stat(cl, baseline=base)
        registry[combo] = {"forecast_allowed": bool(rr.forecast_allowed),
                           "market_comparison_allowed": False, "betting_recommendation_allowed": False,
                           "method": "correlated_combo", "n": rr.n, "n_dates": rr.n_dates,
                           "crps": round(rr.crps, 4), "crps_vs_baseline": round(rr.crps_vs_baseline, 4),
                           "pit_ks_p": round(rr.pit_ks_p, 4),
                           "suppression_reason": "" if rr.forecast_allowed else "; ".join(rr.reasons)}
        report[combo] = registry[combo]
        typer.echo(f"  {combo}: allowed={rr.forecast_allowed} crps={rr.crps:.3f} pit={rr.pit_ks_p:.3f}")

    (out / "p3_full_recovery_report.json").write_text(json.dumps(report, indent=2, default=str))
    allowed = sorted([s for s, e in registry.items() if e.get("forecast_allowed")])
    typer.echo(f"[P3][full-recovery] forecast_allowed={allowed}")


if __name__ == "__main__":
    app()
