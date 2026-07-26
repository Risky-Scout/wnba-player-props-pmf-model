"""PHASE 1 -- decompose the systematic under-projection by prop AND fold (pure-model mission).

Reads the chronological OOF store and, per prop and per outer fold, attributes the total
predicted-count mean bias (E[pmf_mean] - E[actual]) into interpretable components using a
sequential multiplicative (Oaxaca-style) decomposition of count = availability x conditional
minutes x per-active-minute rate, plus a PMF variance/tail diagnostic and (on the exact-quote
subset) a binary-calibration gap:

    availability_bias   = (A_pred - A_act) * M_pred * R_pred
    minutes_bias        =  A_act * (M_pred - M_act) * R_pred
    rate_bias           =  A_act * M_act * (R_pred - R_act)
    residual/pmf_shape  =  E[pmf_mean] - E[actual] - (availability+minutes+rate)

where A=appearance rate, M=conditional minutes, R=per-active-minute stat rate. The binary
calibration gap is mean(model P(over line)) - empirical over-rate on the exact-quote rows.

No market information enters the decomposition (the line is used only to report the binary gap
and the actual-vs-line columns). Writes:
  artifacts/pure_supremacy/ROOT_CAUSE_BY_PROP.csv    (one row per prop x fold)
  artifacts/pure_supremacy/ROOT_CAUSE_SUMMARY.json   (% of total mean error per component)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)
EPS = 1e-9
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _safe_rate(num, den):
    den = np.asarray(den, float)
    return np.where(den > EPS, np.asarray(num, float) / np.maximum(den, EPS), np.nan)


def _p_over(pmf_json, line):
    if not isinstance(pmf_json, str) or not np.isfinite(line):
        return np.nan
    pmf = json_to_pmf(pmf_json)
    k = np.arange(len(pmf))
    return float(pmf[k > line].sum())


def _decompose(g: pd.DataFrame) -> dict:
    """Multiplicative sequential decomposition of the count mean bias for one prop-fold group."""
    did_play = g["did_play"].to_numpy(float)
    appear = did_play > 0.5
    a_pred = float(np.mean(1.0 - g["p_dnp"].to_numpy(float)))
    a_act = float(np.mean(did_play))
    if appear.sum() < 5:
        return {}
    m_pred = float(np.mean(g["minutes_mean"].to_numpy(float)[appear]))
    m_act = float(np.mean(g["actual_minutes"].to_numpy(float)[appear]))
    r_pred = float(np.nanmean(_safe_rate(g["stat_mean"], g["minutes_mean"])[appear]))
    r_act = float(np.nanmean(_safe_rate(g["actual_outcome"], g["actual_minutes"])[appear]))
    pmf_mean = float(np.mean(g["pmf_mean"].to_numpy(float)))
    actual = float(np.mean(g["actual_outcome"].to_numpy(float)))
    total_bias = pmf_mean - actual
    availability_bias = (a_pred - a_act) * m_pred * r_pred
    minutes_bias = a_act * (m_pred - m_act) * r_pred
    rate_bias = a_act * m_act * (r_pred - r_act)
    pmf_shape_residual = total_bias - (availability_bias + minutes_bias + rate_bias)
    # variance / tail diagnostic: predicted PMF variance vs realized squared residual
    sq_resid = float(np.mean((g["actual_outcome"].to_numpy(float) - g["pmf_mean"].to_numpy(float)) ** 2))
    pmf_var = float(np.mean(g["pmf_variance"].to_numpy(float)))
    return {
        "n": int(len(g)), "n_appear": int(appear.sum()),
        "pred_active_minutes": m_pred, "actual_minutes_appear": m_act,
        "pred_dnp": float(np.mean(g["p_dnp"].to_numpy(float))), "actual_dnp": 1.0 - a_act,
        "pred_stat_mean": float(np.mean(g["stat_mean"].to_numpy(float))),
        "pmf_mean": pmf_mean, "actual_outcome": actual,
        "pred_per_active_min_rate": r_pred, "actual_per_active_min_rate": r_act,
        "pmf_variance": pmf_var, "mean_sq_residual": sq_resid,
        "variance_coverage_ratio": (pmf_var / sq_resid) if sq_resid > EPS else np.nan,
        "total_mean_bias": total_bias,
        "availability_bias": availability_bias,
        "minutes_bias": minutes_bias,
        "rate_bias": rate_bias,
        "pmf_shape_residual_bias": pmf_shape_residual,
    }


@app.command()
def main(
    oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet", "--oof"),
    scored: str = typer.Option(
        "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet", "--scored"),
    out_dir: str = typer.Option("artifacts/pure_supremacy", "--out-dir"),
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof).rename(columns={"stat": "prop"})

    # exact-quote binary-calibration gap (market line used ONLY as the query threshold)
    sc = pd.read_parquet(scored)
    for d in (sc,):
        d["game_id"] = d["game_id"].astype(str)
        d["player_id"] = d["player_id"].astype(str)
    binary_gap = {}
    oof_key = df.copy()
    oof_key["game_id"] = oof_key["game_id"].astype(str)
    oof_key["player_id"] = oof_key["player_id"].astype(str)
    j = sc.merge(oof_key[["game_id", "player_id", "prop", "pmf_json"]],
                 on=["game_id", "player_id", "prop"], how="left")
    j["p_over_pmf"] = [_p_over(pj, ln) for pj, ln in zip(j["pmf_json"], j["line"])]
    for prop in DIRECT_PROPS:
        gp = j[(j["prop"] == prop)].dropna(subset=["p_over_pmf"])
        if len(gp):
            binary_gap[prop] = {
                "n_exact_quote_rows": int(len(gp)),
                "mean_model_p_over": float(gp["p_over_pmf"].mean()),
                "empirical_over_rate": float(gp["outcome_over"].mean()),
                "binary_calibration_bias": float(gp["p_over_pmf"].mean() - gp["outcome_over"].mean()),
            }
        else:
            binary_gap[prop] = {"status": "NO_EXACT_QUOTES"}

    rows = []
    summary = {}
    for prop in DIRECT_PROPS:
        pdf = df[df["prop"] == prop]
        agg_overall = _decompose(pdf)
        for fold, g in pdf.groupby("fold_id"):
            d = _decompose(g)
            if d:
                rows.append({"prop": prop, "fold_id": int(fold), **d})
        if agg_overall:
            tot = agg_overall["total_mean_bias"]
            denom = (abs(agg_overall["availability_bias"]) + abs(agg_overall["minutes_bias"])
                     + abs(agg_overall["rate_bias"]) + abs(agg_overall["pmf_shape_residual_bias"]))
            denom = denom if denom > EPS else 1.0
            summary[prop] = {
                **agg_overall,
                "pct_of_abs_error": {
                    "availability": 100.0 * abs(agg_overall["availability_bias"]) / denom,
                    "conditional_minutes": 100.0 * abs(agg_overall["minutes_bias"]) / denom,
                    "opportunity_rate_conversion": 100.0 * abs(agg_overall["rate_bias"]) / denom,
                    "pmf_variance_tail_shape": 100.0 * abs(agg_overall["pmf_shape_residual_bias"]) / denom,
                },
                "binary_calibration": binary_gap.get(prop),
            }

    pd.DataFrame(rows).to_csv(outp / "ROOT_CAUSE_BY_PROP.csv", index=False)
    (outp / "ROOT_CAUSE_SUMMARY.json").write_text(json.dumps({
        "version": "root-cause-decomposition-v1",
        "source_oof": oof,
        "method": ("sequential multiplicative decomposition count = availability x conditional "
                   "minutes x per-active-minute rate; pmf_shape_residual absorbs distribution/tail; "
                   "binary_calibration_bias from exact-quote P(over line) vs empirical over-rate"),
        "no_market_inputs": True,
        "per_prop": summary,
    }, indent=2) + "\n")

    print("=== ROOT-CAUSE DECOMPOSITION (mean count bias attribution, pmf_mean - actual) ===")
    hdr = f"{'prop':9s} {'total':>8s} {'avail':>8s} {'minutes':>8s} {'rate':>8s} {'pmf/tail':>8s} {'binCalib':>9s} {'varCov':>7s}"
    print(hdr)
    for prop in DIRECT_PROPS:
        s = summary.get(prop)
        if not s:
            continue
        bc = s["binary_calibration"].get("binary_calibration_bias") if isinstance(s["binary_calibration"], dict) else None
        bc_s = f"{bc:+.3f}" if bc is not None else "   n/a"
        print(f"{prop:9s} {s['total_mean_bias']:>+8.3f} {s['availability_bias']:>+8.3f} "
              f"{s['minutes_bias']:>+8.3f} {s['rate_bias']:>+8.3f} {s['pmf_shape_residual_bias']:>+8.3f} "
              f"{bc_s:>9s} {s['variance_coverage_ratio']:>7.2f}")


if __name__ == "__main__":
    app()
