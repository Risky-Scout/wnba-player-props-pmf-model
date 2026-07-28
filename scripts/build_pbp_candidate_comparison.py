#!/usr/bin/env python3
"""Compare the pure PBP-opportunity candidate vs P0, market, and the box-opportunity (r1) candidate
on the ONE canonical exact-quote universe, for FG3M and AST (owner directive step D reporting).

Uses the frozen evaluator's metric primitives (LogLoss, Brier, ROC-AUC, ECE, calibration
intercept/slope, full-PMF logscore, CRPS) and its paired date-cluster bootstrap + Holm correction.
Over-probabilities are settled from each candidate's active PMF with the push-safe settlement used
in production. Duplicate keys FAIL closed (no silent drop_duplicates).

Market/P0/outcome/line come from the frozen ``PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet`` (same
universe used for r1/g1). The pure PBP model consumes NO market signal; its cutoff guard is the game
tip (features are strictly prior-game), so the market-leakage cutoff check does not apply.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("opp_eval", REPO / "scripts" / "evaluate_opportunity_oof.py")
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)
from wnba_props_model.opportunity.pmf_builders import settled_over_probability  # noqa: E402

KEY = ["game_id", "player_id", "prop"]
QUOTES = "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet"


def _assert_unique(df: pd.DataFrame, key: list[str], src: str) -> None:
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        raise ValueError(f"[{src}] {int(dup.sum())} duplicate rows on {key}")


def _settle_oof(oof: pd.DataFrame, q: pd.DataFrame, label: str) -> pd.DataFrame:
    """Join an OOF PMF table to the scored quotes and settle candidate over-probability per key."""
    o = oof.copy()
    if "stat" in o.columns and "prop" not in o.columns:
        o = o.rename(columns={"stat": "prop"})
    for c in ("game_id", "player_id"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = o.dropna(subset=["active_pmf_json"])
    _assert_unique(o, KEY, f"oof:{label}")
    j = q.merge(o[KEY + ["active_pmf_json"]], on=KEY, how="inner", validate="one_to_one")

    def _p(js, line):
        arr = np.asarray(json.loads(js), float)
        return settled_over_probability(arr, float(line))[0]

    j[label] = [_p(js, ln) for js, ln in zip(j["active_pmf_json"], j["line"])]
    return j[KEY + [label, "active_pmf_json"]].rename(columns={"active_pmf_json": f"{label}_pmf"})


def _metrics(g: pd.DataFrame, col: str, pmf_col: str | None) -> dict:
    y = g["outcome_over"].to_numpy()
    p = g[col].to_numpy()
    m = {"n": int(len(g)), "log_loss": EV.log_loss(y, p), "brier": EV.brier(y, p),
         "auc": EV.auc(y, p), "ece": EV.expected_calibration_error(y, p)}
    m["calibration_intercept"], m["calibration_slope"] = EV.calibration_intercept_slope(y, p)
    if pmf_col and pmf_col in g.columns and "actual" in g.columns:
        m["full_pmf_log_score"] = EV.full_pmf_log_score(g[pmf_col], g["actual"])
        m["crps"] = EV.crps_discrete(g[pmf_col], g["actual"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp-oof", default="data/oof/opportunity_v2_pbp/oof_pmfs.parquet")
    ap.add_argument("--r1-oof", default="data/oof/opportunity_v2/oof_pmfs.parquet")
    ap.add_argument("--out", default="artifacts/opportunity_v2/CANDIDATE_COMPARISON_PBP.json")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    q = pd.read_parquet(REPO / QUOTES)
    if "stat" in q.columns and "prop" not in q.columns:
        q = q.rename(columns={"stat": "prop"})
    for c in ("game_id", "player_id"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[q["binary_score_eligible"].astype(bool) & q["outcome_over"].isin([0, 1])].copy()
    q = q[q["prop"].isin(["fg3m", "ast"])].copy()
    _assert_unique(q, KEY, "quotes")
    q["outcome_over"] = q["outcome_over"].astype(int)

    base = q[KEY + ["outcome_over", "game_date", "line", "actual", "oof_fold",
                    "model_prob_over_final", "market_prob_over_no_vig"]].rename(
        columns={"model_prob_over_final": "p0", "market_prob_over_no_vig": "market"})

    present = []
    pbp = pd.read_parquet(args.pbp_oof)
    base = base.merge(_settle_oof(pbp, q, "pbp"), on=KEY, how="left")
    present.append("pbp")
    if Path(args.r1_oof).exists():
        r1 = pd.read_parquet(args.r1_oof)
        r1 = r1[r1["stat"].isin(["fg3m"])] if "stat" in r1.columns else r1
        if len(r1):
            base = base.merge(_settle_oof(r1, q, "r1"), on=KEY, how="left")
            present.append("r1")

    series = {"p0": ("p0", None), "market": ("market", None)}
    for lab in present:
        series[lab] = (lab, f"{lab}_pmf")

    results: dict = {}
    for prop, gall in base.groupby("prop"):
        entry = {"n_total": int(len(gall)), "game_dates": int(gall["game_date"].nunique())}
        for name, (col, pmf) in series.items():
            if col not in gall.columns:
                continue
            g = gall.dropna(subset=[col]).copy()
            if len(g):
                entry[name] = _metrics(g, col, pmf)
        vsm_pll, vsm_pbr = {}, {}
        for cand in present:
            if cand not in gall.columns or gall[cand].notna().sum() == 0:
                continue
            for ref in ["p0", "market"] + (["r1"] if cand == "pbp" and "r1" in present else []):
                if ref not in gall.columns:
                    continue
                gi = gall.dropna(subset=[cand, ref]).reset_index(drop=True)
                if len(gi) == 0 or gi["outcome_over"].nunique() < 2:
                    continue
                ci_ll, ci_bs, p_ll, p_brier = EV._paired_bootstrap(
                    gi, cand, ref, gi["game_date"], args.iters, args.seed)
                entry[f"{cand}_minus_{ref}"] = {
                    "n": int(len(gi)),
                    "delta_log_loss": EV.log_loss(gi["outcome_over"], gi[cand]) - EV.log_loss(gi["outcome_over"], gi[ref]),
                    "delta_brier": EV.brier(gi["outcome_over"], gi[cand]) - EV.brier(gi["outcome_over"], gi[ref]),
                    "delta_auc": EV.auc(gi["outcome_over"], gi[cand]) - EV.auc(gi["outcome_over"], gi[ref]),
                    "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs,
                    "p_ll": p_ll, "p_brier": p_brier,
                }
                if ref == "market":
                    vsm_pll[cand] = p_ll
                    vsm_pbr[cand] = p_brier
        if vsm_pll:
            entry["holm_vs_market_p_ll"] = EV.holm(vsm_pll)
            entry["holm_vs_market_p_brier"] = EV.holm(vsm_pbr)
        results[prop] = entry

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"iters": args.iters, "seed": args.seed, "candidates_present": present,
               "quote_universe": QUOTES, "results": results}
    json.dump(payload, open(out, "w"), indent=2, default=float)

    for pr, e in results.items():
        row = {"prop": pr, "n": e["n_total"]}
        for c in ["p0", "pbp", "r1", "market"]:
            if c in e:
                row[c] = {"ll": round(e[c]["log_loss"], 4), "brier": round(e[c]["brier"], 4),
                          "auc": round(e[c]["auc"], 4), "ece": round(e[c]["ece"], 4)}
        if "pbp_minus_market" in e:
            row["pbp_vs_market"] = {"dLL": round(e["pbp_minus_market"]["delta_log_loss"], 4),
                                    "holm_p_ll": round(e.get("holm_vs_market_p_ll", {}).get("pbp", float("nan")), 4)}
        print(json.dumps(row))


if __name__ == "__main__":
    main()
