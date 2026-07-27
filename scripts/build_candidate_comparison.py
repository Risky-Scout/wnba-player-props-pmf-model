#!/usr/bin/env python3
"""P0 vs R1 (OPP_V2_RATE) vs G1 (OPP_V2_TEAM_SHARE) vs market on ONE canonical universe (section D).

Reuses the corrected evaluator's metric + bootstrap functions. For each prop that a candidate
covers, reports n rows/dates, LL, Brier, AUC, ECE, calibration intercept/slope, CRPS, full-PMF log
score, worst fold, 95% date-cluster CIs, raw + Holm p_ll/p_brier vs market, and paired deltas
G1-P0, G1-R1, G1-market. Advancement requires G1 to improve AUC, LL and Brier vs BOTH P0 and R1
without materially worsening PMF metrics.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("opp_eval", REPO / "scripts" / "evaluate_opportunity_oof.py")
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)

KEY = ["game_id", "player_id", "prop"]
ITERS = 4000
SEED = 42


def _settle_series(pmf_json_series, line_series):
    from wnba_props_model.opportunity.pmf_builders import settled_over_probability
    out = []
    for js, ln in zip(pmf_json_series, line_series):
        arr = np.asarray(json.loads(js), float)
        p, _u, _p = settled_over_probability(arr, float(ln))
        out.append(p)
    return np.asarray(out)


def _load_oof(path):
    df = pd.read_parquet(path)
    if "stat" in df.columns and "prop" not in df.columns:
        df = df.rename(columns={"stat": "prop"})
    df = df.dropna(subset=["active_pmf_json"]).drop_duplicates(KEY)
    for c in ("game_id", "player_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _metrics(g, col, pmf_col=None):
    y = g["outcome_over"].to_numpy()
    p = g[col].to_numpy()
    m = {"log_loss": EV.log_loss(y, p), "brier": EV.brier(y, p), "auc": EV.auc(y, p),
         "ece": EV.expected_calibration_error(y, p)}
    ci, sl = EV.calibration_intercept_slope(y, p)
    m["calibration_intercept"], m["calibration_slope"] = ci, sl
    if pmf_col and pmf_col in g.columns and "actual" in g.columns:
        m["full_pmf_log_score"] = EV.full_pmf_log_score(g[pmf_col], g["actual"])
        m["crps"] = EV.crps_discrete(g[pmf_col], g["actual"])
    return m


def _worst_fold(g, model_col, ref_col="market"):
    if "oof_fold" not in g.columns or g["oof_fold"].isna().all():
        return None
    worst = None
    for fold, gf in g.groupby("oof_fold"):
        if len(gf) < 10 or gf["outcome_over"].nunique() < 2:
            continue
        d = EV.log_loss(gf["outcome_over"], gf[model_col]) - EV.log_loss(gf["outcome_over"], gf[ref_col])
        if worst is None or d > worst[1]:
            worst = (str(fold), float(d))
    return {"fold": worst[0], "delta_log_loss_vs_market": worst[1]} if worst else None


def main() -> None:
    q = pd.read_parquet(REPO / "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet")
    if "stat" in q.columns and "prop" not in q.columns:
        q = q.rename(columns={"stat": "prop"})
    for c in ("game_id", "player_id"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[q["binary_score_eligible"].astype(bool) & q["outcome_over"].isin([0, 1])].copy()
    q = q.drop_duplicates(KEY)
    q["outcome_over"] = q["outcome_over"].astype(int)

    r1 = _load_oof(REPO / "data/oof/opportunity_v2/oof_pmfs.parquet")
    g1 = _load_oof(REPO / "data/oof/opportunity_v2_team_share/oof_pmfs.parquet")

    base = q[KEY + ["outcome_over", "game_date", "line", "actual", "oof_fold",
                    "model_prob_over_final", "market_prob_over_no_vig"]].rename(
        columns={"model_prob_over_final": "p0", "market_prob_over_no_vig": "market"})

    # attach R1 and G1 settled probabilities + PMFs
    r1m = base.merge(r1[KEY + ["active_pmf_json"]].rename(columns={"active_pmf_json": "r1_pmf"}),
                     on=KEY, how="left")
    r1m = r1m.merge(g1[KEY + ["active_pmf_json"]].rename(columns={"active_pmf_json": "g1_pmf"}),
                    on=KEY, how="left")
    r1m.loc[r1m["r1_pmf"].notna(), "r1"] = _settle_series(
        r1m.loc[r1m["r1_pmf"].notna(), "r1_pmf"], r1m.loc[r1m["r1_pmf"].notna(), "line"])
    r1m.loc[r1m["g1_pmf"].notna(), "g1"] = _settle_series(
        r1m.loc[r1m["g1_pmf"].notna(), "g1_pmf"], r1m.loc[r1m["g1_pmf"].notna(), "line"])

    results = {}
    for prop, gall in r1m.groupby("prop"):
        entry = {"n_total": int(len(gall)), "game_dates": int(gall["game_date"].nunique())}
        # per-candidate metric universe: restrict to rows where that candidate has a prediction
        cand_cfg = {"p0": ("p0", None), "market": ("market", None),
                    "r1": ("r1", "r1_pmf"), "g1": ("g1", "g1_pmf")}
        for name, (col, pmf) in cand_cfg.items():
            if col not in gall.columns:
                continue
            g = gall.dropna(subset=[col]).copy()
            if len(g) == 0:
                continue
            entry[name] = {"n": int(len(g)), **_metrics(g, col, pmf)}
            entry[name]["worst_fold"] = _worst_fold(g, col) if name in ("r1", "g1") else None
        # paired comparisons on the intersection where G1 exists (fg3m)
        if "g1" in gall.columns and gall["g1"].notna().any():
            gi = gall.dropna(subset=["g1", "r1", "p0", "market"]).reset_index(drop=True)
            entry["paired_universe_n"] = int(len(gi))
            for ref in ("p0", "r1", "market"):
                ci_ll, ci_bs, p_ll, p_brier = EV._paired_bootstrap(gi, "g1", ref, gi["game_date"], ITERS, SEED)
                entry[f"g1_minus_{ref}"] = {
                    "delta_log_loss": EV.log_loss(gi["outcome_over"], gi["g1"]) - EV.log_loss(gi["outcome_over"], gi[ref]),
                    "delta_brier": EV.brier(gi["outcome_over"], gi["g1"]) - EV.brier(gi["outcome_over"], gi[ref]),
                    "delta_auc": EV.auc(gi["outcome_over"], gi["g1"]) - EV.auc(gi["outcome_over"], gi[ref]),
                    "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs,
                    "p_ll": p_ll, "p_brier": p_brier,
                }
            # advancement rule
            adv = (entry["g1_minus_p0"]["delta_auc"] > 0 and entry["g1_minus_r1"]["delta_auc"] > 0 and
                   entry["g1_minus_p0"]["delta_log_loss"] < 0 and entry["g1_minus_r1"]["delta_log_loss"] < 0 and
                   entry["g1_minus_p0"]["delta_brier"] < 0 and entry["g1_minus_r1"]["delta_brier"] < 0)
            entry["g1_advances_vs_p0_and_r1"] = bool(adv)
        results[prop] = entry

    # Holm across props for G1-vs-market p-values (LL and Brier families separately)
    p_ll = {pr: results[pr]["g1_minus_market"]["p_ll"] for pr in results if "g1_minus_market" in results[pr]}
    p_br = {pr: results[pr]["g1_minus_market"]["p_brier"] for pr in results if "g1_minus_market" in results[pr]}
    if p_ll:
        hll, hbr = EV.holm(p_ll), EV.holm(p_br)
        for pr in p_ll:
            results[pr]["g1_vs_market_holm_p_ll"] = hll[pr]
            results[pr]["g1_vs_market_holm_p_brier"] = hbr[pr]

    out = REPO / "artifacts/opportunity_v2/CANDIDATE_COMPARISON_P0_R1_G1.json"
    json.dump({"iters": ITERS, "results": results}, open(out, "w"), indent=2, default=float)
    # concise stdout
    for pr, e in results.items():
        row = {"prop": pr}
        for c in ("p0", "r1", "g1", "market"):
            if c in e:
                row[c] = {"n": e[c]["n"], "auc": round(e[c]["auc"], 4), "ll": round(e[c]["log_loss"], 4)}
        if "g1_advances_vs_p0_and_r1" in e:
            row["g1_advances"] = e["g1_advances_vs_p0_and_r1"]
        print(json.dumps(row))


if __name__ == "__main__":
    main()
