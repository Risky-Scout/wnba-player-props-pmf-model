#!/usr/bin/env python3
"""Exact-quote evaluation for Opportunity V2 (directive section 30).

Joins OPP_V2 active PMFs to the exact deterministic quote rows (which already carry the frozen P0
model probability, no-vig market probability, and settled outcome), settles OPP_V2 over-probability
from the ACTIVE PMF (push-safe), and reports proper-score metrics with paired differences,
game-date cluster bootstrap CIs, and Holm correction across the prop family.

Usage:
  python scripts/evaluate_opportunity_oof.py \
      --oof data/oof/opportunity_v2/oof_pmfs.parquet \
      --quotes artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet \
      --out artifacts/opportunity_v2/OPP_V2_EXACT_QUOTE_METRICS.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.pmf_builders import settled_over_probability

_EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, float), _EPS, 1 - _EPS)


def _log_loss(y, p):
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y, p):
    return float(np.mean((_clip(p) - y) ** 2))


def _auc(y, p):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _settle(js, line):
    arr = np.asarray(json.loads(js), float)
    p_over, _p_under, _p_push = settled_over_probability(arr, float(line))
    return p_over


def _bootstrap_delta(df, col_model, col_ref, dates, iters=10000, seed=42):
    """Date-cluster bootstrap of mean(delta LL) and mean(delta Brier) = model - ref."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(dates.unique()))
    by_date = {d: df.index[dates == d].to_numpy() for d in uniq}
    dll, dbs = [], []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_date[d] for d in pick])
        y = df.loc[idx, "outcome_over"].to_numpy()
        dll.append(_log_loss(y, df.loc[idx, col_model]) - _log_loss(y, df.loc[idx, col_ref]))
        dbs.append(_brier(y, df.loc[idx, col_model]) - _brier(y, df.loc[idx, col_ref]))
    return (np.percentile(dll, [2.5, 97.5]).tolist(), np.percentile(dbs, [2.5, 97.5]).tolist())


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10000)
    args = ap.parse_args()

    oof = pd.read_parquet(args.oof)
    q = pd.read_parquet(args.quotes).rename(columns={"prop": "stat"})
    for c in ("game_id", "player_id"):
        oof[c] = pd.to_numeric(oof[c], errors="coerce")
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[q["binary_score_eligible"] & q["outcome_over"].isin([0, 1])].copy()

    # one active PMF per (game_id, player_id, stat)
    pmf = oof.dropna(subset=["active_pmf_json"]).drop_duplicates(["game_id", "player_id", "stat"])
    j = q.merge(pmf[["game_id", "player_id", "stat", "active_pmf_json"]],
                on=["game_id", "player_id", "stat"], how="inner")
    # temporal guard: cutoff (from OOF) already <= quote; here quotes are pregame by construction.
    j["p_over_opp_v2"] = [ _settle(js, ln) for js, ln in zip(j["active_pmf_json"], j["line"]) ]
    j["outcome_over"] = j["outcome_over"].astype(int)

    results = {}
    pvals_ll = {}
    for prop, g in j.groupby("stat"):
        g = g.reset_index(drop=True)
        y = g["outcome_over"].to_numpy()
        n, ndates = len(g), g["game_date"].nunique()
        model, market, p0 = g["p_over_opp_v2"], g["market_prob_over_no_vig"], g["model_prob_over_final"]
        ll_m, ll_mkt, ll_p0 = _log_loss(y, model), _log_loss(y, market), _log_loss(y, p0)
        bs_m, bs_mkt, bs_p0 = _brier(y, model), _brier(y, market), _brier(y, p0)
        ci_ll, ci_bs = _bootstrap_delta(g, "p_over_opp_v2", "market_prob_over_no_vig",
                                        g["game_date"], iters=args.iters)
        # one-sided bootstrap p-value that delta LL < 0 (model better than market)
        rng = np.random.default_rng(7)
        uniq = np.array(sorted(g["game_date"].unique()))
        by_date = {d: g.index[g["game_date"] == d].to_numpy() for d in uniq}
        deltas = []
        for _ in range(args.iters):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_date[d] for d in pick])
            yy = g.loc[idx, "outcome_over"].to_numpy()
            deltas.append(_log_loss(yy, g.loc[idx, "p_over_opp_v2"]) -
                          _log_loss(yy, g.loc[idx, "market_prob_over_no_vig"]))
        deltas = np.array(deltas)
        p_better = float(np.mean(deltas >= 0.0))  # P(model no better than market)
        pvals_ll[prop] = p_better
        results[prop] = {
            "n": int(n), "game_dates": int(ndates),
            "opp_v2": {"log_loss": ll_m, "brier": bs_m, "auc": _auc(y, model)},
            "market": {"log_loss": ll_mkt, "brier": bs_mkt, "auc": _auc(y, market)},
            "p0_baseline": {"log_loss": ll_p0, "brier": bs_p0, "auc": _auc(y, p0)},
            "delta_vs_market": {"log_loss": ll_m - ll_mkt, "brier": bs_m - bs_mkt,
                                 "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs},
            "delta_vs_p0": {"log_loss": ll_m - ll_p0, "brier": bs_m - bs_p0},
            "beats_market": bool(ll_m < ll_mkt and bs_m < bs_mkt and ci_ll[1] < 0 and ci_bs[1] < 0),
            "beats_p0": bool(ll_m < ll_p0 and bs_m < bs_p0),
        }

    holm = _holm(pvals_ll)
    for prop in results:
        results[prop]["holm_adjusted_p_vs_market"] = holm.get(prop)
        results[prop]["market_superiority_pass"] = bool(
            results[prop]["beats_market"] and results[prop]["n"] >= 300 and
            results[prop]["game_dates"] >= 30 and holm.get(prop, 1.0) < 0.05)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"candidate": "OPP_V2_RAW", "results": results}, open(args.out, "w"), indent=2)
    print(json.dumps({p: {"n": r["n"], "dLL_vs_market": round(r["delta_vs_market"]["log_loss"], 5),
                          "dLL_vs_p0": round(r["delta_vs_p0"]["log_loss"], 5),
                          "auc_opp": round(r["opp_v2"]["auc"], 4), "auc_mkt": round(r["market"]["auc"], 4),
                          "beats_market": r["market_superiority_pass"]}
                      for p, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
