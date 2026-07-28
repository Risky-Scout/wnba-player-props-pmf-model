#!/usr/bin/env python3
"""P0 vs R1 vs G1 vs G2 vs PTS_DECOMP vs market on ONE identical canonical universe.

Fail-closed by construction:
  * NO silent ``drop_duplicates`` anywhere. Duplicate detection RAISES with the offending key(s),
    the source, the duplicate row count, and the conflicting fields.
  * Every candidate OOF is joined to the deterministic scored quotes through the canonical evaluator's
    fail-closed ``EV.build_canonical_scored_rows`` (which itself raises on duplicate OOF predictions,
    duplicate/cross-line market rows, missing quote_pair_id, ambiguous identities, post-cutoff quotes,
    and push/void rows) — never through an ad-hoc dedup.
  * Bootstrap iterations and seed come from the FROZEN promotion contract
    (config/model/opportunity_v2.yaml: promotion.bootstrap_iters / bootstrap_seed). No hard-coded
    weaker value.

Reports, per prop, for each present candidate + P0 + market: n, LL, Brier, AUC, ECE, calibration
intercept/slope, CRPS, full-PMF log score; and paired date-cluster bootstrap deltas (95% CIs, one-sided
p_ll/p_brier) for each candidate vs market / P0 / R1 / G1, with Holm correction across the
candidate-vs-market family per prop.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("opp_eval", REPO / "scripts" / "evaluate_opportunity_oof.py")
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)

KEY = ["game_id", "player_id", "prop"]

# Candidate OOF sources (label -> parquet). A candidate is included only if its OOF exists.
CANDIDATE_OOF = {
    "r1": "data/oof/opportunity_v2/oof_pmfs.parquet",
    "g1": "data/oof/opportunity_v2_team_share/oof_pmfs.parquet",
    "g2": "data/oof/opportunity_v2_g2/oof_pmfs.parquet",
    "pts_decomp": "data/oof/opportunity_v2_pts_decomp/oof_pmfs.parquet",
}
QUOTES = "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet"


def _load_contract() -> tuple[int, int]:
    cfg = yaml.safe_load(open(REPO / "config/model/opportunity_v2.yaml")) or {}
    p = cfg.get("promotion", {})
    return int(p.get("bootstrap_iters", 10000)), int(p.get("bootstrap_seed", 42))


def _assert_unique(df: pd.DataFrame, key: list[str], source: str) -> None:
    """Fail-closed duplicate check. Raises with key, source, row count, and conflicting fields."""
    dup = df.duplicated(subset=key, keep=False)
    if not dup.any():
        return
    d = df[dup]
    conflict_cols: set[str] = set()
    for _, grp in d.groupby(key):
        for c in grp.columns:
            if c in key:
                continue
            if grp[c].nunique(dropna=False) > 1:
                conflict_cols.add(c)
    ex = [dict(zip(key, k if isinstance(k, tuple) else (k,)))
          for k in list(d.groupby(key).groups)[:5]]
    raise ValueError(
        f"[{source}] duplicate rows on {key}: {int(dup.sum())} rows across "
        f"{d.groupby(key).ngroups} keys; example_keys={ex}; conflicting_fields={sorted(conflict_cols)}"
    )


def _prep_quotes() -> pd.DataFrame:
    q = pd.read_parquet(REPO / QUOTES)
    if "stat" in q.columns and "prop" not in q.columns:
        q = q.rename(columns={"stat": "prop"})
    for c in ("game_id", "player_id"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q[q["binary_score_eligible"].astype(bool) & q["outcome_over"].isin([0, 1])].copy()
    _assert_unique(q, KEY, "quotes:PRIMARY_DETERMINISTIC_SCORED_ROWS")  # fail-closed, NO drop
    q["outcome_over"] = q["outcome_over"].astype(int)
    return q


def _metrics(g: pd.DataFrame, col: str, pmf_col: str | None) -> dict:
    y = g["outcome_over"].to_numpy()
    p = g[col].to_numpy()
    m = {"n": int(len(g)), "log_loss": EV.log_loss(y, p), "brier": EV.brier(y, p),
         "auc": EV.auc(y, p), "ece": EV.expected_calibration_error(y, p)}
    ci, sl = EV.calibration_intercept_slope(y, p)
    m["calibration_intercept"], m["calibration_slope"] = ci, sl
    if pmf_col and pmf_col in g.columns and "actual" in g.columns:
        m["full_pmf_log_score"] = EV.full_pmf_log_score(g[pmf_col], g["actual"])
        m["crps"] = EV.crps_discrete(g[pmf_col], g["actual"])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/opportunity_v2/CANDIDATE_COMPARISON_ALL.json")
    args = ap.parse_args()

    iters, seed = _load_contract()
    q = _prep_quotes()

    base = q[KEY + ["outcome_over", "game_date", "line", "actual", "oof_fold",
                    "model_prob_over_final", "market_prob_over_no_vig"]].rename(
        columns={"model_prob_over_final": "p0", "market_prob_over_no_vig": "market"})

    present: list[str] = []
    for label, rel in CANDIDATE_OOF.items():
        p = REPO / rel
        if not p.exists():
            continue
        oof = pd.read_parquet(p)
        # Route the join through the fail-closed canonical evaluator (raises on dup/cross-line/etc.).
        canon = EV.build_canonical_scored_rows(oof, q)
        sub = canon[KEY + ["p_over_opp_v2", "active_pmf_json"]].rename(
            columns={"p_over_opp_v2": label, "active_pmf_json": f"{label}_pmf"})
        _assert_unique(sub, KEY, f"oof:{label}")
        base = base.merge(sub, on=KEY, how="left")
        present.append(label)

    # candidate -> (prob_col, pmf_col). p0/market have no PMF here.
    series = {"p0": ("p0", None), "market": ("market", None)}
    for label in present:
        series[label] = (label, f"{label}_pmf")

    results: dict = {}
    for prop, gall in base.groupby("prop"):
        entry = {"n_total": int(len(gall)), "game_dates": int(gall["game_date"].nunique())}
        for name, (col, pmf) in series.items():
            if col not in gall.columns:
                continue
            g = gall.dropna(subset=[col]).copy()
            if len(g) == 0:
                continue
            entry[name] = _metrics(g, col, pmf)

        # paired comparisons for each present candidate vs its references
        refs_for = {"r1": ["p0", "market"], "g1": ["p0", "r1", "market"],
                    "g2": ["p0", "r1", "g1", "market"], "pts_decomp": ["p0", "r1", "market"]}
        cand_vs_market_pll, cand_vs_market_pbr = {}, {}
        for cand in present:
            if cand not in gall.columns or gall[cand].notna().sum() == 0:
                continue
            for ref in refs_for.get(cand, ["market"]):
                if ref not in gall.columns:
                    continue
                gi = gall.dropna(subset=[cand, ref]).reset_index(drop=True)
                if len(gi) == 0 or gi["outcome_over"].nunique() < 2:
                    continue
                ci_ll, ci_bs, p_ll, p_brier = EV._paired_bootstrap(
                    gi, cand, ref, gi["game_date"], iters, seed)
                entry[f"{cand}_minus_{ref}"] = {
                    "n": int(len(gi)),
                    "delta_log_loss": EV.log_loss(gi["outcome_over"], gi[cand]) - EV.log_loss(gi["outcome_over"], gi[ref]),
                    "delta_brier": EV.brier(gi["outcome_over"], gi[cand]) - EV.brier(gi["outcome_over"], gi[ref]),
                    "delta_auc": EV.auc(gi["outcome_over"], gi[cand]) - EV.auc(gi["outcome_over"], gi[ref]),
                    "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs,
                    "p_ll": p_ll, "p_brier": p_brier,
                }
                if ref == "market":
                    cand_vs_market_pll[cand] = p_ll
                    cand_vs_market_pbr[cand] = p_brier
        # Holm across the candidate-vs-market family within this prop
        if cand_vs_market_pll:
            hll, hbr = EV.holm(cand_vs_market_pll), EV.holm(cand_vs_market_pbr)
            entry["holm_vs_market_p_ll"] = hll
            entry["holm_vs_market_p_brier"] = hbr
        results[prop] = entry

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"iters": iters, "seed": seed, "candidates_present": present, "results": results}
    json.dump(payload, open(out, "w"), indent=2, default=float)
    # keep the legacy filename in sync for existing references
    legacy = REPO / "artifacts/opportunity_v2/CANDIDATE_COMPARISON_P0_R1_G1.json"
    json.dump(payload, open(legacy, "w"), indent=2, default=float)

    for pr, e in results.items():
        row = {"prop": pr}
        for c in ["p0", "r1", "g1", "g2", "pts_decomp", "market"]:
            if c in e:
                row[c] = {"n": e[c]["n"], "auc": round(e[c]["auc"], 4), "ll": round(e[c]["log_loss"], 4)}
        print(json.dumps(row))


if __name__ == "__main__":
    main()
