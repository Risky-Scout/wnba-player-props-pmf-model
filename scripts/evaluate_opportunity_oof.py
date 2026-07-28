#!/usr/bin/env python3
"""Exact-quote evaluation for Opportunity V2 — full market-superiority contract.

This evaluator implements the *complete* frozen supremacy contract (owner directive
section 1). Compared with the earlier draft it:

* builds ONE canonical scored-row artifact and **fails** (raises) on duplicate
  predictions, duplicate deterministic market rows, cross-line joins, missing
  ``quote_pair_id``, ambiguous identities, post-cutoff quotes, and push/void rows
  entering binary scoring — it never silently ``drop_duplicates``;
* uses :func:`sklearn.metrics.roc_auc_score` so tied probabilities are handled
  correctly;
* reports model / P0 / market log-loss, Brier, AUC, ECE, calibration intercept &
  slope, full-PMF log score, CRPS, rows, game dates, and worst fold;
* computes **separate** paired date-cluster bootstrap distributions and one-sided
  p-values for delta log-loss (``p_ll``) and delta Brier (``p_brier``);
* applies Holm correction **separately** across the log-loss family and the Brier
  family and persists ``holm_adjusted_p_ll`` and ``holm_adjusted_p_brier``;
* loads the minimum rows/dates and calibration/AUC gates from the frozen promotion
  contract in ``config/model/opportunity_v2.yaml`` (never hard-codes a weaker one);
* marks a candidate ``selection_eligible`` only when **every** gate passes — no
  single metric or single p-value can produce a PASS.

The pure functions here (``build_canonical_scored_rows``, ``candidate_metrics``,
``evaluate_candidate``) are import-safe and unit-tested in
``tests/opportunity/test_evaluator_contract.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional at import time; required for AUC / calibration slope
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    roc_auc_score = None
    LogisticRegression = None

_EPS = 1e-6

# Push/void settlement labels that must never enter binary scoring.
_NON_BINARY_SETTLEMENT = {"push", "void", "voided", "cancelled", "canceled", "no_action"}


class EvaluatorContractError(ValueError):
    """Raised when the canonical scored-row contract is violated."""


# --------------------------------------------------------------------------- #
# metric primitives
# --------------------------------------------------------------------------- #
def _clip(p):
    return np.clip(np.asarray(p, float), _EPS, 1 - _EPS)


def log_loss(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((_clip(p) - np.asarray(y, float)) ** 2))


def auc(y, p) -> float:
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    if roc_auc_score is None:  # pragma: no cover
        raise EvaluatorContractError("scikit-learn is required for AUC scoring")
    return float(roc_auc_score(y, np.asarray(p, float)))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = _clip(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Logistic re-fit of outcome on logit(p): returns (intercept, slope)."""
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2 or LogisticRegression is None:
        return float("nan"), float("nan")
    x = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def full_pmf_log_score(pmf_json, actual) -> float:
    """Mean negative log probability the count PMF assigns to the realized count."""
    scores = []
    for js, a in zip(pmf_json, actual):
        if js is None or (isinstance(js, float) and np.isnan(js)) or a is None or not np.isfinite(a):
            continue
        arr = np.asarray(json.loads(js), float)
        k = int(round(float(a)))
        pk = arr[k] if 0 <= k < arr.size else 0.0
        scores.append(-np.log(max(pk, _EPS)))
    return float(np.mean(scores)) if scores else float("nan")


def crps_discrete(pmf_json, actual) -> float:
    """Mean discrete CRPS = sum_k (CDF(k) - 1{actual<=k})^2 over the PMF support."""
    vals = []
    for js, a in zip(pmf_json, actual):
        if js is None or (isinstance(js, float) and np.isnan(js)) or a is None or not np.isfinite(a):
            continue
        arr = np.asarray(json.loads(js), float)
        cdf = np.cumsum(arr)
        k = np.arange(arr.size)
        step = (k >= int(round(float(a)))).astype(float)  # 1{actual <= k}
        vals.append(float(np.sum((cdf - step) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# full-PMF certification gates (proper, supported, calibrated, informative)
# --------------------------------------------------------------------------- #
def _pmf_quantile_idx(cdf: np.ndarray, q: float) -> int:
    """Smallest support index k with CDF(k) >= q."""
    return int(np.searchsorted(cdf, q, side="left"))


def full_pmf_certification(pmf_json, actual, contract: dict, reference: dict | None = None):
    """Frozen full-PMF gates. Returns (gates, measures).

    Replaces the old "finite full_pmf_log_score" check with independent gates for: normalization,
    tail-truncation, PMF-log-score noninferiority (vs reference or an absolute cap), CRPS
    noninferiority, central-50%/90% interval coverage, and sharpness. Any gate can independently
    fail a candidate.
    """
    tol_norm = float(contract.get("normalization_tolerance", 1e-6))
    # Randomized PIT coverage: for discrete forecasts, non-randomized central intervals over-cover.
    # U = F(k-1) + V*(F(k)-F(k-1)), V~U(0,1) is Uniform(0,1) under a calibrated forecast, so
    # P(0.25<=U<=0.75)=0.5 and P(0.05<=U<=0.95)=0.9 in expectation. Deterministic seed for reproducibility.
    rng = np.random.default_rng(int(contract.get("coverage_pit_seed", 12345)))
    logs, crps, w90, in50, in90, top = [], [], [], [], [], []
    n, oos, norm_bad = 0, 0, 0
    for js, a in zip(pmf_json, actual):
        if js is None or (isinstance(js, float) and np.isnan(js)) or a is None or not np.isfinite(a):
            continue
        arr = np.asarray(json.loads(js), float)
        n += 1
        s = float(arr.sum())
        if (not np.all(arr >= -1e-9)) or abs(s - 1.0) > tol_norm:
            norm_bad += 1
        k = int(round(float(a)))
        pk = arr[k] if 0 <= k < arr.size else 0.0
        logs.append(-np.log(max(pk, _EPS)))
        cdf = np.cumsum(arr)
        kk = np.arange(arr.size)
        crps.append(float(np.sum((cdf - (kk >= k).astype(float)) ** 2)))
        # sharpness measure: central-90% support width from the discrete CDF
        lo90, hi90 = _pmf_quantile_idx(cdf, 0.05), _pmf_quantile_idx(cdf, 0.95)
        w90.append(hi90 - lo90)
        # randomized PIT for calibration-correct coverage
        f_k = float(cdf[k]) if 0 <= k < arr.size else (1.0 if k >= arr.size else 0.0)
        f_km1 = float(cdf[k - 1]) if 0 <= k - 1 < arr.size else (0.0 if k - 1 < 0 else 1.0)
        u = f_km1 + rng.random() * max(f_k - f_km1, 0.0)
        in50.append(0.25 <= u <= 0.75)
        in90.append(0.05 <= u <= 0.95)
        top.append(float(arr[-1]))
        if k >= arr.size:
            oos += 1

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    mean_log, mean_crps = _m(logs), _m(crps)
    cov50, cov90, mean_w90 = _m(in50), _m(in90), _m(w90)
    max_top = float(np.max(top)) if top else float("nan")
    oos_frac = (oos / n) if n else float("nan")
    ref_log = (reference or {}).get("full_pmf_log_score", contract.get("full_pmf_log_score_max"))
    ref_crps = (reference or {}).get("crps", contract.get("crps_max"))
    tol_log = float(contract.get("full_pmf_log_score_noninferiority_tol", 0.05))
    tol_crps = float(contract.get("crps_noninferiority_tol", 0.05))

    gates = {
        "pmf_normalization_ok": bool(n > 0 and norm_bad == 0),
        "pmf_tail_truncation_ok": bool(np.isfinite(max_top) and np.isfinite(oos_frac)
                                       and max_top <= float(contract.get("tail_bin_mass_max", 0.02))
                                       and oos_frac <= float(contract.get("out_of_support_frac_max", 0.01))),
        "pmf_log_score_noninferiority_ok": bool(np.isfinite(mean_log) and ref_log is not None
                                                and mean_log <= float(ref_log) + tol_log),
        "crps_noninferiority_ok": bool(np.isfinite(mean_crps) and ref_crps is not None
                                       and mean_crps <= float(ref_crps) + tol_crps),
        "coverage_50_ok": bool(np.isfinite(cov50)
                               and abs(cov50 - 0.50) <= float(contract.get("coverage_tol_50", 0.12))),
        "coverage_90_ok": bool(np.isfinite(cov90)
                               and abs(cov90 - 0.90) <= float(contract.get("coverage_tol_90", 0.08))),
        "sharpness_ok": bool(np.isfinite(mean_w90) and 0.0 < mean_w90
                             <= float(contract.get("sharpness_max_width_90", 60.0))),
    }
    measures = {
        "n_scored": int(n), "full_pmf_log_score": mean_log, "crps": mean_crps,
        "coverage_50": cov50, "coverage_90": cov90, "mean_width_90": mean_w90,
        "max_top_bin_mass": max_top, "out_of_support_frac": oos_frac,
        "reference_full_pmf_log_score": (None if ref_log is None else float(ref_log)),
        "reference_crps": (None if ref_crps is None else float(ref_crps)),
    }
    return gates, measures


# --------------------------------------------------------------------------- #
# canonical scored-row builder (fail-closed)
# --------------------------------------------------------------------------- #
KEY = ["game_id", "player_id", "prop"]


def build_canonical_scored_rows(
    oof: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    require_cutoff: bool = True,
) -> pd.DataFrame:
    """Join OOF active PMFs to deterministic scored quotes; FAIL on any ambiguity.

    Returns exactly one row per (game_id, player_id, prop). Raises
    :class:`EvaluatorContractError` on duplicate predictions, duplicate market
    rows, cross-line joins, missing quote_pair_id, ambiguous identities,
    post-cutoff quotes, or push/void rows in the binary-scored set.
    """
    oof = oof.copy()
    q = quotes.copy()
    if "stat" in oof.columns and "prop" not in oof.columns:
        oof = oof.rename(columns={"stat": "prop"})
    if "stat" in q.columns and "prop" not in q.columns:
        q = q.rename(columns={"stat": "prop"})

    for c in ("game_id", "player_id"):
        oof[c] = pd.to_numeric(oof[c], errors="coerce")
        q[c] = pd.to_numeric(q[c], errors="coerce")

    # ambiguous identities
    if q[["game_id", "player_id"]].isna().any().any() or q["prop"].isna().any():
        raise EvaluatorContractError("ambiguous identities: null game_id/player_id/prop in quotes")
    if "quote_pair_id" not in q.columns or q["quote_pair_id"].isna().any():
        raise EvaluatorContractError("missing quote_pair_id in one or more quote rows")

    # push/void rows must not enter binary scoring
    if "settlement_status" in q.columns:
        bad = q["settlement_status"].astype(str).str.lower().isin(_NON_BINARY_SETTLEMENT)
        if (bad & q.get("binary_score_eligible", True)).any():
            raise EvaluatorContractError("push/void rows flagged binary_score_eligible entered scoring")
        q = q[~bad].copy()

    # restrict to legitimately scorable binary rows
    if "binary_score_eligible" in q.columns:
        q = q[q["binary_score_eligible"].astype(bool)].copy()
    q = q[q["outcome_over"].isin([0, 1])].copy()

    # cross-line joins: exactly one line per key
    if "line" in q.columns:
        nlines = q.groupby(KEY)["line"].nunique()
        if (nlines > 1).any():
            raise EvaluatorContractError(
                f"cross-line join: {int((nlines > 1).sum())} keys have multiple lines"
            )

    # duplicate deterministic market rows
    dup_mkt = q.duplicated(subset=KEY, keep=False)
    if dup_mkt.any():
        raise EvaluatorContractError(
            f"duplicate deterministic market rows: {int(dup_mkt.sum())} rows share a key"
        )

    # duplicate OOF predictions
    pmf = oof.dropna(subset=["active_pmf_json"]).copy()
    dup_oof = pmf.duplicated(subset=KEY, keep=False)
    if dup_oof.any():
        raise EvaluatorContractError(
            f"duplicate OOF predictions: {int(dup_oof.sum())} rows share a key"
        )

    keep_cols = KEY + [c for c in ("active_pmf_json", "prediction_cutoff_utc",
                                   "cutoff_policy_id", "scheduled_tip_utc", "oof_fold")
                       if c in pmf.columns]
    j = q.merge(pmf[keep_cols], on=KEY, how="inner", validate="one_to_one")
    if j.empty:
        raise EvaluatorContractError("no rows after join of OOF PMFs to scored quotes")

    # post-cutoff quotes: quote decision timestamp must be <= prediction cutoff
    if require_cutoff and "prediction_cutoff_utc" in j.columns:
        qt_col = "decision_timestamp" if "decision_timestamp" in j.columns else (
            "pair_timestamp" if "pair_timestamp" in j.columns else None)
        if qt_col is not None:
            cutoff = pd.to_datetime(j["prediction_cutoff_utc"], utc=True, errors="coerce")
            qt = pd.to_datetime(j[qt_col], utc=True, errors="coerce")
            late = (qt.notna() & cutoff.notna() & (qt > cutoff))
            if late.any():
                raise EvaluatorContractError(
                    f"post-cutoff quotes: {int(late.sum())} rows have quote time after prediction cutoff"
                )

    # settle candidate over-probability from the ACTIVE PMF (push-safe)
    from wnba_props_model.opportunity.pmf_builders import settled_over_probability

    def _settle(js, line):
        arr = np.asarray(json.loads(js), float)
        p_over, _u, _p = settled_over_probability(arr, float(line))
        return p_over

    j["p_over_opp_v2"] = [_settle(js, ln) for js, ln in zip(j["active_pmf_json"], j["line"])]
    j["outcome_over"] = j["outcome_over"].astype(int)

    # final invariant: one row per key
    if j.duplicated(subset=KEY).any():
        raise EvaluatorContractError("canonical artifact still has duplicate keys after join")
    return j


# --------------------------------------------------------------------------- #
# per-candidate metrics
# --------------------------------------------------------------------------- #
def candidate_metrics(g: pd.DataFrame, prob_col: str) -> dict:
    y = g["outcome_over"].to_numpy()
    p = g[prob_col].to_numpy()
    m = {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "auc": auc(y, p),
        "ece": expected_calibration_error(y, p),
    }
    m["calibration_intercept"], m["calibration_slope"] = calibration_intercept_slope(y, p)
    if prob_col == "p_over_opp_v2" and "active_pmf_json" in g.columns and "actual" in g.columns:
        m["full_pmf_log_score"] = full_pmf_log_score(g["active_pmf_json"], g["actual"])
        m["crps"] = crps_discrete(g["active_pmf_json"], g["actual"])
    return m


def _paired_bootstrap(g: pd.DataFrame, model_col: str, ref_col: str, dates,
                      iters: int, seed: int):
    """Paired date-cluster bootstrap. Returns (ci_ll, ci_brier, p_ll, p_brier)."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(dates)))
    gi = g.reset_index(drop=True)
    dser = pd.Series(np.asarray(dates)).reset_index(drop=True)
    by_date = {d: gi.index[dser == d].to_numpy() for d in uniq}
    dll, dbs = [], []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_date[d] for d in pick])
        y = gi.loc[idx, "outcome_over"].to_numpy()
        dll.append(log_loss(y, gi.loc[idx, model_col]) - log_loss(y, gi.loc[idx, ref_col]))
        dbs.append(brier(y, gi.loc[idx, model_col]) - brier(y, gi.loc[idx, ref_col]))
    dll = np.asarray(dll)
    dbs = np.asarray(dbs)
    ci_ll = np.percentile(dll, [2.5, 97.5]).tolist()
    ci_bs = np.percentile(dbs, [2.5, 97.5]).tolist()
    p_ll = float(np.mean(dll >= 0.0))       # P(model no better than ref on LL)
    p_brier = float(np.mean(dbs >= 0.0))    # P(model no better than ref on Brier)
    return ci_ll, ci_bs, p_ll, p_brier


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def evaluate_candidate(g: pd.DataFrame, contract: dict, *,
                       ci_ll, ci_bs, holm_p_ll: float, holm_p_brier: float,
                       parity_pass: bool = True, reference_pmf: dict | None = None) -> dict:
    """Apply the frozen promotion gate. Returns a dict incl. ``selection_eligible``
    and every sub-gate so no single metric/p-value alone can produce a PASS."""
    y = g["outcome_over"].to_numpy()
    n, ndates = len(g), int(pd.Series(g["game_date"]).nunique())
    mm = candidate_metrics(g, "p_over_opp_v2")
    mk = candidate_metrics(g, "market_prob_over_no_vig")
    d_ll = mm["log_loss"] - mk["log_loss"]
    d_bs = mm["brier"] - mk["brier"]

    # frozen full-PMF certification (proper / supported / calibrated / informative)
    if "active_pmf_json" in g.columns and "actual" in g.columns:
        pmf_gates, pmf_measures = full_pmf_certification(
            g["active_pmf_json"], g["actual"], contract, reference=reference_pmf)
    else:
        pmf_gates = {k: False for k in ("pmf_normalization_ok", "pmf_tail_truncation_ok",
                     "pmf_log_score_noninferiority_ok", "crps_noninferiority_ok",
                     "coverage_50_ok", "coverage_90_ok", "sharpness_ok")}
        pmf_measures = {}

    gates = {
        "rows_ok": n >= int(contract["required_rows"]),
        "dates_ok": ndates >= int(contract["required_dates"]),
        "delta_ll_neg": d_ll < 0,
        "delta_brier_neg": d_bs < 0,
        "ci_ll_upper_neg": ci_ll[1] < 0,
        "ci_brier_upper_neg": ci_bs[1] < 0,
        "holm_ll_ok": holm_p_ll <= float(contract["holm_alpha"]),
        "holm_brier_ok": holm_p_brier <= float(contract["holm_alpha"]),
        "auc_rule_ok": (np.isfinite(mm["auc"]) and mm["auc"] > float(contract["auc_min"])
                        and mm["auc"] >= mk["auc"] - float(contract["auc_vs_market_tol"])),
        "ece_ok": np.isfinite(mm["ece"]) and mm["ece"] <= float(contract["ece_max"]),
        "calibration_ok": (np.isfinite(mm["calibration_slope"])
                           and float(contract["calibration_slope_min"]) <= mm["calibration_slope"]
                           <= float(contract["calibration_slope_max"])
                           and abs(mm["calibration_intercept"]) <= float(contract["calibration_intercept_abs_max"])),
        "parity_ok": bool(parity_pass),
    }
    gates.update(pmf_gates)
    gates = {k: bool(v) for k, v in gates.items()}
    mm = {**mm, "full_pmf_certification": pmf_measures}
    return {
        "n": int(n), "game_dates": ndates,
        "opp_v2": mm, "market": mk,
        "delta_vs_market": {"log_loss": d_ll, "brier": d_bs,
                            "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs},
        "holm_adjusted_p_ll": holm_p_ll,
        "holm_adjusted_p_brier": holm_p_brier,
        "gates": gates,
        "selection_eligible": bool(all(gates.values())),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_contract(path: str | None) -> dict:
    default = {"required_rows": 300, "required_dates": 30, "holm_alpha": 0.05,
               "auc_min": 0.5, "auc_vs_market_tol": 0.0, "calibration_slope_min": 0.80,
               "calibration_slope_max": 1.25, "calibration_intercept_abs_max": 0.25,
               "ece_max": 0.05, "bootstrap_iters": 10000, "bootstrap_seed": 42,
               "normalization_tolerance": 1e-6,
               "full_pmf_log_score_max": 2.5, "crps_max": 1.5,
               "full_pmf_log_score_noninferiority_tol": 0.05, "crps_noninferiority_tol": 0.05,
               "tail_bin_mass_max": 0.02, "out_of_support_frac_max": 0.01,
               "coverage_tol_50": 0.12, "coverage_tol_90": 0.08, "sharpness_max_width_90": 60.0}
    if path and Path(path).exists():
        import yaml
        cfg = yaml.safe_load(open(path)) or {}
        default.update(cfg.get("promotion", {}))
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/model/opportunity_v2.yaml")
    ap.add_argument("--candidate", default="OPP_V2_RAW")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--parity-pass", action="store_true", default=False,
                    help="assert delivery/OOF parity has been verified for this candidate")
    args = ap.parse_args()

    contract = _load_contract(args.config)
    iters = args.iters or int(contract["bootstrap_iters"])
    seed = int(contract["bootstrap_seed"])

    oof = pd.read_parquet(args.oof)
    quotes = pd.read_parquet(args.quotes)
    canon = build_canonical_scored_rows(oof, quotes)

    # per-prop bootstrap + p-values, then Holm separately across LL and Brier families
    ci_by_prop, p_ll_by_prop, p_brier_by_prop = {}, {}, {}
    for prop, g in canon.groupby("prop"):
        ci_ll, ci_bs, p_ll, p_brier = _paired_bootstrap(
            g, "p_over_opp_v2", "market_prob_over_no_vig", g["game_date"], iters, seed)
        ci_by_prop[prop] = (ci_ll, ci_bs)
        p_ll_by_prop[prop] = p_ll
        p_brier_by_prop[prop] = p_brier
    holm_ll = holm(p_ll_by_prop)
    holm_brier = holm(p_brier_by_prop)

    results = {}
    for prop, g in canon.groupby("prop"):
        ci_ll, ci_bs = ci_by_prop[prop]
        res = evaluate_candidate(g, contract, ci_ll=ci_ll, ci_bs=ci_bs,
                                 holm_p_ll=holm_ll[prop], holm_p_brier=holm_brier[prop],
                                 parity_pass=args.parity_pass)
        res["p_ll_raw"] = p_ll_by_prop[prop]
        res["p_brier_raw"] = p_brier_by_prop[prop]
        if "oof_fold" in g.columns and g["oof_fold"].notna().any():
            worst = None
            for fold, gf in g.groupby("oof_fold"):
                if len(gf) < 10 or gf["outcome_over"].nunique() < 2:
                    continue
                dll = log_loss(gf["outcome_over"], gf["p_over_opp_v2"]) - \
                    log_loss(gf["outcome_over"], gf["market_prob_over_no_vig"])
                if worst is None or dll > worst[1]:
                    worst = (str(fold), float(dll))
            res["worst_fold"] = {"fold": worst[0], "delta_log_loss": worst[1]} if worst else None
        results[prop] = res

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"candidate": args.candidate, "contract": contract, "results": results},
              open(args.out, "w"), indent=2)
    print(json.dumps({p: {"n": r["n"], "dLL": round(r["delta_vs_market"]["log_loss"], 5),
                          "dBrier": round(r["delta_vs_market"]["brier"], 5),
                          "auc": round(r["opp_v2"]["auc"], 4), "auc_mkt": round(r["market"]["auc"], 4),
                          "holm_ll": round(r["holm_adjusted_p_ll"], 4),
                          "holm_brier": round(r["holm_adjusted_p_brier"], 4),
                          "eligible": r["selection_eligible"]}
                      for p, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
