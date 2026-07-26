"""PRODUCTION pure-OOF evaluation vs the exact same-book no-vig market (STEP 6-9).

This is the authoritative before/after measurement of the PURE production model. It is meant to
run in GitHub Actions AFTER ``scripts/build_oof_pmfs.py`` has regenerated the walk-forward OOF
with the ``pure_forecast`` config (``config/model/stage5_oof.yaml``), so the OOF probabilities
are a genuine 0%-market baseline.

Lineage (the repaired path, NOT the invalid post-hoc de-DNP shortcut):

    active_pmf  ->  settle_over_from_active_pmf(active_pmf, line)   (push-safe, void-on-DNP)
                ->  monotone pure binary calibration (model-vs-outcome only, nested CV)
                ->  model_prob_over_final

``p_dnp`` is carried separately (availability/void suppression) and is NEVER divided into the
settled over/under probability. See ``models/availability_pmf.py`` for the contract.

It joins the fresh pure OOF (by canonical game_id + player_id + prop) with the exact same-book
decision-time quotes committed in the repo (the G0_v2 primary deterministic scored rows) and
emits, per direct prop, model-vs-market log loss, Brier, AUC, ECE and model-only CRPS, with
date-cluster bootstrap CIs (>=5000 replicates) and worst-fold, then applies the REAL acceptance
contract (STEP 9). It does NOT touch the promotion/edge gates and does NOT claim a win — it
reports honestly. Props that fail the contract are reported as FAIL.

Output: ``artifacts/pure_supremacy/PRODUCTION_PURE_OOF_METRICS.{json,csv}``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.evaluation.rolling_origin import (  # noqa: E402
    all_chronology_pass,
    expanding_window_folds,
    fold_manifest,
)
from wnba_props_model.models.availability_pmf import settle_over_from_active_pmf  # noqa: E402
from wnba_props_model.models.market import (  # noqa: E402
    UndefinedSettledProbabilityError,
)
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf  # noqa: E402

app = typer.Typer(add_completion=False)

EPS = 1e-6
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
QUOTE_COVERED = ["pts", "reb", "ast", "fg3m"]
MIN_TRAIN_DATES = 15
VAL_BLOCK_DATES = 3
ECE_MARGIN = 0.03
WORST_FOLD_LL_MAX = 0.05
N_BOOTSTRAP = 5000
MARKET_COL = "market_prob_over_no_vig"
FINAL_COL = "model_prob_over_final"


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = _clip(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _brier(y, p):
    return float(np.mean((_clip(p) - np.asarray(y, float)) ** 2))


def _ece(y, p, nb=10):
    y = np.asarray(y, int)
    p = _clip(p)
    ed = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, ed[1:-1]), 0, nb - 1)
    n = len(p)
    e = 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _crps_discrete(pmf_json: str, actual: float) -> float:
    """Discrete CRPS (== ranked probability score) of a count PMF vs the observed count.

    CRPS = sum_k (CDF_pmf(k) - 1{actual <= k})^2 over the PMF support. Lower is better; a point
    mass at the true outcome scores 0. This is the model-only full-PMF distributional metric
    (the market has no distribution, only a binary quote).
    """
    pmf = normalize_pmf(json_to_pmf(pmf_json))
    cdf = np.cumsum(pmf)
    k = np.arange(pmf.size)
    step = (k >= float(actual)).astype(float)
    return float(np.sum((cdf - step) ** 2))


# ---------------------------------------------------------------------------
# Active-PMF settlement  (the repaired production lineage)
# ---------------------------------------------------------------------------

def _settled_over_from_active(active_pmf_json: str, line: float) -> float | None:
    """Push-safe P(over) settled from the ACTIVE (conditional-on-play) PMF, or None if
    undefined (all settled mass on the push)."""
    try:
        return float(settle_over_from_active_pmf(active_pmf_json, float(line)).p_over_settled)
    except (UndefinedSettledProbabilityError, ValueError):
        return None


def _fit_calibrator(name, tr):
    """Return (predict_fn, monotone_ok). ``tr`` is a fold's training slice with columns
    ``p_over_settled_active`` and ``outcome_over``."""
    if name == "P0_active_settled_identity":
        return (lambda d: _clip(d["p_over_settled_active"].to_numpy(float))), True
    if name == "P1_active_settled_platt":
        base = _logit(tr["p_over_settled_active"].to_numpy(float)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(
            base, tr["outcome_over"].to_numpy(int))
        slope = float(lr.coef_[0][0])

        def _p(d):
            b = _logit(d["p_over_settled_active"].to_numpy(float)).reshape(-1, 1)
            return _clip(lr.predict_proba(b)[:, 1])
        return _p, slope >= 0
    if name == "P2_active_settled_isotonic":
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(
            tr["p_over_settled_active"].to_numpy(float), tr["outcome_over"].to_numpy(int))

        def _pi(d):
            return _clip(iso.predict(d["p_over_settled_active"].to_numpy(float)))
        return _pi, True  # increasing=True -> monotone by construction
    if name == "P3_active_settled_beta":
        # Beta calibration (Kull et al.): logistic on [ln p, ln(1-p)]. Monotone in p iff both
        # a := coef_lnp >= 0 and b := -coef_ln1mp >= 0.
        p = _clip(tr["p_over_settled_active"].to_numpy(float))
        feats = np.column_stack([np.log(p), np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(feats, tr["outcome_over"].to_numpy(int))
        a, b = float(lr.coef_[0][0]), float(lr.coef_[0][1])

        def _pb(d):
            q = _clip(d["p_over_settled_active"].to_numpy(float))
            return _clip(lr.predict_proba(np.column_stack([np.log(q), np.log(1 - q)]))[:, 1])
        return _pb, (a >= 0 and b <= 0)
    raise ValueError(name)


# Pure recalibration candidate family (AST repair, owner item 7): all consume ONLY the model's
# own active-PMF settled probability vs outcome (zero market input) and are monotone.
PURE_RECAL_CANDIDATES = [
    "P0_active_settled_identity", "P1_active_settled_platt",
    "P2_active_settled_isotonic", "P3_active_settled_beta",
]


def nested_eval(pdf: pd.DataFrame, name: str, *, min_train_dates: int = MIN_TRAIN_DATES,
                val_block_dates: int = VAL_BLOCK_DATES) -> dict | None:
    """Grouped expanding-window nested rolling-origin CV: each fold fits the (pure) calibrator on
    training dates only and predicts held-out dates. The market comparison uses the exact no-vig
    quote. Returns aggregated OOS metrics + per-fold LL deltas for worst-fold analysis."""
    d = pdf["game_date"].to_numpy()
    folds = expanding_window_folds(d, min_train_dates=min_train_dates, val_block_dates=val_block_dates)
    if not folds:
        return None
    pred = np.full(len(pdf), np.nan)
    mono = True
    per_fold = []
    for f in folds:
        tr = pdf[np.isin(d, list(f.train_dates))]
        va_idx = np.where(np.isin(d, list(f.val_dates)))[0]
        if len(tr) < 30 or len(va_idx) == 0:
            continue
        fn, ok = _fit_calibrator(name, tr)
        mono = mono and ok
        pv = fn(pdf.iloc[va_idx])
        pred[va_idx] = pv
        yv = pdf["outcome_over"].to_numpy(int)[va_idx]
        pkv = pdf[MARKET_COL].to_numpy(float)[va_idx]
        per_fold.append(_ll(yv, pv) - _ll(yv, pkv))
    m = np.isfinite(pred)
    if m.sum() == 0:
        return None
    y = pdf["outcome_over"].to_numpy(int)[m]
    pk = pdf[MARKET_COL].to_numpy(float)[m]
    pr = pred[m]
    crps = float(np.mean(pdf["crps_active"].to_numpy(float)[m])) if "crps_active" in pdf else float("nan")
    worst = float(max(per_fold)) if per_fold else float("nan")
    return {
        "candidate": name, "n": int(m.sum()), "n_folds": len(per_fold),
        "model_logloss": _ll(y, pr), "market_logloss": _ll(y, pk),
        "logloss_delta": _ll(y, pr) - _ll(y, pk),
        "model_brier": _brier(y, pr), "market_brier": _brier(y, pk),
        "brier_delta": _brier(y, pr) - _brier(y, pk),
        "model_auc": _auc(y, pr), "market_auc": _auc(y, pk),
        "model_ece": _ece(y, pr), "market_ece": _ece(y, pk),
        "model_crps_active_pmf": crps,
        "worst_fold_logloss_delta": worst, "monotone": bool(mono),
        "_oof_pred": pred, "_mask": m,
    }


def _bootstrap_ci(res: dict, pdf: pd.DataFrame, n_boot=N_BOOTSTRAP, seed=20260726) -> dict:
    """Paired date-cluster bootstrap of (model - market) LL and Brier deltas."""
    m = res["_mask"]
    pr = res["_oof_pred"][m]
    sub = pdf[m].reset_index(drop=True)
    y = sub["outcome_over"].to_numpy(int)
    pk = sub[MARKET_COL].to_numpy(float)
    dates = sub["game_date"].to_numpy()
    uniq = np.array(sorted(set(dates)))
    by = {u: np.where(dates == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    dll, dbr = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        dll.append(_ll(y[idx], pr[idx]) - _ll(y[idx], pk[idx]))
        dbr.append(_brier(y[idx], pr[idx]) - _brier(y[idx], pk[idx]))
    dll, dbr = np.array(dll), np.array(dbr)
    # One-sided paired date-cluster bootstrap p-value for H1: delta < 0 (model better than
    # market). p = P(delta_boot >= 0) with the standard +1/+1 finite-sample correction.
    p_ll = float((int(np.sum(dll >= 0.0)) + 1) / (n_boot + 1))
    p_brier = float((int(np.sum(dbr >= 0.0)) + 1) / (n_boot + 1))
    return {
        "n_bootstrap": int(n_boot),
        "logloss_delta_ci95": [float(np.percentile(dll, 2.5)), float(np.percentile(dll, 97.5))],
        "brier_delta_ci95": [float(np.percentile(dbr, 2.5)), float(np.percentile(dbr, 97.5))],
        "logloss_upper95_below_zero": bool(np.percentile(dll, 97.5) < 0),
        "brier_upper95_below_zero": bool(np.percentile(dbr, 97.5) < 0),
        "logloss_p_onesided": p_ll,
        "brier_p_onesided": p_brier,
    }


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni step-down adjustment across a family of raw one-sided p-values.
    Returns {key: holm_adjusted_p} enforcing monotonicity and capping at 1.0."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)  # step-down monotonicity
        out[k] = running
    return out


def real_selection_contract(res: dict, ci: dict, *, pure_provenance_ok: bool) -> dict:
    """STEP 9 selection contract (development pre-proof screen). A prop is eligible ONLY when:
    pure-input provenance passes; dLL<0; dBrier<0; model_auc>=market_auc; ECE within margin of
    market; no catastrophic fold; AND the date-cluster bootstrap upper-95% CI of BOTH deltas < 0.
    Passing here is NOT certification — the prospective proof (STEP 10) is still required."""
    reasons = []
    if not pure_provenance_ok:
        reasons.append("pure_provenance_fail")
    if not (res["logloss_delta"] < 0):
        reasons.append(f"dLL>=0 ({res['logloss_delta']:+.5f})")
    if not (res["brier_delta"] < 0):
        reasons.append(f"dBrier>=0 ({res['brier_delta']:+.5f})")
    if not (res["model_auc"] >= res["market_auc"]):
        reasons.append(f"AUC<market ({res['model_auc']:.3f}<{res['market_auc']:.3f})")
    if not (res["model_ece"] <= res["market_ece"] + ECE_MARGIN):
        reasons.append(f"ECE>market+{ECE_MARGIN} ({res['model_ece']:.3f}>{res['market_ece']:.3f})")
    if not (np.isfinite(res["worst_fold_logloss_delta"])
            and res["worst_fold_logloss_delta"] <= WORST_FOLD_LL_MAX):
        reasons.append(f"catastrophic_fold ({res['worst_fold_logloss_delta']:+.4f})")
    if not ci["logloss_upper95_below_zero"]:
        reasons.append(f"LL CI upper>=0 {ci['logloss_delta_ci95']}")
    if not ci["brier_upper95_below_zero"]:
        reasons.append(f"Brier CI upper>=0 {ci['brier_delta_ci95']}")
    return {"selection_contract_pass": len(reasons) == 0, "fail_reasons": reasons}


# ---------------------------------------------------------------------------
# Join: fresh pure OOF  x  exact same-book quotes
# ---------------------------------------------------------------------------

def load_joined(oof_path: str, scored_path: str) -> tuple[pd.DataFrame, list[str]]:
    oof = pd.read_parquet(oof_path).rename(columns={"stat": "prop"})
    scored = pd.read_parquet(scored_path)
    missing = [c for c in ("active_pmf_json", "p_dnp") if c not in oof.columns]
    if missing:
        raise ValueError(
            f"pure OOF is missing repaired-lineage column(s) {missing}; regenerate with the "
            "wired build_oof_pmfs.py (active-PMF lineage). Refusing to fall back to the old path.")
    for df in (oof, scored):
        df["game_id"] = df["game_id"].astype(str)
        df["player_id"] = df["player_id"].astype(str)
    key = ["game_id", "player_id", "prop"]
    oof_keep = oof[key + ["active_pmf_json", "p_dnp", "actual_outcome"]].drop_duplicates(key)
    m = scored.merge(oof_keep, on=key, how="inner", suffixes=("", "_oof"))
    return m, key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    oof: str = typer.Option("data/oof/oof_player_stat_pmfs.parquet", "--oof",
                            help="Fresh PURE OOF long PMFs from the wired build_oof_pmfs.py."),
    scored: str = typer.Option(
        "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet",
        "--scored", help="Exact same-book deterministic quote table (one quote per obs)."),
    oof_manifest: str = typer.Option(
        "artifacts/audits/PURE_OOF_RUN_MANIFEST.json", "--oof-manifest",
        help="Provenance manifest written by build_oof_pmfs.py (proves the OOF is pure)."),
    out_dir: str = typer.Option("artifacts/pure_supremacy", "--out-dir"),
    min_train_dates: int = typer.Option(MIN_TRAIN_DATES, "--min-train-dates",
        help="Expanding-window CV min training dates (lower only for synthetic dry-runs)."),
    val_block_dates: int = typer.Option(VAL_BLOCK_DATES, "--val-block-dates",
        help="Expanding-window CV validation block size (dates)."),
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Pure-input provenance: the OOF run manifest must exist and attest pure_forecast.
    prov = {}
    pure_provenance_ok = False
    mp = Path(oof_manifest)
    if mp.exists():
        prov = json.loads(mp.read_text())
        def _zero(v):
            return v is not None and float(v) == 0.0
        pure_provenance_ok = (
            prov.get("information_contract") == "pure_forecast"
            and _zero(prov.get("market_probability_weight"))
            and _zero(prov.get("market_prior_lambda"))
            and not prov.get("clv_head_enabled", True)
            and not prov.get("forbidden_market_columns_present", ["?"])
        )

    m, key = load_joined(oof, scored)

    # Settle P(over) from the ACTIVE PMF (void-on-DNP, push-safe) and full-PMF CRPS.
    m["p_over_settled_active"] = [
        _settled_over_from_active(a, ln) for a, ln in zip(m["active_pmf_json"], m["line"])
    ]
    m["crps_active"] = [
        _crps_discrete(a, y) for a, y in zip(m["active_pmf_json"], m["actual"])
    ]
    # Drop rows where the settled probability is undefined (all mass on the integer push).
    n_before = len(m)
    m = m[m["p_over_settled_active"].notna()].reset_index(drop=True)
    n_undef = n_before - len(m)

    manifests = {}
    for prop in QUOTE_COVERED:
        pdf = m[m["prop"] == prop]
        folds = expanding_window_folds(pdf["game_date"].to_numpy(),
                                       min_train_dates=min_train_dates, val_block_dates=val_block_dates)
        manifests[prop] = {
            "folds": fold_manifest(folds, lambda ds: int(pdf["game_date"].isin(ds).sum())),
            "all_chronology_pass": all_chronology_pass(folds),
        }

    per_prop = {}
    csv_rows = []
    scored_row_frames = []
    pub = lambda r: {k: v for k, v in r.items() if not k.startswith("_")} if r else None
    for prop in DIRECT_PROPS:
        if prop not in QUOTE_COVERED:
            per_prop[prop] = {"status": "NO_EXACT_QUOTES",
                              "note": "market comparison blocked until same-book quotes collected"}
            continue
        pdf = m[m["prop"] == prop].sort_values("game_date").reset_index(drop=True)
        before = nested_eval(pdf, "P0_active_settled_identity",
                             min_train_dates=min_train_dates, val_block_dates=val_block_dates)
        after = nested_eval(pdf, "P1_active_settled_platt",
                            min_train_dates=min_train_dates, val_block_dates=val_block_dates)
        ci = _bootstrap_ci(after, pdf) if after else None
        contract = (real_selection_contract(after, ci, pure_provenance_ok=pure_provenance_ok)
                    if (after and ci) else None)
        # Pure recalibration ladder (AST repair): evaluate every monotone pure calibrator and
        # record the best-by-OOS-LL among those that are monotone. Reporting only — the CI-gated
        # contract still runs on the validated Platt candidate for continuity.
        cand_results = {}
        for cand in PURE_RECAL_CANDIDATES:
            cr = nested_eval(pdf, cand, min_train_dates=min_train_dates,
                             val_block_dates=val_block_dates)
            if cr is not None:
                cand_results[cand] = pub(cr)
        monotone_cands = {k: v for k, v in cand_results.items() if v.get("monotone")}
        best_cand = (min(monotone_cands, key=lambda k: monotone_cands[k]["model_logloss"])
                     if monotone_cands else None)
        per_prop[prop] = {
            "before_active_settled_identity": pub(before),
            "after_active_settled_platt": pub(after),
            "after_bootstrap_ci": ci,
            "real_selection_contract": contract,
            "pure_recalibration_candidates": cand_results,
            "best_pure_recalibration_candidate": best_cand,
            "final_probability_column": FINAL_COL,
            "lineage": "active_pmf -> push_safe_settled -> monotone_platt(model_vs_outcome)",
        }
        # Per-row scored lineage (owner item 2): persist the line-dependent settled + final
        # probabilities that cannot live on the line-free OOF PMF rows.
        if after is not None:
            msk = after["_mask"]
            fin = after["_oof_pred"][msk]
            sub = pdf[msk].reset_index(drop=True)
            scored_row_frames.append(pd.DataFrame({
                "game_id": sub["game_id"].astype(str), "player_id": sub["player_id"].astype(str),
                "prop": prop, "game_date": sub["game_date"].astype(str),
                "line": sub["line"].to_numpy(float), "p_dnp": sub["p_dnp"].to_numpy(float),
                "model_prob_over_settled_from_active_pmf":
                    sub["p_over_settled_active"].to_numpy(float),
                "model_prob_over_final": np.asarray(fin, float),
                "market_prob_over_no_vig": sub[MARKET_COL].to_numpy(float),
                "outcome_over": sub["outcome_over"].to_numpy(int),
            }))

    # Holm-Bonferroni across the frozen prop family, per metric (owner item 4).
    raw_p_ll = {p: per_prop[p]["after_bootstrap_ci"]["logloss_p_onesided"]
                for p in QUOTE_COVERED if per_prop[p].get("after_bootstrap_ci")}
    raw_p_brier = {p: per_prop[p]["after_bootstrap_ci"]["brier_p_onesided"]
                   for p in QUOTE_COVERED if per_prop[p].get("after_bootstrap_ci")}
    holm_ll = _holm(raw_p_ll)
    holm_brier = _holm(raw_p_brier)
    for prop in QUOTE_COVERED:
        ci = per_prop[prop].get("after_bootstrap_ci")
        if ci:
            per_prop[prop]["holm_adjusted_p_ll"] = holm_ll.get(prop)
            per_prop[prop]["holm_adjusted_p_brier"] = holm_brier.get(prop)
            per_prop[prop]["holm_family"] = sorted(raw_p_ll)
        before = per_prop[prop]["before_active_settled_identity"]
        after = per_prop[prop]["after_active_settled_platt"]
        for tag, r in [("BEFORE_active_settled", before), ("AFTER_active_settled_platt", after)]:
            if r:
                row = {"prop": prop, "stage": tag, **{k: r[k] for k in (
                    "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                    "model_brier", "market_brier", "brier_delta", "model_auc", "market_auc",
                    "model_ece", "market_ece", "model_crps_active_pmf",
                    "worst_fold_logloss_delta", "monotone")}}
                if tag == "AFTER_active_settled_platt" and ci:
                    row["raw_p_ll_onesided"] = ci["logloss_p_onesided"]
                    row["raw_p_brier_onesided"] = ci["brier_p_onesided"]
                    row["holm_adjusted_p_ll"] = holm_ll.get(prop)
                    row["holm_adjusted_p_brier"] = holm_brier.get(prop)
                csv_rows.append(row)

    passing = [p for p in QUOTE_COVERED
               if per_prop[p].get("real_selection_contract")
               and per_prop[p]["real_selection_contract"]["selection_contract_pass"]]

    report = {
        "version": "production-pure-oof-v1",
        "lineage": ("active_pmf -> settle_over_from_active_pmf(push-safe, void-on-DNP) -> "
                    "monotone pure Platt (model-vs-outcome, nested CV) -> model_prob_over_final; "
                    "p_dnp kept separate (never divided into the over/under probability)"),
        "cv": "grouped expanding-window nested rolling-origin (max(train)<min(val))",
        "pure_input_provenance_ok": bool(pure_provenance_ok),
        "pure_forecast_provenance": prov or "MISSING (OOF run manifest not found)",
        "rows_joined": int(len(m) + n_undef),
        "rows_undefined_settled_push_dropped": int(n_undef),
        "real_selection_contract": ("pure provenance passes AND dLL<0 AND dBrier<0 AND "
                                    "model_auc>=market_auc AND ECE<=market+0.03 AND no catastrophic "
                                    "fold AND bootstrap upper-95% CI of BOTH deltas < 0"),
        "note_not_certification": ("passing the selection contract is a development pre-proof "
                                   "screen; STEP 10 prospective proof is still required and NO "
                                   "promotion/edge gate is modified here"),
        "multiple_testing": ("one-sided paired date-cluster bootstrap p-value per prop for "
                             "dLL<0 and dBrier<0, Holm-Bonferroni across the quote-covered prop "
                             "family (reporting only; the CI-based selection contract still gates)"),
        "holm_family": sorted(raw_p_ll),
        "fold_manifest": manifests,
        "per_prop": per_prop,
        "props_passing_real_selection_contract": passing,
    }
    (outp / "PRODUCTION_PURE_OOF_METRICS.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    pd.DataFrame(csv_rows).to_csv(outp / "PRODUCTION_PURE_OOF_METRICS.csv", index=False)
    if scored_row_frames:
        pd.concat(scored_row_frames, ignore_index=True).to_parquet(
            outp / "PRODUCTION_PURE_OOF_SCORED_ROWS.parquet", index=False)

    print("\n=== PRODUCTION PURE OOF (active-PMF lineage) vs exact no-vig market ===")
    print(f"pure_input_provenance_ok={pure_provenance_ok}  rows={len(m)}  "
          f"undefined_push_dropped={n_undef}")
    print(f"{'prop':5s} {'stage':22s} {'n':>4s} {'mLL':>7s} {'kLL':>7s} {'dLL':>8s} {'dBr':>8s} "
          f"{'mAUC':>6s} {'kAUC':>6s} {'mECE':>6s} {'CRPS':>6s}")
    for prop in QUOTE_COVERED:
        for tag, k in [("BEFORE", "before_active_settled_identity"),
                       ("AFTER_platt", "after_active_settled_platt")]:
            r = per_prop[prop][k]
            if r:
                print(f"{prop:5s} {tag:22s} {r['n']:>4d} {r['model_logloss']:>7.4f} "
                      f"{r['market_logloss']:>7.4f} {r['logloss_delta']:>+8.4f} "
                      f"{r['brier_delta']:>+8.4f} {r['model_auc']:>6.3f} {r['market_auc']:>6.3f} "
                      f"{r['model_ece']:>6.3f} {r['model_crps_active_pmf']:>6.3f}")
        c = per_prop[prop]["real_selection_contract"]
        if c:
            print(f"      -> REAL selection contract: {'PASS' if c['selection_contract_pass'] else 'FAIL'} "
                  f"{c['fail_reasons']}")
        if per_prop[prop].get("holm_adjusted_p_ll") is not None:
            print(f"      -> Holm-adjusted p: LL={per_prop[prop]['holm_adjusted_p_ll']:.4f} "
                  f"Brier={per_prop[prop]['holm_adjusted_p_brier']:.4f}")
    print(f"\nPROPS PASSING REAL SELECTION CONTRACT: {passing or 'none'}")
    print("stl/blk/turnover: NO_EXACT_QUOTES (market comparison blocked on odds collection)")
    print(f"\nWrote {outp/'PRODUCTION_PURE_OOF_METRICS.json'} and .csv")


if __name__ == "__main__":
    app()
