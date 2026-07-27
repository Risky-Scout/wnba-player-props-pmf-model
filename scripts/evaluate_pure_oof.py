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
from wnba_props_model.models import pure_selection as psel  # noqa: E402
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf  # noqa: E402
from wnba_props_model.models.structural_pmf import (  # noqa: E402
    SUPPORTED_PROPS as SUPPORTED_STRUCTURAL_PROPS,
)

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


def _full_pmf_logscore(pmf_json: str, actual: float) -> float:
    """Full-PMF log score: -log P(actual) under the (active) count PMF (lower is better). This is
    the distributional analogue of the binary log loss and certifies the whole forecast, not just
    the over/under threshold."""
    pmf = normalize_pmf(json_to_pmf(pmf_json))
    a = int(round(float(actual)))
    p = pmf[a] if 0 <= a < pmf.size else 0.0
    return float(-np.log(max(float(p), EPS)))


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


def _fit_calibrator(cid, tr):
    """Return (predict_fn, monotone_ok) for a candidate id (owner item 3 family). Thin wrapper
    over the unified pure-selection engine; ``tr`` needs the candidate's source column
    (``p_over_settled_active`` for direct P*, ``p_over_settled_structural`` for S*) + outcome."""
    fc = psel.fit_candidate(cid, tr)
    if fc is None:
        raise ValueError(f"candidate {cid} unavailable on this training slice")
    return fc.predict, fc.monotone


# Owner item 3 unified candidate family (no market probability in ANY candidate):
#   P0-P3 direct active-PMF (identity/Platt/Beta/isotonic), S1-S3 structural active-PMF
#   (identity/Platt/Beta), E1 pure nonnegative simplex ensemble of eligible direct+structural.
CANDIDATE_FAMILY = list(psel.CANDIDATE_SPECS.keys())
# Back-compat alias: the direct recalibration subset (used by AST-repair reporting/tests).
PURE_RECAL_CANDIDATES = list(psel.DIRECT_CANDIDATES)


def _calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Logistic recalibration curve: fit outcome ~ logit(p). Perfect calibration -> (0, 1)."""
    try:
        lr = LogisticRegression(C=1e12, solver="lbfgs").fit(_logit(p).reshape(-1, 1),
                                                            np.asarray(y, int))
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except (ValueError, Exception):  # noqa: BLE001
        return float("nan"), float("nan")


def _score_stream(pdf: pd.DataFrame, pred_full: np.ndarray, mask: np.ndarray,
                  per_fold_deltas: list[float] | None = None) -> dict:
    """Compute the full metric block (ITEM 5 step 2) on a selected OOF row stream.

    Metrics: LL, Brier, AUC, ECE, calibration intercept+slope, model-only CRPS and full-PMF log
    score, plus worst-fold LL delta. Returns a dict shaped like the legacy nested_eval result so
    the existing bootstrap/CSV/report code consumes it unchanged."""
    y = pdf["outcome_over"].to_numpy(int)[mask]
    pk = pdf[MARKET_COL].to_numpy(float)[mask]
    pr = pred_full[mask]
    crps = float(np.mean(pdf["crps_active"].to_numpy(float)[mask])) if "crps_active" in pdf else float("nan")
    fpls = (float(np.mean(pdf["full_pmf_logscore"].to_numpy(float)[mask]))
            if "full_pmf_logscore" in pdf else float("nan"))
    intc, slope = _calibration_intercept_slope(y, pr)
    worst = float(max(per_fold_deltas)) if per_fold_deltas else float("nan")
    return {
        "n": int(mask.sum()), "n_folds": len(per_fold_deltas) if per_fold_deltas else None,
        "model_logloss": _ll(y, pr), "market_logloss": _ll(y, pk),
        "logloss_delta": _ll(y, pr) - _ll(y, pk),
        "model_brier": _brier(y, pr), "market_brier": _brier(y, pk),
        "brier_delta": _brier(y, pr) - _brier(y, pk),
        "model_auc": _auc(y, pr), "market_auc": _auc(y, pk),
        "model_ece": _ece(y, pr), "market_ece": _ece(y, pk),
        "calibration_intercept": intc, "calibration_slope": slope,
        "model_crps_active_pmf": crps, "model_full_pmf_logscore": fpls,
        "worst_fold_logloss_delta": worst,
        "_oof_pred": pred_full, "_mask": mask,
    }


def nested_eval(pdf: pd.DataFrame, cid: str, *, min_train_dates: int = MIN_TRAIN_DATES,
                val_block_dates: int = VAL_BLOCK_DATES) -> dict | None:
    """Fixed-candidate expanding-window OOS evaluation (per-candidate family reporting). Each outer
    fold fits candidate ``cid`` on training dates only and predicts held-out dates; NO inner
    selection (that is nested_rolling_origin_select). Returns aggregated OOS metrics."""
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
        fc = psel.fit_candidate(cid, tr)
        if fc is None or not fc.eligible:
            continue
        mono = mono and fc.monotone
        pv = fc.predict(pdf.iloc[va_idx])
        pred[va_idx] = pv
        yv = pdf["outcome_over"].to_numpy(int)[va_idx]
        pkv = pdf[MARKET_COL].to_numpy(float)[va_idx]
        per_fold.append(_ll(yv, pv) - _ll(yv, pkv))
    m = np.isfinite(pred)
    if m.sum() == 0:
        return None
    res = _score_stream(pdf, pred, m, per_fold)
    res.update({"candidate": cid, "monotone": bool(mono)})
    return res


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


HOLM_ALPHA = 0.05
MIN_OBS = 100
MIN_DATES = 10


def real_selection_contract(res: dict, ci: dict, *, pure_provenance_ok: bool,
                            holm_p_ll: float | None = None, holm_p_brier: float | None = None,
                            market_prob_weight: float = 0.0, n_dates: int | None = None) -> dict:
    """ITEM 5 final PASS/FAIL, applied ONLY after Holm-adjusted p-values exist.

    A prop PASSES iff ALL hold: market_prob_weight==0; pure provenance; required obs+independent
    dates; dLL<0; dBrier<0; upper95%CI(dLL)<0; upper95%CI(dBrier)<0; Holm p_LL<=0.05; Holm
    p_Brier<=0.05; frozen AUC criterion (model>=market); ECE within margin; calibration slope
    finite & positive; no catastrophic fold. A negative point estimate ALONE is never a pass.
    Passing is a development pre-proof screen; STEP 10 prospective proof is still required."""
    reasons = []
    if float(market_prob_weight or 0.0) != 0.0:
        reasons.append(f"market_prob_weight!=0 ({market_prob_weight})")
    if not pure_provenance_ok:
        reasons.append("pure_provenance_fail")
    if n_dates is not None and n_dates < MIN_DATES:
        reasons.append(f"insufficient_independent_dates ({n_dates}<{MIN_DATES})")
    if res.get("n", 0) < MIN_OBS:
        reasons.append(f"insufficient_obs ({res.get('n')}<{MIN_OBS})")
    if not (res["logloss_delta"] < 0):
        reasons.append(f"dLL>=0 ({res['logloss_delta']:+.5f})")
    if not (res["brier_delta"] < 0):
        reasons.append(f"dBrier>=0 ({res['brier_delta']:+.5f})")
    if not (res["model_auc"] >= res["market_auc"]):
        reasons.append(f"AUC<market ({res['model_auc']:.3f}<{res['market_auc']:.3f})")
    if not (res["model_ece"] <= res["market_ece"] + ECE_MARGIN):
        reasons.append(f"ECE>market+{ECE_MARGIN} ({res['model_ece']:.3f}>{res['market_ece']:.3f})")
    slope = res.get("calibration_slope")
    if not (slope is not None and np.isfinite(slope) and slope > 0):
        reasons.append(f"calibration_slope_not_positive ({slope})")
    if not (np.isfinite(res["worst_fold_logloss_delta"])
            and res["worst_fold_logloss_delta"] <= WORST_FOLD_LL_MAX):
        reasons.append(f"catastrophic_fold ({res['worst_fold_logloss_delta']:+.4f})")
    if not ci["logloss_upper95_below_zero"]:
        reasons.append(f"LL CI upper>=0 {ci['logloss_delta_ci95']}")
    if not ci["brier_upper95_below_zero"]:
        reasons.append(f"Brier CI upper>=0 {ci['brier_delta_ci95']}")
    # ITEM 5: NO verdict may pass before Holm-adjusted p-values are populated.
    if holm_p_ll is None or holm_p_brier is None:
        reasons.append("holm_pvalues_not_populated")
    else:
        if not (holm_p_ll <= HOLM_ALPHA):
            reasons.append(f"holm_p_ll>{HOLM_ALPHA} ({holm_p_ll:.4f})")
        if not (holm_p_brier <= HOLM_ALPHA):
            reasons.append(f"holm_p_brier>{HOLM_ALPHA} ({holm_p_brier:.4f})")
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
    keep_cols = key + ["active_pmf_json", "p_dnp", "actual_outcome"]
    # Optional structural repair candidate PMF (owner item 7): carried when the OOF was built
    # with structural_repair enabled; absent OOFs simply have no structural candidate.
    for opt in ("structural_active_pmf_json", "structural_candidate_id"):
        if opt in oof.columns:
            keep_cols.append(opt)
    oof_keep = oof[keep_cols].drop_duplicates(key)
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
    m["full_pmf_logscore"] = [
        _full_pmf_logscore(a, y) for a, y in zip(m["active_pmf_json"], m["actual"])
    ]
    # Structural repair candidate: settle P(over) + CRPS from the ALTERNATIVE structural active
    # PMF (opportunity×conversion). Rows without a structural PMF carry NaN and are skipped for
    # that candidate only.
    has_structural = "structural_active_pmf_json" in m.columns
    if has_structural:
        m["p_over_settled_structural"] = [
            (_settled_over_from_active(a, ln) if isinstance(a, str) and a else None)
            for a, ln in zip(m["structural_active_pmf_json"], m["line"])]
        m["crps_structural"] = [
            (_crps_discrete(a, y) if isinstance(a, str) and a else float("nan"))
            for a, y in zip(m["structural_active_pmf_json"], m["actual"])]
    # Drop rows where the (direct) settled probability is undefined (all mass on the integer push).
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
    nested_manifest: dict[str, list] = {}
    selected_stream_frames = []
    pub = lambda r: {k: v for k, v in r.items() if not k.startswith("_")} if r else None
    for prop in DIRECT_PROPS:
        if prop not in QUOTE_COVERED:
            per_prop[prop] = {"status": "NO_EXACT_QUOTES",
                              "note": "market comparison blocked until same-book quotes collected"}
            continue
        pdf = m[m["prop"] == prop].sort_values("game_date").reset_index(drop=True)

        # --- ITEM 4: nested rolling-origin selection (inner-chosen candidate per outer fold) ---
        sel = psel.nested_rolling_origin_select(
            pdf, prop, min_train_dates=min_train_dates, val_block_dates=val_block_dates)
        if sel is None:
            per_prop[prop] = {"status": "INSUFFICIENT_FOLDS",
                              "note": "no outer rolling-origin fold could be formed"}
            continue
        nested_manifest[prop] = sel.fold_manifest

        # Per-outer-fold LL delta (vs market) for worst-fold, using the SELECTED predictions.
        y_all = pdf["outcome_over"].to_numpy(int)
        pk_all = pdf[MARKET_COL].to_numpy(float)
        per_fold_deltas = []
        for fm in sel.fold_manifest:
            fmask = sel.selected_mask & (sel.selected_fold_per_row == fm["outer_fold_id"])
            if fmask.any():
                per_fold_deltas.append(
                    _ll(y_all[fmask], sel.selected_pred[fmask]) - _ll(y_all[fmask], pk_all[fmask]))

        # --- ITEM 5 step 2: metrics on the SELECTED outer-OOF row stream ---
        res = _score_stream(pdf, sel.selected_pred, sel.selected_mask, per_fold_deltas)
        # --- ITEM 5 step 3: paired date-cluster CIs + raw one-sided p on the selected stream ---
        ci = _bootstrap_ci(res, pdf)
        n_dates = int(pd.Series(pdf["game_date"].to_numpy()[sel.selected_mask]).nunique())

        # Candidate-family table (reporting): fixed-candidate OOS metrics for every family member.
        cand_results = {}
        for cid in CANDIDATE_FAMILY:
            cr = nested_eval(pdf, cid, min_train_dates=min_train_dates,
                             val_block_dates=val_block_dates)
            if cr is not None:
                cand_results[cid] = pub(cr)
        monotone_direct = {k: v for k, v in cand_results.items()
                           if k in PURE_RECAL_CANDIDATES and v.get("monotone")}
        best_cand = (min(monotone_direct, key=lambda k: monotone_direct[k]["model_logloss"])
                     if monotone_direct else None)
        # Distribution of which candidate the nested selector chose across outer folds.
        chosen_ids = [c for c in sel.selected_candidate_per_row[sel.selected_mask] if c is not None]
        sel_dist = {c: int(chosen_ids.count(c)) for c in sorted(set(chosen_ids))}

        # Structural repair candidate (owner item 7) — reported via the structural source column.
        structural = None
        if has_structural and prop in SUPPORTED_STRUCTURAL_PROPS \
                and "p_over_settled_structural" in pdf.columns:
            spdf = pdf[pdf["p_over_settled_structural"].notna()].copy()
            if len(spdf) >= 30:
                spdf["p_over_settled_active"] = spdf["p_over_settled_structural"].to_numpy(float)
                spdf["crps_active"] = spdf["crps_structural"].to_numpy(float)
                s_before = nested_eval(spdf, "P0", min_train_dates=min_train_dates,
                                       val_block_dates=val_block_dates)
                s_after = nested_eval(spdf, "P1", min_train_dates=min_train_dates,
                                      val_block_dates=val_block_dates)
                s_ci = _bootstrap_ci(s_after, spdf) if s_after else None
                structural = {
                    "candidate_id": (spdf["structural_candidate_id"].dropna().iloc[0]
                                     if "structural_candidate_id" in spdf.columns
                                     and spdf["structural_candidate_id"].notna().any() else None),
                    "n": int(len(spdf)),
                    "before_structural_settled_identity": pub(s_before),
                    "after_structural_settled_platt": pub(s_after),
                    "after_bootstrap_ci": s_ci,
                    "advances": bool(s_after and s_after["logloss_delta"] < 0
                                     and s_after["brier_delta"] < 0
                                     and s_after["model_auc"] >= s_after["market_auc"]),
                    "lineage": ("structural_active_pmf(opportunity×conversion) -> push_safe_settled "
                                "-> monotone_platt(model_vs_outcome)"),
                }

        per_prop[prop] = {
            "selected_outer_oof": pub(res),
            "selected_candidate_distribution": sel_dist,
            "candidates_considered": list(sel.candidates_considered),
            "n_independent_dates": n_dates,
            "after_bootstrap_ci": ci,
            "candidate_family": cand_results,
            "structural_repair_candidate": structural,
            # Back-compat reporting keys (direct identity / Platt subset).
            "before_active_settled_identity": cand_results.get("P0"),
            "after_active_settled_platt": cand_results.get("P1"),
            "pure_recalibration_candidates": {k: cand_results[k] for k in PURE_RECAL_CANDIDATES
                                              if k in cand_results},
            "best_pure_recalibration_candidate": best_cand,
            "real_selection_contract": None,  # ITEM 5: filled AFTER Holm (step 5)
            "final_probability_column": FINAL_COL,
            "lineage": ("active_pmf -> push_safe_settled -> nested-selected pure candidate "
                        "(P0-P3/S1-S3/E1, inner-CV chosen) -> model_prob_over_final"),
            "_res": res,  # kept for contract computation; stripped before serialization
        }

        # Per-row scored lineage (owner item 2): the CHAMPION (nested-selected) prediction.
        msk = sel.selected_mask
        sub = pdf[msk].reset_index(drop=True)
        fin = sel.selected_pred[msk]
        scored_row_frames.append(pd.DataFrame({
            "game_id": sub["game_id"].astype(str), "player_id": sub["player_id"].astype(str),
            "prop": prop, "game_date": sub["game_date"].astype(str),
            "line": sub["line"].to_numpy(float), "p_dnp": sub["p_dnp"].to_numpy(float),
            "model_prob_over_settled_from_active_pmf": sub["p_over_settled_active"].to_numpy(float),
            "model_prob_over_final": np.asarray(fin, float),
            "market_prob_over_no_vig": sub[MARKET_COL].to_numpy(float),
            "outcome_over": sub["outcome_over"].to_numpy(int),
        }))
        # NESTED_SELECTED_OOF_ROWS: selected outer-OOF stream + which candidate produced each row.
        selected_stream_frames.append(pd.DataFrame({
            "game_id": sub["game_id"].astype(str), "player_id": sub["player_id"].astype(str),
            "prop": prop, "game_date": sub["game_date"].astype(str),
            "outer_fold_id": sel.selected_fold_per_row[msk],
            "selected_candidate": [c for c in sel.selected_candidate_per_row[msk]],
            "line": sub["line"].to_numpy(float),
            "model_prob_over_final": np.asarray(fin, float),
            "market_prob_over_no_vig": sub[MARKET_COL].to_numpy(float),
            "outcome_over": sub["outcome_over"].to_numpy(int),
        }))

    # ---- ITEM 5 step 4: Holm-Bonferroni across the frozen quote-covered prop family ----
    raw_p_ll = {p: per_prop[p]["after_bootstrap_ci"]["logloss_p_onesided"]
                for p in QUOTE_COVERED if per_prop[p].get("after_bootstrap_ci")}
    raw_p_brier = {p: per_prop[p]["after_bootstrap_ci"]["brier_p_onesided"]
                   for p in QUOTE_COVERED if per_prop[p].get("after_bootstrap_ci")}
    holm_ll = _holm(raw_p_ll)
    holm_brier = _holm(raw_p_brier)

    # ---- ITEM 5 step 5: THEN (and only then) the PASS/FAIL verdict per prop ----
    for prop in QUOTE_COVERED:
        pp = per_prop[prop]
        ci = pp.get("after_bootstrap_ci")
        if not ci:
            continue
        pp["holm_adjusted_p_ll"] = holm_ll.get(prop)
        pp["holm_adjusted_p_brier"] = holm_brier.get(prop)
        pp["holm_family"] = sorted(raw_p_ll)
        pp["raw_p_ll_onesided"] = ci["logloss_p_onesided"]
        pp["raw_p_brier_onesided"] = ci["brier_p_onesided"]
        res = pp["_res"]
        pp["real_selection_contract"] = real_selection_contract(
            res, ci, pure_provenance_ok=pure_provenance_ok,
            holm_p_ll=holm_ll.get(prop), holm_p_brier=holm_brier.get(prop),
            market_prob_weight=0.0, n_dates=pp.get("n_independent_dates"))

        # CSV: champion selected stream + every candidate-family member.
        champ = pub(res)
        row = {"prop": prop, "stage": "SELECTED_outer_oof",
               "selected_candidate_distribution": json.dumps(pp["selected_candidate_distribution"]),
               "raw_p_ll_onesided": ci["logloss_p_onesided"],
               "raw_p_brier_onesided": ci["brier_p_onesided"],
               "holm_adjusted_p_ll": holm_ll.get(prop),
               "holm_adjusted_p_brier": holm_brier.get(prop),
               "selection_contract_pass": pp["real_selection_contract"]["selection_contract_pass"],
               **{k: champ.get(k) for k in (
                   "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                   "model_brier", "market_brier", "brier_delta", "model_auc", "market_auc",
                   "model_ece", "market_ece", "calibration_intercept", "calibration_slope",
                   "model_crps_active_pmf", "model_full_pmf_logscore",
                   "worst_fold_logloss_delta")}}
        csv_rows.append(row)
        for cid, cr in pp["candidate_family"].items():
            csv_rows.append({"prop": prop, "stage": f"CANDIDATE_{cid}",
                             **{k: cr.get(k) for k in (
                                 "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                                 "model_brier", "market_brier", "brier_delta", "model_auc",
                                 "market_auc", "model_ece", "market_ece", "model_crps_active_pmf",
                                 "model_full_pmf_logscore", "worst_fold_logloss_delta",
                                 "monotone")}})
        sc = pp.get("structural_repair_candidate")
        if sc and sc.get("after_structural_settled_platt"):
            r = sc["after_structural_settled_platt"]
            csv_rows.append({"prop": prop, "stage": "STRUCTURAL_repair_platt",
                             "candidate_id": sc.get("candidate_id"), "advances": sc.get("advances"),
                             **{k: r.get(k) for k in (
                                 "n", "n_folds", "model_logloss", "market_logloss", "logloss_delta",
                                 "model_brier", "market_brier", "brier_delta", "model_auc",
                                 "market_auc", "model_ece", "market_ece", "model_crps_active_pmf",
                                 "worst_fold_logloss_delta", "monotone")}})

    # Strip internal-only keys before serialization.
    for prop in QUOTE_COVERED:
        per_prop[prop].pop("_res", None)

    passing = [p for p in QUOTE_COVERED
               if per_prop[p].get("real_selection_contract")
               and per_prop[p]["real_selection_contract"]["selection_contract_pass"]]

    report = {
        "version": "production-pure-oof-v2-nested-selection",
        "lineage": ("active_pmf -> settle_over_from_active_pmf(push-safe, void-on-DNP) -> "
                    "nested rolling-origin candidate selection (P0-P3/S1-S3/E1, inner-CV chosen, "
                    "no market input) -> model_prob_over_final; p_dnp kept separate"),
        "candidate_family": {cid: f"{src}:{meth}" for cid, (src, meth)
                             in psel.CANDIDATE_SPECS.items()},
        "cv": ("nested rolling-origin: OUTER expanding folds (max(outer_train)<min(outer_val)); "
               "INNER expanding folds from outer-training dates ONLY choose the candidate; refit "
               "on all outer-training dates; score outer-validation once"),
        "contract_order": ("(1) selected outer-OOF row stream -> (2) LL/Brier/AUC/ECE/calibration "
                           "intercept+slope/CRPS/full-PMF log score -> (3) paired date-cluster CIs "
                           "+ raw one-sided p -> (4) Holm across frozen family -> (5) PASS/FAIL"),
        "pure_input_provenance_ok": bool(pure_provenance_ok),
        "pure_forecast_provenance": prov or "MISSING (OOF run manifest not found)",
        "rows_joined": int(len(m) + n_undef),
        "rows_undefined_settled_push_dropped": int(n_undef),
        "real_selection_contract": ("market_prob_weight==0 AND pure provenance AND required "
                                    "obs+independent dates AND dLL<0 AND dBrier<0 AND "
                                    "upper95%CI(dLL)<0 AND upper95%CI(dBrier)<0 AND Holm p_LL<=0.05 "
                                    "AND Holm p_Brier<=0.05 AND model_auc>=market_auc AND "
                                    "ECE<=market+0.03 AND calibration_slope>0 AND no catastrophic "
                                    "fold. No verdict before Holm; a negative point estimate alone "
                                    "is never a pass."),
        "note_not_certification": ("passing the selection contract is a development pre-proof "
                                   "screen; STEP 10 prospective proof is still required and NO "
                                   "promotion/edge gate is modified here"),
        "structural_repair": ("pure opportunity×conversion candidates (pts/reb/fg3m) are S1-S3 in "
                              "the unified family; reported separately for continuity; zero market"),
        "multiple_testing": ("one-sided paired date-cluster bootstrap p-value per prop for dLL<0 "
                             "and dBrier<0, Holm-Bonferroni across the quote-covered prop family; "
                             "Holm p-values GATE the verdict (owner item 5)"),
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
    # ITEM 4 artifacts: nested selection fold manifest + selected outer-OOF row stream.
    (outp / "NESTED_SELECTION_FOLD_MANIFEST.json").write_text(json.dumps({
        "cv": report["cv"], "candidate_family": report["candidate_family"],
        "per_prop_folds": nested_manifest,
    }, indent=2, default=str) + "\n")
    if selected_stream_frames:
        pd.concat(selected_stream_frames, ignore_index=True).to_parquet(
            outp / "NESTED_SELECTED_OOF_ROWS.parquet", index=False)

    print("\n=== PRODUCTION PURE OOF (nested-selected pure candidate) vs exact no-vig market ===")
    print(f"pure_input_provenance_ok={pure_provenance_ok}  rows={len(m)}  "
          f"undefined_push_dropped={n_undef}")
    print(f"{'prop':5s} {'n':>5s} {'mLL':>7s} {'kLL':>7s} {'dLL':>8s} {'dBr':>8s} "
          f"{'mAUC':>6s} {'kAUC':>6s} {'slope':>6s} {'CRPS':>6s} {'fPMFls':>7s}")
    for prop in QUOTE_COVERED:
        pp = per_prop[prop]
        r = pp.get("selected_outer_oof")
        if r:
            print(f"{prop:5s} {r['n']:>5d} {r['model_logloss']:>7.4f} {r['market_logloss']:>7.4f} "
                  f"{r['logloss_delta']:>+8.4f} {r['brier_delta']:>+8.4f} {r['model_auc']:>6.3f} "
                  f"{r['market_auc']:>6.3f} {r['calibration_slope']:>6.2f} "
                  f"{r['model_crps_active_pmf']:>6.3f} {r['model_full_pmf_logscore']:>7.3f}")
            print(f"      selected: {pp['selected_candidate_distribution']}")
        c = pp.get("real_selection_contract")
        if c:
            print(f"      -> REAL selection contract: {'PASS' if c['selection_contract_pass'] else 'FAIL'} "
                  f"{c['fail_reasons']}")
        if pp.get("holm_adjusted_p_ll") is not None:
            print(f"      -> Holm-adjusted p: LL={pp['holm_adjusted_p_ll']:.4f} "
                  f"Brier={pp['holm_adjusted_p_brier']:.4f}")
        sc = pp.get("structural_repair_candidate")
        if sc and sc.get("after_structural_settled_platt"):
            sr = sc["after_structural_settled_platt"]
            print(f"      -> STRUCTURAL {sc.get('candidate_id')}: dLL={sr['logloss_delta']:+.4f} "
                  f"dBr={sr['brier_delta']:+.4f} AUC={sr['model_auc']:.3f} "
                  f"advances={sc.get('advances')}")
    print(f"\nPROPS PASSING REAL SELECTION CONTRACT: {passing or 'none'}")
    print("stl/blk/turnover: NO_EXACT_QUOTES (market comparison blocked on odds collection)")
    print(f"\nWrote {outp/'PRODUCTION_PURE_OOF_METRICS.json'}, .csv, "
          f"NESTED_SELECTION_FOLD_MANIFEST.json, NESTED_SELECTED_OOF_ROWS.parquet")


if __name__ == "__main__":
    app()
