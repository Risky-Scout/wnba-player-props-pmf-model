"""Owner ITEM 3 + ITEM 4 — unified pure candidate-selection engine (nested rolling-origin).

A single, market-free candidate family is evaluated PER PROP and chosen with a proper nested
rolling-origin protocol:

  Candidate family (no market probability enters ANY candidate):
    P0  direct   active-PMF settled identity
    P1  direct   + constrained monotone Platt
    P2  direct   + constrained monotone Beta (Kull et al.)
    P3  direct   + isotonic (only when support passes)
    S1  struct   structural active-PMF settled identity
    S2  struct   + constrained monotone Platt
    S3  struct   + constrained monotone Beta
    E1  ensemble pure NONNEGATIVE (simplex) blend of the eligible direct+structural candidates

  Nested rolling-origin (ITEM 4):
    * OUTER expanding-window folds give the unbiased performance stream (max(outer_train_date) <
      min(outer_val_date));
    * INNER expanding-window folds are built from the OUTER-TRAINING dates ONLY and choose the
      candidate (family/calibration/ensemble weights) by minimizing inner-validation log loss —
      WITHOUT ever reading outer-validation outcomes;
    * the chosen candidate is REFIT on ALL outer-training dates and scored on the untouched outer
      validation block exactly once;
    * the per-outer-fold selected candidate id + date/selection hashes are persisted.

Everything (encoders/imputers/priors/shrinkage/dispersion happen upstream in the PMF build; here
calibration + ensemble weights) is fit on training rows only, so no future date leaks into the
transforms or the candidate choice.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wnba_props_model.evaluation.rolling_origin import expanding_window_folds

EPS = 1e-6
DIRECT_COL = "p_over_settled_active"
STRUCT_COL = "p_over_settled_structural"
OUTCOME_COL = "outcome_over"

# Isotonic "support passes" thresholds (ITEM 3: P3 only when support passes).
ISO_MIN_ROWS = 200
ISO_MIN_UNIQUE = 20

# Candidate registry -> (source, method). No market source exists.
CANDIDATE_SPECS: dict[str, tuple[str, str]] = {
    "P0": ("direct", "identity"),
    "P1": ("direct", "platt"),
    "P2": ("direct", "beta"),
    "P3": ("direct", "isotonic"),
    "S1": ("structural", "identity"),
    "S2": ("structural", "platt"),
    "S3": ("structural", "beta"),
    "E1": ("ensemble", "nonneg_simplex"),
}
DIRECT_CANDIDATES = ("P0", "P1", "P2", "P3")
STRUCT_CANDIDATES = ("S1", "S2", "S3")
ENSEMBLE_CANDIDATE = "E1"
# Ensemble base candidates (one calibrated learner per source; combined on the simplex).
ENSEMBLE_BASES = ("P1", "S2")


def _clip(x) -> np.ndarray:
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def _ll(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def _source_col(source: str) -> str:
    return DIRECT_COL if source == "direct" else STRUCT_COL


# ---------------------------------------------------------------------------
# Fitted calibrators / candidates
# ---------------------------------------------------------------------------

@dataclass
class FittedCandidate:
    cid: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    monotone: bool
    eligible: bool
    detail: dict = field(default_factory=dict)


def _fit_single(cid: str, tr: pd.DataFrame) -> FittedCandidate | None:
    """Fit one non-ensemble candidate on a training slice. Returns None if its source column is
    absent/all-NaN (candidate simply unavailable, never a market fallback)."""
    source, method = CANDIDATE_SPECS[cid]
    col = _source_col(source)
    if col not in tr.columns:
        return None
    x = pd.to_numeric(tr[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(x)
    if ok.sum() < 30:
        return None
    xt = _clip(x[ok])
    y = tr[OUTCOME_COL].to_numpy(int)[ok]
    if len(np.unique(y)) < 2:
        return None

    if method == "identity":
        return FittedCandidate(cid, lambda d: _clip(pd.to_numeric(d[col], errors="coerce")
                                                     .to_numpy(float)), True, True)
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(xt).reshape(-1, 1), y)
        slope = float(lr.coef_[0][0])

        def _p(d):
            b = _logit(pd.to_numeric(d[col], errors="coerce").to_numpy(float)).reshape(-1, 1)
            return _clip(lr.predict_proba(b)[:, 1])
        return FittedCandidate(cid, _p, slope >= 0, True, {"slope": slope})
    if method == "beta":
        feats = np.column_stack([np.log(xt), np.log(1 - xt)])
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(feats, y)
        a, b = float(lr.coef_[0][0]), float(lr.coef_[0][1])

        def _pb(d):
            q = _clip(pd.to_numeric(d[col], errors="coerce").to_numpy(float))
            return _clip(lr.predict_proba(np.column_stack([np.log(q), np.log(1 - q)]))[:, 1])
        return FittedCandidate(cid, _pb, (a >= 0 and b <= 0), True, {"a": a, "b": b})
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        eligible = len(xt) >= ISO_MIN_ROWS and len(np.unique(xt)) >= ISO_MIN_UNIQUE
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(xt, y)

        def _pi(d):
            return _clip(iso.predict(pd.to_numeric(d[col], errors="coerce").to_numpy(float)))
        return FittedCandidate(cid, _pi, True, eligible)
    raise ValueError(cid)


def _nnls_simplex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonnegative simplex weights minimizing training log loss of a convex blend of the columns
    of ``P`` (each column a base candidate's train prediction). Pure: only training data used."""
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    try:
        from scipy.optimize import minimize  # noqa: PLC0415

        def _obj(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            w = w / s if s > 0 else np.ones(k) / k
            return _ll(y, _clip(P @ w))
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, 1.0)] * k
        res = minimize(_obj, np.ones(k) / k, method="SLSQP", bounds=bounds, constraints=cons)
        w = np.clip(res.x, 0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    except Exception:  # noqa: BLE001 — fall back to equal weights, still pure & nonnegative
        return np.ones(k) / k


def _fit_ensemble(tr: pd.DataFrame, base_ids: tuple[str, ...]) -> FittedCandidate | None:
    """Fit the E1 pure nonnegative simplex ensemble over the eligible+monotone base candidates."""
    bases: list[FittedCandidate] = []
    for bid in base_ids:
        fc = _fit_single(bid, tr)
        if fc is not None and fc.eligible and fc.monotone:
            bases.append(fc)
    if len(bases) < 2:
        return None
    P = np.column_stack([fc.predict(tr) for fc in bases])
    w = _nnls_simplex_weights(P, tr[OUTCOME_COL].to_numpy(int))

    def _pe(d):
        Pd = np.column_stack([fc.predict(d) for fc in bases])
        return _clip(Pd @ w)
    return FittedCandidate(
        "E1", _pe, all(fc.monotone for fc in bases), True,
        {"bases": [fc.cid for fc in bases], "weights": [float(x) for x in w]})


def fit_candidate(cid: str, tr: pd.DataFrame) -> FittedCandidate | None:
    if cid == ENSEMBLE_CANDIDATE:
        return _fit_ensemble(tr, ENSEMBLE_BASES)
    return _fit_single(cid, tr)


def available_candidates(pdf: pd.DataFrame) -> list[str]:
    """Candidate ids whose source column is present in the frame."""
    cands = list(DIRECT_CANDIDATES)
    if STRUCT_COL in pdf.columns and pd.to_numeric(pdf[STRUCT_COL], errors="coerce").notna().any():
        cands += list(STRUCT_CANDIDATES) + [ENSEMBLE_CANDIDATE]
    return cands


# ---------------------------------------------------------------------------
# Nested rolling-origin selection (ITEM 4)
# ---------------------------------------------------------------------------

def _hash(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class NestedSelectionResult:
    prop: str
    selected_pred: np.ndarray          # aligned to pdf rows; NaN where not in any outer-val block
    selected_mask: np.ndarray          # bool, rows that received an outer-OOF prediction
    selected_candidate_per_row: np.ndarray  # object array of cid per predicted row (else None)
    selected_fold_per_row: np.ndarray  # int outer fold id per predicted row (else -1)
    fold_manifest: list[dict]
    candidates_considered: list[str]


def _inner_select(outer_tr: pd.DataFrame, candidates: list[str], *,
                  min_train_dates: int, val_block_dates: int) -> tuple[str, dict]:
    """Choose the candidate minimizing inner-validation log loss over expanding-window inner folds
    built from the OUTER-TRAINING dates only. Ineligible/non-monotone fits are excluded. Returns
    (chosen_cid, per_candidate_inner_scores)."""
    d = outer_tr["game_date"].to_numpy()
    inner = expanding_window_folds(d, min_train_dates=min_train_dates,
                                   val_block_dates=val_block_dates)
    scores: dict[str, dict] = {}
    for cid in candidates:
        losses, n_used, mono_all, elig_all = [], 0, True, True
        if not inner:
            # Degenerate (tiny synthetic): score in-sample on the outer-train slice itself.
            fc = fit_candidate(cid, outer_tr)
            if fc is None or not fc.eligible:
                continue
            losses.append(_ll(outer_tr[OUTCOME_COL].to_numpy(int), fc.predict(outer_tr)))
            mono_all = fc.monotone
            n_used = 1
        else:
            for f in inner:
                itr = outer_tr[np.isin(d, list(f.train_dates))]
                iva = outer_tr[np.isin(d, list(f.val_dates))]
                if len(itr) < 30 or len(iva) == 0:
                    continue
                fc = fit_candidate(cid, itr)
                if fc is None:
                    elig_all = False
                    break
                if not fc.eligible:
                    elig_all = False
                    break
                mono_all = mono_all and fc.monotone
                losses.append(_ll(iva[OUTCOME_COL].to_numpy(int), fc.predict(iva)))
                n_used += 1
        if not losses or not elig_all or not mono_all or n_used == 0:
            continue
        scores[cid] = {"inner_mean_logloss": float(np.mean(losses)),
                       "inner_folds_used": int(n_used), "monotone": bool(mono_all)}
    if not scores:
        # Guaranteed-available fallback: direct identity (P0) is always monotone & pure.
        return "P0", {"P0": {"inner_mean_logloss": float("nan"), "inner_folds_used": 0,
                             "monotone": True, "fallback": True}}
    chosen = min(scores, key=lambda c: scores[c]["inner_mean_logloss"])
    return chosen, scores


def nested_rolling_origin_select(
    pdf: pd.DataFrame, prop: str, *, min_train_dates: int, val_block_dates: int,
) -> NestedSelectionResult | None:
    """Full nested rolling-origin selection for one prop. Returns None when no outer fold exists."""
    pdf = pdf.sort_values("game_date").reset_index(drop=True)
    d = pdf["game_date"].to_numpy()
    outer = expanding_window_folds(d, min_train_dates=min_train_dates,
                                   val_block_dates=val_block_dates)
    if not outer:
        return None
    candidates = available_candidates(pdf)
    pred = np.full(len(pdf), np.nan)
    sel_cid = np.full(len(pdf), None, dtype=object)
    sel_fold = np.full(len(pdf), -1, dtype=int)
    mask = np.zeros(len(pdf), dtype=bool)
    manifest: list[dict] = []
    for f in outer:
        outer_tr = pdf[np.isin(d, list(f.train_dates))]
        va_idx = np.where(np.isin(d, list(f.val_dates)))[0]
        if len(outer_tr) < 30 or len(va_idx) == 0:
            continue
        # INNER selection uses outer-training dates ONLY (never the outer-val outcomes).
        chosen, inner_scores = _inner_select(
            outer_tr, candidates, min_train_dates=max(2, min_train_dates - 1),
            val_block_dates=val_block_dates)
        # Refit chosen candidate on ALL outer-training dates, score outer-val ONCE.
        fc = fit_candidate(chosen, outer_tr) or fit_candidate("P0", outer_tr)
        if fc is None:
            continue
        pv = fc.predict(pdf.iloc[va_idx])
        pred[va_idx] = pv
        sel_cid[va_idx] = chosen
        sel_fold[va_idx] = f.fold_id
        mask[va_idx] = True
        manifest.append({
            "outer_fold_id": f.fold_id,
            "outer_train_date_min": f.train_date_min, "outer_train_date_max": f.train_date_max,
            "outer_val_date_min": f.val_date_min, "outer_val_date_max": f.val_date_max,
            "chronology_pass": bool(f.chronology_pass),
            "selected_candidate": chosen,
            "selected_source": CANDIDATE_SPECS[chosen][0],
            "selected_method": CANDIDATE_SPECS[chosen][1],
            "selected_detail": fc.detail,
            "inner_candidate_scores": inner_scores,
            "outer_train_date_hash": _hash(*sorted(f.train_dates)),
            "outer_val_date_hash": _hash(*sorted(f.val_dates)),
            "selection_hash": _hash(prop, f.fold_id, chosen, *sorted(f.train_dates)),
        })
    if not mask.any():
        return None
    return NestedSelectionResult(
        prop=prop, selected_pred=pred, selected_mask=mask,
        selected_candidate_per_row=sel_cid, selected_fold_per_row=sel_fold,
        fold_manifest=manifest, candidates_considered=candidates)
