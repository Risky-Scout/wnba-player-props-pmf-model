"""Chronological injury-conditioned participation model selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from wnba_props_model.data.injury_workbook import assert_no_onset_leakage
from wnba_props_model.sharp_v6.phase3_labels import injury_conditioned_training_cohort

PROHIBITED_FEATURES = frozenset(
    {
        "date_returned",
        "total_games_missed",
        "actual_minutes",
        "minutes",
        "participation",
        "did_play",
        "participation_binary_label",
    }
)

DEFAULT_FEATURE_PREFIXES = (
    "player_minutes_",
    "player_rest",
    "player_days_since",
    "cumulative_minutes",
    "is_home",
)


@dataclass
class ParticipationSelection:
    model: Any
    calibrator: Any | None
    calibration_method: str
    family: str
    feature_cols: list[str]
    feature_hash: str
    metrics: pd.DataFrame
    oof: pd.DataFrame
    applicability_contract: dict[str, Any]
    calibration_report: dict[str, Any]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.feature_cols).to_numpy(float)
        if self.family == "logistic":
            X = _impute_median(X)
        raw = self.model.predict_proba(X)[:, 1]
        if self.family == "logistic":
            raw = _prior_correct(raw, self.calibration_report.get("train_prior", 0.5))
        return _calibrate(raw, self.calibrator, self.calibration_method)


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]


def _features(df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    if feature_cols:
        cols = list(feature_cols)
    else:
        cols = [
            c
            for c in df.columns
            if c.startswith(DEFAULT_FEATURE_PREFIXES)
            or c in {"is_home", "player_rest_days", "team_rest_days", "player_days_since_last_game"}
        ]
    bad = sorted(set(cols) & PROHIBITED_FEATURES)
    assert_no_onset_leakage(cols)
    if bad:
        raise ValueError(f"prohibited participation features: {bad}")
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def chronological_folds(df: pd.DataFrame, *, max_year: int = 2025, n_folds: int = 4):
    dates = pd.to_datetime(df["game_date"])
    unique = np.array(sorted(dates[dates.dt.year <= max_year].dropna().unique()))
    if len(unique) < n_folds + 1:
        return
    # Expanding windows: each fold tests a later date block; train is strictly prior.
    splits = np.array_split(unique, n_folds + 1)
    for fold_i in range(1, len(splits)):
        block = np.concatenate(splits[fold_i : fold_i + 1])
        if not len(block):
            continue
        start, end = block.min(), block.max()
        test = (dates >= start) & (dates <= end) & (dates.dt.year <= max_year)
        train = dates < start
        if train.any() and test.any():
            yield fold_i - 1, np.flatnonzero(train.to_numpy()), np.flatnonzero(test.to_numpy())


def _impute_median(X: np.ndarray) -> np.ndarray:
    """Median-impute columns for estimators that reject NaN (logistic only)."""
    out = np.asarray(X, float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~np.isfinite(col)] = med
        out[:, j] = col
    return out


def _fit_candidate(name: str, X: np.ndarray, y: np.ndarray):
    if name == "logistic":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
        ).fit(_impute_median(X), y)
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=120,
        learning_rate=0.06,
        l2_regularization=2.0,
        random_state=20260730,
    ).fit(X, y)


def _prior_correct(p: np.ndarray, observed_prior: float, fitted_prior: float = 0.5) -> np.ndarray:
    observed_prior = float(np.clip(observed_prior, 1e-3, 1 - 1e-3))
    fitted_prior = float(np.clip(fitted_prior, 1e-3, 1 - 1e-3))
    odds = np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)
    correction = (observed_prior / (1 - observed_prior)) / (fitted_prior / (1 - fitted_prior))
    return np.clip((odds * correction) / (1 + odds * correction), 1e-6, 1 - 1e-6)


def _fit_calibrator(method: str, p: np.ndarray, y: np.ndarray):
    if method == "identity" or len(np.unique(y)) < 2:
        return None
    if method == "platt":
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        return LogisticRegression(C=1.0).fit(z.reshape(-1, 1), y)
    if method == "beta":
        x = np.c_[np.log(np.clip(p, 1e-6, 1 - 1e-6)), np.log(np.clip(1 - p, 1e-6, 1))]
        return LogisticRegression(C=1.0).fit(x, y)
    from sklearn.isotonic import IsotonicRegression

    # Guarded isotonic: require enough mass on both classes.
    if (y == 0).sum() < 20 or (y == 1).sum() < 20:
        return None
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)


def _calibrate(p: np.ndarray, cal: Any | None, method: str) -> np.ndarray:
    if cal is None or method == "identity":
        return np.asarray(p, float)
    if method == "isotonic":
        return np.clip(cal.predict(p), 1e-6, 1 - 1e-6)
    if method == "beta":
        x = np.c_[np.log(np.clip(p, 1e-6, 1 - 1e-6)), np.log(np.clip(1 - p, 1e-6, 1))]
        return np.clip(cal.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return np.clip(cal.predict_proba(z.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(p)
    y, p = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    total = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        total += (len(b) / len(p)) * abs(float(y[b].mean()) - float(p[b].mean()))
    return float(total)


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return 0.0, 1.0
    lr = LogisticRegression(C=1e6, max_iter=500).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def evaluate_chronological(
    labels: pd.DataFrame, *, feature_cols: list[str] | None = None, max_year: int = 2025
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold metrics and chronological OOF predictions; no same-fold calibration."""
    df = injury_conditioned_training_cohort(labels)
    df = df[pd.to_datetime(df["game_date"]).dt.year <= max_year].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    feat = _features(df, feature_cols)
    if not feat:
        raise RuntimeError("no numeric participation features available")
    rows, oof = [], []
    for fold, tr, te in chronological_folds(df, max_year=max_year):
        ytr = df.loc[tr, "participation_binary_label"].astype(int).to_numpy()
        yte = df.loc[te, "participation_binary_label"].astype(int).to_numpy()
        if len(np.unique(ytr)) < 2 or len(te) < 5:
            continue
        Xtr = np.nan_to_num(df.loc[tr, feat].to_numpy(float), nan=np.nan)
        Xte = np.nan_to_num(df.loc[te, feat].to_numpy(float), nan=np.nan)
        for candidate in ("logistic", "hgb"):
            model = _fit_candidate(candidate, Xtr, ytr)
            Xte_use = _impute_median(Xte) if candidate == "logistic" else Xte
            raw = model.predict_proba(Xte_use)[:, 1]
            p = _prior_correct(raw, float(ytr.mean())) if candidate == "logistic" else raw
            intercept, slope = _calibration_slope_intercept(yte, p)
            try:
                auc = float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")
            except ValueError:
                auc = float("nan")
            rows.append(
                {
                    "fold": fold,
                    "candidate": candidate,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "date_count": int(pd.to_datetime(df.loc[te, "game_date"]).nunique()),
                    "base_rate": float(yte.mean()),
                    "pred_mean": float(p.mean()),
                    "nll": float(log_loss(yte, p, labels=[0, 1])),
                    "brier": float(brier_score_loss(yte, p)),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "ece": _ece(yte, p),
                    "roc_auc_diagnostic": auc,
                }
            )
            for i, y, q in zip(te, yte, p, strict=False):
                oof.append(
                    {
                        "row": int(i),
                        "game_id": df.at[i, "game_id"],
                        "player_id": df.at[i, "player_id"],
                        "game_date": df.at[i, "game_date"],
                        "fold": fold,
                        "candidate": candidate,
                        "y": int(y),
                        "raw_p": float(q),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(oof)


def select_and_persist(
    labels: pd.DataFrame, *, feature_cols: list[str] | None = None, max_year: int = 2025
) -> ParticipationSelection:
    metrics, oof = evaluate_chronological(labels, feature_cols=feature_cols, max_year=max_year)
    if metrics.empty:
        raise RuntimeError("insufficient chronological injury-conditioned participation data")
    # Select family by mean NLL, then Brier, preferring logistic when tied.
    ranked = metrics.groupby("candidate")[["nll", "brier"]].mean().sort_values(["nll", "brier"])
    candidate = str(ranked.index[0])
    if (
        abs(float(ranked.iloc[0]["nll"]) - float(ranked.iloc[min(1, len(ranked) - 1)]["nll"]))
        < 1e-4
        and "logistic" in ranked.index
    ):
        candidate = "logistic"

    candidate_oof = oof[oof["candidate"] == candidate].sort_values(["fold", "row"])
    # Calibration: fit only on earlier OOF folds; never calibrate a row with itself.
    choices: list[tuple[float, str, Any]] = []
    for method in ("identity", "platt", "beta", "isotonic"):
        scores = []
        last_cal = None
        for fold in sorted(candidate_oof["fold"].unique()):
            past = candidate_oof[candidate_oof["fold"] < fold]
            cur = candidate_oof[candidate_oof["fold"] == fold]
            if method == "identity":
                scores.append(log_loss(cur.y, cur.raw_p, labels=[0, 1]))
                last_cal = None
                continue
            if len(past) < 40:
                continue
            cal = _fit_calibrator(method, past.raw_p.to_numpy(), past.y.to_numpy())
            if cal is None and method != "identity":
                continue
            scores.append(
                log_loss(cur.y, _calibrate(cur.raw_p.to_numpy(), cal, method), labels=[0, 1])
            )
            last_cal = cal
        choices.append((float(np.mean(scores)) if scores else float("inf"), method, last_cal))
    method = min(choices, key=lambda t: t[0])[1]

    # Final calibrator: fit on all OOF predictions (each row's prediction was OOF).
    # Do NOT refit calibrator on in-sample predictions of the final model.
    final_cal = None
    if method != "identity":
        final_cal = _fit_calibrator(
            method, candidate_oof.raw_p.to_numpy(), candidate_oof.y.to_numpy()
        )
        if final_cal is None:
            method = "identity"

    df = injury_conditioned_training_cohort(labels)
    df = df[pd.to_datetime(df["game_date"]).dt.year <= max_year].reset_index(drop=True)
    feat = _features(df, feature_cols)
    y = df["participation_binary_label"].astype(int).to_numpy()
    X = np.nan_to_num(df[feat].to_numpy(float), nan=np.nan)
    model = _fit_candidate(candidate, X, y)
    train_prior = float(y.mean())

    cal_p = _calibrate(candidate_oof.raw_p.to_numpy(), final_cal, method)
    intercept, slope = _calibration_slope_intercept(candidate_oof.y.to_numpy(), cal_p)
    calibration_report = {
        "selected_family": candidate,
        "selected_calibrator": method,
        "oof_log_loss": float(log_loss(candidate_oof.y, cal_p, labels=[0, 1])),
        "oof_brier": float(brier_score_loss(candidate_oof.y, cal_p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": _ece(candidate_oof.y.to_numpy(), cal_p),
        "train_prior": train_prior,
        "n_oof": len(candidate_oof),
        "worst_fold_nll": float(
            metrics[metrics["candidate"] == candidate].groupby("fold")["nll"].mean().max()
        ),
        "self_calibration_prohibited": True,
        "calibrator_fit_on": "earlier_and_pooled_oof_predictions_only",
    }
    contract = {
        "scope": "injury_conditioned_only",
        "claim": "Does not estimate unconditional P(active|all roster-eligible players)",
        "supported_statuses": ["DOUBTFUL", "QUESTIONABLE", "PROBABLE"],
        "unsupported_statuses": ["NOT_LISTED", "UNKNOWN_SOURCE", "OUT", "SUSPENDED"],
        "not_listed_policy": "ACTIVE_ROSTER_NOT_LISTED_not_historically_calibrated",
        "unknown_source_policy": "ABSTAIN_OR_FAIL_AVAILABILITY_GATE",
        "training_max_year": max_year,
        "feature_cols": feat,
        "feature_hash": _feature_hash(feat),
        "family": candidate,
        "calibration_method": method,
    }
    return ParticipationSelection(
        model=model,
        calibrator=final_cal,
        calibration_method=method,
        family=candidate,
        feature_cols=feat,
        feature_hash=_feature_hash(feat),
        metrics=metrics,
        oof=candidate_oof.assign(calibrated_p=cal_p),
        applicability_contract=contract,
        calibration_report=calibration_report,
    )


def persist_participation_artifacts(selection: ParticipationSelection, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selection.metrics.to_csv(out_dir / "PARTICIPATION_MODEL_COMPARISON.csv", index=False)
    selection.oof.to_parquet(out_dir / "PARTICIPATION_OOF_PREDICTIONS.parquet", index=False)
    (out_dir / "PARTICIPATION_CALIBRATION_REPORT.json").write_text(
        json.dumps(selection.calibration_report, indent=2) + "\n"
    )
    (out_dir / "PARTICIPATION_APPLICABILITY_CONTRACT.json").write_text(
        json.dumps(selection.applicability_contract, indent=2) + "\n"
    )
