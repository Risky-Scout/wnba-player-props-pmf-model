"""Conditional-on-active Phase-3 minute PMF candidates and OOF selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from wnba_props_model.data.participation_labels import LEAKAGE_PROHIBITED_ONSET_FEATURES
from wnba_props_model.sharp_v6.phase3_participation import chronological_folds

ROLE_STATES = (
    "CORE_STARTER",
    "SECONDARY_STARTER",
    "ROTATION",
    "LIMITED_OR_RETURNING",
    "ELEVATED_ROLE",
    "FRINGE_OR_UNCERTAIN",
)
OT_STATUS = "PLAYER_LEVEL_OT_APPROXIMATION_PENDING_SHARED_GAME_OT"
REG_MAX = 40
OT_MAX = 8
SUPPORT = REG_MAX + OT_MAX  # 0..48 inclusive → 49 atoms
_FORBIDDEN = set(LEAKAGE_PROHIBITED_ONSET_FEATURES) | {
    "is_starter",
    "starter",
    "confirmed_starter",
    "market_odds",
    "odds",
    "actual_minutes",
    "minutes",
    "did_play",
}


def minute_features(frame: pd.DataFrame, feature_cols: list[str] | None = None) -> list[str]:
    cols = feature_cols or [
        c
        for c in frame.columns
        if c.startswith(("player_minutes_", "player_pts_mean_", "cumulative_minutes", "opp_"))
        or c
        in {
            "is_home",
            "player_rest_days",
            "team_rest_days",
            "player_days_since_last_game",
        }
    ]
    bad = sorted(set(cols) & _FORBIDDEN)
    if bad:
        raise ValueError(f"forbidden minutes features: {bad}")
    return [c for c in cols if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]


def active_minutes_cohort(df: pd.DataFrame) -> pd.DataFrame:
    active = df.get("participation_label_class", pd.Series("CONFIRMED_ACTIVE", index=df.index)).eq(
        "CONFIRMED_ACTIVE"
    )
    eligible = (
        df.get("training_eligible", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    )
    mins = pd.to_numeric(df["actual_minutes"], errors="coerce")
    # Do not clip OT observations.
    return df[active & eligible & mins.gt(0)].copy()


def role_state_from_minutes(minutes: np.ndarray) -> np.ndarray:
    """Training-only role labels from realized minutes (never used as live features)."""
    return np.select(
        [
            minutes >= 30,
            minutes >= 22,
            minutes >= 12,
            (minutes >= 5) & (minutes < 12),
            minutes >= 18,
        ],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype(int)


def enforce_monotone_survival(survival: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.clip(survival, 0, 1), axis=1)


def survival_to_pmf(survival: np.ndarray, support: int = REG_MAX) -> np.ndarray:
    """Convert P(M>=k) for k=1..support into a PMF on 0..support (reg minutes)."""
    s = enforce_monotone_survival(np.asarray(survival, float))
    pmf = np.empty((len(s), support + 1))
    pmf[:, 0] = 1.0 - s[:, 0]
    pmf[:, 1:support] = s[:, : support - 1] - s[:, 1:support]
    pmf[:, support] = s[:, support - 1]
    pmf = np.clip(pmf, 0, None)
    return pmf / np.clip(pmf.sum(axis=1, keepdims=True), 1e-12, None)


def _attach_ot_tail(reg_pmf: np.ndarray, p_ot: np.ndarray | float) -> np.ndarray:
    """Expand 0..40 regulation PMF with a player-level OT approximation tail to 48."""
    n = len(reg_pmf)
    p_ot_arr = np.full(n, float(p_ot)) if np.isscalar(p_ot) else np.asarray(p_ot, float)
    p_ot_arr = np.clip(p_ot_arr, 0.0, 0.5)
    out = np.zeros((n, SUPPORT + 1))
    out[:, : REG_MAX + 1] = reg_pmf * (1.0 - p_ot_arr)[:, None]
    # Simple geometric-ish OT extra minutes 1..OT_MAX on top of 40.
    weights = np.array([0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02], dtype=float)
    weights = weights / weights.sum()
    for j in range(OT_MAX):
        out[:, REG_MAX + 1 + j] = p_ot_arr * weights[j]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


@dataclass
class Phase3MinutesModel:
    family: str
    feature_cols: list[str]
    models: Any
    residual_sd_by_band: dict[int, float] | float = 6.0
    p_ot: float = 0.05
    status: str = OT_STATUS
    support_max: int = SUPPORT

    def pmf(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.feature_cols).to_numpy(float)
        if self.family == "ordinal_survival":
            from wnba_props_model.sharp_v6.phase3_participation import _impute_median

            X_use = _impute_median(X) if self.models.get("impute") else X
            cols = []
            for m in self.models["thresholds"]:
                if isinstance(m, tuple) and m[0] == "const":
                    cols.append(np.full(len(X_use), float(m[1])))
                else:
                    cols.append(m.predict_proba(X_use)[:, 1])
            surv = np.column_stack(cols)
            reg = survival_to_pmf(surv, support=REG_MAX)
            return _attach_ot_tail(reg, self.p_ot)
        if self.family == "role_mixture":
            probs = self.models["classifier"].predict_proba(X)
            # Ensure columns align to ROLE_STATES indices 0..5
            full = np.zeros((len(X), len(ROLE_STATES)))
            for j, cls in enumerate(self.models["classifier"].classes_):
                full[:, int(cls)] = probs[:, j]
            full = full / np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
            out = np.zeros((len(X), SUPPORT + 1))
            for role, dist in self.models["role_pmfs"].items():
                out += full[:, [role]] * dist[None, :]
            return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        # Candidate A: HGB mean + role-band residual / truncnorm-like discrete
        mu = np.clip(self.models["regressor"].predict(X), 0, REG_MAX)
        bands = role_state_from_minutes(mu)  # proxy band from predicted mean (pregame)
        grid = np.arange(REG_MAX + 1)
        reg = np.zeros((len(X), REG_MAX + 1))
        for i, (m, b) in enumerate(zip(mu, bands, strict=False)):
            sd = (
                self.residual_sd_by_band.get(int(b), 6.0)
                if isinstance(self.residual_sd_by_band, dict)
                else float(self.residual_sd_by_band)
            )
            sd = max(float(sd), 1.5)
            pdf = norm.pdf(grid, loc=m, scale=sd)
            pdf = pdf / max(pdf.sum(), 1e-12)
            reg[i] = pdf
        return _attach_ot_tail(reg, self.p_ot)


def _role_empirical_pmf(y: np.ndarray, support: int = SUPPORT) -> np.ndarray:
    hist = np.bincount(np.clip(np.asarray(y, int), 0, support), minlength=support + 1).astype(float)
    # Laplace smooth
    hist = hist + 0.5
    return hist / hist.sum()


def fit_minutes_candidate(
    train: pd.DataFrame,
    family: str,
    *,
    feature_cols: list[str] | None = None,
    oof_residual_sd_by_band: dict[int, float] | None = None,
) -> Phase3MinutesModel:
    tr = active_minutes_cohort(train)
    feat = minute_features(tr, feature_cols)
    if len(tr) < 20:
        raise RuntimeError("insufficient confirmed-active minutes rows")
    X = tr[feat].to_numpy(float)
    y = tr["actual_minutes"].to_numpy(float)
    p_ot = float(np.mean(y > REG_MAX))
    if family == "ordinal_survival":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from wnba_props_model.sharp_v6.phase3_participation import _impute_median

        y_reg = np.minimum(y, REG_MAX)
        X_imp = _impute_median(X)
        # Regularized logistic thresholds (cheaper than 40 HGBs); monotonicity enforced later.
        models = []
        for k in range(1, REG_MAX + 1):
            yk = y_reg >= k
            if len(np.unique(yk)) < 2:
                # Degenerate threshold: constant survival equal to empirical rate.
                rate = float(np.mean(yk))
                models.append(("const", rate))
            else:
                models.append(
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=400, class_weight="balanced"),
                    ).fit(X_imp, yk)
                )
        return Phase3MinutesModel(
            family, feat, {"thresholds": models, "impute": True}, p_ot=p_ot, status=OT_STATUS
        )
    if family == "role_mixture":
        roles = role_state_from_minutes(y)
        classifier = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, random_state=20260730
        ).fit(X, roles)
        role_pmfs = {}
        for r in range(len(ROLE_STATES)):
            yy = y[roles == r]
            role_pmfs[r] = _role_empirical_pmf(yy if len(yy) else y, SUPPORT)
        return Phase3MinutesModel(
            family,
            feat,
            {"classifier": classifier, "role_pmfs": role_pmfs},
            residual_sd_by_band=float(np.std(y)),
            p_ot=p_ot,
            status=OT_STATUS,
        )
    reg = HistGradientBoostingRegressor(
        max_depth=3, max_iter=160, learning_rate=0.06, random_state=20260730
    ).fit(X, np.minimum(y, REG_MAX))
    if oof_residual_sd_by_band is not None:
        sd_by = oof_residual_sd_by_band
    else:
        # Fallback only; evaluate_chronological supplies OOF residuals for selection.
        resid = np.minimum(y, REG_MAX) - reg.predict(X)
        bands = role_state_from_minutes(y)
        sd_by = {
            int(b): float(np.clip(np.std(resid[bands == b]), 2.0, 11.0))
            if (bands == b).sum() > 20
            else float(np.clip(np.std(resid), 2.0, 11.0))
            for b in range(6)
        }
    return Phase3MinutesModel(
        "hgb_residual",
        feat,
        {"regressor": reg},
        residual_sd_by_band=sd_by,
        p_ot=p_ot,
        status=OT_STATUS,
    )


def pmf_metrics(pmf: np.ndarray, y: np.ndarray) -> dict[str, float]:
    grid = np.arange(pmf.shape[1])
    yy = np.clip(np.asarray(np.rint(y), int), 0, pmf.shape[1] - 1)
    means = pmf @ grid
    nll = -np.log(np.maximum(pmf[np.arange(len(yy)), yy], 1e-12)).mean()
    cdf = pmf.cumsum(axis=1)
    crps = np.mean(np.sum((cdf - (grid[None, :] >= yy[:, None]).astype(float)) ** 2, axis=1))
    # Randomized PIT
    u = np.random.default_rng(20260730).uniform(size=len(yy))
    prev = np.where(yy > 0, cdf[np.arange(len(yy)), yy - 1], 0.0)
    pit = prev + u * (cdf[np.arange(len(yy)), yy] - prev)
    lo50 = np.argmax(cdf >= 0.25, axis=1)
    hi50 = np.argmax(cdf >= 0.75, axis=1)
    lo80 = np.argmax(cdf >= 0.10, axis=1)
    hi80 = np.argmax(cdf >= 0.90, axis=1)
    lo95 = np.argmax(cdf >= 0.025, axis=1)
    hi95 = np.argmax(cdf >= 0.975, axis=1)
    var = (pmf * (grid[None, :] - means[:, None]) ** 2).sum(axis=1)
    empir_var = float(np.var(y))
    return {
        "nll": float(nll),
        "crps": float(crps),
        "mae": float(np.abs(means - y).mean()),
        "rmse": float(np.sqrt(np.mean((means - y) ** 2))),
        "bias": float(np.mean(means - y)),
        "variance_bias": float(np.mean(var) - empir_var),
        "pit_mean": float(np.mean(pit)),
        "coverage_50": float(np.mean((yy >= lo50) & (yy <= hi50))),
        "coverage_80": float(np.mean((yy >= lo80) & (yy <= hi80))),
        "coverage_95": float(np.mean((yy >= lo95) & (yy <= hi95))),
        "tail_exceedance_40": float(np.mean((y > REG_MAX) & (means <= REG_MAX))),
    }


def evaluate_chronological(
    train: pd.DataFrame, *, feature_cols: list[str] | None = None, max_year: int = 2025
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = active_minutes_cohort(train)
    df = df[pd.to_datetime(df["game_date"]).dt.year <= max_year].reset_index(drop=True)
    if "participation_label_class" not in df.columns:
        df["participation_label_class"] = "CONFIRMED_ACTIVE"
    if "training_eligible" not in df.columns:
        df["training_eligible"] = True
    rows, oof = [], []
    for fold, tr, te in chronological_folds(df, max_year=max_year, n_folds=4):
        if len(tr) < 20 or len(te) < 5:
            continue
        # OOF residuals for candidate A control.
        feat = minute_features(df.loc[tr], feature_cols)
        reg = HistGradientBoostingRegressor(
            max_depth=3, max_iter=160, learning_rate=0.06, random_state=20260730
        ).fit(df.loc[tr, feat].to_numpy(float), np.minimum(df.loc[tr, "actual_minutes"], REG_MAX))
        # Residual SD from a nested prior half of train only (not test).
        mid = max(len(tr) // 2, 10)
        tr_fit, tr_res = tr[:mid], tr[mid:]
        if len(tr_res) >= 10:
            reg_half = HistGradientBoostingRegressor(
                max_depth=3, max_iter=120, learning_rate=0.06, random_state=20260730
            ).fit(
                df.loc[tr_fit, feat].to_numpy(float),
                np.minimum(df.loc[tr_fit, "actual_minutes"], REG_MAX),
            )
            resid = np.minimum(df.loc[tr_res, "actual_minutes"], REG_MAX) - reg_half.predict(
                df.loc[tr_res, feat].to_numpy(float)
            )
            bands = role_state_from_minutes(df.loc[tr_res, "actual_minutes"].to_numpy(float))
            sd_by = {
                int(b): float(np.clip(np.std(resid[bands == b]), 2.0, 11.0))
                if (bands == b).sum() > 10
                else float(np.clip(np.std(resid), 2.0, 11.0))
                for b in range(6)
            }
        else:
            sd_by = {b: 6.0 for b in range(6)}

        for family in ("hgb_residual", "ordinal_survival", "role_mixture"):
            model = fit_minutes_candidate(
                df.loc[tr],
                family,
                feature_cols=feature_cols,
                oof_residual_sd_by_band=sd_by if family == "hgb_residual" else None,
            )
            # Ensure regressor for hgb uses full train fold.
            if family == "hgb_residual":
                model.models["regressor"] = reg
            pmf = model.pmf(df.loc[te])
            yte = df.loc[te, "actual_minutes"].to_numpy(float)
            metric = pmf_metrics(pmf, yte)
            # Normalization / monotonicity gate
            ok_norm = bool(np.all(pmf >= -1e-12) and np.allclose(pmf.sum(axis=1), 1.0, atol=1e-6))
            rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "date_count": int(pd.to_datetime(df.loc[te, "game_date"]).nunique()),
                    "pmf_ok": ok_norm,
                    **metric,
                }
            )
            for idx, y, p in zip(te, yte, pmf, strict=False):
                oof.append(
                    {
                        "fold": fold,
                        "family": family,
                        "game_id": df.at[idx, "game_id"] if "game_id" in df.columns else None,
                        "player_id": df.at[idx, "player_id"] if "player_id" in df.columns else None,
                        "game_date": df.at[idx, "game_date"],
                        "actual_minutes": float(y),
                        "pred_mean": float(np.dot(np.arange(p.size), p)),
                        "pmf": p.tolist(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(oof)


def select_and_persist(
    train: pd.DataFrame, *, feature_cols: list[str] | None = None, max_year: int = 2025
) -> tuple[Phase3MinutesModel, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics, oof = evaluate_chronological(train, feature_cols=feature_cols, max_year=max_year)
    if metrics.empty:
        raise RuntimeError("insufficient chronological minutes data")
    metrics = metrics[metrics["pmf_ok"]].copy()
    if metrics.empty:
        raise RuntimeError("no minutes candidate passed PMF normalization")
    # Catastrophic fold filter: drop families with a fold NLL > 2.5x median fold NLL.
    keep_families = []
    for fam, g in metrics.groupby("family"):
        med = g["nll"].median()
        if (g["nll"] > 2.5 * med).any():
            continue
        keep_families.append(fam)
    if keep_families:
        metrics = metrics[metrics["family"].isin(keep_families)]
    avg = metrics.groupby("family")[["nll", "crps"]].mean().sort_values(["nll", "crps"])
    # Prefer simpler family when statistically indistinguishable (NLL delta < 0.01).
    family = str(avg.index[0])
    if len(avg) > 1 and abs(float(avg.iloc[0]["nll"]) - float(avg.iloc[1]["nll"])) < 0.01:
        preference = ["hgb_residual", "ordinal_survival", "role_mixture"]
        candidates = [f for f in preference if f in avg.index]
        if candidates:
            family = candidates[0]
    # Final model fit on <= max_year only (never use 2026 for selection/fit of family).
    fit_df = active_minutes_cohort(train)
    fit_df = fit_df[pd.to_datetime(fit_df["game_date"]).dt.year <= max_year]
    model = fit_minutes_candidate(fit_df, family, feature_cols=feature_cols)
    # Attach OOF residual SDs for hgb control when available.
    if family == "hgb_residual":
        # Approximate from pooled OOF errors of that family.
        fam_oof = oof[oof["family"] == family]
        if not fam_oof.empty:
            err = fam_oof["actual_minutes"] - fam_oof["pred_mean"]
            model.residual_sd_by_band = float(np.clip(err.std(), 2.0, 11.0))
    report = {
        "selected_family": family,
        "ot_status": OT_STATUS,
        "support_max": SUPPORT,
        "oof_nll": float(avg.loc[family, "nll"]),
        "oof_crps": float(avg.loc[family, "crps"]),
        "families_considered": list(avg.index),
        "max_year_for_selection": max_year,
    }
    return model, metrics, oof[oof["family"] == family].copy(), report


def persist_minutes_artifacts(
    model: Phase3MinutesModel,
    metrics: pd.DataFrame,
    oof: pd.DataFrame,
    report: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "MINUTES_CANDIDATE_COMPARISON.csv", index=False)
    oof.drop(columns=["pmf"], errors="ignore").to_parquet(
        out_dir / "MINUTES_OOF_PREDICTIONS.parquet", index=False
    )
    # Role metrics if available via predicted mean bands.
    if not oof.empty:
        bands = role_state_from_minutes(oof["pred_mean"].to_numpy(float))
        tmp = oof.assign(
            predicted_role=[ROLE_STATES[int(b)] for b in bands],
            abs_err=(oof["actual_minutes"] - oof["pred_mean"]).abs(),
        )
        by_role = (
            tmp.groupby("predicted_role")
            .agg(n=("actual_minutes", "size"), mae=("abs_err", "mean"))
            .reset_index()
        )
        by_role.to_csv(out_dir / "MINUTES_METRICS_BY_ROLE.csv", index=False)
    (out_dir / "MINUTES_CALIBRATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    contract = {
        "conditional_on": "CONFIRMED_ACTIVE, training_eligible, actual_minutes>0",
        "family": model.family,
        "feature_cols": model.feature_cols,
        "ot_status": model.status,
        "support_max": model.support_max,
        "dnp_mass_in_minutes_pmf": False,
    }
    (out_dir / "MINUTES_MODEL_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
