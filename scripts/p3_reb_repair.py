"""P3 — Rebounds structural repair, strictly prequential. Candidate D (calibrated
empirical residual PMF with a frozen dispersion-scale grid) evaluated per five-date outer
block; the dispersion scale is chosen using ONLY pre-block dates by interval/PIT
CALIBRATION (not CRPS/ROI). Concatenate + apply the existing corrected gates unchanged.

Candidates A/B/C (HGB Poisson / squared-error conditional-mean, per-minute×fitted-minutes)
require CI retraining on schema-v2 features; Candidate D is the empirical-fallback candidate
evaluable on the existing OOF and is what the prequential calibration selects for Rebounds.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import forecasting as fc  # noqa: E402
from wnba_props_model.evaluation import distribution_calibration as dc  # noqa: E402
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf  # noqa: E402

app = typer.Typer(add_completion=False)
SCALE_GRID = (1.0, 0.9, 0.8, 0.7)          # frozen before scoring
UNDER_TOL, OVER_TOL, PIT_ENV = 0.05, 0.07, 0.10


def _nb_pmf(mu: float, size: float, ms: int) -> np.ndarray:
    """Negative-binomial PMF on 0..ms-1 with mean mu and dispersion `size` (r)."""
    mu = max(float(mu), 1e-3); size = max(float(size), 1e-3)
    k = np.arange(ms)
    p = size / (size + mu)
    from scipy.special import gammaln
    logpmf = gammaln(k + size) - gammaln(size) - gammaln(k + 1) + size * np.log(p) + k * np.log1p(-p)
    out = np.exp(logpmf)
    s = out.sum()
    return out / s if s > 0 else np.ones(ms) / ms


def _fit_minutes_resid(before: pd.DataFrame):
    """Fitted minutes-residual distribution (integer offsets) from prior-fold residuals, per role,
    shrunk to pooled. Returns {role: (offsets, weights)} and pooled (offsets, weights)."""
    def _hist(g):
        e = np.round(g["actual_minutes"].astype(float) - g["minutes_mean"].astype(float)).astype(int)
        e = e[np.isfinite(e)]
        vals, cnts = np.unique(e.values, return_counts=True)
        return vals, cnts / cnts.sum()
    pooled = _hist(before[before["actual_minutes"].notna()])
    per_role = {}
    for role, g in before[before["actual_minutes"].notna()].groupby("role_bucket"):
        if len(g) >= 30:
            per_role[str(role)] = _hist(g)
    return per_role, pooled


def _cand_C(rows, before, ms):
    """Candidate C: per-minute rate x fitted minutes distribution, integrated (no hardcoded quadrature)."""
    per_role, pooled = _fit_minutes_resid(before)
    # NB dispersion (size r) from pre-fold REB residual over-dispersion vs structural mean.
    mu_b = before["_point"].astype(float).clip(lower=1e-3)
    var_b = ((before["actual_outcome"].astype(float) - mu_b) ** 2).mean()
    mean_mu = float(mu_b.mean())
    size = max(mean_mu ** 2 / max(var_b - mean_mu, 1e-3), 0.5)   # method-of-moments NB size
    out = []
    for _, r in rows.iterrows():
        mm = float(r.get("minutes_mean", 0.0))
        rpm = float(r["_point"]) / max(mm, 1.0)                  # rebounds per minute
        off, w = per_role.get(str(r.get("role_bucket", "")), pooled)
        acc = np.zeros(ms)
        for o, wt in zip(off, w):
            m = max(mm + float(o), 1.0)
            acc += wt * _nb_pmf(rpm * m, size, ms)
        s = acc.sum()
        out.append(acc / s if s > 0 else np.ones(ms) / ms)
    return out


def _cand_D(rows, cells, scale, ms):
    return [recalibrate_pmf(
        dc.hierarchical_empirical_pmf(float(r["_point"]),
                                      f"{r.get('role_bucket','')}|{r.get('_minbucket','')}", cells, ms),
        0.0, scale) for _, r in rows.iterrows()]


_ID_COLS = {"game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season",
            "player_name", "team_abbreviation", "opponent_team_abbreviation", "home_away"}


def _feature_cols(feat: pd.DataFrame) -> list:
    from wnba_props_model.features.feature_contract import MODEL_FEATURES, FORBIDDEN_MODEL_FEATURES
    forbidden = set(FORBIDDEN_MODEL_FEATURES)
    return [c for c in MODEL_FEATURES if c in feat.columns and c not in forbidden and c not in _ID_COLS]


def _pooled_role_size(before: pd.DataFrame, pred_col: str):
    """NB dispersion `size` per role, partially pooled toward the global Rebounds estimate."""
    def _mom(g):
        mu = g[pred_col].astype(float).clip(lower=1e-3)
        v = ((g["actual_outcome"].astype(float) - mu) ** 2).mean()
        m = float(mu.mean())
        return max(m ** 2 / max(v - m, 1e-3), 0.5)
    glob = _mom(before)
    out = {}
    for role, g in before.groupby("role_bucket"):
        n = len(g)
        w = n / (n + 50.0)                      # shrink small roles toward global
        out[str(role)] = w * _mom(g) + (1 - w) * glob
    return out, glob


def _cand_AB(before, blk, feat, loss, ms):
    """Candidates A/B — HGB conditional mean E[REB|X] (Poisson / squared-error), NB dispersion
    from pre-fold residuals partially pooled by role. No minutes marginalization, no fixed
    _REB_ROLE_DISPERSION. Requires the schema-v2 feature matrix (available in CI)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cols = _feature_cols(feat)
    key = ["game_id", "player_id"]

    def _join(rows):
        # feature matrix is the source of feature columns; drop any overlap from the OOF side
        left = rows.drop(columns=[c for c in cols if c in rows.columns], errors="ignore")
        return left.merge(feat[key + cols], on=key, how="inner")

    fb = _join(before)
    xb = fb[cols].astype(float).fillna(0.0).values
    yb = fb["actual_outcome"].astype(float).values
    model = HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05,
                                          max_depth=3, random_state=0)
    model.fit(xb, np.clip(yb, 0, None) if loss == "poisson" else yb)
    fb["_pred"] = np.clip(model.predict(xb), 0.0, None)
    size_by_role, glob = _pooled_role_size(fb, "_pred")
    fk = _join(blk)
    mu = np.clip(model.predict(fk[cols].astype(float).fillna(0.0).values), 1e-3, None)
    return [_nb_pmf(m, size_by_role.get(str(rl), glob), ms)
            for m, rl in zip(mu, fk["role_bucket"].astype(str))], fk


def _calib_penalty(pmfs, actuals, gdate):
    """Pre-block calibration penalty: interval-residual excess beyond tolerance + PIT excess."""
    y = actuals.astype(int)
    pen = 0.0
    for cl in (0.5, 0.8, 0.9):
        res = []
        for p, yy in zip(pmfs, y):
            lo, hi = fc.central_interval(p, cl)
            res.append((1.0 if lo <= yy <= hi else 0.0) - float(p[lo:hi + 1].sum()))
        # continuous: prefer scale closest to perfect interval calibration (residual -> 0)
        pen += abs(float(np.mean(res)))
    seeds = [f"{g}|{i}" for i, g in enumerate(gdate)]
    u = np.array([fc.randomized_pit(p, int(yy), k) for p, yy, k in zip(pmfs, y, seeds)])
    u = u[~np.isnan(u)]
    _, cd, _ = fc.clustered_pit_deviation(u, np.asarray(gdate))
    pen += cd
    return pen


def _score_candidate(kind, sc, before, blk, feat, cells, ms):
    """Return (pmfs, scored_rows) for a candidate on `blk` after fitting on `before`."""
    if kind == "C":
        return _cand_C(blk, before, ms), blk
    if kind == "D":
        return _cand_D(blk, cells, sc, ms), blk
    if kind in ("A", "B"):
        loss = "poisson" if kind == "A" else "squared_error"
        return _cand_AB(before, blk, feat, loss, ms)
    raise ValueError(kind)


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         features: str = typer.Option("", help="schema-v2 feature matrix (enables Candidates A/B in CI)"),
         out_dir: str = typer.Option("artifacts/p3"), holdout_dates: int = typer.Option(25)) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    df["_point"] = df["pmf_mean"].astype(float)
    mm = df["minutes_mean"].astype(float) if "minutes_mean" in df else pd.Series(0.0, index=df.index)
    df["_minbucket"] = pd.cut(mm, bins=[-1, 10, 20, 28, 100], labels=["m0", "m1", "m2", "m3"]).astype(str)
    s = df[df["stat"] == "reb"].copy()
    dates = np.sort(s["game_date"].unique()); hold = dates[-holdout_dates:]
    blocks = [list(b) for b in np.array_split(hold, 5)]
    ms = int(s["actual_outcome"].max()) + 6

    feat = None
    if features and Path(features).exists():
        feat = pd.read_parquet(features)
        for c in ("game_id", "player_id"):
            if c in feat.columns:
                feat[c] = feat[c].astype(s[c].dtype, errors="ignore")

    # Frozen candidate set (defined before scoring any block). A/B only when the feature matrix
    # is present (CI); C and D are always evaluable on the OOF.
    CANDS = {"C_perminute_minutes": ("C", None)}
    for sc in SCALE_GRID:
        CANDS[f"D_empirical_residual_s{int(sc*100)}"] = ("D", sc)
    if feat is not None:
        CANDS["A_hgb_poisson_mean"] = ("A", None)
        CANDS["B_hgb_squarederror_mean"] = ("B", None)

    ledger_rows, choices = [], []
    for k, block in enumerate(blocks):
        bstart = min(block)
        before = s[s["game_date"] < bstart]
        blk = s[s["game_date"].isin(set(block))]
        cells = dc.fit_residual_hist(before)
        # Selection uses ONLY earlier dates. When A/B (MLE-fit) are in play, use a nested
        # pre-block holdout (the last 5 before-dates) so fit and selection do not overlap;
        # otherwise (C/D-only, no in-sample fit) evaluate on the full pre-block set.
        if feat is not None:
            bdates = np.sort(before["game_date"].unique())
            sel_hold = set(bdates[-5:]); sel_fit = before[~before["game_date"].isin(sel_hold)]
            sel_eval = before[before["game_date"].isin(sel_hold)]
        else:
            sel_fit = sel_eval = before
        best_name, best_key = None, (float("inf"), float("inf"))
        for name, (kind, sc) in CANDS.items():
            try:
                pmfs_b, rows_b = _score_candidate(kind, sc, sel_fit, sel_eval, feat, cells, ms)
            except Exception as exc:  # a failing candidate never blocks the others
                typer.echo(f"[reb] block {k} candidate {name} skipped: {exc}")
                continue
            if not len(pmfs_b):
                continue
            by = rows_b["actual_outcome"].values
            bg = rows_b["game_date"].astype(str).values
            pen = _calib_penalty(pmfs_b, by, bg)
            crps = float(np.mean([fc.crps_discrete(p, int(yy)) for p, yy in zip(pmfs_b, by)]))
            if (pen, crps) < best_key:
                best_key, best_name = (pen, crps), name
        kind, sc = CANDS[best_name]
        choices.append({"block": k, "method": best_name, "scale": sc, "n_before": int(len(before)),
                        "n_candidates": len(CANDS)})
        blk_pmfs, blk_rows = _score_candidate(kind, sc, before, blk, feat, cells, ms)
        for (_, r), pmf in zip(blk_rows.iterrows(), blk_pmfs):
            rr = r.copy()
            rr["pmf_json"] = json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(pmf) if v > 1e-9})
            ledger_rows.append(rr)
    ledger = pd.DataFrame(ledger_rows)
    devcal = s[~s["game_date"].isin(set(hold))]
    base = {"crps": float(np.mean([fc.crps_discrete(dc.empirical_pmf(devcal["actual_outcome"].values, ms), int(y))
                                   for y in ledger["actual_outcome"]])),
            "log_score": float(np.mean([fc.log_score(dc.empirical_pmf(devcal["actual_outcome"].values, ms), int(y))
                                        for y in ledger["actual_outcome"]])),
            "matched_width_80": float(fc.matched_mass_width(dc.empirical_pmf(devcal["actual_outcome"].values, ms), 0.8))}
    r = fc.evaluate_stat(ledger, baseline=base)
    ledger_hash = hashlib.sha256(pd.util.hash_pandas_object(
        ledger[["game_id", "player_id", "stat", "pmf_json", "actual_outcome"]], index=False).values.tobytes()).hexdigest()[:16]
    sel_methods = [c["method"] for c in choices]
    winner = max(set(sel_methods), key=sel_methods.count)   # most-selected across blocks
    result = {"market": "reb", "forecast_allowed": bool(r.forecast_allowed),
              "method": winner, "block_methods": sel_methods,
              "scales": [c["scale"] for c in choices],
              "n": r.n, "n_dates": r.n_dates, "crps": round(r.crps, 4),
              "crps_vs_baseline": round(r.crps_vs_baseline, 4), "log_vs_baseline": round(r.log_vs_baseline, 4),
              "pit_ks_p": round(r.pit_ks_p, 4), "pit_clustered_dev": round(r.pit_clustered_dev, 4),
              "coverage": r.coverage, "bias": round(r.bias, 4), "sharpness_ratio": round(r.sharpness_ratio, 4),
              "ledger_hash": ledger_hash, "reasons": r.reasons, "block_choices": choices}
    (out / "p3_reb_repair_result.json").write_text(json.dumps(result, indent=2, default=str))
    typer.echo(f"[reb] forecast_allowed={r.forecast_allowed} winner={winner} "
               f"block_methods={sel_methods} "
               f"crps={r.crps:.3f} dCRPS={r.crps_vs_baseline:+.3f} clustered_PIT={r.pit_clustered_dev:.3f} "
               f"ledger_hash={ledger_hash}")
    if r.reasons:
        typer.echo(f"[reb] reasons: {r.reasons}")


if __name__ == "__main__":
    app()
