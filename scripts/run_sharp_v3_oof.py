"""Sharp v3 chronological OOF: participation + conditional minutes + Tier A direct-stat PMFs,
active-conditional, with distributional PIT diagnostics and exact same-time no-vig market
comparison. Development folds are scored first; the 2026 holdout is opened exactly once.

Fail-closed on private-input hash drift. Sportsbook data never enters the PURE feature path.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v3 import core as C

app = typer.Typer(add_completion=False)
OUT = C.REPO / "artifacts" / "sharp_v3"
MODELS = C.REPO / "artifacts" / "sharp_v3" / "fitted"
SEED = 20260730
_HGB = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.06, "min_samples_leaf": 40,
        "l2_regularization": 1.0, "random_state": SEED}


def _num(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _usable(Xtr: np.ndarray) -> np.ndarray:
    """Columns with >=2 distinct finite values in the training slice (HGB BinMapper needs this)."""
    ok = np.zeros(Xtr.shape[1], dtype=bool)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        fin = col[np.isfinite(col)]
        ok[j] = fin.size > 0 and np.unique(fin).size >= 2
    return ok


def _prep(train, eval_, feat):
    """Numeric train/eval matrices restricted to columns usable in the training slice.
    Fail-closed if any same-game label leaks into the estimator matrix."""
    leak = set(feat) & set(C.LABEL_COLS)
    if leak:
        raise ValueError(f"LEAKAGE GUARD: label columns in feature set: {sorted(leak)}")
    Xtr = _num(train, feat)
    mask = _usable(Xtr)
    used = [c for c, m in zip(feat, mask) if m]
    return Xtr[:, mask], _num(eval_, used), used


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _fit_participation(train, eval_, feat):
    Xtr, Xev, _used = _prep(train, eval_, feat)
    ytr = train["participation"].to_numpy(int)
    # chronological inner split for cross-fit calibration (earlier fit -> later calib)
    order = train["game_date"].rank(method="first").to_numpy()
    cut = np.quantile(order, 0.8)
    fitm = order <= cut
    clf = HistGradientBoostingClassifier(**_HGB).fit(Xtr[fitm], ytr[fitm])
    calib = IsotonicRegression(out_of_bounds="clip")
    calib.fit(clf.predict_proba(Xtr[~fitm])[:, 1], ytr[~fitm])
    full = HistGradientBoostingClassifier(**_HGB).fit(Xtr, ytr)
    p_cal = calib.predict(full.predict_proba(Xev)[:, 1])
    return p_cal, full, calib


def _fit_minutes(train, eval_):
    act = train[train["actual_minutes"] > 0]
    feat = C.stat_feature_contract("minutes", list(train.columns))
    Xtr, Xev, _ = _prep(act, eval_, feat)
    ytr = act["actual_minutes"].to_numpy(float)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, ytr)
    sd = float(np.clip(np.std(ytr - reg.predict(Xtr)), 2.0, 12.0))
    mu = np.clip(reg.predict(Xev), 0, 48)
    return mu, sd, reg


def _stat_pmfs(train, eval_, stat):
    feat = C.stat_feature_contract(stat, list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, Xev, used = _prep(act, eval_, feat)
    y = np.clip(act[stat].to_numpy(float), 0, None)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, y)
    mu_tr = np.clip(reg.predict(Xtr), 1e-4, None)
    r = C.residual_dispersion_r(y, mu_tr)
    mu_ev = np.clip(reg.predict(Xev), 1e-4, None)
    pmfs = [C.count_pmf(m, r, C.HARD_CAP[stat]) for m in mu_ev]
    return pmfs, mu_ev, r, reg, C.feature_schema_hash(used), len(used)


def _pure_metrics(pmfs, y, mu, rng):
    return {
        "nll": C.nll(pmfs, y), "crps": C.crps_discrete(pmfs, y),
        "mean_mse": float(np.mean((mu - y) ** 2)), "mean_mae": float(np.mean(np.abs(mu - y))),
        "pit_ks": float(_ks_uniform(C.pit_values(pmfs, y, rng))),
    }


def _ks_uniform(u):
    u = np.sort(np.asarray(u)); n = len(u)
    if n == 0:
        return float("nan")
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - u)))


def _market_compare(eval_, pmfs_by_stat, settled, stat, min_val_col="actual_outcome"):
    """Join eval predictions to exact same-time no-vig settled pairs; model vs market on identical
    non-push, non-void settled rows."""
    sp = settled[settled["prop"] == stat].copy()
    if sp.empty:
        return None
    idx_map = {(int(r.game_id), int(r.player_id)): i for i, r in
               zip(eval_.index, eval_[["game_id", "player_id"]].itertuples())}
    rows = []
    for r in sp.itertuples():
        key = (int(r.game_id), int(r.player_id))
        if key not in idx_map:
            continue
        pos = list(eval_.index).index(idx_map[key])
        pmf = pmfs_by_stat[pos]
        line = float(r.line)
        actual = getattr(r, min_val_col)
        if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
            continue
        if getattr(r, "did_play", True) in (False, 0):
            continue
        if float(actual).is_integer() and float(actual) == line:   # push -> excluded from binary
            continue
        mkt = C.no_vig_over(r.over_odds, r.under_odds)
        if not np.isfinite(mkt):
            continue
        model_over = C.prob_over(pmf, line)
        rows.append({"game_date": eval_.loc[idx_map[key], "game_date"], "outcome": int(actual > line),
                     "model_over": model_over, "market_over": mkt})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    eps = 1e-6
    d["mp"] = d["model_over"].clip(eps, 1 - eps); d["kp"] = d["market_over"].clip(eps, 1 - eps)
    ll_model = float(log_loss(d["outcome"], d["mp"], labels=[0, 1]))
    ll_market = float(log_loss(d["outcome"], d["kp"], labels=[0, 1]))
    delta = ll_model - ll_market
    # game-date clustered bootstrap for delta CI
    gd = d["game_date"].to_numpy()
    uniq = np.unique(gd)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.where(gd == g)[0] for g in samp])
        dd = d.iloc[mask]
        boots.append(float(log_loss(dd["outcome"], dd["mp"], labels=[0, 1]) -
                           log_loss(dd["outcome"], dd["kp"], labels=[0, 1])))
    ci_hi = float(np.quantile(boots, 0.95))    # one-sided 95% upper bound
    return {"stat": stat, "rows": len(d), "game_dates": len(uniq),
            "logloss_model": ll_model, "logloss_market": ll_market, "delta_logloss": delta,
            "delta_ci95_upper": ci_hi, "brier_model": float(brier_score_loss(d["outcome"], d["mp"])),
            "brier_market": float(brier_score_loss(d["outcome"], d["kp"])),
            "ece_model": C.ece(d["mp"].to_numpy(), d["outcome"].to_numpy()),
            "market_beaten": bool(delta < 0 and ci_hi < 0)}


@app.command()
def main(open_holdout: bool = typer.Option(False, "--open-holdout")) -> None:
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    _, df = C.load_verified()
    settled = pd.read_parquet(C.REPO / "data/atomic_quotes/settled_quote_pairs.parquet")
    rng = np.random.default_rng(SEED)
    part_feat = C.stat_feature_contract("participation", list(df.columns))

    folds = list(C.DEV_FOLDS) + ([C.HOLDOUT] if open_holdout else [])
    part_rows, min_rows, stat_rows, market_rows = [], [], [], []
    holdout_flag = OUT / "HOLDOUT_OPENED.flag"
    if open_holdout and holdout_flag.exists():
        raise SystemExit("FINAL HOLDOUT already opened once — refusing second use (single-use guard).")

    for fold in folds:
        tr_idx, ev_idx = C.split(df, fold)
        if len(tr_idx) < 500 or len(ev_idx) < 50:
            continue
        train, eval_ = df.loc[tr_idx], df.loc[ev_idx]
        # participation
        p_active, _, _ = _fit_participation(train, eval_, part_feat)
        ya = eval_["participation"].to_numpy(int)
        part_rows.append({"fold": fold.name, "rows": len(eval_),
                          "log_loss": float(log_loss(ya, np.clip(p_active, 1e-6, 1 - 1e-6), labels=[0, 1])),
                          "brier": float(brier_score_loss(ya, p_active)),
                          "ece": C.ece(p_active, ya), "base_rate": float(ya.mean()),
                          "pred_rate": float(p_active.mean()), "is_holdout": fold.is_holdout})
        # minutes (active eval rows)
        act_ev = eval_[eval_["actual_minutes"] > 0]
        mu_min, sd_min, _ = _fit_minutes(train, act_ev) if len(act_ev) else (np.array([]), 0, None)
        if len(act_ev):
            ym = act_ev["actual_minutes"].to_numpy(float)
            min_rows.append({"fold": fold.name, "rows": len(act_ev), "mae": float(np.mean(np.abs(mu_min - ym))),
                             "rmse": float(np.sqrt(np.mean((mu_min - ym) ** 2))), "resid_sd": sd_min,
                             "is_holdout": fold.is_holdout})
        # Tier A stat PMFs on active eval rows
        act_mask = eval_["actual_minutes"] > 0
        eval_act = eval_[act_mask]
        for stat in C.TIER_A:
            pmfs, mu_ev, r, _, fh, nfeat = _stat_pmfs(train, eval_act, stat)
            y = np.clip(eval_act[stat].to_numpy(float), 0, None)
            m = _pure_metrics(pmfs, y, mu_ev, rng)
            m.update({"fold": fold.name, "stat": stat, "rows": len(y), "dispersion_r": r,
                      "n_features": nfeat, "feature_hash": fh, "is_holdout": fold.is_holdout})
            stat_rows.append(m)
            mc = _market_compare(eval_act, pmfs, settled, stat)
            if mc:
                mc.update({"fold": fold.name, "is_holdout": fold.is_holdout})
                market_rows.append(mc)

    dev = pd.DataFrame([r for r in stat_rows if not r["is_holdout"]])
    hold = pd.DataFrame([r for r in stat_rows if r["is_holdout"]])
    _agg(dev).to_csv(OUT / "DEVELOPMENT_OOF_METRICS_BY_STAT.csv", index=False)
    if len(hold):
        _agg(hold).to_csv(OUT / "FINAL_HOLDOUT_METRICS_BY_STAT.csv", index=False)
        holdout_flag.write_text(f"opened {ts}\n")
    pd.DataFrame(part_rows).to_json(OUT / "PARTICIPATION_MODEL_REPORT.json", orient="records", indent=2)
    pd.DataFrame(min_rows).to_json(OUT / "MINUTES_MODEL_REPORT.json", orient="records", indent=2)
    pd.DataFrame(stat_rows).to_json(OUT / "COUNT_MODEL_REPORT.json", orient="records", indent=2)
    mdf = pd.DataFrame(market_rows)
    mdf.to_json(OUT / "MARKET_PMF_RECONSTRUCTION_AUDIT.json", orient="records", indent=2)

    # activation registry (dev-only certification; production behavior)
    reg = {}
    for stat in C.TIER_A:
        pure = "TRAINED_AND_EVALUATED" if stat in dev.get("stat", pd.Series()).values else "NO_VALID_LABELS"
        sub = mdf[(mdf["stat"] == stat) & (~mdf["is_holdout"])] if len(mdf) else pd.DataFrame()
        if len(sub):
            beaten = bool((sub["delta_logloss"].mean() < 0) and (sub["delta_ci95_upper"].mean() < 0))
            market = "CERTIFIED_RESIDUAL" if beaten else "MARKET_FALLBACK"
        else:
            market = "PURE_UNCERTIFIED"   # no exact market pairs for this stat
        reg[stat] = {"pure_track": pure, "market_track": market,
                     "production_behavior": ("certified_residual" if market == "CERTIFIED_RESIDUAL"
                                             else "exact_market_fallback" if market == "MARKET_FALLBACK"
                                             else "pure_uncertified_or_abstain")}
    for stat in ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]:
        reg[stat] = {"pure_track": "TRAINED_NOT_CERTIFIED", "market_track": "MARKET_FALLBACK",
                     "production_behavior": "derived_from_joint_or_market_fallback"}
    (OUT / "ACTIVATION_REGISTRY.json").write_text(json.dumps(
        {"artifact": "ACTIVATION_REGISTRY", "generated_at_utc": ts, "tier_A": reg,
         "note": "Certification requires delta_logloss<0 AND clustered 95% upper bound<0 on dev."},
        indent=2, default=str))

    (OUT / "FOLD_BOUNDARY_AUDIT.json").write_text(json.dumps(
        {"artifact": "FOLD_BOUNDARY_AUDIT", "frozen_from": "dates+coverage only",
         "dev_folds": [f.__dict__ for f in C.DEV_FOLDS], "holdout": C.HOLDOUT.__dict__,
         "holdout_opened": bool(open_holdout)}, indent=2, default=str))
    (OUT / "MODEL_LINEAGE.json").write_text(json.dumps(
        {"artifact": "MODEL_LINEAGE", "code_sha": _git_sha(), "design_hash":
         json.loads((OUT / "V3_FREEZE_MANIFEST.json").read_text())["modeling_design_v3_sha256"],
         "seed": SEED, "generated_at_utc": ts}, indent=2, default=str))

    typer.echo("============ SHARP V3 OOF ============")
    typer.echo(f"  dev stat-rows: {len(dev)}  holdout: {len(hold)}  market-compares: {len(mdf)}")
    if len(dev):
        typer.echo(_agg(dev)[["stat", "nll", "crps", "mean_mae", "pit_ks"]].to_string(index=False))
    if len(mdf):
        col = ["stat", "rows", "logloss_model", "logloss_market", "delta_logloss", "delta_ci95_upper", "market_beaten"]
        typer.echo("---- market comparison (dev) ----")
        typer.echo(mdf[~mdf["is_holdout"]][col].to_string(index=False))


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("stat").agg(
        rows=("rows", "sum"), nll=("nll", "mean"), crps=("crps", "mean"),
        mean_mse=("mean_mse", "mean"), mean_mae=("mean_mae", "mean"),
        pit_ks=("pit_ks", "mean"), dispersion_r=("dispersion_r", "mean"),
        n_features=("n_features", "max")).reset_index()
    return g


if __name__ == "__main__":
    app()
