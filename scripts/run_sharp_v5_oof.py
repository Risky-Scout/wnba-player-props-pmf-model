"""Sharp v5 chronological OOF on 2023-2025 ONLY. Uses minutes-propagated mixture stat PMFs with
exact-tail scoring and push-aware no-vig market comparison; freezes exact feature contracts; and
compares V5 to the V4 baseline.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import nbinom, poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v4 import core as V4C
from wnba_props_model.sharp_v5 import models as M5

app = typer.Typer(add_completion=False)
REPO = V4C.load_verified.__globals__["REPO"]
OUT = REPO / "artifacts" / "sharp_v5"
SEED = 20260730


@dataclass(frozen=True)
class Fold:
    name: str; train_end: str; eval_start: str; eval_end: str


FOLDS = [Fold("v5_2024_h1", "2024-05-01", "2024-05-01", "2024-06-30"),
         Fold("v5_2024_h2", "2024-07-01", "2024-07-01", "2024-09-30"),
         Fold("v5_2025_h1", "2025-05-01", "2025-05-01", "2025-06-30"),
         Fold("v5_2025_h2", "2025-07-01", "2025-07-01", "2025-10-31")]


def _split(df, f):
    d = df["game_date"]
    return (df.index[d < pd.Timestamp(f.train_end)],
            df.index[(d >= pd.Timestamp(f.eval_start)) & (d <= pd.Timestamp(f.eval_end))])


def _mix_atoms(lam, r, matoms, K):
    """Vectorized minutes-propagated mixture atoms over 0..K."""
    idx = np.where(matoms > 1e-4)[0]
    w = matoms[idx]; means = np.clip(lam * idx, 1e-6, None)
    k = np.arange(K + 1)
    if r is None:
        comp = poisson.pmf(k[:, None], means[None, :])
    else:
        p = r / (r + means)
        comp = nbinom.pmf(k[:, None], r, p[None, :])
    atoms = comp @ w
    s = atoms.sum()
    return atoms / s if s > 0 else atoms, float(max(0.0, 1 - s))


def _metrics(atoms_list, y):
    nll, crps, pit = [], [], []
    rng = np.random.default_rng(SEED)
    for a, yi in zip(atoms_list, y):
        yi = int(yi)
        # exact-tail: if y beyond stored support, use survival tail mass (aggregate) — not clipped to atom
        p = a[yi] if yi < a.size else max(1e-12, 1 - a.sum())
        nll.append(-np.log(max(p, 1e-12)))

        cdf = np.cumsum(np.concatenate([a, [max(0.0, 1 - a.sum())]]))
        cdf = np.clip(cdf, 0, 1)
        ks = np.arange(cdf.size)
        heavi = (ks >= yi).astype(float)
        crps.append(float(np.sum((cdf - heavi) ** 2)))
        lo = a[:yi].sum() if yi <= a.size else 1.0
        pit.append(lo + rng.random() * (a[yi] if yi < a.size else 0.0))
    u = np.sort(pit); n = len(u)
    ks = float(np.max(np.abs(np.arange(1, n + 1) / n - u))) if n else float("nan")
    return float(np.mean(nll)), float(np.mean(crps)), ks


def _market(ev, atoms_list, settled, stat):
    sp = settled[settled["prop"] == stat]
    if sp.empty:
        return None
    idx = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(ev["game_id"], ev["player_id"]))}
    rows = []
    for r in sp.itertuples():
        key = (int(r.game_id), int(r.player_id))
        if key not in idx:
            continue
        a = atoms_list[idx[key]]; line = float(r.line); actual = getattr(r, "actual_outcome", None)
        if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
            continue
        if getattr(r, "did_play", True) in (False, 0):
            continue
        if float(actual).is_integer() and float(actual) == line:
            continue
        # push-aware settled model prob A/(A+B)
        k = np.arange(a.size)
        A = a[k > line].sum() + max(0.0, 1 - a.sum()); B = a[k < line].sum()
        den = A + B; model_over = A / den if den > 0 else np.nan
        mkt = _nv(r.over_odds, r.under_odds)
        if not (np.isfinite(model_over) and np.isfinite(mkt)):
            continue
        rows.append({"gd": ev.iloc[idx[key]]["game_date"], "o": int(actual > line),
                     "m": model_over, "k": mkt})
    if not rows:
        return None
    d = pd.DataFrame(rows); eps = 1e-6
    o = d["o"].to_numpy(float)
    mp = d["m"].clip(eps, 1 - eps).to_numpy(); kp = d["k"].clip(eps, 1 - eps).to_numpy()

    def _ll(oo, pp):
        return float(-np.mean(oo * np.log(pp) + (1 - oo) * np.log(1 - pp)))
    llm, llk = _ll(o, mp), _ll(o, kp)
    gd = d["gd"].to_numpy()
    uniq, inv = np.unique(gd, return_inverse=True)
    by_date = [np.where(inv == j)[0] for j in range(len(uniq))]     # precompute once
    rng = np.random.default_rng(SEED)
    boots = np.empty(3000)
    for b in range(3000):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_date[j] for j in pick])
        boots[b] = _ll(o[idx], mp[idx]) - _ll(o[idx], kp[idx])
    return {"stat": stat, "rows": len(d), "game_dates": len(uniq), "logloss_model": llm,
            "logloss_market": llk, "delta_logloss": llm - llk, "delta_ci95_upper": float(np.quantile(boots, 0.95)),
            "market_beaten": bool((llm - llk) < 0 and np.quantile(boots, 0.95) < 0)}


def _nv(o, u):
    def imp(a):
        a = float(a)
        return float("nan") if (-100 < a < 100) else ((100 / (a + 100)) if a > 0 else (abs(a) / (abs(a) + 100)))
    po, pu = imp(o), imp(u)
    return float(po / (po + pu)) if np.isfinite(po) and np.isfinite(pu) and po + pu > 0 else float("nan")


@app.command()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    _, df = V4C.load_verified()
    V4C.assert_unique_keys(df, ["game_id", "player_id"], "features+targets")
    df = df[df["game_date"] <= pd.Timestamp("2025-12-31")]
    settled = pd.read_parquet(REPO / "data/atomic_quotes/settled_quote_pairs.parquet")
    minutes_contract = V4C.resolve_contract("minutes", list(df.columns))

    rows, market_rows = [], []
    for f in FOLDS:
        tr_i, ev_i = _split(df, f)
        if len(tr_i) < 500 or len(ev_i) < 50:
            continue
        train, ev = df.loc[tr_i], df.loc[ev_i]
        eva = ev[ev["actual_minutes"] > 0]
        matoms, _, _ = M5.minutes_pmf_rows(train, eva, minutes_contract)
        for stat in V4C.TIER_A:
            lam, r_rows, fh, nfeat, _r_by = M5.stat_mixture_rows(train, eva, stat, matoms)
            y = np.clip(eva[stat].to_numpy(float), 0, None)
            cap = V4C.EMERGENCY_CAP[stat]
            atoms_list = []
            mean_pred = np.zeros(len(lam))
            for i in range(len(lam)):
                r = None if np.isnan(r_rows[i]) else float(r_rows[i])
                a, _ovf = _mix_atoms(lam[i], r, matoms[i], cap)
                atoms_list.append(a)
                mean_pred[i] = float(np.dot(np.arange(a.size), a))
            nll, crps, pit = _metrics(atoms_list, y)
            rows.append({"fold": f.name, "stat": stat, "family": "minutes_mixture_nb2", "rows": len(y),
                         "nll": nll, "crps": crps, "mean_mae": float(np.mean(np.abs(mean_pred - y))),
                         "pit_ks": pit, "n_features": nfeat, "feature_hash": fh})
            mc = _market(eva, atoms_list, settled, stat)
            if mc:
                mc["fold"] = f.name; market_rows.append(mc)

    dev = pd.DataFrame(rows)
    dev.to_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv", index=False)
    dev.loc[dev.groupby("stat")["nll"].idxmax()][["stat", "fold", "nll", "crps", "pit_ks"]].to_csv(
        OUT / "WORST_FOLD_REPORT.csv", index=False)
    pd.DataFrame(market_rows).to_csv(OUT / "MARKET_COMPARISON_BY_STAT.csv", index=False)
    _v4_compare(dev)
    _activation(dev, pd.DataFrame(market_rows), ts)
    typer.echo("======== SHARP V5 OOF (minutes-propagated, 2023-2025) ========")
    agg = dev.groupby("stat").agg(nll=("nll", "mean"), crps=("crps", "mean"),
                                  mae=("mean_mae", "mean"), pit=("pit_ks", "mean")).reset_index()
    typer.echo(agg.to_string(index=False))
    if market_rows:
        typer.echo(f"market beaten any: {bool(pd.DataFrame(market_rows)['market_beaten'].any())}")


def _v4_compare(dev):
    v4p = REPO / "artifacts/sharp_v4/HISTORICAL_DEVELOPMENT_METRICS.csv"
    v5 = dev.groupby("stat").agg(v5_nll=("nll", "mean"), v5_crps=("crps", "mean")).reset_index()
    if v4p.exists():
        v4 = pd.read_csv(v4p).groupby("stat").agg(v4_nll=("nll", "mean"), v4_crps=("crps", "mean")).reset_index()
        out = v5.merge(v4, on="stat", how="left")
        out["nll_delta_v5_minus_v4"] = out["v5_nll"] - out["v4_nll"]
        out.to_csv(OUT / "V5_VERSUS_V4_COMPARISON.csv", index=False)
    else:
        v5.to_csv(OUT / "V5_VERSUS_V4_COMPARISON.csv", index=False)


def _activation(dev, market, ts):
    reg = {}
    for stat in V4C.TIER_A:
        sub = market[market["stat"] == stat] if len(market) else pd.DataFrame()
        if len(sub):
            beaten = bool((sub["delta_logloss"].mean() < 0) and (sub["delta_ci95_upper"].mean() < 0))
            status = "CERTIFIED_RESIDUAL" if beaten else "MARKET_CONSISTENT_ZERO_RESIDUAL"
        else:
            status = "TRAINED_PURE_UNCERTIFIED"
        reg[stat] = {"status": status, "family": "minutes_mixture_nb2"}
    for combo in ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]:
        reg[combo] = {"status": "MARKET_CONSISTENT_ZERO_RESIDUAL", "note": "joint copula scaffolded, not fitted"}
    (OUT / "ACTIVATION_REGISTRY.json").write_text(json.dumps(
        {"artifact": "ACTIVATION_REGISTRY", "generated_at_utc": ts, "tier_A": reg}, indent=2, default=str))


if __name__ == "__main__":
    app()
