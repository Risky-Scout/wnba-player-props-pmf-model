"""Sharp v4 chronological OOF on 2023-2025 ONLY (the consumed 2026 V3 holdout is never used for
selection). Fits minutes-as-a-distribution, exact-tail hierarchical-dispersion direct PMFs, a
hurdle challenger for rare stats, and structural rebounds; selects family by OOF NLL; compares to
the exact same-time no-vig market and to the V3 baseline.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v4 import core as C  # noqa: E402
from wnba_props_model.sharp_v4 import models as M  # noqa: E402

app = typer.Typer(add_completion=False)
OUT = C.load_verified.__globals__["REPO"] / "artifacts" / "sharp_v4"
REPO = C.load_verified.__globals__["REPO"]
SEED = 20260730


@dataclass(frozen=True)
class Fold:
    name: str; train_end: str; eval_start: str; eval_end: str


V4_FOLDS = [
    Fold("v4_dev_2024_h1", "2024-05-01", "2024-05-01", "2024-06-30"),
    Fold("v4_dev_2024_h2", "2024-07-01", "2024-07-01", "2024-09-30"),
    Fold("v4_dev_2025_h1", "2025-05-01", "2025-05-01", "2025-06-30"),
    Fold("v4_dev_2025_h2", "2025-07-01", "2025-07-01", "2025-10-31"),
]


def _split(df, f):
    d = df["game_date"]
    return (df.index[d < pd.Timestamp(f.train_end)],
            df.index[(d >= pd.Timestamp(f.eval_start)) & (d <= pd.Timestamp(f.eval_end))])


def _component_targets():
    s = pd.read_parquet(REPO / "data/recovered_v2/wnba_player_game_stats.parquet")
    C.assert_unique_keys(s, ["game_id", "player_id"], "player_game_stats")
    return s[["game_id", "player_id", "oreb", "dreb", "fg3a", "fta"]].copy()  # fg3m already a target


@app.command()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    _, df = C.load_verified()
    C.assert_unique_keys(df, ["game_id", "player_id"], "features+targets")
    comp = _component_targets()
    before = len(df)
    df = C.safe_merge(df, comp, ["game_id", "player_id"], how="left")
    join_audit = {"artifact": "JOIN_CARDINALITY_AUDIT", "generated_at_utc": ts,
                  "features_targets_rows": before, "after_component_join": len(df),
                  "keys": ["game_id", "player_id"], "validate": "one_to_one", "duplicates": 0}
    (OUT / "JOIN_CARDINALITY_AUDIT.json").write_text(json.dumps(join_audit, indent=2))
    df = df[df["game_date"] <= pd.Timestamp("2025-12-31")]     # selection uses 2023-2025 ONLY
    settled = pd.read_parquet(REPO / "data/atomic_quotes/settled_quote_pairs.parquet")
    rng = np.random.default_rng(SEED)

    rows, part_rows, minr, market_rows, overflow_stat = [], [], [], [], []
    for f in V4_FOLDS:
        tr_i, ev_i = _split(df, f)
        if len(tr_i) < 500 or len(ev_i) < 50:
            continue
        train, ev = df.loc[tr_i], df.loc[ev_i]
        eva = ev[ev["actual_minutes"] > 0]
        # participation with calibration-family selection (earlier OOF)
        part_rows.append(_participation(train, ev))
        # minutes distribution
        mp, _, sdb = M.fit_minutes(train, eva)
        mnll, mcrps = M.minutes_metrics(mp, eva["actual_minutes"].to_numpy(float))
        minr.append({"fold": f.name, "rows": len(eva), "nll": mnll, "crps": mcrps, "sd_by_band": sdb})
        # Tier A stats: direct exact-tail hierarchical dispersion; family challengers
        for stat in C.TIER_A:
            y = np.clip(eva[stat].to_numpy(float), 0, None)
            cands = {}
            dpm, fh, nfeat, rband = M.fit_direct(train, eva, stat)
            cands["direct_nb2_exacttail"] = dpm
            if stat in ("stl", "blk", "turnover"):
                hpm, _, _ = M.fit_hurdle(train, eva, stat)
                cands["hurdle_nb2"] = hpm
            if stat == "reb":
                cands["structural_oreb_dreb"] = M.fit_structural_reb(train, eva)
            best_name, best_nll = None, np.inf
            for name, pmfs in cands.items():
                nll = C.nll_exact(pmfs, y)
                if nll < best_nll:
                    best_nll, best_name = nll, name
            pmfs = cands[best_name]
            rows.append({"fold": f.name, "stat": stat, "family": best_name, "rows": len(y),
                         "nll": best_nll, "crps": C.crps_exact(pmfs, y),
                         "mean_mae": float(np.mean(np.abs([p.prob_over(-0.5) * 0 + _mean(p) for p in pmfs] - y))),
                         "pit_ks": _pit(pmfs, y, rng),
                         "dispersion_by_band": {str(k): (round(v, 2) if v else None) for k, v in rband.items()},
                         "n_features": nfeat, "feature_hash": fh})
            overflow_stat.append({"stat": stat, "max_overflow": float(max(getattr(p, "overflow", 0.0) for p in pmfs))})
            mc = _market(eva, pmfs, settled, stat)
            if mc:
                mc["fold"] = f.name; market_rows.append(mc)

    dev = pd.DataFrame(rows)
    dev.to_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv", index=False)
    _worst(dev).to_csv(OUT / "WORST_BLOCK_REPORT.csv", index=False)
    pd.DataFrame(part_rows).to_json(OUT / "PARTICIPATION_REPORT.json", orient="records", indent=2)
    pd.DataFrame(minr).to_json(OUT / "MINUTES_DISTRIBUTION_REPORT.json", orient="records", indent=2)
    pd.DataFrame(market_rows).to_json(OUT / "MARKET_PMF_AUDIT.json", orient="records", indent=2)
    _v3_compare(dev).to_csv(OUT / "V3_BASELINE_COMPARISON.csv", index=False)
    _dispersion_report(dev, ts)
    _overflow_report(overflow_stat, ts)
    _activation(dev, pd.DataFrame(market_rows), ts)

    typer.echo("========= SHARP V4 OOF (2023-2025) =========")
    agg = dev.groupby(["stat", "family"]).agg(nll=("nll", "mean"), crps=("crps", "mean"),
                                              mae=("mean_mae", "mean"), pit=("pit_ks", "mean")).reset_index()
    typer.echo(agg.to_string(index=False))
    if market_rows:
        m = pd.DataFrame(market_rows)
        typer.echo("---- market (dev) beaten any: %s ----" % bool(m["market_beaten"].any()))


def _mean(p):
    a = getattr(p, "atoms", None)
    if a is not None:
        return float(np.dot(np.arange(a.size), a))
    return float(p.mu)


def _pit(pmfs, y, rng):
    u = []
    for p, yi in zip(pmfs, y):
        yi = int(yi)
        lo = p.cdf(yi - 1)
        px = float(np.exp(p.logpmf(yi)))
        u.append(lo + rng.random() * px)
    u = np.sort(np.asarray(u)); n = len(u)
    return float(np.max(np.abs(np.arange(1, n + 1) / n - u))) if n else float("nan")


def _participation(train, ev):
    feat = C.resolve_contract("participation", list(train.columns))
    Xtr, Xev, _ = M.prep(train, ev, feat)
    ytr = train["participation"].to_numpy(int); yev = ev["participation"].to_numpy(int)
    order = train["game_date"].rank(method="first").to_numpy(); cut = np.quantile(order, 0.8)
    fitm = order <= cut
    clf = HistGradientBoostingClassifier(**M._HGBC).fit(Xtr[fitm], ytr[fitm])
    raw_c = clf.predict_proba(Xtr[~fitm])[:, 1]; yc = ytr[~fitm]
    # compare calibration families on the held-out calibration slice
    fams = {}
    fams["isotonic"] = IsotonicRegression(out_of_bounds="clip").fit(raw_c, yc)
    fams["platt"] = LogisticRegression().fit(_logit(raw_c).reshape(-1, 1), yc)
    best, best_ll = None, np.inf
    for name, cal in fams.items():
        pc = _apply(cal, name, raw_c)
        ll = log_loss(yc, np.clip(pc, 1e-6, 1 - 1e-6), labels=[0, 1])
        if ll < best_ll:
            best_ll, best = ll, name
    full = HistGradientBoostingClassifier(**M._HGBC).fit(Xtr, ytr)
    p = _apply(fams[best], best, full.predict_proba(Xev)[:, 1])
    return {"fold": "agg", "calibrator": best, "log_loss": float(log_loss(yev, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(yev, p)), "ece": C.ece(p, yev),
            "base_rate": float(yev.mean()), "pred_rate": float(p.mean())}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _apply(cal, name, raw):
    if name == "isotonic":
        return cal.predict(raw)
    return cal.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def _market(ev, pmfs, settled, stat):
    sp = settled[settled["prop"] == stat]
    if sp.empty:
        return None
    idx = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(ev["game_id"], ev["player_id"]))}
    rows = []
    for r in sp.itertuples():
        key = (int(r.game_id), int(r.player_id))
        if key not in idx:
            continue
        pmf = pmfs[idx[key]]; line = float(r.line); actual = getattr(r, "actual_outcome", None)
        if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
            continue
        if getattr(r, "did_play", True) in (False, 0):
            continue
        if float(actual).is_integer() and float(actual) == line:
            continue
        mkt = C.no_vig_over(r.over_odds, r.under_odds)
        if not np.isfinite(mkt):
            continue
        rows.append({"gd": ev.iloc[idx[key]]["game_date"], "o": int(actual > line),
                     "m": pmf.prob_over(line), "k": mkt})
    if not rows:
        return None
    d = pd.DataFrame(rows); eps = 1e-6
    d["mp"] = d["m"].clip(eps, 1 - eps); d["kp"] = d["k"].clip(eps, 1 - eps)
    llm = float(log_loss(d["o"], d["mp"], labels=[0, 1])); llk = float(log_loss(d["o"], d["kp"], labels=[0, 1]))
    gd = d["gd"].to_numpy(); uniq = np.unique(gd); rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        s = rng.choice(uniq, len(uniq), replace=True)
        mask = np.concatenate([np.where(gd == g)[0] for g in s])
        dd = d.iloc[mask]
        boots.append(log_loss(dd["o"], dd["mp"], labels=[0, 1]) - log_loss(dd["o"], dd["kp"], labels=[0, 1]))
    return {"stat": stat, "rows": len(d), "game_dates": int(len(uniq)), "logloss_model": llm,
            "logloss_market": llk, "delta_logloss": llm - llk, "delta_ci95_upper": float(np.quantile(boots, 0.95)),
            "market_beaten": bool((llm - llk) < 0 and np.quantile(boots, 0.95) < 0)}


def _worst(dev):
    return dev.loc[dev.groupby("stat")["nll"].idxmax()][["stat", "fold", "family", "nll", "crps", "pit_ks"]]


def _v3_compare(dev):
    v3p = REPO / "artifacts/sharp_v3/DEVELOPMENT_OOF_METRICS_BY_STAT.csv"
    v4 = dev.groupby("stat").agg(v4_nll=("nll", "mean"), v4_crps=("crps", "mean"),
                                 v4_family=("family", lambda s: s.mode().iloc[0])).reset_index()
    if v3p.exists():
        v3 = pd.read_csv(v3p)[["stat", "nll", "crps"]].rename(columns={"nll": "v3_nll", "crps": "v3_crps"})
        out = v4.merge(v3, on="stat", how="left")
        out["nll_delta_v4_minus_v3"] = out["v4_nll"] - out["v3_nll"]
        return out
    return v4


def _dispersion_report(dev, ts):
    (OUT / "DISPERSION_REPORT.json").write_text(json.dumps({
        "artifact": "DISPERSION_REPORT", "generated_at_utc": ts,
        "method": "hierarchical partial-pool player-role dispersion in phi (role band from lagged season minutes)",
        "by_stat_family": dev.groupby("stat")["family"].agg(lambda s: s.mode().iloc[0]).to_dict(),
        "note": "one global scalar replaced by per-role shrunk NB2 dispersion; hurdle challenger for rare stats."},
        indent=2, default=str))


def _overflow_report(overflow_stat, ts):
    o = pd.DataFrame(overflow_stat).groupby("stat")["max_overflow"].max().to_dict()
    (OUT / "PMF_SUPPORT_AND_OVERFLOW_AUDIT.json").write_text(json.dumps({
        "artifact": "PMF_SUPPORT_AND_OVERFLOW_AUDIT", "generated_at_utc": ts, "tail_tolerance": 1e-6,
        "max_overflow_by_stat": o, "outcome_clipping": "PROHIBITED — exact analytic tail used in NLL/CRPS/pricing",
        "all_within_tol": bool(all(v < 1e-6 for v in o.values()))}, indent=2, default=str))


def _activation(dev, market, ts):
    reg = {}
    for stat in C.TIER_A:
        trained = stat in dev["stat"].values
        sub = market[market["stat"] == stat] if len(market) else pd.DataFrame()
        if len(sub):
            beaten = bool((sub["delta_logloss"].mean() < 0) and (sub["delta_ci95_upper"].mean() < 0))
            status = "CERTIFIED_RESIDUAL" if beaten else "MARKET_CONSISTENT_ZERO_RESIDUAL"
        else:
            status = "TRAINED_PURE_UNCERTIFIED"
        reg[stat] = {"status": status, "trained": trained,
                     "family": dev[dev.stat == stat]["family"].mode().iloc[0] if trained else None}
    for combo in ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]:
        reg[combo] = {"status": "MARKET_CONSISTENT_ZERO_RESIDUAL",
                      "note": "derive from joint sim (copula scaffolded, not certified) or market fallback"}
    (OUT / "ACTIVATION_REGISTRY.json").write_text(json.dumps({
        "artifact": "ACTIVATION_REGISTRY", "generated_at_utc": ts, "tier_A": reg,
        "prospective": "PROSPECTIVE_EVIDENCE_ACCUMULATING (V4 prospective holdout begins post-freeze)"},
        indent=2, default=str))


if __name__ == "__main__":
    app()
