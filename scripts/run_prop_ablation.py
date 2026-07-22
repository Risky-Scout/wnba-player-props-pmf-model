"""Run the per-prop feature ablation (G0/S1-S5) and pick the optimal feature set per prop.

Leakage-safe: expanding chronological folds over a DEV period; the final holdout dates are
scored once for the winner. Each candidate's feature subset comes from
config/feature_ablation_maps_v1.json and is trained identically (HGB Poisson conditional
mean -> NB count PMF with method-of-moments dispersion), so differences isolate the FEATURES.

Selection metric priority: full-count NLL (primary), then RPS. AUC/accuracy are not used.
Outputs artifacts/feature_ablation/ablation_verdict.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation.diagnostics import pmf_nll, rps  # noqa: E402

app = typer.Typer(add_completion=False)
PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
_ID = {"game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season",
       "player_name", "team_abbreviation", "opponent_team_abbreviation", "home_away"}


def _nb_pmf(mu: float, size: float, ms: int) -> np.ndarray:
    from scipy.special import gammaln
    mu = max(float(mu), 1e-3); size = max(float(size), 1e-3)
    k = np.arange(ms); p = size / (size + mu)
    lp = gammaln(k + size) - gammaln(size) - gammaln(k + 1) + size * np.log(p) + k * np.log1p(-p)
    out = np.exp(lp); s = out.sum()
    return out / s if s > 0 else np.ones(ms) / ms


def _score_candidate(df, prop, cols, folds, ms):
    """Mean OOS NLL + RPS across folds for one candidate feature set."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    target = f"actual_{prop}"
    use = [c for c in cols if c in df.columns and c not in _ID and c != target]
    if len(use) < 4:
        return None
    nlls, rpss, n = [], [], 0
    for tr_end, va_end in folds:
        tr = df[df["game_date"] < tr_end]
        va = df[(df["game_date"] >= tr_end) & (df["game_date"] < va_end)]
        tr = tr[tr[target].notna()]; va = va[va[target].notna()]
        if len(tr) < 400 or len(va) < 80:
            continue
        Xtr = tr[use].astype(float).fillna(0.0).values
        ytr = np.clip(tr[target].astype(float).values, 0, None)
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.05,
                                          max_depth=3, random_state=0).fit(Xtr, ytr)
        mu_tr = np.clip(m.predict(Xtr), 1e-3, None)
        v = ((ytr - mu_tr) ** 2).mean(); mbar = mu_tr.mean()
        size = max(mbar ** 2 / max(v - mbar, 1e-3), 0.5)   # MoM NB dispersion (train only)
        mu_va = np.clip(m.predict(va[use].astype(float).fillna(0.0).values), 1e-3, None)
        for mu, y in zip(mu_va, va[target].astype(int).values):
            pmf = _nb_pmf(mu, size, ms)
            nlls.append(pmf_nll(pmf, int(y))); rpss.append(rps(pmf, int(y)))
        n += len(va)
    if not nlls:
        return None
    return {"nll": float(np.mean(nlls)), "rps": float(np.mean(rpss)), "n": n, "n_features": len(use)}


@app.command()
def main(features: str = typer.Option("data/processed/wnba_player_game_features_wide.parquet"),
         maps: str = typer.Option("config/feature_ablation_maps_v1.json"),
         out_dir: str = typer.Option("artifacts/feature_ablation"),
         holdout_dates: int = typer.Option(25)) -> None:
    df = pd.read_parquet(features)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712
    elif "actual_minutes" in df.columns:
        df = df[df["actual_minutes"].fillna(0) > 0]
    cand_maps = json.loads(Path(maps).read_text())["candidates"]

    dates = np.sort(df["game_date"].unique())
    dev = dates[:-holdout_dates]
    cuts = [dev[int(len(dev) * f)] for f in (0.55, 0.7, 0.85)]
    folds = [(cuts[i], cuts[i + 1] if i + 1 < len(cuts) else dev[-1] + np.timedelta64(1, "D"))
             for i in range(len(cuts))]

    verdict = {}
    for prop in PROPS:
        ms = int(df[f"actual_{prop}"].max()) + 8 if f"actual_{prop}" in df.columns else 60
        res = {}
        for cand, pm in cand_maps.items():
            cols = pm.get(prop, [])
            r = _score_candidate(df, prop, cols, folds, ms)
            if r:
                res[cand] = r
        if not res or "G0" not in res:
            verdict[prop] = {"status": "no_result"}; continue
        best = min(res, key=lambda c: (res[c]["nll"], res[c]["rps"]))
        g0 = res["G0"]
        verdict[prop] = {
            "winner": best, "n_features_winner": res[best]["n_features"],
            "g0_nll": round(g0["nll"], 4), "winner_nll": round(res[best]["nll"], 4),
            "nll_delta_vs_g0": round(res[best]["nll"] - g0["nll"], 4),
            "g0_rps": round(g0["rps"], 4), "winner_rps": round(res[best]["rps"], 4),
            "rps_delta_vs_g0": round(res[best]["rps"] - g0["rps"], 4),
            "improves_over_g0": bool(res[best]["nll"] < g0["nll"] - 1e-4),
            "all": {c: {k: round(v, 4) for k, v in r.items()} for c, r in res.items()},
        }
        typer.echo(f"[{prop}] winner={best} NLL {g0['nll']:.4f}->{res[best]['nll']:.4f} "
                   f"({res[best]['nll'] - g0['nll']:+.4f}) improves={verdict[prop]['improves_over_g0']}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    typer.echo(f"\n[ablation] verdict -> {out}/ablation_verdict.json")


if __name__ == "__main__":
    app()
