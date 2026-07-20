"""Per-prop feature selection (leakage-safe, chronological).

For each stat, ranks features by permutation importance aggregated over expanding
chronological folds (train earlier -> score later), keeps the features that are stably
useful, and writes an optimal per-stat feature map. The map is consumed by
training.stat_feature_subset (OFF unless the map is supplied), so this never changes the
live global-feature model until a challenger retrain is validated and promoted.

Selection uses ONLY pre-holdout chronological folds; the final holdout is never touched here.

Output:
  config/prop_feature_map.json   {stat: [selected_features]}
  artifacts/feature_selection/prop_feature_importance.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)
STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    from wnba_props_model.features.feature_contract import MODEL_FEATURES, FORBIDDEN_MODEL_FEATURES
    forb = set(FORBIDDEN_MODEL_FEATURES)
    return [c for c in MODEL_FEATURES if c in df.columns and c not in forb]


def select_for_stat(df: pd.DataFrame, stat: str, cols: list[str], folds: list[tuple],
                    keep_frac: float, min_cols: int) -> tuple[list[str], dict]:
    """Return (selected_cols, importance_by_col) via chronological permutation importance."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    target = f"actual_{stat}"
    if target not in df.columns:
        return cols, {}
    imp_acc = {c: [] for c in cols}
    for tr_end, va_start, va_end in folds:
        tr = df[df["game_date"] < tr_end]
        va = df[(df["game_date"] >= va_start) & (df["game_date"] < va_end)]
        tr = tr[tr[target].notna()]; va = va[va[target].notna()]
        if len(tr) < 300 or len(va) < 100:
            continue
        Xtr = tr[cols].astype(float).fillna(0.0); ytr = tr[target].astype(float)
        Xva = va[cols].astype(float).fillna(0.0); yva = va[target].astype(float)
        m = HistGradientBoostingRegressor(loss="poisson" if (ytr >= 0).all() else "squared_error",
                                          max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
        m.fit(Xtr, np.clip(ytr, 0, None))
        r = permutation_importance(m, Xva, yva, n_repeats=5, random_state=0, scoring="neg_mean_squared_error")
        for c, mval in zip(cols, r.importances_mean):
            imp_acc[c].append(float(mval))
    # aggregate: mean importance across folds; a feature is "stable" if positive in the majority
    summary = {}
    for c in cols:
        vals = imp_acc[c]
        summary[c] = {"mean_importance": float(np.mean(vals)) if vals else 0.0,
                      "frac_positive": float(np.mean([v > 0 for v in vals])) if vals else 0.0,
                      "n_folds": len(vals)}
    ranked = sorted(cols, key=lambda c: summary[c]["mean_importance"], reverse=True)
    stable = [c for c in ranked if summary[c]["mean_importance"] > 0 and summary[c]["frac_positive"] >= 0.5]
    n_keep = max(min_cols, int(np.ceil(keep_frac * len(cols))), len(stable) and min(len(stable), len(cols)))
    selected = (stable[:n_keep] if len(stable) >= min_cols else ranked[:max(min_cols, n_keep)])
    return selected, summary


@app.command()
def main(features: str = typer.Option("data/processed/wnba_player_game_features_wide.parquet", "--features"),
         out_map: str = typer.Option("config/prop_feature_map.json", "--out-map"),
         report_dir: str = typer.Option("artifacts/feature_selection", "--report-dir"),
         holdout_dates: int = typer.Option(25, "--holdout-dates"),
         keep_frac: float = typer.Option(0.6, "--keep-frac"),
         min_cols: int = typer.Option(12, "--min-cols")) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    df = pd.read_parquet(features)
    df["game_date"] = pd.to_datetime(df["game_date"])
    cols = _feature_cols(df)
    typer.echo(f"[select] {len(df)} rows, {len(cols)} candidate features")
    dates = np.sort(df["game_date"].unique())
    dev = dates[:-holdout_dates] if len(dates) > holdout_dates else dates
    # expanding chronological folds over the DEV period only (never the holdout)
    n = len(dev)
    cuts = [dev[int(n * f)] for f in (0.5, 0.65, 0.8)]
    folds = []
    for i, c in enumerate(cuts):
        va_end = cuts[i + 1] if i + 1 < len(cuts) else (dev[-1] + np.timedelta64(1, "D"))
        folds.append((c, c, va_end))
    fmap, rows = {}, []
    for stat in STATS:
        sel, summary = select_for_stat(df, stat, cols, folds, keep_frac, min_cols)
        fmap[stat] = sel
        typer.echo(f"[select] {stat}: {len(sel)}/{len(cols)} features kept")
        for c, s in summary.items():
            rows.append({"stat": stat, "feature": c, "selected": c in set(sel), **s})
    Path(out_map).write_text(json.dumps(fmap, indent=2))
    rep = Path(report_dir); rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rep / "prop_feature_importance.csv", index=False)
    typer.echo(f"[select] wrote {out_map} + {rep}/prop_feature_importance.csv")


if __name__ == "__main__":
    app()
