"""Hyperparameter optimization for WNBA PMF stat and minutes models (P3.2).

Uses Optuna to minimize OOF Ignorance Score over 3-fold chronological splits.
Each fold trains on the earliest N% of data and validates on the next 20%.

CLI usage:
    python scripts/tune_hyperparams.py --stat pts --n-trials 75
    python scripts/tune_hyperparams.py --stat all --n-trials 50 --out artifacts/hyperparams/best_params_all.json

After tuning, set ``use_tuned_hyperparams: true`` and
``hgb_hyperparams_path: artifacts/hyperparams/best_params_all.json`` in
config/model/stage4_baseline.yaml to load these parameters at training time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_enable=False)

_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "minutes"]
_FOLDS = [(0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]


def _ignorance_score(pmf: np.ndarray, actual: int) -> float:
    """Log loss (ignorance score) for a single observation."""
    k = min(int(actual), len(pmf) - 1)
    p = max(pmf[k], 1e-12)
    return -np.log2(p)


def _oof_is_for_params(
    stat: str,
    wide_df: pd.DataFrame,
    model_cols: list[str],
    params: dict[str, Any],
    seed: int,
) -> float:
    """Compute mean OOF Ignorance Score for given HGB hyperparams."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415
    from wnba_props_model.models.pmf_utils import negbinom_pmf_batch, dispersion_from_moments  # noqa: PLC0415

    target_col = f"actual_{stat}"
    if target_col not in wide_df.columns:
        return float("inf")

    # Filter to rows with actual data
    df = wide_df.dropna(subset=[target_col]).copy()
    if "did_play" in df.columns:
        df = df[df["did_play"].fillna(1).astype(bool)]
    if len(df) < 200:
        return float("inf")

    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)

    available_cols = [c for c in model_cols if c in df.columns]
    X_all = df[available_cols].copy()
    y_all = df[target_col].values

    is_scores: list[float] = []
    for train_end_pct, val_end_pct in _FOLDS:
        train_end = int(n * train_end_pct)
        val_end   = int(n * val_end_pct)
        X_train = X_all.iloc[:train_end].fillna(0)
        y_train = y_all[:train_end]
        X_val   = X_all.iloc[train_end:val_end].fillna(0)
        y_val   = y_all[train_end:val_end]

        if len(X_train) < 100 or len(X_val) < 20:
            continue

        mdl = HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 200),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            learning_rate=params.get("learning_rate", 0.1),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.0),
            random_state=seed,
        )
        mdl.fit(X_train, y_train)
        preds = np.clip(mdl.predict(X_val), 0.01, None)

        # Estimate dispersion from training residuals
        resid_mean = float(np.mean(y_train))
        resid_var  = float(np.var(y_train))
        r = dispersion_from_moments(resid_mean, resid_var)

        pmf_cap = {"pts": 60, "reb": 30, "ast": 25, "fg3m": 15,
                   "stl": 10, "blk": 10, "turnover": 12, "minutes": 50}.get(stat, 30)
        for mu, actual in zip(preds, y_val):
            if r is not None:
                pmf = negbinom_pmf_batch(np.array([mu]), r, pmf_cap)[0]
            else:
                from wnba_props_model.models.pmf_utils import poisson_pmf_batch  # noqa: PLC0415
                pmf = poisson_pmf_batch(np.array([mu]), pmf_cap)[0]
            is_scores.append(_ignorance_score(pmf, int(actual)))

    return float(np.mean(is_scores)) if is_scores else float("inf")


def _tune_stat(
    stat: str,
    wide_df: pd.DataFrame,
    model_cols: list[str],
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Run Optuna study for a single stat. Returns best params dict."""
    try:
        import optuna  # noqa: PLC0415
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        typer.echo("[WARN] optuna not installed; returning defaults. Run: pip install optuna")
        return {
            "max_iter": 200, "max_leaf_nodes": 31,
            "learning_rate": 0.1, "min_samples_leaf": 20, "l2_regularization": 0.0,
        }

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_iter":          trial.suggest_int("max_iter", 150, 400),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 63),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
        }
        return _oof_is_for_params(stat, wide_df, model_cols, params, seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    typer.echo(f"    best IS={study.best_value:.4f}  params={best}")
    return best


@app.command()
def main(
    stat: str = typer.Option("all", "--stat", help="Stat to tune (or 'all')"),
    features_wide: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_wide.parquet"),
        "--features-wide"
    ),
    manifest: Path = typer.Option(
        Path("data/processed/feature_schema_manifest.json"),
        "--manifest"
    ),
    out: Path = typer.Option(
        Path("artifacts/hyperparams/best_params_all.json"),
        "--out"
    ),
    n_trials: int = typer.Option(75, "--n-trials"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Tune HGB hyperparameters via Optuna chronological CV."""
    typer.echo("=== Hyperparameter Tuning (Optuna) ===")
    t0 = time.time()

    wide = pd.read_parquet(features_wide)
    if "game_date" in wide.columns:
        wide["game_date"] = pd.to_datetime(wide["game_date"], utc=True, errors="coerce")

    model_cols: list[str] = []
    if manifest.exists():
        with open(manifest) as f:
            model_cols = json.load(f).get("model_feature_columns", [])
    if not model_cols:
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES  # noqa: PLC0415
        model_cols = [
            c for c in wide.select_dtypes(include="number").columns
            if c not in FORBIDDEN_MODEL_FEATURES and not c.startswith("actual_")
        ]

    stats_to_tune = _STATS if stat == "all" else [stat]
    best_params: dict[str, Any] = {}

    # Load existing params if out file exists (allows incremental updates)
    if out.exists():
        with open(out) as f:
            best_params = json.load(f)

    for s in stats_to_tune:
        typer.echo(f"\nTuning {s} ({n_trials} trials)...")
        result = _tune_stat(s, wide, model_cols, n_trials, seed)
        best_params[s] = result

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(best_params, f, indent=2)
    typer.echo(f"\nSaved → {out}  (elapsed: {time.time()-t0:.1f}s)")
    typer.echo("Set use_tuned_hyperparams: true in config/model/stage4_baseline.yaml to activate.")


if __name__ == "__main__":
    app()
