"""Feature importance audit for WNBA PMF stat models (P5.1).

Loads trained StatRateModel artifacts, extracts HGB feature_importances_,
runs sklearn permutation importance on a held-out sample, and writes a ranked
report to artifacts/audits/feature_importance_report.json.

CLI usage:
    python scripts/audit_feature_importance.py
    python scripts/audit_feature_importance.py --model-dir artifacts/models/stage4_baseline
    python scripts/audit_feature_importance.py --n-permutations 30 --out my_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.inspection import permutation_importance

app = typer.Typer(pretty_exceptions_enable=False)

_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
_LOW_IMPORTANCE_THRESHOLD = 0.0001


def _load_models(model_dir: Path) -> dict[str, Any]:
    """Load stat_rate_models.joblib — returns dict of stat → model."""
    path = model_dir / "stat_rate_models.joblib"
    if not path.exists():
        typer.echo(f"[WARN] Model file not found: {path}")
        return {}
    return joblib.load(path)


def _load_features(
    features_wide: Path,
    manifest: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load feature table and derive model_feature_columns from manifest."""
    wide = pd.read_parquet(features_wide)
    model_cols: list[str] = []
    if manifest.exists():
        with open(manifest) as f:
            mf = json.load(f)
        model_cols = mf.get("model_feature_columns", [])
    if not model_cols:
        # Fallback: use numeric cols excluding forbidden
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES  # noqa: PLC0415
        model_cols = [
            c for c in wide.select_dtypes(include="number").columns
            if c not in FORBIDDEN_MODEL_FEATURES and not c.startswith("actual_")
        ]
    return wide, model_cols


def _audit_stat(
    stat: str,
    model: Any,
    wide: pd.DataFrame,
    model_cols: list[str],
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Return importance report for a single stat."""
    # Align feature matrix
    usable_cols = getattr(model, "_usable_cols", model_cols)
    available_cols = [c for c in usable_cols if c in wide.columns]
    X = wide[available_cols].copy()

    target_col = f"actual_{stat}"
    if target_col not in wide.columns:
        return {"stat": stat, "error": f"missing {target_col}"}

    # Filter to did_play rows for relevant assessment
    did_play_mask = wide.get("did_play", pd.Series(1, index=wide.index)).fillna(1).astype(bool)
    X_val = X[did_play_mask].fillna(0)
    y_val = wide.loc[did_play_mask, target_col].fillna(0)

    if len(X_val) < 50:
        return {"stat": stat, "error": f"too few rows ({len(X_val)})"}

    # Sample for speed (max 3000 rows)
    if len(X_val) > 3000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_val), 3000, replace=False)
        X_val = X_val.iloc[idx].reset_index(drop=True)
        y_val = y_val.iloc[idx].reset_index(drop=True)

    # HGB built-in importances (gain-based)
    builtin_importances: dict[str, float] = getattr(model, "_feature_importances", {})

    # Permutation importance (MAE as scorer)
    from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

    def _scorer(est: Any, X_: pd.DataFrame, y_: pd.Series) -> float:
        return -mean_absolute_error(y_, est.predict_mean(X_))

    try:
        perm_result = permutation_importance(
            type("_Wrapper", (), {
                "predict": lambda self, X_: model.predict_mean(pd.DataFrame(X_, columns=available_cols)),
            })(),
            X_val.values,
            y_val.values,
            n_repeats=n_permutations,
            random_state=seed,
            scoring="neg_mean_absolute_error",
        )
        perm_means = dict(zip(available_cols, perm_result.importances_mean.tolist()))
    except Exception as exc:
        perm_means = {c: float("nan") for c in available_cols}
        typer.echo(f"  [WARN] permutation importance failed for {stat}: {exc}")

    # Rank by built-in importance, fall back to permutation
    ranking = sorted(
        available_cols,
        key=lambda c: builtin_importances.get(c, perm_means.get(c, 0.0)),
        reverse=True,
    )

    low_importance = [
        c for c in available_cols
        if abs(perm_means.get(c, 0.0)) < _LOW_IMPORTANCE_THRESHOLD
    ]

    return {
        "stat": stat,
        "n_features": len(available_cols),
        "n_rows_evaluated": len(X_val),
        "top_features": [
            {
                "feature": c,
                "builtin_importance": round(builtin_importances.get(c, 0.0), 6),
                "permutation_importance": round(perm_means.get(c, float("nan")), 6),
            }
            for c in ranking[:30]
        ],
        "low_importance_features": low_importance,
        "n_low_importance": len(low_importance),
    }


@app.command()
def main(
    model_dir: Path = typer.Option(
        Path("artifacts/models/stage4_baseline"),
        "--model-dir", help="Directory with stat_rate_models.joblib"
    ),
    features_wide: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_wide.parquet"),
        "--features-wide"
    ),
    manifest: Path = typer.Option(
        Path("data/processed/feature_schema_manifest.json"),
        "--manifest"
    ),
    out: Path = typer.Option(
        Path("artifacts/audits/feature_importance_report.json"),
        "--out", help="Output report path"
    ),
    n_permutations: int = typer.Option(10, "--n-permutations"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Audit feature importances for all stat models."""
    typer.echo("=== Feature Importance Audit ===")
    t0 = time.time()

    stat_models = _load_models(model_dir)
    if not stat_models:
        typer.echo("[FAIL] No models loaded.", err=True)
        raise typer.Exit(1)

    wide, model_cols = _load_features(features_wide, manifest)
    typer.echo(f"  Loaded {len(wide):,} rows × {len(model_cols)} model features")

    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_dir": str(model_dir),
        "n_rows": len(wide),
        "n_model_features": len(model_cols),
        "stats": {},
    }

    for stat in _STATS:
        model = stat_models.get(stat)
        if model is None:
            typer.echo(f"  [SKIP] {stat}: model not found")
            continue
        typer.echo(f"  Auditing {stat}...")
        stat_report = _audit_stat(stat, model, wide, model_cols, n_permutations, seed)
        report["stats"][stat] = stat_report
        n_low = stat_report.get("n_low_importance", 0)
        top3 = [f["feature"] for f in stat_report.get("top_features", [])[:3]]
        typer.echo(f"    top-3: {top3}  |  low-importance: {n_low}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    typer.echo(f"\nReport → {out}  (elapsed: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    app()
