"""Compute TreeSHAP values and write explainability parquet (blueprint §7).

Loads the production HGB models from artifacts/models/stage4_baseline/,
runs TreeSHAP on today's feature matrix, and writes:

  artifacts/explainability/shap_{game_date}.parquet
  artifacts/explainability/shap_{game_date}_top5.json  (top-5 per player/stat)

The top-5 SHAP values are embedded into the pre-game output JSON by
format_pregame_output.py (via the output_schema.py build_player_record helper).

Feature human-readable names (blueprint §7.3):
  usage_pct            → "usage_pct"
  season_avg_pts       → "season_avg_pts"
  projected_minutes    → "projected_minutes"
  opponent_def_rating  → "opponent_def_rating"
  pace_differential    → "pace_differential"
  rest_advantage       → "rest_advantage"
  home_court           → "home_court"
  recent_form_pts_5g   → "recent_form_pts_5g"
  utm_usage_boost      → "utm_usage_boost"
  fg3_pct              → "fg3_pct"

Usage:
    python scripts/compute_shap_explainability.py \\
        --game-date 2026-06-25 \\
        --features-wide data/processed/wnba_player_game_features_wide.parquet \\
        --model-dir artifacts/models/stage4_baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

STATS_TO_EXPLAIN = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
TOP_N = 5

# Human-readable feature labels (blueprint §7.3)
_FEATURE_LABELS: dict[str, str] = {
    "usage_pct": "usage_pct",
    "season_avg_pts": "season_avg_pts",
    "season_avg_reb": "season_avg_reb",
    "season_avg_ast": "season_avg_ast",
    "projected_minutes": "projected_minutes",
    "opp_def_rtg": "opponent_def_rating",
    "opp_def_rating": "opponent_def_rating",
    "pace_diff": "pace_differential",
    "pace_differential": "pace_differential",
    "rest_advantage": "rest_advantage",
    "home_court": "home_court",
    "is_home": "home_court",
    "recent_pts_5g": "recent_form_pts_5g",
    "rolling_pts_5": "recent_form_pts_5g",
    "utm_usage_boost": "utm_usage_boost",
    "fg3_pct": "fg3_pct",
    "fg3_pct_season": "fg3_pct",
}


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="ISO game date YYYY-MM-DD."),
    features_wide: str = typer.Option(
        "data/processed/wnba_player_game_features_wide.parquet",
        "--features-wide",
    ),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline", "--model-dir"),
    pmfs_parquet: str = typer.Option(
        "", "--pmfs-parquet",
        help="PMF slate to join player_id/game_id. Auto-detected if not set.",
    ),
    out_dir: str = typer.Option("artifacts/explainability", "--out-dir"),
    max_players: int = typer.Option(
        200, "--max-players",
        help="Maximum players to compute SHAP for (performance guard).",
    ),
) -> None:
    """Compute TreeSHAP explainability values for today's projections."""
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        typer.echo("[ERROR] shap not installed. Run: pip install shap", err=True)
        raise typer.Exit(1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix ───────────────────────────────────────────────
    feat_path = Path(features_wide)
    if not feat_path.exists():
        typer.echo(f"[ERROR] Features file not found: {feat_path}", err=True)
        raise typer.Exit(1)

    feat_df = pd.read_parquet(feat_path)
    typer.echo(f"Loaded {len(feat_df):,} feature rows from {feat_path}")

    # Filter to game_date if column exists
    if "game_date" in feat_df.columns:
        feat_df["game_date_str"] = feat_df["game_date"].astype(str).str[:10]
        today_df = feat_df[feat_df["game_date_str"] == game_date].copy()
        if today_df.empty:
            typer.echo(f"[WARN] No features for {game_date} — using full dataset for SHAP baseline.", err=True)
            today_df = feat_df.tail(200).copy()
    else:
        today_df = feat_df.tail(200).copy()

    today_df = today_df.head(max_players)
    typer.echo(f"Computing SHAP for {len(today_df)} rows")

    # ── Load models ───────────────────────────────────────────────────────
    models = _load_models(model_dir)
    if not models:
        typer.echo(f"[ERROR] No models found in {model_dir}", err=True)
        raise typer.Exit(1)

    # ── Load feature manifest ─────────────────────────────────────────────
    feat_cols = _load_feature_cols(model_dir)

    # ── Compute SHAP per stat ─────────────────────────────────────────────
    all_shap_rows: list[dict] = []
    top5_out: dict = {}

    for stat, model in models.items():
        if stat not in STATS_TO_EXPLAIN:
            continue
        typer.echo(f"  SHAP for stat={stat} ...")

        # Prepare feature matrix
        X = _prepare_X(today_df, feat_cols)
        if X is None or X.shape[1] == 0:
            typer.echo(f"  [WARN] Empty feature matrix for {stat} — skipping.", err=True)
            continue

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                sv = shap_values[0]
            else:
                sv = shap_values
        except Exception as exc:
            typer.echo(f"  [WARN] SHAP failed for {stat}: {exc}", err=True)
            continue

        actual_cols = feat_cols[:X.shape[1]] if len(feat_cols) >= X.shape[1] else list(range(X.shape[1]))

        for row_idx, row in enumerate(today_df.itertuples()):
            pid = int(getattr(row, "player_id", row_idx))
            gid = int(getattr(row, "game_id", 0))
            sv_row = sv[row_idx] if row_idx < len(sv) else np.zeros(X.shape[1])

            # Top-5 features by |SHAP|
            top_idx = np.argsort(np.abs(sv_row))[::-1][:TOP_N]
            top5 = [
                {
                    "feature": _humanize(str(actual_cols[i])),
                    "shap_value": round(float(sv_row[i]), 4),
                }
                for i in top_idx
                if i < len(actual_cols)
            ]

            for feat_i, col in enumerate(actual_cols):
                if feat_i >= len(sv_row):
                    break
                all_shap_rows.append({
                    "game_date": game_date,
                    "player_id": pid,
                    "game_id": gid,
                    "stat": stat,
                    "feature": _humanize(str(col)),
                    "feature_raw": str(col),
                    "shap_value": round(float(sv_row[feat_i]), 4),
                })

            key = f"{gid}_{pid}_{stat}"
            top5_out[key] = top5

    # ── Write outputs ─────────────────────────────────────────────────────
    if all_shap_rows:
        shap_df = pd.DataFrame(all_shap_rows)
        shap_path = out / f"shap_{game_date}.parquet"
        shap_df.to_parquet(shap_path, index=False)
        typer.echo(f"Wrote {len(shap_df):,} SHAP rows → {shap_path}")

    top5_path = out / f"shap_{game_date}_top5.json"
    top5_path.write_text(json.dumps(top5_out, indent=2))
    typer.echo(f"Wrote top-5 SHAP → {top5_path}")
    typer.echo(f"Done: {len(top5_out)} player/stat combinations computed")


def _load_models(model_dir: str) -> dict:
    """Load HGB models from joblib files in model_dir."""
    import joblib  # noqa: PLC0415

    md = Path(model_dir)
    models: dict = {}
    for stat in STATS_TO_EXPLAIN:
        for fname in [f"hgb_{stat}.pkl", f"model_{stat}.pkl", f"{stat}_model.pkl", "model.pkl"]:
            p = md / fname
            if p.exists():
                try:
                    models[stat] = joblib.load(p)
                    typer.echo(f"  Loaded model for {stat} from {p.name}")
                    break
                except Exception:
                    pass

    if not models:
        # Try to load a single multi-output model
        for fname in ["model.pkl", "stage4_model.pkl", "hgb_model.pkl"]:
            p = md / fname
            if p.exists():
                try:
                    m = joblib.load(p)
                    # Assign the same model for all stats as fallback
                    for s in STATS_TO_EXPLAIN:
                        models[s] = m
                    typer.echo(f"  Loaded shared model from {p.name}")
                    break
                except Exception:
                    pass

    return models


def _load_feature_cols(model_dir: str) -> list[str]:
    md = Path(model_dir)
    for fname in ["feature_manifest.json", "feature_schema_manifest.json"]:
        p = md / fname
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    return [str(c) for c in data]
                if isinstance(data, dict):
                    cols = data.get("model_cols") or data.get("feature_columns") or data.get("columns") or []
                    return [str(c) for c in cols]
            except Exception:
                pass
    return []


def _prepare_X(df: pd.DataFrame, feat_cols: list[str]) -> "np.ndarray | None":
    """Build numeric feature matrix from DataFrame."""
    available = [c for c in feat_cols if c in df.columns]
    if not available:
        # Fallback: use all numeric columns except known ID/meta columns
        skip = {"player_id", "game_id", "team_id", "opponent_team_id",
                "season", "game_date", "game_date_str"}
        available = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]

    if not available:
        return None

    X = df[available].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return X.values.astype(np.float32)


def _humanize(feat: str) -> str:
    return _FEATURE_LABELS.get(feat, feat)


if __name__ == "__main__":
    app()
