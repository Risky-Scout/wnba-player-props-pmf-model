"""Explain what drives a specific player's projection.

Outputs a plain-English driver explanation with top feature contributions,
minutes change flags, and risk indicators.

Usage:
    # Explain a specific player + stat
    python scripts/explain_projection.py \
        --player-id 341 \
        --stat pts \
        --game-date 2026-06-16

    # Explain all stats for a player
    python scripts/explain_projection.py \
        --player-id 341 \
        --game-date 2026-06-16

    # Explain all players on a team
    python scripts/explain_projection.py \
        --team PHX \
        --game-date 2026-06-16 \
        --out artifacts/explanations/PHX_2026-06-16.json

    # Explain what drives a player's projected MINUTES specifically
    python scripts/explain_projection.py explain-minutes \
        --player-id 341 \
        --game-date 2026-06-16

    # Show what changed vs. yesterday's projection and why
    python scripts/explain_projection.py explain-change \
        --player-id 341 \
        --game-date 2026-06-16 \
        --reference-date 2026-06-15
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.evaluation.explain import build_explanations

app = typer.Typer(add_completion=False)

_DEFAULT_FEATURES = "data/processed/wnba_player_game_features_wide.parquet"
_DEFAULT_MODEL_DIR = "artifacts/models/stage4_baseline"
_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

# Human-readable labels for feature names
_FEATURE_LABELS: dict[str, str] = {
    "player_minutes_mean_l5": "Minutes (L5 avg)",
    "player_minutes_mean_l10": "Minutes (L10 avg)",
    "player_rest_days": "Days since last game",
    "player_back_to_back_flag": "Back-to-back game",
    "player_dnp_streak_l3": "Recent DNP streak",
    "team_pace": "Team pace",
    "opp_pts_allowed_mean_l10": "Opp points allowed (L10)",
    "opp_reb_allowed_mean_l10": "Opp rebounds allowed (L10)",
    "is_home": "Home game advantage",
    "player_mins_home_mean_l5": "Home minutes (L5)",
    "player_mins_away_mean_l5": "Away minutes (L5)",
    "player_injured_l1": "Player missed last game",
    "vacated_minutes_l1": "Teammate vacated minutes (L1)",
    "player_heavy_minutes_b2b": "Heavy minutes in prior B2B",
    "player_usage_proxy_l5": "Usage proxy (L5)",
}


def _load_pmfs(pmfs_path: str | None, game_date: str) -> pd.DataFrame | None:
    """Load PMF parquet, searching common locations if not specified."""
    if pmfs_path and Path(pmfs_path).exists():
        return pd.read_parquet(pmfs_path)

    candidates = [
        Path(f"deliveries/next_game/full_pmfs_wide.parquet"),
        Path(f"deliveries/next_game/player_projections_{game_date}.parquet"),
        Path(f"deliveries/today/full_pmfs_wide.parquet"),
    ]
    for c in candidates:
        if c.exists():
            return pd.read_parquet(c)
    return None


def _get_minutes_feature_importances(model_dir: str) -> pd.DataFrame | None:
    """Extract and rank minutes model feature importances.

    Returns DataFrame with columns: feature, importance, rank
    """
    import joblib

    model_path = Path(model_dir) / "minutes_model.joblib"
    if not model_path.exists():
        model_path = Path(model_dir) / "minutes_model.pkl"
    if not model_path.exists():
        return None

    try:
        minutes_model_obj = joblib.load(model_path)
        # MinutesModel wraps an HGB regressor; get the underlying sklearn model
        if hasattr(minutes_model_obj, "_model"):
            hgb = minutes_model_obj._model
        elif hasattr(minutes_model_obj, "model"):
            hgb = minutes_model_obj.model
        else:
            hgb = minutes_model_obj

        if not hasattr(hgb, "feature_importances_"):
            return None

        importances = hgb.feature_importances_
        feature_names = getattr(hgb, "feature_names_in_", None)
        if feature_names is None:
            # Try to get from model metadata
            meta_path = Path(model_dir) / "model_feature_cols.json"
            if meta_path.exists():
                feature_names = json.loads(meta_path.read_text())
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            "feature": list(feature_names),
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df
    except Exception:
        return None


@app.command()
def main(
    player_id: int | None = typer.Option(None, help="BDL player_id to explain."),
    team: str | None = typer.Option(None, help="Team abbreviation to explain all players."),
    stat: str | None = typer.Option(None, help="Specific stat (pts/reb/ast/fg3m/stl/blk/turnover). Default: all."),
    game_date: str | None = typer.Option(None, help="Game date YYYY-MM-DD (default: tomorrow)."),
    features_wide: str = typer.Option(_DEFAULT_FEATURES),
    pmfs_path: str | None = typer.Option(None, help="PMF parquet. Auto-detected from deliveries/ if not set."),
    model_dir: str = typer.Option(_DEFAULT_MODEL_DIR),
    out: str | None = typer.Option(None, help="Output JSON path. Prints to console if not set."),
) -> None:
    """Explain what is driving a player's projection."""
    target = game_date or (date.today() + timedelta(days=1)).isoformat()

    features = pd.read_parquet(features_wide)
    pmfs = _load_pmfs(pmfs_path, target)
    if pmfs is None:
        typer.echo("[ERROR] No PMF file found. Run predict_today.py first.")
        raise typer.Exit(1)

    feat_filtered = features.copy()
    if player_id:
        feat_filtered = feat_filtered[feat_filtered["player_id"] == player_id]
    if team:
        feat_filtered = feat_filtered[feat_filtered["team_abbreviation"] == team.upper()]
    if feat_filtered.empty:
        typer.echo("[WARN] No matching players found in feature table")
        raise typer.Exit(1)

    pids = feat_filtered["player_id"].unique().tolist()
    pmfs_filtered = pmfs[pmfs["player_id"].isin(pids)]
    stats = [stat] if stat else _STATS

    explanations = build_explanations(
        features=feat_filtered,
        pmfs=pmfs_filtered,
        model_dir=model_dir,
        stats=stats,
    )

    if not explanations:
        typer.echo("[WARN] No explanations generated — check player_id and that PMFs exist")
        raise typer.Exit(1)

    for exp in explanations:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {exp['player_name']} | {exp['stat'].upper()} | {target}")
        typer.echo(f"{'='*60}")
        typer.echo(f"  Minutes: {exp['projected_minutes']} min (L5 avg: {exp['minutes_l5_avg']} min)")
        if exp["minutes_change_flag"]:
            typer.echo(f"  ⚠ Minutes change: {exp['minutes_change_vs_l5']:+.1f} min vs. L5 average")
        typer.echo(f"  Projected {exp['stat']}: {exp['projected_mean']:.1f}")
        if exp.get("stat_l5_avg"):
            typer.echo(f"  L5 avg {exp['stat']}: {exp['stat_l5_avg']:.1f}")
        typer.echo(f"  Role: {exp['role_bucket']} | DNP risk: {exp['dnp_risk']} | Injury: {exp['injury_flag']}")
        typer.echo(f"\n  Minutes narrative:\n    {exp['minutes_narrative']}")
        typer.echo(f"\n  {exp['stat'].upper()} narrative:\n    {exp['stat_narrative']}")
        if exp["top_minutes_drivers"]:
            typer.echo(f"\n  Top minutes drivers:")
            for d in exp["top_minutes_drivers"][:3]:
                typer.echo(f"    • {d['label']}: {d['value']:.2f} ({d['direction']})")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(explanations, indent=2))
        typer.echo(f"\nExplanations written → {out}")


@app.command(name="explain-minutes")
def explain_minutes(
    player_id: int = typer.Option(..., "--player-id", help="BDL player_id."),
    game_date: str | None = typer.Option(None, "--game-date", help="Game date YYYY-MM-DD (default: tomorrow)."),
    features_wide: str = typer.Option(_DEFAULT_FEATURES),
    model_dir: str = typer.Option(_DEFAULT_MODEL_DIR),
    top_n: int = typer.Option(10, "--top-n", help="Number of top features to show."),
) -> None:
    """Explain what is driving a player's projected minutes.

    Uses the minutes model's feature importances (HGB feature_importances_)
    to rank features, then reads the player's actual feature values to show
    whether each feature is pushing minutes UP or DOWN vs. the league average.

    Works entirely from pre-fitted artifacts — no retraining required.
    """
    target = game_date or (date.today() + timedelta(days=1)).isoformat()

    # Load model feature importances
    fi_df = _get_minutes_feature_importances(model_dir)
    if fi_df is None:
        typer.echo(
            "[ERROR] Could not extract feature importances from minutes model.\n"
            "        Ensure the model has been trained and artifacts exist in "
            f"{model_dir}"
        )
        raise typer.Exit(1)

    # Load player features
    features = pd.read_parquet(features_wide)
    player_rows = features[features["player_id"] == player_id]
    if player_rows.empty:
        typer.echo(f"[ERROR] Player {player_id} not found in features table.")
        raise typer.Exit(1)

    # Use most recent row for this player
    if "game_date" in player_rows.columns:
        player_rows = player_rows.sort_values("game_date")
    row = player_rows.iloc[-1]
    player_name = row.get("player_name", f"Player {player_id}")

    # Get league-average values for comparison
    numeric_features = features.select_dtypes(include="number")
    league_means = numeric_features.mean()

    typer.echo(f"\n{'='*65}")
    typer.echo(f"  Minutes Model Drivers: {player_name} | {target}")
    typer.echo(f"{'='*65}")
    typer.echo(f"  Model: {model_dir}/minutes_model.joblib")
    typer.echo(f"  Top {top_n} features by importance:\n")
    typer.echo(f"  {'Rank':<5} {'Feature':<40} {'Importance':>12} {'Player':>10} {'League Avg':>12} {'Direction'}")
    typer.echo(f"  {'-'*95}")

    shown = 0
    for _, feat_row in fi_df.iterrows():
        if shown >= top_n:
            break
        fname = feat_row["feature"]
        if fname not in row.index:
            continue
        player_val = float(row[fname]) if pd.notna(row.get(fname)) else None
        league_val = float(league_means.get(fname, np.nan)) if fname in league_means else None
        importance = feat_row["importance"]

        if player_val is None:
            direction = "N/A"
        elif league_val is None or np.isnan(league_val):
            direction = "—"
        elif player_val > league_val * 1.05:
            direction = "↑ ABOVE avg"
        elif player_val < league_val * 0.95:
            direction = "↓ BELOW avg"
        else:
            direction = "≈ avg"

        label = _FEATURE_LABELS.get(fname, fname)[:38]
        pval_str = f"{player_val:.3f}" if player_val is not None else "N/A"
        lval_str = f"{league_val:.3f}" if (league_val is not None and not np.isnan(league_val)) else "—"

        typer.echo(
            f"  {int(feat_row['rank']):<5} {label:<40} {importance:>12.5f} "
            f"{pval_str:>10} {lval_str:>12}  {direction}"
        )
        shown += 1

    # Show minutes projection summary
    min_col = next(
        (c for c in ("pred_minutes_mean", "player_minutes_mean_l5", "player_minutes_rolling_mean_l5")
         if c in row.index and pd.notna(row.get(c))),
        None,
    )
    if min_col:
        typer.echo(f"\n  Projected minutes ({min_col}): {row[min_col]:.1f}")
    rest = row.get("player_rest_days")
    b2b = row.get("player_back_to_back_flag")
    inj = row.get("player_injured_l1")
    if rest is not None:
        typer.echo(f"  Rest days: {rest:.0f} | Back-to-back: {'Yes' if b2b else 'No'} | Prior DNP: {'Yes' if inj else 'No'}")


@app.command(name="explain-change")
def explain_change(
    player_id: int = typer.Option(..., "--player-id", help="BDL player_id."),
    game_date: str | None = typer.Option(None, "--game-date", help="Target game date (default: tomorrow)."),
    reference_date: str | None = typer.Option(
        None,
        "--reference-date",
        help="Reference date to compare against (default: yesterday's prediction).",
    ),
    features_wide: str = typer.Option(_DEFAULT_FEATURES),
    pmfs_path: str | None = typer.Option(None, "--pmfs-path", help="Today's PMF parquet."),
    reference_pmfs_path: str | None = typer.Option(None, "--reference-pmfs-path", help="Reference PMF parquet."),
) -> None:
    """Show what changed between two projection dates and why.

    Compares today's projection for a player against a reference date
    (default: yesterday) and identifies which features changed that would
    explain the difference in projected minutes and per-stat means.

    Works entirely from pre-fitted artifacts — reads parquets from disk.
    """
    target = game_date or (date.today() + timedelta(days=1)).isoformat()
    ref_date = reference_date or (date.today() - timedelta(days=1)).isoformat()

    features = pd.read_parquet(features_wide)

    # Load today's and yesterday's PMFs
    today_pmfs = _load_pmfs(pmfs_path, target)
    if today_pmfs is None:
        typer.echo(f"[ERROR] No PMF file found for target date {target}.")
        raise typer.Exit(1)

    # Reference PMFs: try common paths
    if reference_pmfs_path and Path(reference_pmfs_path).exists():
        ref_pmfs = pd.read_parquet(reference_pmfs_path)
    else:
        ref_candidates = [
            Path(f"deliveries/{ref_date}/full_pmfs_wide.parquet"),
            Path(f"data/clv_tracking/scored_{ref_date}.parquet"),
        ]
        ref_pmfs = None
        for c in ref_candidates:
            if c.exists():
                ref_pmfs = pd.read_parquet(c)
                break

    today_player = today_pmfs[today_pmfs["player_id"] == player_id]
    ref_player = ref_pmfs[ref_pmfs["player_id"] == player_id] if ref_pmfs is not None else pd.DataFrame()

    player_name = today_player["player_name"].iloc[0] if not today_player.empty else f"Player {player_id}"

    typer.echo(f"\n{'='*65}")
    typer.echo(f"  Change Report: {player_name} ({player_id})")
    typer.echo(f"  {ref_date} → {target}")
    typer.echo(f"{'='*65}")

    # Minutes change
    today_mins = float(today_player["minutes_mean"].iloc[0]) if "minutes_mean" in today_player.columns and not today_player.empty else None
    ref_mins = float(ref_player["minutes_mean"].iloc[0]) if "minutes_mean" in ref_player.columns and not ref_player.empty else None

    if today_mins is not None and ref_mins is not None:
        delta_mins = today_mins - ref_mins
        typer.echo(f"\n  Projected minutes: {ref_mins:.1f} → {today_mins:.1f} ({delta_mins:+.1f})")
    elif today_mins is not None:
        typer.echo(f"\n  Projected minutes: {today_mins:.1f} (no reference available)")

    # Per-stat projection changes
    typer.echo(f"\n  {'Stat':<12} {'Reference':>12} {'Today':>12} {'Delta':>10}")
    typer.echo(f"  {'-'*50}")

    mean_col = "pmf_mean" if "pmf_mean" in today_player.columns else "mean"
    ref_mean_col = "pmf_mean" if (not ref_player.empty and "pmf_mean" in ref_player.columns) else "mean"

    for stat in _STATS:
        today_stat = today_player[today_player["stat"] == stat]
        ref_stat = ref_player[ref_player["stat"] == stat] if not ref_player.empty else pd.DataFrame()

        today_v = float(today_stat[mean_col].iloc[0]) if not today_stat.empty and mean_col in today_stat.columns else None
        ref_v = float(ref_stat[ref_mean_col].iloc[0]) if not ref_stat.empty and ref_mean_col in ref_stat.columns else None

        if today_v is None:
            continue
        ref_str = f"{ref_v:.3f}" if ref_v is not None else "—"
        delta_str = f"{today_v - ref_v:+.3f}" if ref_v is not None else "—"
        typer.echo(f"  {stat:<12} {ref_str:>12} {today_v:>12.3f} {delta_str:>10}")

    # Feature-level changes (compare player's features between dates)
    player_today_feat = features[
        (features["player_id"] == player_id) &
        (features.get("game_date", pd.Series(dtype=str)).astype(str) == target)
    ] if "game_date" in features.columns else features[features["player_id"] == player_id].tail(1)

    player_ref_feat = features[
        (features["player_id"] == player_id) &
        (features.get("game_date", pd.Series(dtype=str)).astype(str) == ref_date)
    ] if "game_date" in features.columns else pd.DataFrame()

    if not player_today_feat.empty and not player_ref_feat.empty:
        typer.echo(f"\n  Key feature changes ({ref_date} → {target}):")
        typer.echo(f"  {'Feature':<40} {'Reference':>12} {'Today':>12} {'Delta':>10}")
        typer.echo(f"  {'-'*78}")

        key_features = [
            "player_minutes_mean_l5", "player_rest_days", "player_back_to_back_flag",
            "player_injured_l1", "vacated_minutes_l1", "player_dnp_streak_l3",
            "team_pace", "opp_pts_allowed_mean_l10",
        ]
        for feat in key_features:
            if feat not in player_today_feat.columns:
                continue
            today_val = player_today_feat[feat].iloc[0]
            ref_val = player_ref_feat[feat].iloc[0] if feat in player_ref_feat.columns else None

            if pd.isna(today_val):
                continue
            today_str = f"{float(today_val):.3f}"
            if ref_val is not None and not pd.isna(ref_val):
                ref_str = f"{float(ref_val):.3f}"
                delta_val = float(today_val) - float(ref_val)
                delta_str = f"{delta_val:+.3f}"
                # Only print if changed
                if abs(delta_val) > 0.001:
                    label = _FEATURE_LABELS.get(feat, feat)[:38]
                    typer.echo(f"  {label:<40} {ref_str:>12} {today_str:>12} {delta_str:>10}")
            else:
                label = _FEATURE_LABELS.get(feat, feat)[:38]
                typer.echo(f"  {label:<40} {'—':>12} {today_str:>12} {'—':>10}")
    elif player_today_feat.empty:
        typer.echo(f"\n  [INFO] No feature row for {player_id} on {target} — features may not be filtered by date.")

    typer.echo("")


if __name__ == "__main__":
    app()

