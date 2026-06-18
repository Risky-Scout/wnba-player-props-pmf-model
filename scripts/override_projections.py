"""Fast manual override for player projections.

Loads pre-fitted Stage 4 model artifacts from disk (NO retraining) and
recomputes PMFs only for players whose projected minutes changed, making
the whole operation complete in seconds rather than minutes.

Override types (JSON payload keys per player_id):
  "status": "out"              → set projected_minutes = 0, redistribute
  "status": "questionable"     → no change to minutes (informational)
  "status": "limited"          → apply minutes_multiplier = 0.65 if no cap given
  "status": "active"           → restore player (remove prior out flag)
  "projected_minutes": <float> → hard override to specific minute value
  "minutes_multiplier": <float> → scale current projection (0.8 = 80% of baseline)
  "minutes_cap": <float>       → ceiling on projected minutes

Usage:
    python scripts/override_projections.py \\
        --slate deliveries/next_game/full_pmfs_wide.parquet \\
        --features-wide data/processed/wnba_player_game_features_wide.parquet \\
        --overrides '{"123": {"status": "out"}, "456": {"projected_minutes": 28}}' \\
        --out-dir deliveries/overrides \\
        --game-date 2026-06-18
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.team_score import WNBATeamScoreModel
from wnba_props_model.pipeline.predict import predict_player_pmfs

app = typer.Typer(add_completion=False)

# Directory where the fitted WNBATeamScoreModel lives
_DEFAULT_GAME_TOTALS_DIR = "artifacts/models/game_totals"

# Minutes cap implied by "limited" status when no explicit cap is provided
_LIMITED_MULTIPLIER = 0.65
# Minimum residual team minutes before redistribution is attempted
_MIN_TEAM_MINUTES = 1.0
# Default model paths
_DEFAULT_MODEL_DIR = "artifacts/models/stage4_baseline"
_DEFAULT_CAL_DIR = "artifacts/models/calibration"
_DEFAULT_CONFIG = "config/model/stage4_baseline.yaml"


def _parse_overrides(overrides_raw: str) -> dict[int, dict]:
    """Parse JSON override string → {player_id_int: override_dict}."""
    raw = json.loads(overrides_raw)
    return {int(k): v for k, v in raw.items()}


def _apply_minutes_overrides(
    features_wide: pd.DataFrame,
    slate_minutes: pd.DataFrame,  # player_id → minutes_mean from the existing slate
    overrides: dict[int, dict],
) -> tuple[pd.DataFrame, set[int], dict[str, str]]:
    """Apply overrides to the features_wide table.

    Returns:
        (modified_features_wide, affected_player_ids, change_log)
    """
    feat = features_wide.copy()
    change_log: dict[str, str] = {}

    # Map player_id → current projected minutes from the slate
    cur_mins = slate_minutes.set_index("player_id")["minutes_mean"].to_dict()

    # Determine teams of affected players
    player_team_map = feat.set_index("player_id")["team_id"].to_dict() if "team_id" in feat.columns else {}

    # Step 1: Apply direct overrides to get new minutes per player
    new_minutes: dict[int, float] = {pid: float(cur_mins.get(pid, 0.0)) for pid in cur_mins}

    for pid, ov in overrides.items():
        orig = new_minutes.get(pid, 0.0)
        status = ov.get("status", "").lower()

        if status == "out":
            new_minutes[pid] = 0.0
            change_log[str(pid)] = f"OUT: minutes {orig:.1f} → 0.0"
        elif status == "limited":
            cap = float(ov.get("minutes_cap", orig * _LIMITED_MULTIPLIER))
            new_minutes[pid] = min(orig, cap)
            change_log[str(pid)] = f"LIMITED: minutes {orig:.1f} → {new_minutes[pid]:.1f}"
        elif status in ("active", "questionable"):
            change_log[str(pid)] = f"{status.upper()}: minutes unchanged {orig:.1f}"
        elif "projected_minutes" in ov:
            new_minutes[pid] = float(ov["projected_minutes"])
            change_log[str(pid)] = f"MINUTES_OVERRIDE: {orig:.1f} → {new_minutes[pid]:.1f}"
        elif "minutes_multiplier" in ov:
            new_minutes[pid] = orig * float(ov["minutes_multiplier"])
            change_log[str(pid)] = (
                f"MULTIPLIER x{ov['minutes_multiplier']}: "
                f"{orig:.1f} → {new_minutes[pid]:.1f}"
            )
        if "minutes_cap" in ov and pid in new_minutes:
            cap = float(ov["minutes_cap"])
            if new_minutes[pid] > cap:
                new_minutes[pid] = cap
                change_log[str(pid)] = change_log.get(str(pid), "") + f" (capped at {cap:.1f})"

    # Step 2: Redistribute vacated minutes to teammates
    affected_pids: set[int] = set()
    teams_to_redistribute: set = {
        player_team_map.get(pid) for pid in overrides
        if player_team_map.get(pid) is not None
    }

    for team_id in teams_to_redistribute:
        team_pids = [
            pid for pid, tid in player_team_map.items()
            if tid == team_id and pid in cur_mins
        ]
        if not team_pids:
            continue

        original_team_total = sum(cur_mins.get(pid, 0.0) for pid in team_pids)
        new_team_total = sum(new_minutes.get(pid, cur_mins.get(pid, 0.0)) for pid in team_pids)
        vacated = original_team_total - new_team_total

        # Players who didn't get an override and weren't set to 0
        active_teammates = [
            pid for pid in team_pids
            if pid not in overrides or overrides.get(pid, {}).get("status", "") not in ("out",)
        ]

        if vacated > _MIN_TEAM_MINUTES and active_teammates:
            # Redistribute proportionally to active teammates' current minute shares
            active_minutes = {
                pid: max(new_minutes.get(pid, cur_mins.get(pid, 0.0)), 0.1)
                for pid in active_teammates
            }
            total_active = sum(active_minutes.values())
            for pid in active_teammates:
                share = active_minutes[pid] / total_active
                old_val = new_minutes.get(pid, cur_mins.get(pid, 0.0))
                new_minutes[pid] = old_val + vacated * share
                affected_pids.add(pid)
                change_log[str(pid)] = (
                    change_log.get(str(pid), "")
                    + f" [+{vacated * share:.1f} redistributed from teammates]"
                )

        affected_pids.update(overrides.keys())

    # Step 3: Write new minutes back to features_wide
    # Update the rolling-average minutes feature most likely used by the PMF engine
    minutes_feature_candidates = [
        c for c in feat.columns
        if c in ("pred_minutes_mean", "minutes_mean", "player_minutes_mean_l5",
                  "player_minutes_mean_l10", "player_minutes_rolling_mean_l5")
    ]

    for pid in affected_pids:
        mask = feat["player_id"] == pid
        if not mask.any():
            continue
        for col in minutes_feature_candidates:
            feat.loc[mask, col] = new_minutes.get(pid, feat.loc[mask, col].iloc[0])

    return feat, affected_pids, change_log


@app.command()
def main(
    slate: str = typer.Option(..., help="Existing PMF slate parquet (full_pmfs_wide.parquet)."),
    features_wide: str = typer.Option(..., help="Wide feature table from build_features.py."),
    overrides: str = typer.Option(..., help='JSON string: {"player_id": {"status": "out"}, ...}'),
    out_dir: str = typer.Option("deliveries/overrides", help="Output directory for revised files."),
    game_date: str | None = typer.Option(None, help="ISO game date (YYYY-MM-DD)."),
    model_dir: str = typer.Option(_DEFAULT_MODEL_DIR),
    cal_dir: str = typer.Option(_DEFAULT_CAL_DIR),
    config: str = typer.Option(_DEFAULT_CONFIG),
    raw_props: str | None = typer.Option(None, help="BDL player props parquet for edge recalculation."),
    game_totals_dir: str = typer.Option(
        _DEFAULT_GAME_TOTALS_DIR,
        help="Directory with team_score_model.pkl for game-total recalculation.",
    ),
    games_parquet: str | None = typer.Option(
        None,
        help="Games parquet for pace-adjustment lookup (optional).",
    ),
    recalc_game_totals: bool = typer.Option(
        True,
        "--recalc-game-totals/--no-recalc-game-totals",
        help="Recalculate WNBATeamScorePMFGrid for affected games after player override.",
    ),
) -> None:
    """Recompute PMFs for overridden players (seconds, not minutes).

    Only players whose minutes projection changed (and their teammates who
    received redistributed minutes) have their PMFs recomputed. All other
    players in the slate are carried over unchanged.
    """
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ────────────────────────────────────────────────────────
    slate_df = pd.read_parquet(slate)
    typer.echo(f"Loaded slate: {len(slate_df):,} PMF rows from {slate}")

    feat_df = pd.read_parquet(features_wide)
    if game_date and "game_date" in feat_df.columns:
        filt = feat_df[feat_df["game_date"].astype(str) == game_date].copy()
        if not filt.empty:
            feat_df = filt
    typer.echo(f"Loaded features: {len(feat_df):,} player-game rows")

    override_map = _parse_overrides(overrides)
    typer.echo(f"Overrides received for {len(override_map)} player(s): {list(override_map.keys())}")

    # ── Minutes from existing slate (deduplicated per player) ────────────
    if "minutes_mean" in slate_df.columns:
        slate_mins = (
            slate_df.groupby("player_id")["minutes_mean"].first().reset_index()
        )
    else:
        # Fall back to feature table
        min_col = next(
            (c for c in feat_df.columns
             if c in ("pred_minutes_mean", "player_minutes_mean_l5", "player_minutes_rolling_mean_l5")),
            None,
        )
        if min_col:
            slate_mins = feat_df[["player_id", min_col]].copy()
            slate_mins = slate_mins.rename(columns={min_col: "minutes_mean"})
            slate_mins = slate_mins.groupby("player_id")["minutes_mean"].first().reset_index()
        else:
            slate_mins = pd.DataFrame(columns=["player_id", "minutes_mean"])

    # ── Apply overrides + redistribute minutes ──────────────────────────
    modified_feat, affected_pids, change_log = _apply_minutes_overrides(
        feat_df, slate_mins, override_map
    )

    typer.echo(f"\n=== Override Summary ===")
    for pid_str, msg in sorted(change_log.items()):
        typer.echo(f"  Player {pid_str}: {msg}")
    typer.echo(f"Total affected players: {len(affected_pids)}")

    if not affected_pids:
        typer.echo("\n[INFO] No players affected — writing original slate unchanged.")
        slate_df.to_parquet(out / "full_pmfs_wide.parquet", index=False)
        elapsed = time.perf_counter() - t0
        typer.echo(f"\nCompleted in {elapsed:.2f}s (no changes)")
        return

    # ── Recompute PMFs only for affected rows ───────────────────────────
    affected_feat = modified_feat[modified_feat["player_id"].isin(affected_pids)].copy()
    typer.echo(f"\nRecomputing PMFs for {len(affected_feat):,} affected player-game rows...")

    new_pmfs = predict_player_pmfs(
        feature_df=affected_feat,
        model_dir=model_dir,
        config_path=config,
        cal_dir=cal_dir if Path(cal_dir).exists() else None,
        apply_calibration=True,
    )
    typer.echo(f"Recomputed {len(new_pmfs):,} PMF rows")

    # ── Merge back into full slate ───────────────────────────────────────
    # Remove old rows for affected players, append new ones
    unaffected = slate_df[~slate_df["player_id"].isin(affected_pids)].copy()
    revised_slate = pd.concat([unaffected, new_pmfs], ignore_index=True)

    # ── Print before/after summary ────────────────────────────────────────
    typer.echo("\n=== PMF Change Summary ===")
    typer.echo(f"{'Player':<10} {'Stat':<12} {'Old Mean':>10} {'New Mean':>10} {'Delta':>10}")
    typer.echo("-" * 55)

    old_means = (
        slate_df[slate_df["player_id"].isin(affected_pids)]
        .set_index(["player_id", "stat"])["pmf_mean"]
        if "pmf_mean" in slate_df.columns
        else pd.Series(dtype=float)
    )
    new_means = new_pmfs.set_index(["player_id", "stat"])["pmf_mean"] if "pmf_mean" in new_pmfs.columns else pd.Series(dtype=float)

    for pid in sorted(affected_pids):
        for stat in sorted(new_means.index.get_level_values("stat").unique()):
            try:
                old_v = float(old_means.get((pid, stat), np.nan))
                new_v = float(new_means.get((pid, stat), np.nan))
                delta = new_v - old_v
                if abs(delta) > 0.05:  # only print meaningful changes
                    typer.echo(f"{pid:<10} {stat:<12} {old_v:>10.3f} {new_v:>10.3f} {delta:>+10.3f}")
            except Exception:
                pass

    # ── Write outputs ─────────────────────────────────────────────────────
    # Compute diff_report: before/after for all affected player × stat lines
    diff_records = []
    for pid in sorted(affected_pids):
        for stat in sorted(new_means.index.get_level_values("stat").unique()
                           if hasattr(new_means.index, 'get_level_values') else []):
            try:
                old_v = float(old_means.get((pid, stat), np.nan))
                new_v = float(new_means.get((pid, stat), np.nan))
                diff_records.append({
                    "player_id": pid,
                    "stat": stat,
                    "old_mean": round(old_v, 4) if not np.isnan(old_v) else None,
                    "new_mean": round(new_v, 4) if not np.isnan(new_v) else None,
                    "delta": round(new_v - old_v, 4) if not (np.isnan(old_v) or np.isnan(new_v)) else None,
                    "change_log": change_log.get(str(pid), ""),
                })
            except Exception:
                pass

    diff_report = {
        "game_date": game_date,
        "n_overrides": len(override_map),
        "n_affected_players": len(affected_pids),
        "override_map": {str(k): v for k, v in override_map.items()},
        "change_log": change_log,
        "stat_changes": diff_records,
    }

    slate_out = out / "player_pmfs_override.parquet"
    revised_slate.to_parquet(slate_out, index=False)
    # Also write the standard filename for compatibility
    (out / "full_pmfs_wide.parquet").unlink(missing_ok=True)
    revised_slate.to_parquet(out / "full_pmfs_wide.parquet", index=False)
    typer.echo(f"\nWrote revised slate → {slate_out} ({len(revised_slate):,} rows)")

    diff_path = out / "diff_report.json"
    with open(diff_path, "w") as f:
        json.dump(diff_report, f, indent=2, default=str)
    typer.echo(f"Wrote diff report → {diff_path}")

    # ── Game total recalculation ───────────────────────────────────────────
    if recalc_game_totals:
        model_path = Path(game_totals_dir) / "team_score_model.pkl"
        if model_path.exists():
            try:
                typer.echo("\nRecalculating game totals for affected games...")
                ts_model = WNBATeamScoreModel.load(str(model_path))

                # Identify affected game_ids from the revised slate
                affected_game_ids: set = set()
                if "game_id" in revised_slate.columns:
                    affected_game_ids = set(
                        revised_slate[revised_slate["player_id"].isin(affected_pids)]["game_id"].dropna().unique()
                    )

                # Identify home/away teams for affected games
                game_team_map: dict = {}  # game_id → {home, away}
                if "game_id" in revised_slate.columns and "team_id" in revised_slate.columns:
                    for game_id, grp in revised_slate.groupby("game_id"):
                        teams = grp["team_id"].dropna().unique().tolist()
                        if len(teams) >= 2:
                            home_abbrs = []
                            away_abbrs = []
                            if "home_away" in grp.columns:
                                home_abbrs = grp[grp["home_away"] == "home"]["team_abbreviation"].dropna().unique().tolist() \
                                    if "team_abbreviation" in grp.columns else []
                                away_abbrs = grp[grp["home_away"] == "away"]["team_abbreviation"].dropna().unique().tolist() \
                                    if "team_abbreviation" in grp.columns else []
                            if home_abbrs and away_abbrs:
                                game_team_map[game_id] = {
                                    "home": home_abbrs[0],
                                    "away": away_abbrs[0],
                                }

                gt_rows = []
                for game_id in affected_game_ids:
                    if game_id not in game_team_map:
                        typer.echo(f"  [SKIP] game_id={game_id}: team mapping unavailable")
                        continue
                    home = game_team_map[game_id]["home"]
                    away = game_team_map[game_id]["away"]
                    try:
                        grid = ts_model.predict(home, away, game_id=game_id, game_date=game_date)
                        d = grid.to_dict()
                        d["override_applied"] = True
                        gt_rows.append(d)
                        diff_report["game_total_changes"] = diff_report.get("game_total_changes", [])
                        diff_report["game_total_changes"].append({
                            "game_id": game_id,
                            "home": home,
                            "away": away,
                            "total_mean": d["total_mean"],
                        })
                        typer.echo(f"  Recalculated: {home} vs {away} → E[total]={d['total_mean']:.1f}")
                    except Exception as exc:
                        typer.echo(f"  [WARN] {home} vs {away}: {exc}")

                if gt_rows:
                    gt_df = pd.DataFrame([{
                        "game_id": r.get("game_id"),
                        "game_date": r.get("game_date"),
                        "home_team": r.get("home_team"),
                        "away_team": r.get("away_team"),
                        "total_mean": r.get("total_mean"),
                        "home_mean": r.get("home_mean"),
                        "away_mean": r.get("away_mean"),
                        "override_applied": True,
                    } for r in gt_rows])
                    gt_path = out / "game_total_override.parquet"
                    gt_df.to_parquet(gt_path, index=False)
                    typer.echo(f"Wrote game total override → {gt_path}")

                # Refresh diff report with game total changes
                with open(diff_path, "w") as f:
                    json.dump(diff_report, f, indent=2, default=str)

            except Exception as exc:
                typer.echo(f"[WARN] Game total recalculation failed: {exc}", err=True)
        else:
            typer.echo(f"[INFO] No game totals model found at {model_path} — skipping")

    # ── Rebuild edge report if props available ─────────────────────────────
    if raw_props and Path(raw_props).exists():
        try:
            from wnba_props_model.pipeline.deliver import build_market_comparison
            props_df = pd.read_parquet(raw_props)
            comp = build_market_comparison(revised_slate, props_df)
            if not comp.empty:
                edges = comp[comp["edge_over"].abs() >= 0.04].sort_values(
                    "edge_over", key=np.abs, ascending=False
                )
                edges.to_parquet(out / "publishable_edges.parquet", index=False)
                typer.echo(f"Wrote revised edges → {out / 'publishable_edges.parquet'} ({len(edges):,} edges)")
        except Exception as exc:
            typer.echo(f"[WARN] Edge report rebuild failed: {exc}", err=True)

    elapsed = time.perf_counter() - t0
    typer.echo(f"\n✓ Override complete in {elapsed:.2f}s")
    if elapsed > 30:
        typer.echo(f"[WARN] Runtime exceeded 30s target ({elapsed:.1f}s)", err=True)


if __name__ == "__main__":
    app()
