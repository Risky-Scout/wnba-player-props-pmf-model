"""Injury update engine — upstream PMF rebuild (blueprint §5).

Architecture (corrected)
------------------------
This script applies point-in-time injury statuses BEFORE the PMF
distribution is considered final.  It replaces the former mean-only
scaling approach with a full distributional rebuild:

  1. Load pregame feature_df and PMF slate (for before/after comparison)
  2. Load or fetch point-in-time injury statuses
  3. Build availability table (per-player: status, probability, multiplier)
  4. Apply minutes adjustments to feature_df (scale minutes feature cols)
  5. Redistribute freed-up team minutes via UTM
  6. Rebuild ALL affected base-stat PMFs by rerunning predict_player_pmfs()
     with the updated feature_df
  7. Rebuild ALL affected combo PMFs
  8. Run integrity validation (fatal error on non-inactive zero-means)
  9. Write updated slate
  10. Write availability table and injury report
  11. Write deterministic before/after comparison report

What this script does NOT do
-----------------------------
- Does NOT scale pmf_mean / stat_mean / mean as primary adjustment
- Does NOT transform PMF support indices or multiply probabilities in-place
- Does NOT drop rows using pmf_mean > 0 as the removal criterion
- Does NOT set pmf_json={"0":1.0} except for confirmed-OUT players (whose
  PMF is legitimately a Dirac at zero)

Confirmed-inactive (OUT) players
---------------------------------
- Rows are RETAINED in the full PMF parquet (availability_status="OUT")
- pmf_json is set to {"0":1.0}, pmf_mean=0.0
- is_market_actionable=False
- Edge report filters these rows using the explicit confirmed_inactive_mask

Fatal integrity errors (non-inactive rows)
------------------------------------------
- pmf_mean <= 0  → fatal
- NaN pmf_mean   → fatal
- Invalid pmf_json → fatal

Usage:
    python scripts/apply_injury_updates.py \\
        --game-date 2026-06-25 \\
        --slate deliveries/tonight/full_pmfs_wide.parquet \\
        --features data/processed/wnba_player_game_features_wide.parquet

    # Dry-run (no writes):
    python scripts/apply_injury_updates.py --game-date 2026-06-25 --dry-run
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="ISO game date YYYY-MM-DD."),
    slate: str = typer.Option(
        "", "--slate",
        help="PMF parquet to update. Auto-detected if not set.",
    ),
    features: str = typer.Option(
        "", "--features",
        help="Wide feature parquet. Auto-detected if not set.",
    ),
    injuries_json: str = typer.Option(
        "", "--injuries-json",
        help="Pre-fetched injuries JSON (skips BDL API call if provided).",
    ),
    prediction_timestamp_utc: str = typer.Option(
        "",
        "--prediction-timestamp-utc",
        help=(
            "UTC prediction timestamp (ISO format) for injury timestamp validation. "
            "Defaults to the pipeline run time (pulled_at_utc) when fetching from BDL, "
            "or datetime.now(UTC) when loading from file. "
            "All injury source_updated_at values must be ≤ this timestamp."
        ),
    ),
    usage_parquet: str = typer.Option(
        "data/processed/player_season_adv_usage.parquet",
        "--usage-parquet",
        help="Season usage% parquet for UTM.",
    ),
    model_dir: str = typer.Option(
        "artifacts/models/stage4_baseline",
        "--model-dir",
        help="Stage 4 model artifact directory.",
    ),
    config_path: str = typer.Option(
        "",
        "--config-path",
        help=(
            "Explicit path to stage4_baseline.yaml model config. "
            "Required for PMF rebuild. "
            "Example: config/model/stage4_baseline.yaml"
        ),
    ),
    cal_dir: str = typer.Option(
        "artifacts/models/calibration",
        "--cal-dir",
        help="Calibration artifact directory.",
    ),
    out_dir: str = typer.Option("deliveries/tonight", "--out-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print changes, do not write."),
    skip_rebuild: bool = typer.Option(
        False, "--skip-rebuild",
        help="Skip PMF rebuild (for debugging availability table only).",
    ),
) -> None:
    """Apply injury statuses → rebuild affected PMFs with full distributional update."""
    from wnba_props_model.pipeline.injury_pipeline import (
        InjuryFetchResult,
        fetch_bdl_injuries,
        build_availability_table,
        apply_injury_to_feature_df,
        rebuild_affected_pmfs,
        rebuild_combos_for_affected,
        validate_injury_adjusted_pmfs,
        validate_injury_timestamps,
        build_before_after_report,
        build_confirmed_inactive_mask,
        INACTIVE_THRESHOLD,
    )

    # ── Validate config path (Blocker 5) ──────────────────────────────────
    resolved_config_path = _resolve_config_path(config_path, model_dir)
    if resolved_config_path is None:
        typer.echo(
            "[FATAL] --config-path is required and the file does not exist.\n"
            "  Provide: --config-path config/model/stage4_baseline.yaml\n"
            "  Do not construct the config path relative to --model-dir.",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"[apply_injury] Config path: {resolved_config_path}")

    # ── Resolve paths (fail-closed when games exist) ───────────────────────
    slate_path = _resolve_slate(slate, game_date)
    if slate_path is None:
        typer.echo(
            "[FATAL] No PMF slate found for this date — cannot proceed. "
            "Run predict_today.py first.",
            err=True,
        )
        raise typer.Exit(1)

    features_path = _resolve_features(features)
    if features_path is None:
        typer.echo(
            "[FATAL] No feature parquet found — cannot rebuild PMFs. "
            "Ensure data/processed/wnba_player_game_features_wide.parquet exists.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"[apply_injury] Loading PMF slate: {slate_path}")
    pmfs_before = pd.read_parquet(slate_path)
    typer.echo(f"[apply_injury] Loaded {len(pmfs_before):,} PMF rows")

    typer.echo(f"[apply_injury] Loading feature matrix: {features_path}")
    feature_df = pd.read_parquet(features_path)
    typer.echo(f"[apply_injury] Loaded {len(feature_df):,} feature rows")

    # Filter feature_df to upcoming game date only
    if "game_date" in feature_df.columns:
        feature_df = feature_df[
            pd.to_datetime(feature_df["game_date"]).dt.strftime("%Y-%m-%d") == game_date
        ].copy()
        typer.echo(
            f"[apply_injury] Feature rows for {game_date}: {len(feature_df):,}"
        )

    if feature_df.empty:
        typer.echo(
            f"[WARN] No feature rows found for game_date={game_date}. "
            "No games scheduled — slate unchanged.",
            err=True,
        )
        raise typer.Exit(0)

    # ── Load or fetch injuries ─────────────────────────────────────────────
    # Track the pulled_at_utc for prediction timestamp default
    _pulled_at_utc: str | None = None

    if injuries_json:
        injuries = _load_injuries_file(injuries_json)
        if not injuries:
            # Empty file = verified empty injury snapshot (no injuries to apply).
            typer.echo("[apply_injury] Injury JSON file is empty. Verified empty — slate unchanged.")
            raise typer.Exit(0)
        _pulled_at_utc = datetime.now(timezone.utc).isoformat()
    else:
        # Defect 1: Use typed InjuryFetchResult from injury_pipeline.
        # FAILURE (missing key, timeout, HTTP error, malformed JSON) → nonzero exit.
        # SUCCESS_EMPTY → verified empty snapshot → zero exit.
        # SUCCESS_WITH_ROWS → normalize records and continue.
        api_key = os.environ.get("BDL_API_KEY", "")
        team_ids = list(set(
            feature_df["team_id"].dropna().astype(int).unique().tolist()
            + (feature_df["opponent_team_id"].dropna().astype(int).unique().tolist()
               if "opponent_team_id" in feature_df.columns else [])
        ))
        fetch_result: InjuryFetchResult = fetch_bdl_injuries(
            api_key=api_key, team_ids=team_ids
        )

        if fetch_result.status == "FAILURE":
            typer.echo(
                f"[FATAL] BDL injury fetch failed: {fetch_result.error}",
                err=True,
            )
            raise typer.Exit(1)
        elif fetch_result.status == "SUCCESS_EMPTY":
            typer.echo(
                "[apply_injury] Verified empty injury snapshot — "
                "no injuries reported today. Slate unchanged."
            )
            raise typer.Exit(0)
        else:
            # SUCCESS_WITH_ROWS: normalize each record, preserving per-record timestamps
            injuries = [_normalize_bdl_injury(r) for r in fetch_result.records]
            _pulled_at_utc = (
                fetch_result.pulled_at_utc.isoformat()
                if fetch_result.pulled_at_utc is not None
                else datetime.now(timezone.utc).isoformat()
            )
            typer.echo(f"[apply_injury] BDL: {len(injuries)} injury records fetched")

    typer.echo(f"[apply_injury] Processing {len(injuries)} injury records")

    # ── Save raw injury file ───────────────────────────────────────────────
    inj_dir = Path("data/injuries")
    if not dry_run:
        inj_dir.mkdir(parents=True, exist_ok=True)
        inj_out = inj_dir / f"{game_date}.json"
        inj_out.write_text(json.dumps(injuries, indent=2))
        typer.echo(f"[apply_injury] Saved raw injuries → {inj_out}")

    # ── Build availability table ───────────────────────────────────────────
    # Each record's source_updated_at is preserved per-record (Defect 2).
    # Do NOT use the earliest timestamp as a substitute for all record timestamps.
    availability = build_availability_table(
        injuries=injuries,
        feature_df=feature_df,
        # source_updated_at omitted: each record carries its own timestamp;
        # build_availability_table uses pulled_ts as the snapshot-level fallback.
    )

    # ── Validate per-record source timestamps (Defect 2) ──────────────────
    # prediction_timestamp_utc defaults to pulled_at_utc (pipeline run time).
    pred_ts_for_validation = (
        prediction_timestamp_utc
        or _pulled_at_utc
        or datetime.now(timezone.utc).isoformat()
    )
    typer.echo(f"[apply_injury] Validating timestamps against prediction_ts={pred_ts_for_validation}")
    try:
        validate_injury_timestamps(
            availability_table=availability,
            prediction_timestamp_utc=pred_ts_for_validation,
        )
    except ValueError as exc:
        typer.echo(f"[FATAL] Injury timestamp validation failed: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"[apply_injury] Availability table: {len(availability):,} rows, "
        f"{availability['is_confirmed_inactive'].sum()} confirmed inactive"
    )

    # Print dry-run summary
    if dry_run:
        _print_dry_run_summary(availability, feature_df)
        raise typer.Exit(0)

    # ── Load UTM ──────────────────────────────────────────────────────────
    utm = _load_utm(usage_parquet, feature_df)

    # ── Apply injury feature adjustments ──────────────────────────────────
    feature_df_adjusted = apply_injury_to_feature_df(
        feature_df=feature_df,
        availability_table=availability,
        utm=utm,
    )

    # Identify all affected players (injury + UTM teammates)
    affected_player_ids: set[int] = set()
    mult_changed = feature_df_adjusted["_injury_minutes_multiplier"] != 1.0
    if mult_changed.any():
        affected_player_ids = set(
            feature_df_adjusted.loc[mult_changed, "player_id"].astype(int).unique()
        )
    # Also include confirmed-inactive players (their PMF must be {0:1.0})
    inactive_pids = set(
        availability.loc[
            availability["is_confirmed_inactive"], "player_id"
        ].astype(int).unique()
    )
    affected_player_ids |= inactive_pids

    typer.echo(
        f"[apply_injury] Affected players (injury + UTM): {len(affected_player_ids)}"
    )

    # ── Rebuild PMFs for affected players ────────────────────────────────
    if skip_rebuild or not affected_player_ids:
        pmfs_after = pmfs_before.copy()
        typer.echo("[apply_injury] PMF rebuild skipped")
    else:
        typer.echo(
            f"[apply_injury] Rebuilding PMFs for {len(affected_player_ids)} players …"
        )
        new_atom_pmfs = rebuild_affected_pmfs(
            feature_df_adjusted=feature_df_adjusted,
            affected_player_ids=affected_player_ids,
            model_dir=model_dir,
            config_path=resolved_config_path,
            cfg={},
            cal_dir=cal_dir if Path(cal_dir).exists() else None,
            apply_calibration=Path(cal_dir).exists(),
            apply_shrinkage=True,
        )
        typer.echo(
            f"[apply_injury] PMF rebuild complete: {len(new_atom_pmfs):,} new atom rows"
        )

        # Merge new atom PMFs back into the full slate
        COMBO_STATS = frozenset(
            {"stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"}
        )
        ATOM_STATS = frozenset(
            {"pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"}
        )
        # Remove old atom + combo rows for affected players
        keep_mask = ~(
            pmfs_before["player_id"].isin(affected_player_ids)
            & pmfs_before["stat"].isin(ATOM_STATS | COMBO_STATS)
        )
        pmfs_base = pmfs_before[keep_mask].copy()

        # Append new atom PMFs
        pmfs_after_atoms = pd.concat(
            [pmfs_base, new_atom_pmfs], ignore_index=True
        )

        # Rebuild combos for affected players using updated atoms.
        # Defect 5: pass model_dir so the same correlation map, position map,
        # and IPF settings are used as live inference.  Fails clearly if the
        # production correlation artifact is expected but missing.
        pmfs_after = rebuild_combos_for_affected(
            full_pmfs_with_new_atoms=pmfs_after_atoms,
            affected_player_ids=affected_player_ids,
            model_dir=model_dir,
        )
        typer.echo(
            f"[apply_injury] Full slate after rebuild: {len(pmfs_after):,} rows"
        )

    # ── Apply confirmed-OUT players: set pmf_json = {0:1.0} ─────────────
    _ZERO_PMF = json.dumps({"0": 1.0})
    if inactive_pids and "pmf_json" in pmfs_after.columns:
        inact_mask = pmfs_after["player_id"].isin(inactive_pids)
        pmfs_after.loc[inact_mask, "pmf_json"] = _ZERO_PMF
        pmfs_after.loc[inact_mask, "pmf_mean"] = 0.0
        if "mean" in pmfs_after.columns:
            pmfs_after.loc[inact_mask, "mean"] = 0.0
        if "stat_mean" in pmfs_after.columns:
            pmfs_after.loc[inact_mask, "stat_mean"] = 0.0
        if "pmf_mean_full_precision" in pmfs_after.columns:
            pmfs_after.loc[inact_mask, "pmf_mean_full_precision"] = 0.0
        typer.echo(
            f"[apply_injury] Set pmf_json={{0:1.0}} for "
            f"{inact_mask.sum()} confirmed-inactive rows"
        )

    # ── Attach availability columns ───────────────────────────────────────
    pmfs_after = _attach_availability_columns(pmfs_after, availability)

    # ── Integrity validation ──────────────────────────────────────────────
    try:
        validate_injury_adjusted_pmfs(pmfs_after, availability)
        typer.echo("[apply_injury] Integrity validation PASS")
    except ValueError as exc:
        typer.echo(f"[FATAL] PMF integrity error: {exc}", err=True)
        raise typer.Exit(1)

    # ── Build before/after comparison report ─────────────────────────────
    before_after_report_df = build_before_after_report(
        old_pmfs=pmfs_before,
        new_pmfs=pmfs_after,
        availability_table=availability,
    )
    if not before_after_report_df.empty:
        # Only report rows where something actually changed
        changed = before_after_report_df[
            (before_after_report_df["pmf_mean_before"] - before_after_report_df["pmf_mean_after"]).abs() > 1e-6
        ]
        if not changed.empty:
            typer.echo("\n[apply_injury] Before/after PMF changes:")
            typer.echo(
                changed[[
                    "player_name", "stat", "injury_status",
                    "minutes_before", "minutes_after",
                    "pmf_mean_before", "pmf_mean_after",
                    "P(over)_before", "P(over)_after",
                ]].to_string(index=False)
            )

    # ── Verify PMF rebuild is real ─────────────────────────────────────────
    # For each affected player, verify pmf_json changed (when material)
    _verify_pmf_rebuild(
        pmfs_before=pmfs_before,
        pmfs_after=pmfs_after,
        affected_player_ids=affected_player_ids,
        availability=availability,
    )

    # ── Write outputs ────────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pmfs_after.to_parquet(slate_path, index=False)
    typer.echo(f"[apply_injury] Updated slate written → {slate_path}")

    # Write canonical injury adjustments parquet (Step 3 schema)
    # Field names conform to the point-in-time spec:
    #   game_id, player_id, raw_status, normalized_status,
    #   availability_probability, starter_probability,
    #   minutes_multiplier, minutes_cap,
    #   is_confirmed_inactive, is_market_actionable,
    #   source_updated_at, pulled_at_utc
    adj_path = Path("data/processed") / f"injury_adjustments_{game_date}.parquet"
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_parquet(adj_path, index=False)
    typer.echo(f"[apply_injury] Injury adjustments → {adj_path}")

    # Also write the legacy availability_table path for downstream compatibility
    avail_path = out_path / f"availability_table_{game_date}.parquet"
    availability.to_parquet(avail_path, index=False)
    typer.echo(f"[apply_injury] Availability table → {avail_path}")

    # Write before/after report
    if not before_after_report_df.empty:
        ba_path = out_path / f"injury_pmf_changes_{game_date}.parquet"
        before_after_report_df.to_parquet(ba_path, index=False)
        typer.echo(f"[apply_injury] Before/after report → {ba_path}")

    # Write human-readable injury report JSON
    impact_report = _build_impact_report(availability, before_after_report_df, game_date)
    report_path = out_path / f"injury_report_{game_date}.json"
    report_path.write_text(json.dumps(impact_report, indent=2, default=str))
    typer.echo(f"[apply_injury] Injury report → {report_path}")

    n_inactive = int(availability["is_confirmed_inactive"].sum())
    n_adjusted = int((availability["minutes_multiplier"] != 1.0).sum())
    typer.echo(
        f"\n[apply_injury] Summary: {n_inactive} confirmed inactive, "
        f"{n_adjusted} players adjusted, "
        f"{len(affected_player_ids)} PMFs rebuilt"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_config_path(config_path_arg: str, model_dir: str) -> Path | None:
    """Resolve the explicit config path for stage4_baseline.yaml.

    Requires an explicit path; does NOT construct relative to model_dir.
    Returns None if the file does not exist.
    """
    if config_path_arg:
        p = Path(config_path_arg)
        return p if p.exists() else None
    # Attempt the canonical default location as a convenience fallback
    default = Path("config/model/stage4_baseline.yaml")
    return default if default.exists() else None



def _resolve_slate(slate_arg: str, game_date: str) -> Path | None:
    if slate_arg:
        return Path(slate_arg)
    for c in [
        "deliveries/tonight/full_pmfs_wide.parquet",
        f"deliveries/tonight/player_projections_{game_date}.parquet",
        "deliveries/next_game/full_pmfs_wide.parquet",
    ]:
        p = Path(c)
        if p.exists():
            return p
    return None


def _resolve_features(features_arg: str) -> Path | None:
    if features_arg:
        return Path(features_arg)
    for c in [
        "data/processed/wnba_player_game_features_wide.parquet",
        "data/processed/features_wide.parquet",
    ]:
        p = Path(c)
        if p.exists():
            return p
    return None


def _normalize_bdl_injury(raw: dict) -> dict:
    """Normalize a raw BDL injury record to internal format.

    Preserves per-record source_updated_at (Defect 2: each record must carry its
    own timestamp so build_availability_table can store distinct values per player).
    """
    player = raw.get("player") or {}
    pid = raw.get("player_id") or player.get("id") or 0
    name = (
        raw.get("player_name")
        or f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
    )
    # Preserve the per-record source timestamp so build_availability_table
    # stores distinct timestamps per player (not one shared snapshot timestamp).
    src_ts = (
        raw.get("source_updated_at")
        or raw.get("updated_at")
        or raw.get("created_at")
    )
    return {
        "player_id": int(pid),
        "player_name": name,
        "status": str(raw.get("status") or "available").lower(),
        "return_date": raw.get("return_date"),
        "comment": raw.get("comment"),
        "source_updated_at": str(src_ts) if src_ts else None,
    }


def _load_injuries_file(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    if isinstance(raw, list):
        return raw
    return raw.get("data", [raw])


def _load_utm(usage_parquet: str, feature_df: pd.DataFrame):
    from wnba_props_model.models.usage_transfer import UsageTransferMatrix  # noqa: PLC0415

    p = Path(usage_parquet)
    if p.exists():
        try:
            usg_df = pd.read_parquet(p)
            return UsageTransferMatrix(usg_df)
        except Exception as exc:
            typer.echo(f"[WARN] Could not load UTM from {p}: {exc}", err=True)

    # Fallback: uniform UTM
    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    return UsageTransferMatrix(usg_df)


def _attach_availability_columns(
    pmfs_df: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Attach availability metadata columns to the PMF DataFrame."""
    if availability.empty:
        return pmfs_df

    # Support both old column name (normalized_availability_status) and
    # new column name (normalized_status) introduced in Step 3.
    norm_col = (
        "normalized_status"
        if "normalized_status" in availability.columns
        else "normalized_availability_status"
    )
    avail_cols = [
        "player_id", "game_id",
        norm_col,
        "availability_probability",
        "is_confirmed_inactive",
        "is_market_actionable",
    ]
    avail_sub = availability[[c for c in avail_cols if c in availability.columns]].copy()
    avail_sub = avail_sub.rename(
        columns={norm_col: "availability_status"}
    )

    # Drop existing availability columns to avoid duplicates
    drop_cols = [
        c for c in avail_sub.columns
        if c not in ("player_id", "game_id") and c in pmfs_df.columns
    ]
    pmfs_df = pmfs_df.drop(columns=drop_cols, errors="ignore")

    merged = pmfs_df.merge(avail_sub, on=["player_id", "game_id"], how="left")

    # Default: available for players not in availability table
    if "availability_status" in merged.columns:
        merged["availability_status"] = merged["availability_status"].fillna("AVAILABLE")
    if "availability_probability" in merged.columns:
        merged["availability_probability"] = merged["availability_probability"].fillna(1.0)
    if "is_confirmed_inactive" in merged.columns:
        merged["is_confirmed_inactive"] = merged["is_confirmed_inactive"].fillna(False)
    if "is_market_actionable" in merged.columns:
        merged["is_market_actionable"] = merged["is_market_actionable"].fillna(True)

    return merged


def _verify_pmf_rebuild(
    pmfs_before: pd.DataFrame,
    pmfs_after: pd.DataFrame,
    affected_player_ids: set[int],
    availability: pd.DataFrame,
) -> None:
    """Log warnings if a material adjustment did not change pmf_json."""
    if "pmf_json" not in pmfs_before.columns or "pmf_json" not in pmfs_after.columns:
        return

    before_idx = pmfs_before.set_index(["player_id", "game_id", "stat"])["pmf_json"]
    after_idx  = pmfs_after.set_index(["player_id", "game_id", "stat"])["pmf_json"]

    avail_lu = (
        availability
        .set_index("player_id")[["minutes_multiplier", "is_confirmed_inactive"]]
        .to_dict("index")
    ) if not availability.empty else {}

    n_checked = 0
    n_unchanged = 0
    unchanged_examples: list[str] = []

    for pid in affected_player_ids:
        info = avail_lu.get(int(pid), {})
        mult = float(info.get("minutes_multiplier", 1.0))
        is_inactive = bool(info.get("is_confirmed_inactive", False))

        # Skip dual-scenario (GTD) and unchanged players
        if mult == -1.0 or mult == 1.0:
            continue

        keys_before = [(idx, gid, stat) for idx, gid, stat in before_idx.index if idx == pid]
        for key in keys_before:
            if key not in after_idx.index:
                continue
            old_j = str(before_idx.loc[key])
            new_j = str(after_idx.loc[key])
            n_checked += 1
            if old_j == new_j and not is_inactive:
                n_unchanged += 1
                if len(unchanged_examples) < 3:
                    unchanged_examples.append(
                        f"player_id={key[0]} game={key[1]} stat={key[2]}"
                    )

    if n_unchanged > 0:
        logger.warning(
            "[apply_injury] %d / %d material adjustments did not change pmf_json "
            "(examples: %s) — verify PMF rebuild is working correctly",
            n_unchanged, n_checked, unchanged_examples,
        )
    else:
        logger.info(
            "[apply_injury] PMF rebuild verified: all %d checked rows changed pmf_json",
            n_checked,
        )


def _print_dry_run_summary(
    availability: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    """Print a dry-run summary of what would change."""
    typer.echo("\n[DRY RUN] Injury adjustment summary:")
    typer.echo(f"  Total players: {len(availability):,}")
    typer.echo(
        f"  Confirmed inactive: {availability['is_confirmed_inactive'].sum()}"
    )

    non_trivial = availability[
        availability["minutes_multiplier"].between(0.01, 0.99)
    ]
    if not non_trivial.empty:
        typer.echo("  Partial availability:")
        for _, row in non_trivial.iterrows():
            mins_col = next(
                (c for c in ["player_minutes_mean_l5", "player_minutes_mean_season"]
                 if c in feature_df.columns),
                None,
            )
            base_mins = (
                float(feature_df.loc[
                    feature_df["player_id"] == row["player_id"],
                    mins_col
                ].mean())
                if mins_col else float("nan")
            )
            new_mins = base_mins * float(row["minutes_multiplier"])
            raw_col = "raw_status" if "raw_status" in non_trivial.columns else "raw_injury_status"
            typer.echo(
                f"    player_id={row['player_id']} "
                f"status={row.get(raw_col, 'unknown')} "
                f"multiplier={row['minutes_multiplier']:.2f} "
                f"mins {base_mins:.1f}→{new_mins:.1f}"
            )


def _build_impact_report(
    availability: pd.DataFrame,
    before_after: pd.DataFrame,
    game_date: str,
) -> dict:
    adjustments = []
    raw_col = "raw_status" if "raw_status" in availability.columns else "raw_injury_status"
    for _, row in availability.iterrows():
        if row["minutes_multiplier"] == 1.0 and not row["is_confirmed_inactive"]:
            continue
        adjustments.append({
            "player_id": int(row["player_id"]),
            "raw_status": str(row.get(raw_col, "available")),
            "availability_probability": float(row["availability_probability"]),
            "minutes_multiplier": float(row["minutes_multiplier"]),
            "is_confirmed_inactive": bool(row["is_confirmed_inactive"]),
            "is_market_actionable": bool(row["is_market_actionable"]),
        })

    pmf_changes = []
    if not before_after.empty:
        for _, row in before_after.iterrows():
            if abs(float(row.get("pmf_mean_before", 0)) - float(row.get("pmf_mean_after", 0))) < 1e-6:
                continue
            pmf_changes.append({
                "player_id": int(row["player_id"]),
                "player_name": str(row.get("player_name", row["player_id"])),
                "stat": str(row["stat"]),
                "pmf_mean_before": float(row.get("pmf_mean_before", float("nan"))),
                "pmf_mean_after": float(row.get("pmf_mean_after", float("nan"))),
                "p_over_before": float(row.get("P(over)_before", float("nan"))),
                "p_over_after": float(row.get("P(over)_after", float("nan"))),
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "players_adjusted": len(adjustments),
        "pmf_rows_changed": len(pmf_changes),
        "adjustments": adjustments,
        "pmf_changes": pmf_changes,
    }


if __name__ == "__main__":
    app()
