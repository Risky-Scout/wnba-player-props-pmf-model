"""Automated BDL injury poll → UTM redistribution engine (blueprint §5).

Polls the BDL /wnba/v1/player_injuries endpoint for all teams in tomorrow's
games, applies the status-to-minutes mapping from blueprint Table 5.1, runs
UTM redistribution for affected players, and writes:

  data/injuries/{date}.json         — raw BDL injury responses
  deliveries/tonight/full_pmfs_wide.parquet  — updated PMF slate (in-place)
  deliveries/tonight/injury_report_{date}.json — human-readable impact summary

Also outputs GTD dual-scenario records per blueprint §5.3.

Status → pipeline action mapping (blueprint §5.1):
  out          → minutes = 0,  full UTM redistribution
  doubtful     → minutes = 0,  full UTM redistribution + uncertainty flag
  questionable → minutes × 0.50, proportional UTM
  probable     → minutes × 0.85, proportional UTM
  available / null → no change
  GTD          → dual scenario: IN + OUT

Usage:
    python scripts/apply_injury_updates.py \\
        --game-date 2026-06-25 \\
        --slate deliveries/tonight/full_pmfs_wide.parquet

    # Dry-run (no writes):
    python scripts/apply_injury_updates.py --game-date 2026-06-25 --dry-run
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

_STATUS_MINUTES_MULTIPLIER: dict[str, float] = {
    "out": 0.0,
    "inactive": 0.0,
    "dnp": 0.0,
    "doubtful": 0.0,
    "unlikely": 0.0,
    "questionable": 0.50,
    "probable": 0.85,
    "limited": 0.65,
    "gtd": -1.0,      # sentinel → dual scenario
    "game-time decision": -1.0,
    "available": 1.0,
    "active": 1.0,
}

_FULL_REDISTRIBUTION_STATUSES = {"out", "inactive", "dnp", "doubtful", "unlikely"}
_DUAL_SCENARIO_STATUSES = {"gtd", "game-time decision"}

# Stat columns scaled linearly with minutes when redistribution is applied
_SCALABLE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                   "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "stocks"]


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="ISO game date YYYY-MM-DD."),
    slate: str = typer.Option(
        "", "--slate",
        help="PMF parquet to update. Auto-detected if not set.",
    ),
    injuries_json: str = typer.Option(
        "", "--injuries-json",
        help="Pre-fetched injuries JSON (skips BDL API call if provided).",
    ),
    usage_parquet: str = typer.Option(
        "data/processed/player_season_adv_usage.parquet",
        "--usage-parquet",
        help="Season usage% parquet for UTM.",
    ),
    out_dir: str = typer.Option("deliveries/tonight", "--out-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print changes, do not write."),
) -> None:
    """Poll BDL injuries and apply UTM redistribution to the PMF slate."""
    # ── Resolve slate ─────────────────────────────────────────────────────
    slate_path = _resolve_slate(slate, game_date)
    if slate_path is None:
        typer.echo("[WARN] No PMF slate found — nothing to update. Run predict_today.py first.", err=True)
        raise typer.Exit(0)

    pmfs_df = pd.read_parquet(slate_path)
    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows from {slate_path}")

    # ── Load or fetch injuries ─────────────────────────────────────────────
    if injuries_json:
        injuries = _load_injuries_file(injuries_json)
    else:
        injuries = _fetch_bdl_injuries(pmfs_df)

    if not injuries:
        typer.echo("No injury data found. Slate unchanged.")
        raise typer.Exit(0)

    typer.echo(f"Processing {len(injuries)} injury records")

    # ── Save raw injury file ───────────────────────────────────────────────
    inj_dir = Path("data/injuries")
    inj_dir.mkdir(parents=True, exist_ok=True)
    inj_out = inj_dir / f"{game_date}.json"
    if not dry_run:
        inj_out.write_text(json.dumps(injuries, indent=2))
        typer.echo(f"Saved raw injuries → {inj_out}")

    # ── Build UTM ──────────────────────────────────────────────────────────
    utm = _load_utm(usage_parquet, pmfs_df)

    # ── Apply injury adjustments per game ─────────────────────────────────
    impact_report: list[dict] = []
    gtd_scenarios: list[dict] = []

    for game_id in pmfs_df["game_id"].unique():
        g_mask = pmfs_df["game_id"] == game_id
        game_players = pmfs_df[g_mask]["player_id"].unique()
        game_inj = [r for r in injuries if r.get("player_id") in game_players]
        if not game_inj:
            continue

        for inj in game_inj:
            pid = int(inj.get("player_id", 0))
            raw_status = str(inj.get("status") or "available").lower().strip()
            multiplier = _STATUS_MINUTES_MULTIPLIER.get(raw_status, 1.0)

            p_mask = g_mask & (pmfs_df["player_id"] == pid)
            if not p_mask.any():
                continue

            base_mins = float(pmfs_df.loc[p_mask & (pmfs_df["stat"] == "pts"), "minutes_mean"].mean())
            if pd.isna(base_mins):
                base_mins = float(pmfs_df.loc[p_mask, "minutes_mean"].mean())
            if pd.isna(base_mins):
                continue

            if multiplier == -1.0:  # GTD → dual scenario
                gtd_rec = _build_gtd_scenario(pid, base_mins, pmfs_df, p_mask, inj, utm, game_id, pmfs_df[g_mask])
                gtd_scenarios.append(gtd_rec)
                impact_report.append({"player_id": pid, "status": raw_status, "action": "dual_gtd_scenario"})
                continue

            if multiplier == 1.0:
                continue  # no change

            new_mins = base_mins * multiplier
            delta_mins = base_mins - new_mins

            if dry_run:
                typer.echo(
                    f"[DRY] {inj.get('player_name', pid)} status={raw_status} "
                    f"mins {base_mins:.1f} → {new_mins:.1f} (Δ={delta_mins:.1f})"
                )
            else:
                # Update the player's minutes_mean
                pmfs_df.loc[p_mask, "minutes_mean"] = new_mins
                pmfs_df.loc[p_mask, "injury_flag"] = True
                pmfs_df.loc[p_mask, "override_source"] = "injury_update"

                # Scale stat_mean proportionally
                if base_mins > 0:
                    scale = new_mins / base_mins
                    for col in ["stat_mean", "pmf_mean", "mean"]:
                        if col in pmfs_df.columns:
                            pmfs_df.loc[p_mask, col] = pmfs_df.loc[p_mask, col] * scale

            # UTM redistribution for freed-up minutes
            if delta_mins > 0:
                teammates = _get_teammates(pmfs_df, g_mask, pid)
                redistribution = utm.redistribute(
                    roster=teammates,
                    out_player_ids=[pid],
                    out_minutes_dict={pid: delta_mins},
                )
                if not dry_run:
                    updated_roster, transfer_report = redistribution
                    _apply_utm_to_df(pmfs_df, g_mask, updated_roster, base_mins_map={
                        t["player_id"]: t.get("projected_minutes", 0.0) for t in teammates
                    })
                    impact_report.append({
                        "player_id": pid,
                        "player_name": inj.get("player_name"),
                        "status": raw_status,
                        "multiplier": multiplier,
                        "base_minutes": round(base_mins, 1),
                        "new_minutes": round(new_mins, 1),
                        "delta_minutes": round(delta_mins, 1),
                        "utm_transfers": transfer_report,
                    })

    # ── Write updated slate ────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        pmfs_df.to_parquet(slate_path, index=False)
        typer.echo(f"Updated slate written → {slate_path}")

        report_path = out_path / f"injury_report_{game_date}.json"
        report_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "game_date": game_date,
            "injuries_processed": len(injuries),
            "players_adjusted": len(impact_report),
            "gtd_scenarios": len(gtd_scenarios),
            "adjustments": impact_report,
            "gtd_scenarios_detail": gtd_scenarios,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, default=str))
        typer.echo(f"Saved injury report → {report_path}")

    typer.echo(f"\nSummary: {len(impact_report)} players adjusted, {len(gtd_scenarios)} GTD scenarios")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _fetch_bdl_injuries(pmfs_df: pd.DataFrame) -> list[dict]:
    """Fetch injuries from BDL API for all teams in the slate."""
    api_key = os.environ.get("BDL_API_KEY", "")
    if not api_key:
        typer.echo("[WARN] BDL_API_KEY not set — skipping API fetch.", err=True)
        return []

    try:
        import requests
    except ImportError:
        typer.echo("[WARN] requests not installed.", err=True)
        return []

    team_ids = list(set(
        pmfs_df["team_id"].dropna().astype(int).unique().tolist() +
        pmfs_df["opponent_team_id"].dropna().astype(int).unique().tolist()
    ))

    params: dict = {"per_page": 100}
    params.update({f"team_ids[{i}]": tid for i, tid in enumerate(team_ids)})
    headers = {"Authorization": api_key}

    try:
        resp = requests.get(
            "https://api.balldontlie.io/wnba/v1/player_injuries",
            params=params, headers=headers, timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("data", data) if isinstance(data, dict) else data
        typer.echo(f"BDL injuries: {len(raw)} records fetched")
        return [_normalize_bdl_injury(r) for r in raw]
    except Exception as exc:
        typer.echo(f"[WARN] BDL injury fetch failed: {exc}", err=True)
        return []


def _normalize_bdl_injury(raw: dict) -> dict:
    player = raw.get("player") or {}
    pid = raw.get("player_id") or player.get("id") or 0
    name = raw.get("player_name") or f"{player.get('first_name','')} {player.get('last_name','')}".strip()
    return {
        "player_id": int(pid),
        "player_name": name,
        "status": str(raw.get("status") or "available").lower(),
        "return_date": raw.get("return_date"),
        "comment": raw.get("comment"),
    }


def _load_injuries_file(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    if isinstance(raw, list):
        return raw
    return raw.get("data", [raw])


def _load_utm(usage_parquet: str, pmfs_df: pd.DataFrame):
    """Load UsageTransferMatrix from parquet, falling back to uniform."""
    from wnba_props_model.models.usage_transfer import UsageTransferMatrix  # noqa: PLC0415

    p = Path(usage_parquet)
    if p.exists():
        try:
            usg_df = pd.read_parquet(p)
            return UsageTransferMatrix(usg_df)
        except Exception as exc:
            typer.echo(f"[WARN] Could not load UTM from {p}: {exc}", err=True)

    # Fallback: build uniform UTM from pmfs_df player IDs
    usg_df = pd.DataFrame({
        "player_id": pmfs_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    return UsageTransferMatrix(usg_df)


def _get_teammates(pmfs_df: pd.DataFrame, g_mask: "pd.Series", exclude_pid: int) -> list[dict]:
    """Build roster dict list for UTM.redistribute()."""
    game_df = pmfs_df[g_mask]
    pts_rows = game_df[game_df["stat"] == "pts"]
    roster = []
    for _, row in pts_rows.iterrows():
        pid = int(row["player_id"])
        if pid == exclude_pid:
            continue
        roster.append({
            "player_id": pid,
            "projected_minutes": float(row.get("minutes_mean") or 0.0),
        })
    return roster


def _apply_utm_to_df(
    pmfs_df: pd.DataFrame,
    g_mask: "pd.Series",
    updated_roster: list[dict],
    base_mins_map: dict[int, float],
) -> None:
    """Write UTM-boosted minutes back into the DataFrame."""
    for p in updated_roster:
        pid = int(p["player_id"])
        new_mins = float(p.get("projected_minutes") or 0.0)
        old_mins = base_mins_map.get(pid, new_mins)
        p_mask = g_mask & (pmfs_df["player_id"] == pid)
        if not p_mask.any():
            continue
        pmfs_df.loc[p_mask, "minutes_mean"] = new_mins
        if old_mins > 0 and new_mins != old_mins:
            scale = new_mins / old_mins
            for col in ["stat_mean", "pmf_mean", "mean"]:
                if col in pmfs_df.columns:
                    pmfs_df.loc[p_mask, col] = pmfs_df.loc[p_mask, col] * scale


def _build_gtd_scenario(
    pid: int,
    base_mins: float,
    pmfs_df: pd.DataFrame,
    p_mask: "pd.Series",
    inj: dict,
    utm,
    game_id: int,
    game_df: pd.DataFrame,
) -> dict:
    """Build a GTD dual-scenario record per blueprint §5.3."""
    # Scenario IN: no change to player's projections
    scenario_in_pts = float(pmfs_df.loc[p_mask & (pmfs_df["stat"] == "pts"), "pmf_mean"].mean())

    # Scenario OUT: player is removed, UTM redistributed
    teammates = _get_teammates(pmfs_df, pmfs_df["game_id"] == game_id, pid)
    updated_roster, transfer_report = utm.redistribute(
        roster=teammates, out_player_ids=[pid], out_minutes_dict={pid: base_mins}
    )

    teammate_impact = {}
    for t in updated_roster:
        tpid = t["player_id"]
        old_mins = next((x.get("projected_minutes", 0.0) for x in teammates if x["player_id"] == tpid), 0.0)
        delta = t.get("projected_minutes", 0.0) - old_mins
        if abs(delta) > 0.5:
            t_name_row = pmfs_df[(pmfs_df["player_id"] == tpid) & (pmfs_df["stat"] == "pts")]
            t_name = str(t_name_row.iloc[0]["player_name"]) if not t_name_row.empty else str(tpid)
            pts_per_min = float(t_name_row.iloc[0]["pmf_mean"] / max(old_mins, 1e-3)) if not t_name_row.empty else 0.0
            teammate_impact[f"player_{tpid}"] = {
                "player_name": t_name,
                "added_minutes": round(delta, 1),
                "points_boost": round(delta * pts_per_min, 1),
            }

    return {
        "player_id": pid,
        "player_name": inj.get("player_name", str(pid)),
        "injury_status": "GTD",
        "comment": inj.get("comment", ""),
        "scenario_in": {
            "projected_minutes": {"mean": round(base_mins, 1)},
            "stat_projections": {"points": {"mean": round(scenario_in_pts, 1)}},
            "teammate_impact": None,
        },
        "scenario_out": {
            "projected_minutes": {"mean": 0.0},
            "stat_projections": {"points": {"mean": 0.0}},
            "teammate_impact": teammate_impact,
        },
        "official_status_pending": True,
    }


if __name__ == "__main__":
    app()
