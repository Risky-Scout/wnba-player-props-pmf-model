"""Player scratch simulation (blueprint §6.2).

Removes a player entirely from a game's projection slate, applies full UTM
redistribution to teammates, and outputs a structured scratch_impact JSON.

Usage:
    python scripts/scratch_simulation.py \\
        --player-id 842 \\
        --game-date 2026-07-16 \\
        --output-format json

Output example (blueprint §6.2):
    {
      "removed_player_id": 842,
      "removed_player_name": "A'ja Wilson",
      "original_projections": {...},
      "teammate_impacts": [...],
      "team_totals_delta": {...}
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)


@app.command()
def main(
    player_id: int = typer.Option(..., "--player-id", help="BDL player ID to scratch."),
    game_date: str = typer.Option("", "--game-date", help="ISO game date YYYY-MM-DD."),
    slate: str = typer.Option("", "--slate", help="PMF slate parquet. Auto-detected if not set."),
    usage_parquet: str = typer.Option(
        "data/processed/player_season_adv_usage.parquet", "--usage-parquet"
    ),
    output_format: str = typer.Option("json", "--output-format", help="json | table"),
    out_file: str = typer.Option("", "--out", help="Write JSON to file instead of stdout."),
) -> None:
    """Simulate removing a player and show teammate impact via UTM redistribution."""
    from wnba_props_model.models.usage_transfer import UsageTransferMatrix  # noqa: PLC0415

    # ── Load slate ────────────────────────────────────────────────────────
    slate_path = _resolve_slate(slate, game_date)
    if slate_path is None:
        typer.echo("[ERROR] No PMF slate found. Run predict_today.py first.", err=True)
        raise typer.Exit(1)

    pmfs_df = pd.read_parquet(slate_path)

    # Find the player
    p_mask = pmfs_df["player_id"] == player_id
    if not p_mask.any():
        typer.echo(f"[ERROR] player_id={player_id} not found in slate.", err=True)
        raise typer.Exit(1)

    player_name = str(pmfs_df.loc[p_mask, "player_name"].iloc[0])
    game_ids = pmfs_df.loc[p_mask, "game_id"].unique()
    if len(game_ids) == 0:
        typer.echo("[ERROR] Player has no game in slate.", err=True)
        raise typer.Exit(1)
    game_id = int(game_ids[0])

    # Original projections for the removed player
    orig_projections = _get_player_stats(pmfs_df, player_id)
    removed_mins = float(pmfs_df.loc[p_mask & (pmfs_df["stat"] == "pts"), "minutes_mean"].mean())
    if pd.isna(removed_mins):
        removed_mins = float(pmfs_df.loc[p_mask, "minutes_mean"].mean())

    # ── Load UTM ──────────────────────────────────────────────────────────
    utm = _load_utm(usage_parquet, pmfs_df)

    # ── Get teammates ─────────────────────────────────────────────────────
    g_mask = pmfs_df["game_id"] == game_id
    teammates_raw = _get_teammates(pmfs_df, g_mask, player_id)
    base_mins_map = {t["player_id"]: t["projected_minutes"] for t in teammates_raw}
    base_stats_map = {
        t["player_id"]: _get_player_stats(pmfs_df, t["player_id"])
        for t in teammates_raw
    }

    # ── Apply redistribution ──────────────────────────────────────────────
    updated_roster, transfer_report = utm.redistribute(
        roster=teammates_raw,
        out_player_ids=[player_id],
        out_minutes_dict={player_id: removed_mins},
    )

    # ── Build teammate impact list ────────────────────────────────────────
    teammate_impacts = []
    team_totals_delta = {s: 0.0 for s in ["points", "rebounds", "assists",
                                           "steals", "blocks", "threes"]}

    for t in sorted(updated_roster, key=lambda x: -(x.get("projected_minutes", 0) - base_mins_map.get(x["player_id"], 0))):
        tpid = int(t["player_id"])
        new_mins = float(t.get("projected_minutes") or 0.0)
        old_mins = base_mins_map.get(tpid, new_mins)
        delta_mins = new_mins - old_mins

        if abs(delta_mins) < 0.1:
            continue

        base = base_stats_map.get(tpid, {})
        scale = (new_mins / old_mins) if old_mins > 0 else 1.0

        t_name_rows = pmfs_df[(pmfs_df["player_id"] == tpid) & (pmfs_df["stat"] == "pts")]
        t_name = str(t_name_rows.iloc[0]["player_name"]) if not t_name_rows.empty else str(tpid)
        t_team = str(t_name_rows.iloc[0]["team_abbreviation"]) if not t_name_rows.empty else ""
        usg = utm.get_usage(tpid)

        impact = {
            "player_id": tpid,
            "player_name": t_name,
            "team": t_team,
            "minutes_delta": round(delta_mins, 1),
            "usage_pct_delta": round(usg * abs(delta_mins) / max(removed_mins, 1e-3) * 100, 1),
        }
        for stat_key, stat_display in [("pts","points"),("reb","rebounds"),("ast","assists"),
                                       ("stl","steals"),("blk","blocks"),("fg3m","threes")]:
            base_v = base.get(stat_key, 0.0)
            delta_v = base_v * (scale - 1.0)
            impact[f"{stat_display}_delta"] = round(delta_v, 1)
            team_totals_delta[stat_display] -= delta_v  # removed player loss

        teammate_impacts.append(impact)
        for stat_key, stat_display in [("pts","points"),("reb","rebounds"),("ast","assists"),
                                       ("stl","steals"),("blk","blocks"),("fg3m","threes")]:
            team_totals_delta[stat_display] += impact.get(f"{stat_display}_delta", 0.0)

    # Adjust team totals for removed player's original contributions
    for stat_key, stat_display in [("pts","points"),("reb","rebounds"),("ast","assists"),
                                   ("stl","steals"),("blk","blocks"),("fg3m","threes")]:
        orig_v = orig_projections.get(stat_key, 0.0)
        team_totals_delta[stat_display] = round(
            team_totals_delta[stat_display] - orig_v, 1
        )

    # ── Build output ──────────────────────────────────────────────────────
    result = {
        "removed_player_id": player_id,
        "removed_player_name": player_name,
        "game_id": game_id,
        "removed_projected_minutes": round(removed_mins, 1),
        "original_projections": {
            "points_mean": round(orig_projections.get("pts", 0.0), 1),
            "rebounds_mean": round(orig_projections.get("reb", 0.0), 1),
            "assists_mean": round(orig_projections.get("ast", 0.0), 1),
            "steals_mean": round(orig_projections.get("stl", 0.0), 1),
            "blocks_mean": round(orig_projections.get("blk", 0.0), 1),
            "threes_mean": round(orig_projections.get("fg3m", 0.0), 1),
        },
        "teammate_impacts": teammate_impacts,
        "team_totals_delta": {k: round(v, 1) for k, v in team_totals_delta.items()},
    }

    payload = json.dumps(result, indent=2)

    if out_file:
        Path(out_file).write_text(payload)
        typer.echo(f"Scratch simulation written → {out_file}")
    else:
        if output_format == "table":
            _print_table(result)
        else:
            typer.echo(payload)


def _print_table(result: dict) -> None:
    typer.echo(f"\nScratch Simulation: {result['removed_player_name']} (id={result['removed_player_id']})")
    typer.echo(f"Projected minutes removed: {result['removed_projected_minutes']:.1f}")
    op = result["original_projections"]
    typer.echo(f"Original: pts={op['points_mean']} reb={op['rebounds_mean']} ast={op['assists_mean']}")
    typer.echo(f"\nTeammate impacts (sorted by minutes gained):")
    typer.echo(f"  {'Player':<25} {'min Δ':>7} {'pts Δ':>7} {'reb Δ':>7} {'ast Δ':>7} {'usg% Δ':>7}")
    typer.echo("  " + "-" * 65)
    for t in result["teammate_impacts"]:
        typer.echo(
            f"  {t['player_name']:<25} {t['minutes_delta']:>+7.1f} "
            f"{t.get('points_delta',0):>+7.1f} {t.get('rebounds_delta',0):>+7.1f} "
            f"{t.get('assists_delta',0):>+7.1f} {t.get('usage_pct_delta',0):>+7.1f}"
        )
    d = result["team_totals_delta"]
    typer.echo(f"\nTeam totals delta: pts={d.get('points',0):+.1f} reb={d.get('rebounds',0):+.1f} ast={d.get('assists',0):+.1f}")


def _resolve_slate(slate: str, game_date: str) -> Path | None:
    if slate:
        return Path(slate)
    for c in [
        "deliveries/tonight/full_pmfs_wide.parquet",
        f"deliveries/tonight/player_projections_{game_date}.parquet",
        "deliveries/next_game/full_pmfs_wide.parquet",
    ]:
        p = Path(c)
        if p.exists():
            return p
    return None


def _load_utm(usage_parquet: str, pmfs_df: pd.DataFrame):
    from wnba_props_model.models.usage_transfer import UsageTransferMatrix  # noqa: PLC0415
    p = Path(usage_parquet)
    if p.exists():
        try:
            return UsageTransferMatrix(pd.read_parquet(p))
        except Exception:
            pass
    usg_df = pd.DataFrame({"player_id": pmfs_df["player_id"].unique(), "usage_pct": 0.20})
    return UsageTransferMatrix(usg_df)


def _get_teammates(pmfs_df: pd.DataFrame, g_mask: "pd.Series", exclude_pid: int) -> list[dict]:
    pts_rows = pmfs_df[g_mask & (pmfs_df["stat"] == "pts")]
    return [
        {"player_id": int(r["player_id"]), "projected_minutes": float(r.get("minutes_mean") or 0.0)}
        for _, r in pts_rows.iterrows()
        if int(r["player_id"]) != exclude_pid
    ]


def _get_player_stats(pmfs_df: pd.DataFrame, player_id: int) -> dict:
    p_rows = pmfs_df[pmfs_df["player_id"] == player_id]
    stats: dict[str, float] = {}
    for stat in ["pts", "reb", "ast", "stl", "blk", "fg3m"]:
        r = p_rows[p_rows["stat"] == stat]
        if not r.empty:
            stats[stat] = float(r.iloc[0].get("pmf_mean") or r.iloc[0].get("stat_mean") or 0.0)
    return stats


if __name__ == "__main__":
    app()
