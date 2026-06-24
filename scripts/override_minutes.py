"""Manual minutes override CLI (blueprint §6.1).

Writes an entry to config/player_overrides.json, then immediately applies the
override to the PMF slate via UTM redistribution and re-exports the pre-game
JSON output.

Safety rules (§6.3):
  - minutes must be in [0, 48]
  - stat projections must be non-negative (enforced downstream)
  - overrides expire after game_date
  - multiple overrides for same player/date are rejected (last-write-wins with warning)
  - every override is logged with timestamp, author, reason

Usage:
    python scripts/override_minutes.py \\
        --player-id 842 \\
        --game-date 2026-07-16 \\
        --minutes 28.0 \\
        --reason "Back-to-back rest management"

    # List active overrides:
    python scripts/override_minutes.py --list

    # Remove an override:
    python scripts/override_minutes.py --remove --player-id 842 --game-date 2026-07-16
"""
from __future__ import annotations

import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

_OVERRIDES_FILE = Path("config/player_overrides.json")
_MINUTES_BOUNDS = (0.0, 48.0)


@app.command()
def main(
    player_id: int = typer.Option(0, "--player-id", help="BDL player ID."),
    game_date: str = typer.Option("", "--game-date", help="ISO game date YYYY-MM-DD."),
    minutes: float = typer.Option(-1.0, "--minutes", help="Override projected minutes [0, 48]."),
    reason: str = typer.Option("", "--reason", help="Reason for override (logged)."),
    author: str = typer.Option("", "--author", help="Author (defaults to system user)."),
    apply_to_slate: bool = typer.Option(
        True, "--apply/--no-apply",
        help="Immediately apply override to the PMF slate (default True).",
    ),
    slate: str = typer.Option("", "--slate", help="PMF slate parquet. Auto-detected if not set."),
    usage_parquet: str = typer.Option(
        "data/processed/player_season_adv_usage.parquet", "--usage-parquet"
    ),
    list_overrides: bool = typer.Option(False, "--list", help="List active overrides and exit."),
    remove: bool = typer.Option(False, "--remove", help="Remove an override for player/date."),
) -> None:
    """Add, remove, or list manual minutes overrides (blueprint §6.1)."""
    overrides_data = _load_overrides()

    if list_overrides:
        _list(overrides_data)
        raise typer.Exit(0)

    if not player_id:
        typer.echo("[ERROR] --player-id is required.", err=True)
        raise typer.Exit(1)
    if not game_date:
        typer.echo("[ERROR] --game-date is required.", err=True)
        raise typer.Exit(1)

    if remove:
        _remove_override(overrides_data, player_id, game_date)
        raise typer.Exit(0)

    if minutes < 0:
        typer.echo("[ERROR] --minutes is required for adding an override.", err=True)
        raise typer.Exit(1)

    # Validate bounds
    lo, hi = _MINUTES_BOUNDS
    if not (lo <= minutes <= hi):
        typer.echo(f"[ERROR] minutes={minutes} out of valid range [{lo}, {hi}].", err=True)
        raise typer.Exit(1)

    author = author or _get_author()
    existing = [
        o for o in overrides_data["overrides"]
        if o["player_id"] == player_id and o["game_date"] == game_date
    ]
    if existing:
        typer.echo(
            f"[WARN] Override already exists for player {player_id} on {game_date}. "
            "Overwriting (last-write-wins).", err=True
        )
        overrides_data["overrides"] = [
            o for o in overrides_data["overrides"]
            if not (o["player_id"] == player_id and o["game_date"] == game_date)
        ]

    record = {
        "player_id": player_id,
        "game_date": game_date,
        "override_minutes": round(minutes, 1),
        "reason": reason or "manual",
        "applied_by": author,
        "applied_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    overrides_data["overrides"].append(record)
    _save_overrides(overrides_data)
    typer.echo(f"Override saved: player {player_id}, date {game_date}, minutes={minutes}")

    if apply_to_slate:
        _apply_to_slate(player_id, game_date, minutes, reason, slate, usage_parquet)


def _apply_to_slate(
    player_id: int,
    game_date: str,
    override_minutes: float,
    reason: str,
    slate: str,
    usage_parquet: str,
) -> None:
    """Apply the override to the live PMF slate using UTM redistribution."""
    import pandas as pd

    slate_path = _resolve_slate(slate, game_date)
    if slate_path is None:
        typer.echo("[WARN] No PMF slate found — override saved but not applied to slate.", err=True)
        return

    pmfs_df = pd.read_parquet(slate_path)
    p_mask = pmfs_df["player_id"] == player_id
    if not p_mask.any():
        typer.echo(f"[WARN] player_id={player_id} not found in slate.", err=True)
        return

    # Current model-projected minutes
    current_mins = float(pmfs_df.loc[p_mask, "minutes_mean"].iloc[0])
    delta_mins = override_minutes - current_mins
    typer.echo(f"  model_minutes={current_mins:.1f} → override_minutes={override_minutes:.1f} (Δ={delta_mins:+.1f})")

    # Update player's minutes and scale stats
    pmfs_df.loc[p_mask, "minutes_mean"] = override_minutes
    pmfs_df.loc[p_mask, "override_applied"] = True
    pmfs_df.loc[p_mask, "override_source"] = f"manual:{reason[:50]}"
    if current_mins > 0:
        scale = override_minutes / current_mins
        for col in ["stat_mean", "pmf_mean", "mean"]:
            if col in pmfs_df.columns:
                pmfs_df.loc[p_mask, col] = (pmfs_df.loc[p_mask, col] * scale).clip(lower=0)

    # UTM redistribution for delta
    if abs(delta_mins) > 0.1:
        game_ids = pmfs_df.loc[p_mask, "game_id"].unique()
        for gid in game_ids:
            g_mask = pmfs_df["game_id"] == gid
            utm = _load_utm(usage_parquet, pmfs_df)
            teammates = _get_teammates(pmfs_df, g_mask, player_id)
            base_mins_map = {t["player_id"]: t["projected_minutes"] for t in teammates}
            if delta_mins < 0:  # player gaining minutes from reduced teammates
                # Simple linear scale: teammates lose proportionally
                total_teammate_mins = sum(t["projected_minutes"] for t in teammates) or 1e-6
                for t in teammates:
                    frac = t["projected_minutes"] / total_teammate_mins
                    t["projected_minutes"] = max(0, t["projected_minutes"] + delta_mins * frac)
                updated_roster = teammates
            else:
                updated_roster, _ = utm.redistribute(
                    roster=teammates, out_player_ids=[player_id],
                    out_minutes_dict={player_id: delta_mins}
                )
            _apply_utm_to_df(pmfs_df, g_mask, updated_roster, base_mins_map)

    pmfs_df.to_parquet(slate_path, index=False)
    typer.echo(f"Slate updated → {slate_path}")


def _list(overrides_data: dict) -> None:
    ovs = overrides_data.get("overrides", [])
    if not ovs:
        typer.echo("No active overrides.")
        return
    typer.echo(f"{'player_id':>10}  {'game_date':>12}  {'minutes':>8}  reason")
    typer.echo("-" * 60)
    for o in ovs:
        typer.echo(
            f"{o['player_id']:>10}  {o['game_date']:>12}  {o['override_minutes']:>8.1f}  {o.get('reason','')}"
        )


def _remove_override(overrides_data: dict, player_id: int, game_date: str) -> None:
    before = len(overrides_data["overrides"])
    overrides_data["overrides"] = [
        o for o in overrides_data["overrides"]
        if not (o["player_id"] == player_id and o["game_date"] == game_date)
    ]
    after = len(overrides_data["overrides"])
    _save_overrides(overrides_data)
    typer.echo(f"Removed {before - after} override(s) for player {player_id} on {game_date}")


def _load_overrides() -> dict:
    if not _OVERRIDES_FILE.exists():
        return {"overrides": []}
    try:
        return json.loads(_OVERRIDES_FILE.read_text())
    except Exception:
        return {"overrides": []}


def _save_overrides(data: dict) -> None:
    _OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDES_FILE.write_text(json.dumps(data, indent=2))


def _get_author() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "system"


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


def _load_utm(usage_parquet: str, pmfs_df: "pd.DataFrame"):
    from wnba_props_model.models.usage_transfer import UsageTransferMatrix  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    p = Path(usage_parquet)
    if p.exists():
        try:
            return UsageTransferMatrix(pd.read_parquet(p))
        except Exception:
            pass
    usg_df = pd.DataFrame({
        "player_id": pmfs_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    return UsageTransferMatrix(usg_df)


def _get_teammates(pmfs_df: "pd.DataFrame", g_mask: "pd.Series", exclude_pid: int) -> list[dict]:
    game_df = pmfs_df[g_mask]
    pts_rows = game_df[game_df["stat"] == "pts"]
    return [
        {"player_id": int(r["player_id"]), "projected_minutes": float(r.get("minutes_mean") or 0.0)}
        for _, r in pts_rows.iterrows()
        if int(r["player_id"]) != exclude_pid
    ]


def _apply_utm_to_df(pmfs_df: "pd.DataFrame", g_mask: "pd.Series",
                     updated_roster: list[dict], base_mins_map: dict) -> None:
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
                    pmfs_df.loc[p_mask, col] = (pmfs_df.loc[p_mask, col] * scale).clip(lower=0)


if __name__ == "__main__":
    app()
