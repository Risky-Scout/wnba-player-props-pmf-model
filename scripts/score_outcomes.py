#!/usr/bin/env python3
"""Score model predictions against actual game outcomes.

Joins published edges (from deliveries/tonight/ or artifact) against
BDL box scores fetched after game completion. Outputs outcome_tracking.parquet
with columns: date, player_name, stat, line, direction, model_p_over,
market_p_over, edge_pp, kelly_fraction, actual_value, hit (bool),
pnl_units (kelly × outcome), clv_edge (vs closing market odds).

Run nightly after games complete (wired in daily workflow post-game step).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
import requests

app = typer.Typer(add_completion=False)


def fetch_box_scores(game_date: str, bdl_api_key: str) -> pd.DataFrame:
    """Fetch player box scores from BDL for a given date."""
    url = f"https://api.balldontlie.io/wnba/v1/stats?dates[]={game_date}&per_page=100"
    headers = {"Authorization": bdl_api_key}
    rows = []
    next_cursor = None
    while True:
        params: dict = {}
        if next_cursor:
            params["cursor"] = next_cursor
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows.extend(data.get("data", []))
        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Flatten player info
    df["player_id"] = df["player"].apply(
        lambda x: x.get("id") if isinstance(x, dict) else None
    )
    df["player_name"] = df["player"].apply(
        lambda x: f"{x.get('first_name', '')} {x.get('last_name', '')}".strip()
        if isinstance(x, dict)
        else ""
    )
    return df


def actual_for_stat(box: pd.Series, stat: str) -> float | None:
    """Extract the actual value for a given stat from a box score row."""
    _MAP = {
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "fg3m": "fg3m",
        "turnover": "turnover",
        "pts_reb": None,
        "pts_ast": None,
        "reb_ast": None,
        "pts_reb_ast": None,
        "stocks": None,
    }
    if stat == "pts_reb":
        return (box.get("pts") or 0) + (box.get("reb") or 0)
    if stat == "pts_ast":
        return (box.get("pts") or 0) + (box.get("ast") or 0)
    if stat == "reb_ast":
        return (box.get("reb") or 0) + (box.get("ast") or 0)
    if stat == "pts_reb_ast":
        return (box.get("pts") or 0) + (box.get("reb") or 0) + (box.get("ast") or 0)
    if stat == "stocks":
        return (box.get("stl") or 0) + (box.get("blk") or 0)
    col = _MAP.get(stat)
    return float(box.get(col, 0) or 0) if col else None


@app.command()
def main(
    game_date: str = typer.Option(..., help="Date to score (YYYY-MM-DD)"),
    edges_path: Path = typer.Option(
        Path("deliveries/tonight/publishable_edges.parquet")
    ),
    output_path: Path = typer.Option(Path("data/outcome_tracking.parquet")),
    bdl_api_key: str = typer.Option("", envvar="BDL_API_KEY"),
) -> None:
    """Score yesterday's model edges against actual BDL box scores."""
    if not edges_path.exists():
        typer.echo(
            f"[score_outcomes] No edges file at {edges_path}, skipping"
        )
        raise typer.Exit(0)

    edges = pd.read_parquet(edges_path)
    typer.echo(f"[score_outcomes] Loaded {len(edges)} edges for {game_date}")

    if not bdl_api_key:
        typer.echo("[score_outcomes] No BDL_API_KEY — cannot fetch box scores")
        raise typer.Exit(0)

    box_df = fetch_box_scores(game_date, bdl_api_key)
    if box_df.empty:
        typer.echo("[score_outcomes] No box scores returned, skipping")
        raise typer.Exit(0)

    records = []
    for _, edge in edges.iterrows():
        pid = edge.get("player_id")
        stat = edge.get("stat", "")
        line = edge.get("line", 0)
        direction = "OVER" if float(edge.get("edge_over", 0) or 0) >= 0 else "UNDER"
        model_p = float(edge.get("model_prob_over", 0.5) or 0.5)
        market_p = float(edge.get("market_prob_over_no_vig", 0.5) or 0.5)
        kelly = float(edge.get("kelly_fraction", 0) or 0)

        player_box = box_df[box_df["player_id"] == pid]
        if player_box.empty:
            continue

        actual = actual_for_stat(player_box.iloc[0], str(stat))
        if actual is None:
            continue

        hit = bool((actual > line) if direction == "OVER" else (actual <= line))
        pnl = kelly if hit else -kelly

        records.append(
            {
                "date": game_date,
                "player_id": pid,
                "player_name": edge.get("player_name", ""),
                "stat": stat,
                "line": line,
                "direction": direction,
                "model_p_over": round(model_p, 4),
                "market_p_over": round(market_p, 4),
                "edge_pp": round((model_p - market_p) * 100, 2),
                "kelly_fraction": round(kelly, 4),
                "actual_value": actual,
                "hit": hit,
                "pnl_units": round(pnl, 4),
                "role_bucket": edge.get("role_bucket", ""),
                "confidence_tier": edge.get("confidence_tier", ""),
            }
        )

    if not records:
        typer.echo("[score_outcomes] No matchable outcomes found")
        raise typer.Exit(0)

    scored = pd.DataFrame(records)
    hit_rate = float(scored["hit"].mean())
    total_pnl = float(scored["pnl_units"].sum())
    typer.echo(
        f"[score_outcomes] {len(scored)} outcomes: "
        f"hit_rate={hit_rate:.1%}, pnl={total_pnl:+.2f} units"
    )

    # Append to existing tracking file (replace today's rows if re-running)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        existing = existing[existing["date"] != game_date]
        scored = pd.concat([existing, scored], ignore_index=True)

    scored.to_parquet(output_path, index=False)
    typer.echo(
        f"[score_outcomes] Saved {len(scored)} total records to {output_path}"
    )


if __name__ == "__main__":
    app()
