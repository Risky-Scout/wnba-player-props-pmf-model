"""Export a self-contained HTML report with embedded PMF plots.

Sections:
  1. Edge table sorted by edge descending
  2. Per-player PMF bar charts (one per stat with market line)
  3. Game total distributions

All plots are base64-embedded PNGs — no external dependencies at view time.

Usage:
    python scripts/export_html_report.py \
        --pmfs deliveries/next_game/full_pmfs_wide.parquet \
        --game-totals deliveries/next_game/game_totals_2026-06-18.parquet \
        --edges deliveries/next_game/publishable_edges.parquet \
        --game-date 2026-06-18 \
        --out deliveries/next_game/report_2026-06-18.html
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.pmf_grid import WNBAPMFGrid, pmfs_df_to_grids
from wnba_props_model.models.simulation import json_to_pmf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f4f6f9; color: #1a1a2e; line-height: 1.5; }
header { background: #1a1a2e; color: #fff; padding: 18px 32px; display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 1.4rem; letter-spacing: 0.5px; }
header .meta { font-size: 0.85rem; color: #aaa; }
main { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
section { background: #fff; border-radius: 10px; padding: 20px 24px; margin-bottom: 28px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
h2 { font-size: 1.1rem; color: #1a1a2e; margin-bottom: 14px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; }
h3 { font-size: 0.95rem; color: #444; margin: 16px 0 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #1a1a2e; color: #fff; padding: 8px 10px; text-align: left; font-weight: 600; }
td { padding: 7px 10px; border-bottom: 1px solid #eee; }
tr:hover td { background: #f8f9fa; }
.edge-pos { color: #2d8e4e; font-weight: 700; }
.edge-neg { color: #c0392b; }
.pmf-img { max-width: 100%; border-radius: 6px; margin: 6px 0; }
.player-card { border-left: 4px solid #4C72B0; padding-left: 12px; margin-bottom: 20px; }
.player-meta { font-size: 0.82rem; color: #888; margin-bottom: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; background: #e9ecef; color: #555; margin-right: 4px; }
.badge-starter { background: #d4edda; color: #155724; }
.badge-bench { background: #fff3cd; color: #856404; }
.badge-risk { background: #f8d7da; color: #721c24; }
.game-card { border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
footer { text-align: center; color: #888; font-size: 0.78rem; padding: 24px; }
@media (max-width: 700px) { table { font-size: 0.78rem; } }
"""

_BADGE_MAP = {
    "starter": "badge-starter", "star": "badge-starter",
    "bench": "badge-bench", "reserve": "badge-bench",
    "inactive_risk": "badge-risk", "dnp_risk": "badge-risk",
}


def _badge(role: str) -> str:
    cls = _BADGE_MAP.get(role, "")
    return f'<span class="badge {cls}">{role}</span>'


def _edge_cell(edge: float) -> str:
    cls = "edge-pos" if edge > 0 else "edge-neg"
    return f'<span class="{cls}">{edge:+.1%}</span>'


def _build_edge_table_html(edges_df: pd.DataFrame) -> str:
    if edges_df.empty:
        return "<p>No edges available.</p>"

    df = edges_df.copy()
    df = df.sort_values("edge_over", ascending=False)
    # Keep top 50 edges for readability
    df = df.head(50)

    cols = {
        "player_name": "Player",
        "stat": "Stat",
        "line": "Line",
        "model_prob_over": "Model P(Over)",
        "market_prob_over_no_vig": "Market P(Over)",
        "edge_over": "Edge",
        "fair_over_american": "Fair (Over)",
        "role_bucket": "Role",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}

    rows_html = ""
    for _, r in df.iterrows():
        cells = ""
        for col in available:
            val = r.get(col, "")
            if col == "edge_over":
                cells += f"<td>{_edge_cell(float(val))}</td>"
            elif col in ("model_prob_over", "market_prob_over_no_vig") and pd.notna(val):
                cells += f"<td>{float(val):.1%}</td>"
            elif col == "role_bucket":
                cells += f"<td>{_badge(str(val))}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{v}</th>" for v in available.values())
    return f"""
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


def _build_player_sections_html(
    grids: list[WNBAPMFGrid],
    edges_df: pd.DataFrame,
) -> str:
    """Build one section per player with PMF plots for each stat."""
    try:
        from wnba_props_model.visualization.pmf_plots import fig_to_base64, plot_player_pmf
        plots_available = True
    except ImportError:
        plots_available = False

    # Build market line lookup: (player_id, stat) → line
    line_lookup: dict[tuple, float] = {}
    if not edges_df.empty and "player_id" in edges_df.columns and "stat" in edges_df.columns and "line" in edges_df.columns:
        for _, r in edges_df.iterrows():
            key = (r["player_id"], r["stat"])
            if key not in line_lookup:
                line_lookup[key] = float(r["line"])

    html_parts = []
    for grid in sorted(grids, key=lambda g: g.pmf_mean("pts") if g.has_stat("pts") else 0, reverse=True):
        game_date = grid.game_context.get("game_date", "")
        role_badge = _badge(grid.role_bucket)
        mins_str = f"{grid.projected_minutes:.1f} min" if grid.projected_minutes > 0 else ""

        stat_plots_html = ""
        for stat in sorted(grid.stats):
            market_line = line_lookup.get((grid.player_id, stat))
            narrative = grid.narrative(stat, market_line)

            if plots_available:
                try:
                    fig = plot_player_pmf(grid, stat, market_line=market_line)
                    b64 = fig_to_base64(fig)
                    img_tag = f'<img class="pmf-img" src="data:image/png;base64,{b64}" alt="{grid.player_name} {stat} PMF">'
                except Exception as exc:
                    img_tag = f"<p><em>Plot error: {exc}</em></p>"
            else:
                img_tag = ""

            stat_plots_html += f"""
            <div>
              <h3>{stat.upper()}</h3>
              <p class="player-meta">{narrative}</p>
              {img_tag}
            </div>
            """

        html_parts.append(f"""
        <div class="player-card">
          <h3>{grid.player_name} {role_badge}</h3>
          <p class="player-meta">{mins_str} · {game_date}</p>
          {stat_plots_html}
        </div>
        """)

    return "\n".join(html_parts)


def _build_game_totals_html(game_totals_df: pd.DataFrame) -> str:
    """Build one card per game with total distribution summary."""
    if game_totals_df.empty:
        return "<p>No game totals available.</p>"

    try:
        from wnba_props_model.visualization.pmf_plots import fig_to_base64
        from wnba_props_model.models.team_score import WNBATeamScorePMFGrid
        from wnba_props_model.models.simulation import normalize_pmf
        plots_available = True
    except ImportError:
        plots_available = False

    html_parts = []
    for _, r in game_totals_df.iterrows():
        home = r.get("home_team", "Home")
        away = r.get("away_team", "Away")
        total_mean = r.get("total_mean", r.get("expected_total", "?"))
        home_mean = r.get("home_mean", r.get("expected_home", "?"))
        away_mean = r.get("away_mean", r.get("expected_away", "?"))
        game_id = r.get("game_id", "")

        html_parts.append(f"""
        <div class="game-card">
          <h3>{home} vs {away}
            <span class="badge">{r.get('game_date', '')}</span>
          </h3>
          <p class="player-meta">
            E[{home}]={float(home_mean):.1f} · E[{away}]={float(away_mean):.1f} · E[Total]={float(total_mean):.1f}
          </p>
        </div>
        """)

    return "\n".join(html_parts)


def _build_backtest_html(backtest_dir: Path) -> str:
    """Embed equity curve and backtest summary from latest backtest run."""
    import glob  # noqa: PLC0415

    # Find latest backtest JSON
    json_files = sorted(glob.glob(str(backtest_dir / "backtest_*.json")))
    if not json_files:
        return "<p>No backtest results found. Run <code>scripts/backtest_strategy.py</code> first.</p>"

    with open(json_files[-1]) as f:
        import json as _json  # noqa: PLC0415
        summary = _json.load(f)

    # Find equity curve PNG
    equity_png = backtest_dir / "backtest_equity_curve.png"
    roi_png    = backtest_dir / "backtest_per_stat_roi.png"
    kelly_png  = backtest_dir / "backtest_kelly_dist.png"

    try:
        from wnba_props_model.visualization.pmf_plots import fig_to_base64  # noqa: PLC0415
        import base64  # noqa: PLC0415

        def _png_b64(path: Path) -> str | None:
            if not path.exists():
                return None
            with open(path, "rb") as f2:
                return base64.b64encode(f2.read()).decode("ascii")
    except ImportError:
        def _png_b64(path: Path) -> str | None:  # type: ignore[misc]
            return None

    equity_b64 = _png_b64(equity_png)
    roi_b64    = _png_b64(roi_png)
    kelly_b64  = _png_b64(kelly_png)

    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.4f if isinstance(v, float) else v}</td></tr>"
        for k, v in summary.items()
    )
    table_html = f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

    charts_html = ""
    for b64, caption in [
        (equity_b64, "Equity Curve"),
        (roi_b64, "Per-Stat ROI"),
        (kelly_b64, "Kelly Distribution"),
    ]:
        if b64:
            charts_html += f'<div class="chart"><p>{caption}</p><img src="data:image/png;base64,{b64}" style="max-width:100%"></div>\n'

    return f"<div>{table_html}{charts_html}</div>"


def export_html_report(
    game_date: str,
    player_grids: list[WNBAPMFGrid],
    edges_df: pd.DataFrame,
    game_totals_df: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
    backtest_dir: str | Path | None = None,
) -> str:
    """Build a self-contained HTML report string.

    Parameters
    ----------
    game_date : str
        ISO date string (e.g. '2026-06-18').
    player_grids : list[WNBAPMFGrid]
    edges_df : pd.DataFrame
        Publishable edges from build_edge_report.py.
    game_totals_df : pd.DataFrame, optional
        Game totals summary from predict_game_totals.py.
    out_path : str | Path, optional
        If provided, write the HTML to this file.

    Returns
    -------
    str : The full HTML document.
    """
    edge_table = _build_edge_table_html(edges_df)
    player_sections = _build_player_sections_html(player_grids, edges_df)
    gt_html = _build_game_totals_html(game_totals_df) if game_totals_df is not None else "<p>Not available.</p>"
    bt_html = _build_backtest_html(Path(backtest_dir)) if backtest_dir and Path(backtest_dir).exists() else "<p>No backtest results available.</p>"

    n_players = len(player_grids)
    n_edges = len(edges_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WNBA PMF Report — {game_date}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>WNBA PMF Report — {game_date}</h1>
  <div class="meta">{n_players} players · {n_edges} edges</div>
</header>
<main>

<section>
  <h2>📊 Edge Table (Top 50 by Model Edge)</h2>
  {edge_table}
</section>

<section>
  <h2>🏀 Game Total Distributions</h2>
  {gt_html}
</section>

<section>
  <h2>📈 Strategy Backtest</h2>
  {bt_html}
</section>

<section>
  <h2>📈 Player PMF Distributions</h2>
  {player_sections}
</section>

</main>
<footer>
  Generated {game_date} · WNBA PMF Elite Build v1 · Push-correct probabilities via WNBAPMFGrid
</footer>
</body>
</html>"""

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html, encoding="utf-8")
        logger.info("HTML report written → %s (%d bytes)", out_path, len(html))

    return html


@app.command()
def main(
    pmfs: str = typer.Option(..., help="full_pmfs_wide.parquet from predict_today.py."),
    game_date: str = typer.Option(..., "--game-date", help="ISO date (YYYY-MM-DD)."),
    out: str = typer.Option(..., "--out", help="Output HTML file path."),
    game_totals: str | None = typer.Option(None, "--game-totals",
        help="game_totals_{date}.parquet from predict_game_totals.py."),
    edges: str | None = typer.Option(None, "--edges",
        help="publishable_edges.parquet from build_edge_report.py."),
    bt_dir: str | None = typer.Option(None, "--backtest-dir",
        help="Directory containing backtest results (artifacts/reports)."),
    max_players: int = typer.Option(30, help="Max players to render in full PMF detail."),
) -> None:
    """Build a self-contained HTML report with PMF plots and edge table."""
    typer.echo(f"Loading PMFs from {pmfs}...")
    pmfs_df = pd.read_parquet(pmfs)
    typer.echo(f"  {len(pmfs_df):,} PMF rows")

    # Convert PMF DataFrame to WNBAPMFGrid objects
    ctx_cols = ["game_id", "game_date", "team_id", "opponent_team_id", "home_away"]
    grids = pmfs_df_to_grids(pmfs_df, game_context_cols=ctx_cols)
    typer.echo(f"  {len(grids)} player PMF grids")

    # Limit players for rendering speed
    if len(grids) > max_players:
        grids = grids[:max_players]
        typer.echo(f"  [INFO] Limiting to {max_players} players for HTML report")

    edges_df = pd.DataFrame()
    if edges and Path(edges).exists():
        edges_df = pd.read_parquet(edges)
        typer.echo(f"  {len(edges_df)} publishable edges")

    gt_df = None
    if game_totals and Path(game_totals).exists():
        gt_df = pd.read_parquet(game_totals)
        typer.echo(f"  {len(gt_df)} game total rows")

    html = export_html_report(
        game_date=game_date,
        player_grids=grids,
        edges_df=edges_df,
        game_totals_df=gt_df,
        out_path=out,
        backtest_dir=bt_dir,
    )
    typer.echo(f"\nHTML report → {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    app()
