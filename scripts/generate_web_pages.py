"""Generate static web pages for the WizardOfOdds WNBA predictions site.

Produces three output directories (mirrors of the gh-pages deploy targets):

  tools/odds-scanner/predictions/WNBA/Pre-Game/Edge/
    index.html          — Edge Board dashboard (interactive table)
    latest.json         — Today's edge data (re-written daily)
    {date}.json         — Date-stamped archive copy

  tools/odds-scanner/predictions/WNBA/Pre-Game/PMF-Distributions/
    index.html          — PMF Chart dashboard (interactive visualizations)
    latest.json         — Today's full PMF data with moments
    {date}.json         — Date-stamped archive copy

The HTML pages fetch `latest.json` (or `{date}.json` when passed ?date=YYYY-MM-DD)
via a relative URL and render entirely client-side—no build step required.

Usage:
    python scripts/generate_web_pages.py \\
        --game-date 2026-06-28 \\
        --projections deliveries/tonight/player_projections_2026-06-28.parquet \\
        --edges deliveries/tonight/publishable_edges.parquet \\
        --out-dir tools/odds-scanner/predictions/WNBA/Pre-Game
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# PMF helpers
# ---------------------------------------------------------------------------

def _parse_pmf(pmf_json_str: str | None) -> tuple[list[list], float, float, float, float, float, int, float]:
    """Return (pairs, mu, var, std, skew, kurt, mode, median) from a pmf_json string."""
    empty = ([], 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    if not pmf_json_str:
        return empty
    try:
        d = json.loads(pmf_json_str)
        k = np.array([int(x) for x in d.keys()], dtype=float)
        p = np.array(list(d.values()), dtype=float)
        p = p / p.sum()
        mu = float(np.dot(k, p))
        var = float(np.dot((k - mu) ** 2, p))
        std = float(np.sqrt(max(var, 1e-9)))
        skew = float(np.dot(((k - mu) / std) ** 3, p))
        kurt = float(np.dot(((k - mu) / std) ** 4, p) - 3.0)
        mode_k = int(k[np.argmax(p)])
        cum = np.cumsum(p)
        median_k = float(k[np.searchsorted(cum, 0.5)])
        pairs = [[int(kk), round(float(pp), 5)] for kk, pp in zip(k, p) if pp > 0.001]
        return pairs, mu, var, std, skew, kurt, mode_k, median_k
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# JSON data builders
# ---------------------------------------------------------------------------

def _build_edge_json(edges_df: pd.DataFrame, proj_df: pd.DataFrame, game_date: str) -> dict:
    """Build the payload for Pre-Game/Edge/latest.json."""
    merged = edges_df.merge(
        proj_df[["player_id", "stat", "pmf_mean", "median"]],
        on=["player_id", "stat"],
        how="left",
        suffixes=("", "_proj"),
    )
    rows = []
    for _, r in merged.iterrows():
        edge = float(r.get("edge_over", 0) or 0)
        kelly = float(r.get("kelly_fraction", 0) or 0)
        rows.append({
            "player": r["player_name"],
            "stat": str(r["stat"]).upper(),
            "direction": "OVER" if edge >= 0 else "UNDER",
            "model_mean": round(float(r.get("pmf_mean", 0) or 0), 2),
            "median": round(float(r.get("median", 0) or 0), 1),
            "market_line": round(float(r.get("line", 0) or 0), 1),
            "model_p_over": round(float(r.get("model_prob_over", 0) or 0), 4),
            "market_p_over": round(float(r.get("market_prob_over_no_vig", 0) or 0), 4),
            "edge": round(edge, 4),
            "kelly_pct": round(kelly * 100, 2),
            "abs_edge": abs(edge),
            "clv_adj_edge": round(float(r.get("clv_decay_adjusted_edge", edge) or edge), 4),
            "line_moved_toward_over": bool(r.get("line_moved_toward_over", False)),
            "reverse_line_movement": bool(r.get("reverse_line_movement_flag", False)),
            "model_market_ratio": round(float(r.get("model_market_ratio", 1) or 1), 3),
        })
    rows.sort(key=lambda x: -x["abs_edge"])
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(rows),
        "over_signals": sum(1 for r in rows if r["direction"] == "OVER"),
        "under_signals": sum(1 for r in rows if r["direction"] == "UNDER"),
        "props": rows,
    }


def _build_pmf_json(edges_df: pd.DataFrame, proj_df: pd.DataFrame, game_date: str) -> dict:
    """Build the payload for Pre-Game/PMF-Distributions/latest.json."""
    merged = edges_df.merge(
        proj_df[["player_id", "stat", "pmf_json", "pmf_mean", "pmf_variance", "median", "mode"]],
        on=["player_id", "stat"],
        how="left",
        suffixes=("", "_proj"),
    )
    props = []
    for _, r in merged.iterrows():
        pmf_str = r.get("pmf_json", "{}") or "{}"
        pairs, mu, var, std, skew, kurt, mode_k, median_k = _parse_pmf(pmf_str)
        if not pairs:
            continue
        edge = float(r.get("edge_over", 0) or 0)
        props.append({
            "player": r["player_name"],
            "stat": str(r["stat"]).upper(),
            "stat_raw": str(r["stat"]),
            "line": round(float(r.get("line", 0) or 0), 1),
            "mean": round(mu, 2),
            "median": round(median_k, 1),
            "mode": mode_k,
            "variance": round(var, 3),
            "std_dev": round(std, 3),
            "skewness": round(skew, 3),
            "excess_kurtosis": round(kurt, 3),
            "model_p_over": round(float(r.get("model_prob_over", 0) or 0), 4),
            "market_p_over": round(float(r.get("market_prob_over_no_vig", 0) or 0), 4),
            "edge": round(edge, 4),
            "kelly_pct": round(float(r.get("kelly_fraction", 0) or 0) * 100, 2),
            "pmf": pairs,
        })
    props.sort(key=lambda x: -abs(x["edge"]))
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(props),
        "props": props,
    }


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_EDGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA Pre-Game Edge Board — WizardOfOdds</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e4e4e4;line-height:1.5;min-height:100vh}
header{background:#141622;border-bottom:1px solid #2a2d3e;padding:14px 28px;display:flex;align-items:center;gap:16px}
.logo{font-size:1rem;font-weight:700;color:#fff;letter-spacing:.3px}
.logo span{color:#599ce7}
.header-meta{font-size:.78rem;color:#888;margin-left:auto}
.header-date{font-size:.85rem;font-weight:600;color:#aaa;margin-left:8px}
main{max-width:1300px;margin:24px auto;padding:0 18px}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.stat-card{background:#141622;border:1px solid #2a2d3e;border-radius:8px;padding:14px 18px}
.stat-card .value{font-size:1.6rem;font-weight:700;color:#e4e4e4}
.stat-card .label{font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.stat-card .value.green{color:#3fa266}
.stat-card .value.red{color:#e05a6a}
.stat-card .value.blue{color:#599ce7}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.filter-group{display:flex;gap:6px;flex-wrap:wrap}
.pill{background:#1e2130;border:1px solid #2a2d3e;color:#aaa;border-radius:100px;padding:4px 14px;font-size:.78rem;cursor:pointer;transition:all .15s;white-space:nowrap}
.pill:hover{background:#2a2d3e;color:#e4e4e4}
.pill.active{background:#599ce7;border-color:#599ce7;color:#fff;font-weight:600}
.pill.active.green{background:#3fa266;border-color:#3fa266}
.pill.active.red{background:#e05a6a;border-color:#e05a6a}
.search{background:#1e2130;border:1px solid #2a2d3e;color:#e4e4e4;border-radius:6px;padding:5px 12px;font-size:.82rem;outline:none;width:200px}
.search:focus{border-color:#599ce7}
.sort-hint{font-size:.75rem;color:#666;margin-left:auto}
.table-wrap{overflow-x:auto;border-radius:8px;border:1px solid #2a2d3e}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead{position:sticky;top:0;z-index:1}
th{background:#141622;color:#888;padding:10px 14px;text-align:left;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;cursor:pointer;white-space:nowrap;border-bottom:1px solid #2a2d3e;user-select:none}
th:hover{color:#e4e4e4}
th.sorted{color:#599ce7}
th .sort-arrow{margin-left:4px;opacity:.6}
td{padding:9px 14px;border-bottom:1px solid #1e2130;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1a1d2b}
.player{font-weight:500;color:#e4e4e4}
.stat-badge{display:inline-block;background:#1e2130;border:1px solid #2a2d3e;border-radius:4px;padding:1px 7px;font-size:.72rem;font-weight:600;color:#aaa}
.dir-over{color:#3fa266;font-weight:700;font-size:.78rem}
.dir-under{color:#e05a6a;font-weight:700;font-size:.78rem}
.edge-pos{color:#3fa266;font-weight:700}
.edge-neg{color:#e05a6a;font-weight:700}
.kelly-val{color:#599ce7;font-weight:600}
.kelly-zero{color:#555}
.prob{color:#c4c4c4}
.numeric{text-align:right;font-variant-numeric:tabular-nums}
.badge-rlm{display:inline-block;background:#7b4fcf22;border:1px solid #7b4fcf44;color:#9386f2;border-radius:3px;font-size:.65rem;padding:1px 5px;margin-left:4px}
.footer{text-align:center;font-size:.75rem;color:#555;padding:32px 0 20px;border-top:1px solid #1e2130;margin-top:24px}
.loading{text-align:center;color:#666;padding:60px;font-size:.9rem}
.error{text-align:center;color:#e05a6a;padding:40px;font-size:.85rem}
@media(max-width:768px){.summary{grid-template-columns:repeat(2,1fr)}.search{width:140px}}
</style>
</head>
<body>
<header>
  <div class="logo">WizardOf<span>Odds</span></div>
  <div style="font-size:.8rem;color:#666;border-left:1px solid #2a2d3e;margin-left:8px;padding-left:14px">WNBA · Pre-Game Edge Board</div>
  <div class="header-meta">Model projections vs. no-vig market · Sorted by edge magnitude</div>
  <div class="header-date" id="hdrDate">—</div>
</header>
<main>
  <div class="summary" id="summary">
    <div class="stat-card"><div class="value" id="s-total">—</div><div class="label">Props Analyzed</div></div>
    <div class="stat-card"><div class="value green" id="s-over">—</div><div class="label">Over Signals</div></div>
    <div class="stat-card"><div class="value red" id="s-under">—</div><div class="label">Under Signals</div></div>
    <div class="stat-card"><div class="value blue" id="s-top">—</div><div class="label">Top Kelly</div></div>
  </div>

  <div class="controls">
    <div class="filter-group" id="statFilters">
      <button class="pill active" data-stat="ALL">All</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
      <button class="pill" data-stat="3PM">3PM</button>
    </div>
    <div class="filter-group">
      <button class="pill active green" data-dir="ALL">Both</button>
      <button class="pill" data-dir="OVER">Over only</button>
      <button class="pill" data-dir="UNDER">Under only</button>
    </div>
    <input class="search" type="text" placeholder="Filter player..." id="searchInput">
    <div class="sort-hint" id="sortHint">Click column headers to sort</div>
  </div>

  <div class="table-wrap">
    <table id="edgeTable">
      <thead>
        <tr>
          <th data-col="player">Player<span class="sort-arrow"></span></th>
          <th data-col="stat">Stat<span class="sort-arrow"></span></th>
          <th data-col="direction">Dir<span class="sort-arrow"></span></th>
          <th data-col="model_mean" class="numeric">Mdl Mean<span class="sort-arrow"></span></th>
          <th data-col="median" class="numeric">Median<span class="sort-arrow"></span></th>
          <th data-col="market_line" class="numeric">Mkt Line<span class="sort-arrow"></span></th>
          <th data-col="model_p_over" class="numeric">Mdl P%<span class="sort-arrow"></span></th>
          <th data-col="market_p_over" class="numeric">Mkt P%<span class="sort-arrow"></span></th>
          <th data-col="edge" class="numeric sorted">Edge%<span class="sort-arrow">↓</span></th>
          <th data-col="kelly_pct" class="numeric">Kelly%<span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody id="tableBody"><tr><td colspan="10" class="loading">Loading projections…</td></tr></tbody>
    </table>
  </div>

  <div class="footer">
    <div>Generated <span id="genTime">—</span> · WNBA Pre-Game Model · WizardOfOdds Sports Analytics</div>
    <div style="margin-top:6px;color:#444">Edge = Model P(over) − Market P(over) no-vig · Kelly criterion sizing based on fractional Kelly · Positive edge = model favors OVER</div>
  </div>
</main>

<script>
(function() {
  let allProps = [];
  let sortCol = 'edge';
  let sortAsc = false; // start descending by |edge|
  let statFilter = 'ALL';
  let dirFilter = 'ALL';
  let searchFilter = '';

  // Load data
  const params = new URLSearchParams(location.search);
  const dateParam = params.get('date');
  const dataUrl = dateParam ? `${dateParam}.json` : 'latest.json';

  fetch(dataUrl)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(data => {
      allProps = data.props || [];
      document.getElementById('s-total').textContent = data.total_props || 0;
      document.getElementById('s-over').textContent = data.over_signals || 0;
      document.getElementById('s-under').textContent = data.under_signals || 0;
      const topK = allProps.find(p => p.kelly_pct > 0);
      document.getElementById('s-top').textContent = topK ? `${topK.kelly_pct.toFixed(1)}%` : '—';
      document.getElementById('hdrDate').textContent = data.game_date || '';
      document.getElementById('genTime').textContent = data.generated_at ? new Date(data.generated_at).toLocaleString() : '—';
      render();
    })
    .catch(err => {
      document.getElementById('tableBody').innerHTML = `<tr><td colspan="10" class="error">Failed to load data: ${err.message}. <a href="latest.json" style="color:#599ce7">latest.json</a></td></tr>`;
    });

  function filtered() {
    return allProps.filter(p => {
      if (statFilter !== 'ALL') {
        const st = p.stat === 'FG3M' ? '3PM' : p.stat;
        if (st !== statFilter) return false;
      }
      if (dirFilter !== 'ALL' && p.direction !== dirFilter) return false;
      if (searchFilter && !p.player.toLowerCase().includes(searchFilter)) return false;
      return true;
    });
  }

  function sorted(rows) {
    return [...rows].sort((a, b) => {
      let av = a[sortCol], bv = b[sortCol];
      if (sortCol === 'edge') { av = Math.abs(av); bv = Math.abs(bv); }
      if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortAsc ? av - bv : bv - av;
    });
  }

  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function sign(v) { return v >= 0 ? '+' : ''; }

  function render() {
    const rows = sorted(filtered());
    const tbody = document.getElementById('tableBody');
    if (!rows.length) { tbody.innerHTML = '<tr><td colspan="10" class="loading">No props match the current filters.</td></tr>'; return; }
    tbody.innerHTML = rows.map(p => {
      const isOver = p.direction === 'OVER';
      const edgeFmt = `${sign(p.edge)}${(p.edge * 100).toFixed(1)}%`;
      const rlm = p.reverse_line_movement ? '<span class="badge-rlm">RLM</span>' : '';
      return `<tr>
        <td class="player">${p.player}${rlm}</td>
        <td><span class="stat-badge">${p.stat === 'FG3M' ? '3PM' : p.stat}</span></td>
        <td><span class="${isOver ? 'dir-over' : 'dir-under'}">${p.direction}</span></td>
        <td class="numeric">${p.model_mean.toFixed(2)}</td>
        <td class="numeric">${p.median.toFixed(1)}</td>
        <td class="numeric">${p.market_line}</td>
        <td class="numeric prob">${pct(p.model_p_over)}</td>
        <td class="numeric prob">${pct(p.market_p_over)}</td>
        <td class="numeric ${isOver ? 'edge-pos' : 'edge-neg'}">${edgeFmt}</td>
        <td class="numeric ${p.kelly_pct > 0 ? 'kelly-val' : 'kelly-zero'}">${p.kelly_pct > 0 ? p.kelly_pct.toFixed(1) + '%' : '—'}</td>
      </tr>`;
    }).join('');
  }

  // Stat filter
  document.getElementById('statFilters').addEventListener('click', e => {
    const btn = e.target.closest('[data-stat]');
    if (!btn) return;
    statFilter = btn.dataset.stat;
    document.querySelectorAll('[data-stat]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  });

  // Dir filter
  document.querySelectorAll('[data-dir]').forEach(btn => {
    btn.addEventListener('click', () => {
      dirFilter = btn.dataset.dir;
      document.querySelectorAll('[data-dir]').forEach(b => b.classList.remove('active', 'green', 'red'));
      btn.classList.add('active');
      if (dirFilter === 'OVER') btn.classList.add('green');
      if (dirFilter === 'UNDER') btn.classList.add('red');
      render();
    });
  });

  // Search
  document.getElementById('searchInput').addEventListener('input', e => {
    searchFilter = e.target.value.toLowerCase().trim();
    render();
  });

  // Column sort
  document.querySelectorAll('th[data-col]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) { sortAsc = !sortAsc; }
      else { sortCol = col; sortAsc = col === 'player' || col === 'stat'; }
      document.querySelectorAll('th').forEach(t => { t.classList.remove('sorted'); t.querySelector('.sort-arrow').textContent = ''; });
      th.classList.add('sorted');
      th.querySelector('.sort-arrow').textContent = sortAsc ? '↑' : '↓';
      render();
    });
  });
})();
</script>
</body>
</html>"""

_PMF_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA PMF Distributions — WizardOfOdds</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e4e4e4;line-height:1.5;min-height:100vh}
header{background:#141622;border-bottom:1px solid #2a2d3e;padding:14px 28px;display:flex;align-items:center;gap:16px}
.logo{font-size:1rem;font-weight:700;color:#fff;letter-spacing:.3px}
.logo span{color:#599ce7}
.header-meta{font-size:.78rem;color:#888;margin-left:auto}
.header-date{font-size:.85rem;font-weight:600;color:#aaa;margin-left:8px}
main{max-width:1400px;margin:24px auto;padding:0 18px}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.filter-group{display:flex;gap:6px;flex-wrap:wrap}
.pill{background:#1e2130;border:1px solid #2a2d3e;color:#aaa;border-radius:100px;padding:4px 14px;font-size:.78rem;cursor:pointer;transition:all .15s;white-space:nowrap}
.pill:hover{background:#2a2d3e;color:#e4e4e4}
.pill.active{background:#599ce7;border-color:#599ce7;color:#fff;font-weight:600}
.search{background:#1e2130;border:1px solid #2a2d3e;color:#e4e4e4;border-radius:6px;padding:5px 12px;font-size:.82rem;outline:none;width:200px}
.search:focus{border-color:#599ce7}
.legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;background:#141622;border:1px solid #2a2d3e;border-radius:6px;padding:8px 14px;margin-bottom:14px;font-size:.75rem;color:#aaa}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-line{width:22px;height:2px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#141622;border:1px solid #2a2d3e;border-radius:10px;overflow:hidden}
.card-header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #1e2130;background:#0f1117}
.card-title{font-size:.88rem;font-weight:600;color:#e4e4e4}
.stat-tag{background:#1e2130;border:1px solid #2a2d3e;border-radius:4px;padding:1px 7px;font-size:.72rem;font-weight:700;color:#aaa;margin-left:6px}
.edge-badge{font-size:.78rem;font-weight:700;padding:2px 10px;border-radius:100px}
.edge-over{background:#3fa26622;color:#3fa266;border:1px solid #3fa26644}
.edge-under{background:#e05a6a22;color:#e05a6a;border:1px solid #e05a6a44}
.card-body{display:flex;gap:0;padding:12px 14px 14px}
.chart-area{flex:0 0 auto;position:relative}
.stats-panel{flex:1 1 0;min-width:0;padding-left:14px;border-left:1px solid #1e2130}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:.78rem}
.stat-label{color:#888}
.stat-value{font-weight:500;color:#e4e4e4;font-variant-numeric:tabular-nums}
.stat-value.green{color:#3fa266}
.stat-value.red{color:#e05a6a}
.stat-value.blue{color:#599ce7}
.stat-value.amber{color:#f1b467}
.divider{height:1px;background:#1e2130;margin:5px 0}
.footer{text-align:center;font-size:.75rem;color:#555;padding:32px 0 20px;border-top:1px solid #1e2130;margin-top:24px}
.loading{text-align:center;color:#666;padding:60px;font-size:.9rem}
</style>
</head>
<body>
<header>
  <div class="logo">WizardOf<span>Odds</span></div>
  <div style="font-size:.8rem;color:#666;border-left:1px solid #2a2d3e;margin-left:8px;padding-left:14px">WNBA · Full PMF Distributions</div>
  <div class="header-meta">Probability mass functions with market analysis · Computed from model posteriors</div>
  <div class="header-date" id="hdrDate">—</div>
</header>
<main>
  <div class="controls">
    <div class="filter-group" id="statFilters">
      <button class="pill active" data-stat="ALL">All</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
      <button class="pill" data-stat="3PM">3PM</button>
    </div>
    <input class="search" type="text" placeholder="Filter player..." id="searchInput">
  </div>

  <div class="legend">
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#e05a6a" stroke-width="2" stroke-dasharray="5 3"/></svg>Mkt Line</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#599ce7" stroke-width="2" stroke-dasharray="4 2"/></svg>Mean</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#f1b467" stroke-width="2" stroke-dasharray="4 2"/></svg>Median</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#81a1c1" stroke-width="2" stroke-dasharray="4 2"/></svg>Mode</div>
    <div class="legend-item"><svg width="14" height="10"><rect x="0" y="1" width="14" height="8" fill="#3fa266" fill-opacity="0.5"/></svg>Over region</div>
    <div class="legend-item"><svg width="14" height="10"><rect x="0" y="1" width="14" height="8" fill="#d08770" fill-opacity="0.45"/></svg>Under region</div>
  </div>

  <div class="grid" id="pmfGrid"><div class="loading" style="grid-column:1/-1">Loading PMF data…</div></div>

  <div class="footer">
    <div>Generated <span id="genTime">—</span> · WNBA Pre-Game Model · WizardOfOdds Sports Analytics</div>
    <div style="margin-top:6px;color:#444">PMFs computed from Negative Binomial posterior · Vertical lines: market line (pink), model mean (blue), median (amber), mode (teal) · Green bars = over-line outcomes</div>
  </div>
</main>

<script>
(function(){
  let allProps = [];
  let statFilter = 'ALL';
  let searchFilter = '';

  const params = new URLSearchParams(location.search);
  const dateParam = params.get('date');
  const dataUrl = dateParam ? `${dateParam}.json` : 'latest.json';

  fetch(dataUrl)
    .then(r => { if(!r.ok) throw new Error(r.status); return r.json(); })
    .then(data => {
      allProps = data.props || [];
      document.getElementById('hdrDate').textContent = data.game_date || '';
      document.getElementById('genTime').textContent = data.generated_at ? new Date(data.generated_at).toLocaleString() : '—';
      render();
    })
    .catch(err => {
      document.getElementById('pmfGrid').innerHTML = `<div class="loading" style="grid-column:1/-1;color:#e05a6a">Failed to load: ${err.message}</div>`;
    });

  function filtered() {
    return allProps.filter(p => {
      const st = p.stat === 'FG3M' ? '3PM' : p.stat;
      if (statFilter !== 'ALL' && st !== statFilter) return false;
      if (searchFilter && !p.player.toLowerCase().includes(searchFilter)) return false;
      return true;
    });
  }

  // --- PMF SVG chart ---
  function buildSVG(prop) {
    const W = 300, H = 150, PL = 26, PR = 6, PT = 6, PB = 22;
    const pw = W - PL - PR, ph = H - PT - PB;
    const pmf = prop.pmf;
    if (!pmf || !pmf.length) return '';
    const kMin = pmf[0][0], kMax = pmf[pmf.length-1][0];
    const numBins = kMax - kMin + 1;
    const bw = pw / numBins;
    const pMax = Math.max(...pmf.map(([,p]) => p));
    const vx = v => PL + (v - kMin) * bw;
    const barH = p => (p / pMax) * ph;

    const bars = pmf.map(([k,p]) => {
      const x = vx(k), h = barH(p);
      const isOver = k > prop.line;
      const fill = isOver ? '#3fa266' : '#d08770';
      const opacity = isOver ? 0.6 : 0.45;
      return `<rect x="${(x+0.5).toFixed(1)}" y="${(PT+ph-h).toFixed(1)}" width="${Math.max(bw-1,1).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" fill-opacity="${opacity}"/>`;
    }).join('');

    const line = `<line x1="${vx(prop.line).toFixed(1)}" y1="${PT}" x2="${vx(prop.line).toFixed(1)}" y2="${PT+ph}" stroke="#e05a6a" stroke-width="2" stroke-dasharray="5 3"/>`;
    const meanX = (vx(prop.mean)+bw/2).toFixed(1);
    const meanL = `<line x1="${meanX}" y1="${PT}" x2="${meanX}" y2="${PT+ph}" stroke="#599ce7" stroke-width="1.5" stroke-dasharray="4 2"/>`;
    const medX = (vx(prop.median)+bw/2).toFixed(1);
    const medL = `<line x1="${medX}" y1="${PT}" x2="${medX}" y2="${PT+ph}" stroke="#f1b467" stroke-width="1.5" stroke-dasharray="4 2"/>`;
    const modeX = (vx(prop.mode)+bw/2).toFixed(1);
    const modeL = `<line x1="${modeX}" y1="${PT}" x2="${modeX}" y2="${PT+ph}" stroke="#81a1c1" stroke-width="1.5" stroke-dasharray="4 2"/>`;

    const axis = `<line x1="${PL}" y1="${PT+ph}" x2="${W-PR}" y2="${PT+ph}" stroke="#2a2d3e" stroke-width="0.5"/>`;

    // X-axis labels: kMin, line, kMax
    const labels = [];
    const seen = new Set();
    const addLabel = (v, txt) => {
      const x = (vx(v)+bw/2).toFixed(1);
      const key = Math.round(vx(v));
      if (!seen.has(key)) { seen.add(key); labels.push(`<text x="${x}" y="${H-5}" text-anchor="middle" font-size="8" fill="#666" font-family="system-ui">${txt}</text>`); }
    };
    addLabel(kMin, kMin);
    addLabel(prop.line, prop.line);
    addLabel(kMax, kMax);

    return `<svg width="${W}" height="${H}" style="display:block">${bars}${line}${meanL}${medL}${modeL}${axis}${labels.join('')}</svg>`;
  }

  function pct(v) { return (v*100).toFixed(1)+'%'; }
  function sign(v) { return v>=0 ? '+' : ''; }

  function buildCard(p) {
    const isOver = p.edge >= 0;
    const edgePct = (Math.abs(p.edge)*100).toFixed(1);
    const st = p.stat === 'FG3M' ? '3PM' : p.stat;
    const skewFlagged = Math.abs(p.skewness) > 1;
    const kurtFlagged = Math.abs(p.excess_kurtosis) > 3;
    const svg = buildSVG(p);

    return `<div class="card">
      <div class="card-header">
        <div><span class="card-title">${p.player}</span><span class="stat-tag">${st} ${p.line}</span></div>
        <span class="edge-badge ${isOver?'edge-over':'edge-under'}">${isOver?'OVER':'UNDER'} ${sign(p.edge)}${edgePct}%</span>
      </div>
      <div class="card-body">
        <div class="chart-area">${svg}</div>
        <div class="stats-panel">
          <div class="stat-row"><span class="stat-label">EV (mean)</span><span class="stat-value">${p.mean.toFixed(2)}</span></div>
          <div class="stat-row"><span class="stat-label">Median</span><span class="stat-value">${p.median.toFixed(1)}</span></div>
          <div class="stat-row"><span class="stat-label">Mode</span><span class="stat-value">${p.mode}</span></div>
          <div class="divider"></div>
          <div class="stat-row"><span class="stat-label">Variance</span><span class="stat-value">${p.variance.toFixed(2)}</span></div>
          <div class="stat-row"><span class="stat-label">Std Dev</span><span class="stat-value">${p.std_dev.toFixed(2)}</span></div>
          <div class="stat-row"><span class="stat-label">Skewness</span><span class="stat-value ${skewFlagged?'amber':''}">${p.skewness.toFixed(3)}</span></div>
          <div class="stat-row"><span class="stat-label">Ex. Kurtosis</span><span class="stat-value ${kurtFlagged?'amber':''}">${p.excess_kurtosis.toFixed(3)}</span></div>
          <div class="divider"></div>
          <div class="stat-row"><span class="stat-label">Mdl P(over)</span><span class="stat-value ${isOver?'green':'red'}">${pct(p.model_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">Mkt P(over)</span><span class="stat-value">${pct(p.market_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">Edge</span><span class="stat-value ${isOver?'green':'red'}">${sign(p.edge)}${(p.edge*100).toFixed(1)}%</span></div>
          ${p.kelly_pct>0?`<div class="stat-row"><span class="stat-label">Kelly %</span><span class="stat-value blue">${p.kelly_pct.toFixed(1)}%</span></div>`:''}
        </div>
      </div>
    </div>`;
  }

  function render() {
    const rows = filtered();
    const grid = document.getElementById('pmfGrid');
    if (!rows.length) { grid.innerHTML = '<div class="loading" style="grid-column:1/-1">No props match filters.</div>'; return; }
    grid.innerHTML = rows.map(buildCard).join('');
  }

  // Stat filter
  document.getElementById('statFilters').addEventListener('click', e => {
    const btn = e.target.closest('[data-stat]');
    if (!btn) return;
    statFilter = btn.dataset.stat;
    document.querySelectorAll('[data-stat]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  });

  // Search
  document.getElementById('searchInput').addEventListener('input', e => {
    searchFilter = e.target.value.toLowerCase().trim();
    render();
  });
})();
</script>
</body>
</html>"""

_LIVE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA Live In-Play Edges — WizardOfOdds</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e4e4e4;line-height:1.5;min-height:100vh}
header{background:#141622;border-bottom:1px solid #2a2d3e;padding:14px 28px;display:flex;align-items:center;gap:16px}
.logo{font-size:1rem;font-weight:700;color:#fff;letter-spacing:.3px}
.logo span{color:#599ce7}
.live-dot{width:8px;height:8px;border-radius:50%;background:#3fa266;animation:pulse 1.4s ease-in-out infinite;margin-left:4px}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}
.header-meta{font-size:.78rem;color:#888;margin-left:auto}
main{max-width:1200px;margin:24px auto;padding:0 18px}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.stat-card{background:#141622;border:1px solid #2a2d3e;border-radius:8px;padding:14px 18px}
.stat-card .value{font-size:1.5rem;font-weight:700;color:#e4e4e4}
.stat-card .value.green{color:#3fa266}
.stat-card .value.red{color:#e05a6a}
.stat-card .value.blue{color:#599ce7}
.stat-card .label{font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.pill{background:#1e2130;border:1px solid #2a2d3e;color:#aaa;border-radius:100px;padding:4px 14px;font-size:.78rem;cursor:pointer;transition:all .15s}
.pill:hover{background:#2a2d3e;color:#e4e4e4}
.pill.active{background:#599ce7;border-color:#599ce7;color:#fff;font-weight:600}
.search{background:#1e2130;border:1px solid #2a2d3e;color:#e4e4e4;border-radius:6px;padding:5px 12px;font-size:.82rem;outline:none;width:200px}
.search:focus{border-color:#599ce7}
.refresh-info{font-size:.75rem;color:#555;margin-left:auto}
.table-wrap{overflow-x:auto;border-radius:8px;border:1px solid #2a2d3e}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead{position:sticky;top:0;z-index:1}
th{background:#141622;color:#888;padding:10px 14px;text-align:left;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #2a2d3e;white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid #1e2130;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#1a1d2b}
.player{font-weight:500;color:#e4e4e4}
.stat-badge{background:#1e2130;border:1px solid #2a2d3e;border-radius:4px;padding:1px 7px;font-size:.72rem;font-weight:600;color:#aaa}
.dir-over{color:#3fa266;font-weight:700;font-size:.78rem}
.dir-under{color:#e05a6a;font-weight:700;font-size:.78rem}
.edge-pos{color:#3fa266;font-weight:700}
.edge-neg{color:#e05a6a;font-weight:700}
.kelly-val{color:#599ce7;font-weight:600}
.numeric{text-align:right;font-variant-numeric:tabular-nums}
.game-header{background:#0f1117;padding:8px 14px;font-size:.78rem;color:#888;border-bottom:1px solid #1e2130;display:flex;align-items:center;gap:8px}
.game-title{font-weight:600;color:#aaa}
.qtr-badge{background:#1e2130;border:1px solid #2a2d3e;border-radius:4px;padding:1px 7px;font-size:.72rem;color:#aaa}
.empty{text-align:center;color:#666;padding:60px;font-size:.9rem}
.footer{text-align:center;font-size:.75rem;color:#555;padding:32px 0 20px;border-top:1px solid #1e2130;margin-top:24px}
.last-update{font-size:.75rem;color:#599ce7}
@media(max-width:768px){.summary{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header>
  <div class="logo">WizardOf<span>Odds</span></div>
  <div style="font-size:.8rem;color:#666;border-left:1px solid #2a2d3e;margin-left:8px;padding-left:14px">WNBA</div>
  <div style="display:flex;align-items:center;gap:6px;font-size:.82rem;color:#3fa266;font-weight:600"><div class="live-dot"></div>LIVE In-Play Edges</div>
  <div class="header-meta">Bayesian in-game updates · Gamma-Poisson posterior · Refreshes every 2 min</div>
</header>
<main>
  <div class="summary">
    <div class="stat-card"><div class="value" id="s-games">—</div><div class="label">Active Games</div></div>
    <div class="stat-card"><div class="value" id="s-props">—</div><div class="label">Live Props</div></div>
    <div class="stat-card"><div class="value green" id="s-over">—</div><div class="label">Over Edges</div></div>
    <div class="stat-card"><div class="value red" id="s-under">—</div><div class="label">Under Edges</div></div>
  </div>

  <div class="controls">
    <div style="display:flex;gap:6px" id="statFilters">
      <button class="pill active" data-stat="ALL">All</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
    </div>
    <input class="search" type="text" placeholder="Filter player..." id="searchInput">
    <div class="refresh-info">Auto-refreshes every 2 min · <span class="last-update" id="lastUpdate">—</span></div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Game</th>
          <th>Player</th>
          <th>Stat</th>
          <th>Dir</th>
          <th class="numeric">Pace Adj. Mean</th>
          <th class="numeric">Projected Final</th>
          <th class="numeric">Mkt Line</th>
          <th class="numeric">Live P%</th>
          <th class="numeric">Open P%</th>
          <th class="numeric">Edge%</th>
          <th class="numeric">Kelly%</th>
        </tr>
      </thead>
      <tbody id="tableBody"><tr><td colspan="11" class="empty">Waiting for live game data…<br><small style="color:#444;margin-top:8px;display:block">This page activates when WNBA games are in progress (approx 7 PM–11 PM ET).</small></td></tr></tbody>
    </table>
  </div>

  <div class="footer">
    <div>Live model powered by Gamma-Poisson Bayesian online updating · WizardOfOdds Sports Analytics</div>
    <div style="margin-top:6px;color:#444">Pace-adjusted projections account for remaining possessions, foul trouble, and score differential · Data updates every 2 minutes during games</div>
  </div>
</main>

<script>
(function(){
  let allProps = [];
  let statFilter = 'ALL';
  let searchFilter = '';

  function load() {
    fetch('latest.json?t=' + Date.now())
      .then(r => r.json())
      .then(data => {
        // Flatten games → props
        allProps = [];
        const games = data.games || data.props || [];
        if (Array.isArray(games) && games.length && games[0].players) {
          // Structured game envelope
          games.forEach(game => {
            (game.players || []).forEach(player => {
              Object.entries(player.live_props || player.props || {}).forEach(([stat, pd]) => {
                if (!pd.market_line) return;
                const edge = (pd.live_p_over || pd.model_p_over || 0) - (pd.market_p_over || 0);
                allProps.push({
                  game: `${game.away_team} @ ${game.home_team}`,
                  quarter: game.period || game.quarter || '—',
                  player: player.player_name,
                  stat: stat.toUpperCase(),
                  direction: edge >= 0 ? 'OVER' : 'UNDER',
                  pace_mean: pd.pace_adj_mean || pd.mean || 0,
                  projected_final: pd.projected_final || pd.mean || 0,
                  market_line: pd.market_line || 0,
                  live_p: pd.live_p_over || pd.model_p_over || 0,
                  open_p: pd.open_p_over || pd.market_p_over || 0,
                  edge, kelly_pct: (pd.live_kelly || pd.kelly_fraction || 0) * 100,
                });
              });
            });
          });
        } else if (Array.isArray(games)) {
          // Already flat
          allProps = games.map(p => ({...p, direction: (p.edge||0) >= 0 ? 'OVER' : 'UNDER'}));
        }
        allProps.sort((a,b) => Math.abs(b.edge) - Math.abs(a.edge));
        document.getElementById('s-games').textContent = data.active_games || games.length || 0;
        document.getElementById('s-props').textContent = allProps.length;
        document.getElementById('s-over').textContent = allProps.filter(p => p.direction === 'OVER').length;
        document.getElementById('s-under').textContent = allProps.filter(p => p.direction === 'UNDER').length;
        document.getElementById('lastUpdate').textContent = 'Updated ' + new Date().toLocaleTimeString();
        render();
      })
      .catch(() => { /* keep old data */ });
  }

  function filtered() {
    return allProps.filter(p => {
      if (statFilter !== 'ALL' && p.stat !== statFilter) return false;
      if (searchFilter && !p.player.toLowerCase().includes(searchFilter)) return false;
      return true;
    });
  }

  function pct(v) { return (v*100).toFixed(1)+'%'; }
  function sign(v) { return v>=0?'+':''; }
  function f2(v) { return typeof v === 'number' ? v.toFixed(2) : '—'; }

  function render() {
    const rows = filtered();
    const tbody = document.getElementById('tableBody');
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="11" class="empty">No live edges match current filters.<br><small style="color:#444;margin-top:8px;display:block">Games in progress will appear here automatically.</small></td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const isOver = p.direction === 'OVER';
      const edgeFmt = `${sign(p.edge)}${(p.edge*100).toFixed(1)}%`;
      return `<tr>
        <td style="font-size:.75rem;color:#666">${p.game||'—'} <span class="qtr-badge">${p.quarter||''}</span></td>
        <td class="player">${p.player}</td>
        <td><span class="stat-badge">${p.stat}</span></td>
        <td><span class="${isOver?'dir-over':'dir-under'}">${p.direction}</span></td>
        <td class="numeric">${f2(p.pace_mean)}</td>
        <td class="numeric">${f2(p.projected_final)}</td>
        <td class="numeric">${p.market_line}</td>
        <td class="numeric">${pct(p.live_p)}</td>
        <td class="numeric" style="color:#666">${pct(p.open_p)}</td>
        <td class="numeric ${isOver?'edge-pos':'edge-neg'}">${edgeFmt}</td>
        <td class="numeric ${p.kelly_pct>0?'kelly-val':''}">${p.kelly_pct>0?p.kelly_pct.toFixed(1)+'%':'—'}</td>
      </tr>`;
    }).join('');
  }

  // Stat filter
  document.getElementById('statFilters').addEventListener('click', e => {
    const btn = e.target.closest('[data-stat]');
    if (!btn) return;
    statFilter = btn.dataset.stat;
    document.querySelectorAll('[data-stat]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  });

  document.getElementById('searchInput').addEventListener('input', e => {
    searchFilter = e.target.value.toLowerCase().trim();
    render();
  });

  load();
  setInterval(load, 120000); // refresh every 2 minutes
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="YYYY-MM-DD"),
    projections: str = typer.Option("", "--projections", help="player_projections_{date}.parquet"),
    edges: str = typer.Option("", "--edges", help="publishable_edges.parquet"),
    out_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA/Pre-Game",
        "--out-dir",
        help="Root output directory for Pre-Game pages",
    ),
    live_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA/Inplay/Edges",
        "--live-dir",
        help="Output directory for Inplay/Edges page",
    ),
    skip_live_html: bool = typer.Option(False, "--skip-live-html", help="Skip writing the live page HTML"),
) -> None:
    """Generate all three web page directories (Edge, PMF-Distributions, Inplay/Edges)."""
    out = Path(out_dir)
    edge_dir = out / "Edge"
    pmf_dir = out / "PMF-Distributions"
    live_out = Path(live_dir)

    edge_dir.mkdir(parents=True, exist_ok=True)
    pmf_dir.mkdir(parents=True, exist_ok=True)
    live_out.mkdir(parents=True, exist_ok=True)

    # --- Load source data ---
    proj_path = projections or f"deliveries/tonight/player_projections_{game_date}.parquet"
    edges_path = edges or "deliveries/tonight/publishable_edges.parquet"

    typer.echo(f"[generate_web_pages] game_date={game_date}")
    typer.echo(f"  projections : {proj_path}")
    typer.echo(f"  edges       : {edges_path}")

    try:
        proj_df = pd.read_parquet(proj_path)
        typer.echo(f"  Loaded projections: {len(proj_df)} rows")
    except Exception as exc:
        typer.echo(f"  [WARN] Could not load projections: {exc} — using empty DataFrame")
        proj_df = pd.DataFrame(columns=["player_id", "stat", "pmf_json", "pmf_mean", "pmf_variance", "median", "mode"])

    try:
        edges_df = pd.read_parquet(edges_path)
        typer.echo(f"  Loaded edges: {len(edges_df)} rows")
    except Exception as exc:
        typer.echo(f"  [WARN] Could not load edges: {exc} — using empty DataFrame")
        edges_df = pd.DataFrame(columns=["player_name", "player_id", "stat", "line", "edge_over",
                                          "kelly_fraction", "model_prob_over", "market_prob_over_no_vig"])

    # --- Build JSON ---
    edge_json = _build_edge_json(edges_df, proj_df, game_date)
    pmf_json = _build_pmf_json(edges_df, proj_df, game_date)

    # Write Edge page JSON
    (edge_dir / "latest.json").write_text(json.dumps(edge_json, separators=(",", ":")))
    (edge_dir / f"{game_date}.json").write_text(json.dumps(edge_json, separators=(",", ":")))
    typer.echo(f"  Edge JSON → {edge_dir}/latest.json ({edge_json['total_props']} props)")

    # Write PMF page JSON
    (pmf_dir / "latest.json").write_text(json.dumps(pmf_json, separators=(",", ":")))
    (pmf_dir / f"{game_date}.json").write_text(json.dumps(pmf_json, separators=(",", ":")))
    typer.echo(f"  PMF JSON → {pmf_dir}/latest.json ({pmf_json['total_props']} props with distributions)")

    # --- Write HTML templates (overwrite with canonical version each run) ---
    (edge_dir / "index.html").write_text(_EDGE_HTML)
    typer.echo(f"  Edge HTML → {edge_dir}/index.html")

    (pmf_dir / "index.html").write_text(_PMF_HTML)
    typer.echo(f"  PMF HTML → {pmf_dir}/index.html")

    # --- Live page: always write fresh index.html; JSON is written by live_inplay.yml ---
    if not skip_live_html:
        (live_out / "index.html").write_text(_LIVE_HTML)
        typer.echo(f"  Live HTML → {live_out}/index.html")

    typer.echo("[generate_web_pages] Done.")


if __name__ == "__main__":
    app()
