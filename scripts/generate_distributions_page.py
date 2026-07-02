#!/usr/bin/env python3
"""
Generate the WNBA Pre-Game/Distributions/ page — pure PMF visualization.

Reads Pre-Game/PMF-Distributions/latest.json and renders a clean, interactive
probability distribution chart page for every modeled player prop.

No edges table, no bookmaker buttons — purely the statistical distributions.

Outputs:
  Pre-Game/Distributions/latest.json   — cleaned PMF schema
  Pre-Game/Distributions/{date}.json   — dated archive copy
  Pre-Game/Distributions/index.html    — premium PMF visualization page

Usage:
    python scripts/generate_distributions_page.py \\
        --game-date 2026-07-01 \\
        --base-dir tools/odds-scanner/predictions/WNBA
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)

_STAT_LABELS: dict[str, str] = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "FG3M": "3-Pointers Made", "STL": "Steals", "BLK": "Blocks",
    "TURNOVER": "Turnovers", "STOCKS": "Stl + Blk",
    "PA": "Pts + Ast", "PR": "Pts + Reb",
    "RA": "Reb + Ast", "PRA": "Pts + Reb + Ast",
}


def _build_json(pmf_path: Path, game_date: str) -> dict:
    """Read PMF-Distributions/latest.json, emit cleaned schema."""
    try:
        pmf_data = json.loads(pmf_path.read_text())
    except Exception as exc:
        typer.echo(f"[WARN] PMF JSON not found: {exc}")
        pmf_data = {"props": []}

    props = []
    for p in pmf_data.get("props", []):
        stat_up = p.get("stat", "").upper()
        edge_frac = float(p.get("edge") or 0)
        props.append({
            "player": p.get("player", ""),
            "stat": stat_up,
            "stat_label": _STAT_LABELS.get(stat_up, stat_up),
            "stat_raw": p.get("stat_raw", stat_up.lower()),
            "line": float(p.get("line") or 0),
            "mean": p.get("mean"),
            "median": p.get("median"),
            "mode": p.get("mode"),
            "std_dev": p.get("std_dev"),
            "variance": p.get("variance"),
            "skewness": p.get("skewness"),
            "excess_kurtosis": p.get("excess_kurtosis"),
            "model_p_over": round(float(p.get("model_p_over") or 0), 4),
            "market_p_over": round(float(p.get("market_p_over") or 0), 4),
            "edge_pp": round(edge_frac * 100, 2),
            "kelly_pct": round(float(p.get("kelly_pct") or 0), 2),
            "pmf": p.get("pmf", []),
        })

    return {
        "schema_version": "3.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(props),
        "props": props,
    }


# ---------------------------------------------------------------------------
# HTML template — pure PMF visualization
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA PMF Distributions — WizardOfOdds</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0A0E1A;--surface:#0F1529;--surface2:#141B30;--surface3:#1A2238;
  --border:#1E2640;--border2:#2A3350;
  --gold:#D4AF37;--gold-dim:rgba(212,175,55,.15);--gold-border:rgba(212,175,55,.3);
  --text:#E8EAF2;--text2:#9BA3BC;--text3:#5A6380;
  --green:#2ECC71;--green-dim:rgba(46,204,113,.15);--green-border:rgba(46,204,113,.3);
  --red:#E74C3C;--red-dim:rgba(231,76,60,.15);--red-border:rgba(231,76,60,.3);
  --blue:#3B82F6;--amber:#F59E0B;
}
html{font-size:14px}
body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5}

/* Header */
header{background:var(--surface);border-bottom:1px solid var(--border);padding:11px 22px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100}
.brand{font-family:'Fraunces',serif;font-size:1.1rem;font-weight:700;color:var(--text)}
.brand .acc{color:var(--gold)}
.sep{width:1px;height:16px;background:var(--border2)}
.nav-links{display:flex;gap:2px}
.nav-link{font-size:.7rem;color:var(--text3);padding:3px 10px;border-radius:4px;text-decoration:none;transition:color .15s}
.nav-link:hover,.nav-link.active{color:var(--gold)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:14px;font-size:.7rem;color:var(--text3)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px;animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* Main */
main{max-width:1400px;margin:20px auto;padding:0 20px}

/* KPI strip */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 18px}
.kpi .val{font-family:'Fraunces',serif;font-size:1.55rem;font-weight:700;line-height:1.1;color:var(--text)}
.kpi .val.gold{color:var(--gold)}
.kpi .lbl{font-size:.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}

/* Filters */
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:18px}
.filter-row{display:flex;gap:4px;flex-wrap:wrap}
.pill{background:var(--surface2);border:1px solid var(--border2);color:var(--text3);border-radius:20px;padding:3px 13px;font-size:.7rem;cursor:pointer;transition:all .15s;font-family:inherit;white-space:nowrap}
.pill:hover{border-color:var(--gold);color:var(--text)}
.pill.active{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}
.search-box{background:var(--surface2);border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:4px 12px;font-size:.76rem;font-family:inherit;outline:none;width:200px}
.search-box:focus{border-color:var(--gold)}
.flt-spacer{flex:1}
.count-lbl{font-size:.7rem;color:var(--text3)}

/* Sort bar */
.sort-bar{display:flex;align-items:center;gap:8px;margin-bottom:14px;font-size:.72rem;color:var(--text3)}
.sort-btn{background:var(--surface2);border:1px solid var(--border2);color:var(--text3);border-radius:6px;padding:3px 11px;font-size:.7rem;cursor:pointer;font-family:inherit;transition:all .15s}
.sort-btn:hover{border-color:var(--gold);color:var(--text)}
.sort-btn.active{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}

/* PMF grid */
.pmf-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}

/* PMF card */
.pmf-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;transition:border-color .2s,box-shadow .2s}
.pmf-card:hover{border-color:var(--border2);box-shadow:0 4px 24px rgba(0,0,0,.3)}
.card-header{padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.card-title-row{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:6px}
.player-name{font-family:'Fraunces',serif;font-size:1rem;font-weight:600;color:var(--text)}
.edge-badge{font-size:.72rem;font-weight:700;padding:3px 9px;border-radius:5px;white-space:nowrap;flex-shrink:0}
.edge-pos{background:var(--green-dim);border:1px solid var(--green-border);color:var(--green)}
.edge-neg{background:var(--red-dim);border:1px solid var(--red-border);color:var(--red)}
.card-meta{display:flex;align-items:center;gap:8px;font-size:.7rem;color:var(--text3)}
.stat-chip{background:var(--surface3);border:1px solid var(--border2);border-radius:4px;padding:1px 8px;font-size:.68rem;font-weight:600;color:var(--text2)}
.line-chip{background:var(--gold-dim);border:1px solid var(--gold-border);border-radius:4px;padding:1px 8px;font-size:.68rem;font-weight:600;color:var(--gold)}

/* Chart area */
.chart-wrap{padding:12px 14px 8px;position:relative;height:130px}
.chart-wrap canvas{width:100%!important}

/* Probability strip */
.prob-strip{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--border);border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.prob-cell{padding:8px 12px;background:var(--surface);text-align:center}
.prob-cell .pv{font-size:.82rem;font-weight:700}
.prob-cell .pk{font-size:.6rem;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.prob-model{color:var(--green)}
.prob-mkt{color:var(--text3)}
.prob-edge-pos{color:var(--green)}
.prob-edge-neg{color:var(--red)}

/* Stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border)}
.stat-cell{padding:7px 6px;background:var(--surface2);text-align:center}
.stat-cell .sv{font-size:.78rem;font-weight:600;color:var(--text)}
.stat-cell .sk{font-size:.58rem;color:var(--text3);margin-top:1px}

/* Probability calculator */
.prob-calc{padding:10px 14px 12px}
.calc-label{font-size:.67rem;color:var(--text3);margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}
.calc-row{display:flex;align-items:center;gap:10px}
.calc-row input[type=range]{flex:1;accent-color:var(--gold);height:3px}
.calc-result{font-size:.76rem;font-weight:600;color:var(--text);white-space:nowrap;min-width:140px;text-align:right}

/* Empty state */
.no-data{text-align:center;padding:80px 20px;color:var(--text3)}
.no-data h2{font-family:'Fraunces',serif;font-size:1.4rem;color:var(--text2);margin-bottom:8px}

/* Footer */
footer{text-align:center;font-size:.67rem;color:var(--text3);padding:26px 0 18px;border-top:1px solid var(--border);margin-top:28px;line-height:1.9}

/* Responsive */
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.pmf-grid{grid-template-columns:1fr}.search-box{width:140px}}
</style>
</head>
<body>

<header>
  <div class="brand">WizardOf<span class="acc">Odds</span></div>
  <div class="sep"></div>
  <nav class="nav-links">
    <a href="../Edge/" class="nav-link">Prop Edges</a>
    <a href="" class="nav-link active">Distributions</a>
    <a href="../Pricer/" class="nav-link">Market X-Ray</a>
    <a href="../../In-Play/Edges/" class="nav-link">Live In-Play</a>
  </nav>
  <div class="hdr-right">
    <span id="dateLabel">—</span>
    <span><span class="dot"></span>LIVE DATA · Refresh in <span id="cdSec" style="color:var(--gold)">300</span>s</span>
  </div>
</header>

<main>
  <!-- KPIs -->
  <div class="kpis">
    <div class="kpi"><div class="val" id="kTotal">—</div><div class="lbl">Props Modeled</div></div>
    <div class="kpi"><div class="val gold" id="kPos">—</div><div class="lbl">Positive Edges</div></div>
    <div class="kpi"><div class="val" id="kGames">—</div><div class="lbl">Stats Covered</div></div>
    <div class="kpi"><div class="val" id="kUpdated" style="font-size:.85rem">—</div><div class="lbl">Last Updated</div></div>
  </div>

  <!-- Filters -->
  <div class="filters">
    <div class="filter-row" id="statPills">
      <button class="pill active" data-stat="">All Stats</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
      <button class="pill" data-stat="FG3M">3PM</button>
      <button class="pill" data-stat="STL">STL</button>
      <button class="pill" data-stat="BLK">BLK</button>
      <button class="pill" data-stat="PRA">PRA</button>
    </div>
    <div class="filter-row" id="dirPills">
      <button class="pill active" data-dir="">Both</button>
      <button class="pill" data-dir="OVER">Over edge</button>
      <button class="pill" data-dir="UNDER">Under edge</button>
    </div>
    <input class="search-box" type="text" id="search" placeholder="Search player…">
    <div class="flt-spacer"></div>
    <span class="count-lbl" id="countLbl">—</span>
  </div>

  <!-- Sort bar -->
  <div class="sort-bar">
    <span>Sort by:</span>
    <button class="sort-btn active" data-sort="edge">Edge magnitude</button>
    <button class="sort-btn" data-sort="player">Player A–Z</button>
    <button class="sort-btn" data-sort="stat">Stat type</button>
    <button class="sort-btn" data-sort="std_dev">Std Dev ↓</button>
  </div>

  <!-- PMF grid -->
  <div class="pmf-grid" id="pmfGrid">
    <div class="no-data"><h2>Loading distributions…</h2></div>
  </div>
</main>

<footer>
  <div>WNBA Pre-Game Probability Distributions — WizardOfOdds Sports Analytics</div>
  <div>PMF computed from Bayesian hierarchical model with isotonic distribution regression (IDR) calibration</div>
  <div>Bars: green = over market line · red = under · dashed line = market line · Edge = Model P(over) − Market P(no-vig)</div>
  <div style="margin-top:4px;color:var(--text3)">For entertainment and research purposes only. Gamble responsibly. 21+</div>
</footer>

<script>
(function(){
'use strict';

let ALL = [];
let statFilt = '', dirFilt = '', searchFilt = '', sortKey = 'edge';
const charts = {};
let cdTimer;

// ── Fetch ──────────────────────────────────────────────────────────
function load() {
  fetch('latest.json?t=' + Date.now())
    .then(r => r.json())
    .then(data => {
      ALL = data.props || [];
      updateKPIs(data);
      render();
      startCountdown(300);
    })
    .catch(err => console.warn('[WOO] fetch failed:', err));
}

function updateKPIs(data) {
  document.getElementById('kTotal').textContent = data.total_props ?? ALL.length;
  const pos = ALL.filter(p => p.edge_pp > 0).length;
  document.getElementById('kPos').textContent = pos;
  const statTypes = new Set(ALL.map(p => p.stat)).size;
  document.getElementById('kGames').textContent = statTypes;
  const gen = data.generated_at ? new Date(data.generated_at) : null;
  document.getElementById('kUpdated').textContent = gen
    ? gen.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
    : '—';
  document.getElementById('dateLabel').textContent = data.game_date || '—';
}

// ── Filter + sort ──────────────────────────────────────────────────
function filtered() {
  return ALL.filter(p => {
    if (statFilt && p.stat !== statFilt) return false;
    if (dirFilt) {
      const isOver = p.edge_pp >= 0;
      if (dirFilt === 'OVER' && !isOver) return false;
      if (dirFilt === 'UNDER' && isOver) return false;
    }
    if (searchFilt && !p.player.toLowerCase().includes(searchFilt)) return false;
    if (!p.pmf || p.pmf.length === 0) return false;
    return true;
  });
}

function sorted(rows) {
  return [...rows].sort((a, b) => {
    if (sortKey === 'edge') return Math.abs(b.edge_pp) - Math.abs(a.edge_pp);
    if (sortKey === 'player') return a.player.localeCompare(b.player);
    if (sortKey === 'stat') return a.stat.localeCompare(b.stat) || Math.abs(b.edge_pp) - Math.abs(a.edge_pp);
    if (sortKey === 'std_dev') return (b.std_dev || 0) - (a.std_dev || 0);
    return 0;
  });
}

// ── Render grid ────────────────────────────────────────────────────
function render() {
  destroyCharts();
  const rows = sorted(filtered());
  const grid = document.getElementById('pmfGrid');
  document.getElementById('countLbl').textContent = rows.length + ' distributions shown';

  if (!rows.length) {
    grid.innerHTML = '<div class="no-data"><h2>No distributions match filters</h2><p>PMF data is generated during the pre-game pipeline run.</p></div>';
    return;
  }

  grid.innerHTML = rows.map((p, i) => buildCard(p, i)).join('');

  // Wire sliders
  document.querySelectorAll('.prob-slider').forEach(inp => {
    inp.addEventListener('input', () => updateSlider(inp));
    updateSlider(inp);
  });

  // Draw charts after paint
  requestAnimationFrame(() => drawCharts(rows));
}

function buildCard(p, i) {
  const ep = p.edge_pp || 0;
  const epCls = ep >= 0 ? 'edge-pos' : 'edge-neg';
  const epLabel = (ep >= 0 ? '+' : '') + ep.toFixed(1) + '% edge';
  const statLabel = p.stat_label || p.stat;

  const f2 = v => v != null ? (+v).toFixed(2) : '—';
  const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';

  const modelP = pct(p.model_p_over);
  const mktP = pct(p.market_p_over);
  const edgePctLabel = (ep >= 0 ? '+' : '') + ep.toFixed(1) + '%';
  const edgePCls = ep >= 0 ? 'prob-edge-pos' : 'prob-edge-neg';

  const minK = p.pmf.length ? p.pmf[0][0] : 0;
  const maxK = p.pmf.length ? p.pmf[p.pmf.length - 1][0] : 30;
  const initN = Math.ceil(p.line);

  return `
<div class="pmf-card">
  <div class="card-header">
    <div class="card-title-row">
      <span class="player-name">${p.player}</span>
      <span class="edge-badge ${epCls}">${epLabel}</span>
    </div>
    <div class="card-meta">
      <span class="stat-chip">${statLabel}</span>
      <span class="line-chip">Line: ${p.line}</span>
    </div>
  </div>

  <div class="chart-wrap"><canvas id="ch_${i}"></canvas></div>

  <div class="prob-strip">
    <div class="prob-cell">
      <div class="pv prob-model">${modelP}</div>
      <div class="pk">Model P(Over)</div>
    </div>
    <div class="prob-cell">
      <div class="pv prob-mkt">${mktP}</div>
      <div class="pk">Market (no-vig)</div>
    </div>
    <div class="prob-cell">
      <div class="pv ${edgePCls}">${edgePctLabel}</div>
      <div class="pk">Edge</div>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-cell"><div class="sv">${f2(p.mean)}</div><div class="sk">Mean</div></div>
    <div class="stat-cell"><div class="sv">${f2(p.median)}</div><div class="sk">Median</div></div>
    <div class="stat-cell"><div class="sv">${f2(p.std_dev)}</div><div class="sk">Std Dev</div></div>
    <div class="stat-cell"><div class="sv">${f2(p.skewness)}</div><div class="sk">Skewness</div></div>
    <div class="stat-cell"><div class="sv">${f2(p.excess_kurtosis)}</div><div class="sk">Ex. Kurt.</div></div>
  </div>

  <div class="prob-calc">
    <div class="calc-label">Interactive — P(X ≥ n)</div>
    <div class="calc-row">
      <input class="prob-slider" type="range"
             min="${minK}" max="${maxK}" step="1" value="${initN}"
             data-pidx="${i}" data-line="${p.line}">
      <span class="calc-result" id="cr_${i}">—</span>
    </div>
  </div>
</div>`;
}

// ── Probability slider ─────────────────────────────────────────────
function updateSlider(inp) {
  const i = parseInt(inp.dataset.pidx);
  const line = parseFloat(inp.dataset.line);
  const n = parseInt(inp.value);
  const p = sorted(filtered())[i];
  if (!p || !p.pmf.length) return;
  const prob = p.pmf.filter(([k]) => k >= n).reduce((s, [, v]) => s + v, 0);
  const vsLine = n === Math.ceil(line) ? ' (at line)' : n > line ? ' (above)' : ' (below)';
  document.getElementById('cr_' + i).textContent =
    'P(X ≥ ' + n + ') = ' + (prob * 100).toFixed(1) + '%' + vsLine;
}

// ── Chart drawing ──────────────────────────────────────────────────
function destroyCharts() {
  Object.keys(charts).forEach(id => {
    try { charts[id].destroy(); } catch(e) {}
    delete charts[id];
  });
}

function drawCharts(rows) {
  rows.forEach((p, i) => {
    const canvas = document.getElementById('ch_' + i);
    if (!canvas || !p.pmf.length) return;

    const line = p.line;
    const labels = p.pmf.map(([k]) => k);
    const vals = p.pmf.map(([, v]) => +(v * 100).toFixed(4));

    // Color each bar: green if k > line (over), red if k <= line (under)
    const bgColors = labels.map(k =>
      k > line ? 'rgba(46,204,113,.65)' : 'rgba(231,76,60,.55)'
    );
    const borderColors = labels.map(k =>
      k > line ? 'rgba(46,204,113,.9)' : 'rgba(231,76,60,.8)'
    );

    // Find index closest to market line for annotation
    const lineIdx = labels.findIndex(k => k >= line);

    try {
      charts['ch_' + i] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              data: vals,
              backgroundColor: bgColors,
              borderColor: borderColors,
              borderWidth: 1,
              borderRadius: 2,
              barPercentage: 0.9,
              categoryPercentage: 0.95,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: ctx => 'X = ' + ctx[0].label,
                label: ctx => 'P(' + ctx.label + ') = ' + ctx.parsed.y.toFixed(3) + '%'
              }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(30,38,64,.5)', lineWidth: 1 },
              ticks: {
                color: '#5A6380',
                font: { size: 9, family: 'JetBrains Mono' },
                maxRotation: 0,
                maxTicksLimit: 10,
              }
            },
            y: {
              grid: { color: 'rgba(30,38,64,.5)', lineWidth: 1 },
              ticks: {
                color: '#5A6380',
                font: { size: 9, family: 'JetBrains Mono' },
                callback: v => v + '%'
              }
            }
          }
        },
        plugins: [{
          id: 'linemarker',
          afterDraw(chart) {
            // Draw dashed vertical line at market line
            const { ctx, chartArea, scales } = chart;
            if (lineIdx < 0) return;
            const xScale = scales.x;
            // find pixel for the first bar at or above line
            let xPos = null;
            for (let li = 0; li < labels.length; li++) {
              if (labels[li] >= line) {
                const meta = chart.getDatasetMeta(0);
                if (meta.data[li]) {
                  xPos = meta.data[li].x - (meta.data[li].width || 8) / 2;
                }
                break;
              }
            }
            if (xPos === null) return;
            ctx.save();
            ctx.strokeStyle = 'rgba(212,175,55,.8)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(xPos, chartArea.top);
            ctx.lineTo(xPos, chartArea.bottom);
            ctx.stroke();
            ctx.setLineDash([]);
            // Label
            ctx.fillStyle = 'rgba(212,175,55,.9)';
            ctx.font = '9px JetBrains Mono';
            ctx.fillText('Line ' + line, xPos + 3, chartArea.top + 9);
            ctx.restore();
          }
        }]
      });
    } catch(e) {
      console.warn('[WOO] Chart error:', e);
    }
  });
}

// ── Filters ────────────────────────────────────────────────────────
document.getElementById('statPills').addEventListener('click', e => {
  const btn = e.target.closest('[data-stat]');
  if (!btn) return;
  statFilt = btn.dataset.stat;
  document.querySelectorAll('#statPills .pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
});

document.getElementById('dirPills').addEventListener('click', e => {
  const btn = e.target.closest('[data-dir]');
  if (!btn) return;
  dirFilt = btn.dataset.dir;
  document.querySelectorAll('#dirPills .pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
});

document.getElementById('search').addEventListener('input', e => {
  searchFilt = e.target.value.trim().toLowerCase();
  render();
});

document.querySelectorAll('.sort-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    sortKey = btn.dataset.sort;
    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  });
});

// ── Countdown ──────────────────────────────────────────────────────
function startCountdown(sec) {
  clearInterval(cdTimer);
  let t = Math.round(sec);
  document.getElementById('cdSec').textContent = t;
  cdTimer = setInterval(() => {
    t--;
    document.getElementById('cdSec').textContent = Math.max(t, 0);
    if (t <= 0) { clearInterval(cdTimer); load(); }
  }, 1000);
}

// ── Boot ───────────────────────────────────────────────────────────
load();
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    game_date: str = typer.Option("", "--game-date", help="YYYY-MM-DD (defaults to today)"),
    base_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA",
        "--base-dir",
        help="Root WNBA predictions directory",
    ),
    json_only: bool = typer.Option(False, "--json-only", help="Write only JSON data files, skip index.html regeneration."),
) -> None:
    """Generate pure PMF visualization page at Pre-Game/Distributions/."""
    if not game_date:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base = Path(base_dir)
    pmf_path = base / "Pre-Game" / "PMF-Distributions" / "latest.json"
    out_dir  = base / "Pre-Game" / "Distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[generate_distributions_page] game_date={game_date}")
    typer.echo(f"  pmf_path : {pmf_path}")
    typer.echo(f"  out_dir  : {out_dir}")

    payload = _build_json(pmf_path, game_date)

    (out_dir / "latest.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out_dir / f"{game_date}.json").write_text(json.dumps(payload, separators=(",", ":")))
    if not json_only:
        (out_dir / "index.html").write_text(_HTML)

    typer.echo(f"  → latest.json ({payload['total_props']} props)")
    typer.echo(f"  → {game_date}.json")
    if not json_only:
        typer.echo(f"  → index.html")
    typer.echo("[generate_distributions_page] Done.")


if __name__ == "__main__":
    app()
