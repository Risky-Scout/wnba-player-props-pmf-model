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
import math
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

_STAT_LABELS: dict[str, str] = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "FG3M": "3-Pointers Made", "STL": "Steals", "BLK": "Blocks",
    "TURNOVER": "Turnovers", "STOCKS": "Stl + Blk",
    # Short-form aliases
    "PA": "Pts + Ast", "PR": "Pts + Reb",
    "RA": "Reb + Ast", "PRA": "Pts + Reb + Ast",
    # Pipeline stat keys (underscore-separated)
    "PTS_REB": "Pts + Reb", "PTS_AST": "Pts + Ast",
    "REB_AST": "Reb + Ast", "PTS_REB_AST": "Pts + Reb + Ast",
}


def _compute_push_aware_probs(pmf_pairs: list, line: float) -> tuple[float, float, float]:
    """Compute (p_over, p_under, p_push) from PMF pairs at a given line.

    For integer lines: p_push = P(X == line); p_under = P(X < line) = 1 - p_over - p_push
    For half-point lines: p_push = 0; p_under = 1 - p_over
    Ensures p_over + p_under + p_push = 1 within 1e-12.
    """
    if not pmf_pairs:
        return 0.0, 0.0, 0.0
    k_vals = [int(pair[0]) for pair in pmf_pairs]
    p_vals = [float(pair[1]) for pair in pmf_pairs]
    total = sum(p_vals)
    if total > 0:
        p_vals = [v / total for v in p_vals]
    p_over = sum(p for k, p in zip(k_vals, p_vals) if k > float(line))
    is_integer = (float(line) == math.floor(float(line)))
    p_push = 0.0
    if is_integer:
        p_push = sum(p for k, p in zip(k_vals, p_vals) if k == int(line))
    p_under = max(0.0, 1.0 - p_over - p_push)
    return round(p_over, 6), round(p_under, 6), round(p_push, 6)


def _build_json(
    pmf_path: Path,
    game_date: str,
    release_id: str = "",
    git_commit: str = "",
) -> dict:
    """Read PMF-Distributions/latest.json, emit cleaned schema.

    After building props from the PMF data, merges in any Edge Board rows that
    have a real market line but are absent from the Distributions at that exact
    (player, stat, line) triple.  The PMF shape is borrowed from the matching
    player/stat entry (distributions don't change with the line — only the
    reference cut-point changes).

    Fails closed (raises FileNotFoundError) when the PMF source is missing.
    """
    if not pmf_path.exists():
        raise FileNotFoundError(
            f"Source PMF-Distributions file not found: {pmf_path}. "
            "Cannot build Distributions page without a current-run PMF source."
        )
    try:
        pmf_data = json.loads(pmf_path.read_text())
    except Exception as exc:
        raise ValueError(
            f"Source PMF-Distributions file is unreadable: {pmf_path}: {exc}"
        ) from exc

    # Validate source lineage when release_id is provided
    if release_id:
        src_release = pmf_data.get("release_id", "")
        if src_release and src_release != release_id:
            raise ValueError(
                f"Source PMF-Distributions release_id={src_release!r} "
                f"does not match expected release_id={release_id!r}. "
                "Stale or mismatched source — refusing to build Distributions page."
            )

    props = []
    for p in pmf_data.get("props", []):
        stat_up = p.get("stat", "").upper()
        raw_line = p.get("line")
        market_line = float(raw_line) if raw_line is not None else 0.0
        has_market_line = market_line > 0

        # edge_pp is only meaningful when there is a real sportsbook line.
        # Props with line=0/None have no market reference; setting edge=0 is
        # misleading (model trivially has P(over 0)≈100%), so we null it out.
        # edge_pp is stored as PERCENTAGE POINTS (e.g. -37.44 means -37.44pp).
        raw_edge = p.get("edge")
        if has_market_line and raw_edge is not None:
            edge_pp: float | None = round(float(raw_edge) * 100, 2)
        else:
            edge_pp = None

        pmf_pairs = p.get("pmf", [])
        # Compute push-aware probabilities from the full PMF pairs
        p_over_computed, p_under_computed, p_push_computed = _compute_push_aware_probs(
            pmf_pairs, market_line
        ) if has_market_line and pmf_pairs else (
            round(float(p.get("model_p_over") or 0), 4), 0.0, 0.0
        )
        # Use source model_p_over if no market line (can't compute vs a line)
        if not has_market_line:
            p_over_computed = round(float(p.get("model_p_over") or 0), 4)
            p_under_computed = round(1.0 - p_over_computed, 4)
            p_push_computed  = 0.0

        # pmf_full = complete probability pairs summing to 1 (never filtered)
        # pmf_chart = filtered pairs for chart rendering (omits tiny masses)
        _CHART_THRESHOLD = 0.001
        pmf_chart = [[k, v] for k, v in pmf_pairs if v >= _CHART_THRESHOLD]
        omitted_mass = round(sum(v for _, v in pmf_pairs) - sum(v for _, v in pmf_chart), 8)

        props.append({
            "player": p.get("player", ""),
            "stat": stat_up,
            "stat_label": _STAT_LABELS.get(stat_up, stat_up),
            "stat_raw": p.get("stat_raw", stat_up.lower()),
            "line": market_line,
            "has_market_line": has_market_line,
            "mean": p.get("mean"),
            "median": p.get("median"),
            "mode": p.get("mode"),
            "std_dev": p.get("std_dev"),
            "variance": p.get("variance"),
            "skewness": p.get("skewness"),
            "excess_kurtosis": p.get("excess_kurtosis"),
            "model_p_over": p_over_computed,
            "model_p_under": p_under_computed,
            "model_p_push": p_push_computed,
            "market_p_over": round(float(p.get("market_p_over") or 0), 4),
            "no_vig_over_prob": round(float(p.get("no_vig_over_prob") or p.get("market_p_over") or 0), 4),
            "no_vig_under_prob": round(float(p.get("no_vig_under_prob") or 0), 4),
            "edge_pp": edge_pp,
            "kelly_pct": round(float(p.get("kelly_pct") or 0), 2),
            # pmf_full: complete mass (sum=1); use this for all probability calculations
            "pmf_full": pmf_pairs,
            # pmf_chart: filtered for rendering (may exclude small tail mass)
            "pmf_chart": pmf_chart,
            "omitted_chart_mass": omitted_mass if omitted_mass > 0 else None,
            # Legacy field alias for backward compatibility
            "pmf": pmf_pairs,
        })

    # ── Merge missing Edge Board rows ─────────────────────────────────────────
    # The Edge Board may contain rows for (player, stat, market_line) triples
    # that don't appear in PMF-Distributions because build_edge_report picks up
    # market lines that the PMF merge step missed (e.g. when drop_duplicates on
    # player_id+stat discards alternate lines).  Ensure every Edge Board row
    # that has a real market line is represented in this page at that exact line.
    edge_json_path = pmf_path.parent.parent / "Edge" / "latest.json"
    try:
        edge_data = json.loads(edge_json_path.read_text())
        edge_props = edge_data.get("props", [])
    except Exception as exc:
        typer.echo(f"[WARN] Could not load Edge/latest.json for merge: {exc}")
        edge_props = []

    if edge_props:
        # Index distributions by (player, stat, line) for fast lookup
        dist_keys: set[tuple] = {(p["player"], p["stat"], p["line"]) for p in props}

        # Also index by (player, stat) → first matching dist prop index (for PMF clone)
        player_stat_idx: dict[tuple, int] = {}
        for i, p in enumerate(props):
            key = (p["player"], p["stat"])
            if key not in player_stat_idx:
                player_stat_idx[key] = i

        merged_count = 0
        for ep in edge_props:
            player = ep.get("player", "")
            stat   = ep.get("stat", "")
            market_line = float(ep.get("market_line") or 0)

            if market_line <= 0:
                continue  # no real market line on this edge row — skip
            if (player, stat, market_line) in dist_keys:
                continue  # already present — nothing to do

            # edge is stored as decimal in Edge Board JSON (e.g. -0.374 = -37.4pp)
            raw_edge_decimal = float(ep.get("edge") or 0)
            edge_pp_val: float | None = round(raw_edge_decimal * 100, 2)

            ps_key = (player, stat)
            if ps_key in player_stat_idx:
                # Clone the matching distribution entry and patch the line/edge fields.
                # The PMF shape is line-independent — only the rendering cut-point changes.
                base = dict(props[player_stat_idx[ps_key]])
                base["line"]           = market_line
                base["has_market_line"] = True
                base["edge_pp"]        = edge_pp_val
                base["model_p_over"]   = round(float(ep.get("model_p_over") or 0), 4)
                base["market_p_over"]  = round(float(ep.get("no_vig_over_prob") or ep.get("market_p_over") or 0), 4)
                base["no_vig_over_prob"] = base["market_p_over"]
                base["no_vig_under_prob"] = round(1.0 - base["market_p_over"], 4)
                # Recompute median_vs_line if median is available
                median = base.get("median")
                if median is not None:
                    base["median_vs_line"] = round(float(median) - market_line, 2)
                base["kelly_pct"] = round(float(ep.get("kelly_pct") or 0), 2)
            else:
                # No PMF entry at all — create a minimal row with edge-board data.
                # pmf will be empty so the chart won't render, but the row is present.
                base = {
                    "player":          player,
                    "stat":            stat,
                    "stat_label":      _STAT_LABELS.get(stat, stat),
                    "stat_raw":        stat.lower(),
                    "line":            market_line,
                    "has_market_line": True,
                    "mean":            ep.get("model_mean"),
                    "median":          ep.get("median"),
                    "mode":            None,
                    "std_dev":         None,
                    "variance":        None,
                    "skewness":        None,
                    "excess_kurtosis": None,
                    "model_p_over":    round(float(ep.get("model_p_over") or 0), 4),
                    "market_p_over":   round(float(ep.get("no_vig_over_prob") or 0), 4),
                    "no_vig_over_prob": round(float(ep.get("no_vig_over_prob") or 0), 4),
                    "no_vig_under_prob": round(1.0 - float(ep.get("no_vig_over_prob") or 0), 4),
                    "edge_pp":         edge_pp_val,
                    "kelly_pct":       round(float(ep.get("kelly_pct") or 0), 2),
                    "pmf":             [],
                }

            props.append(base)
            dist_keys.add((player, stat, market_line))
            merged_count += 1

        if merged_count:
            typer.echo(f"  [merge] Added {merged_count} Edge Board rows missing from Distributions")

    payload: dict = {
        "schema_version": "3.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(props),
        "props": props,
    }
    if release_id:
        payload["release_id"] = release_id
    if git_commit:
        payload["git_commit"] = git_commit
    # Carry through source PMF lineage fields
    for field in ("model_version", "calibration_version", "prediction_timestamp_utc",
                  "market_timestamp_utc"):
        val = pmf_data.get(field)
        if val:
            payload[field] = val
    return payload


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
.edge-neg{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.3);color:var(--amber)}
.edge-none{background:var(--surface3);border:1px solid var(--border2);color:var(--text3)}
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
.prob-edge-neg{color:var(--amber)}

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
      <button class="pill" data-stat="PTS_REB_AST">PRA</button>
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
  <div>Bars: green = over market line · red = under · gold dashed = market line · blue dashed = slider query threshold · Edge = Model P(over) − Market P(no-vig) · UNDER edge = amber</div>
  <div style="margin-top:4px;color:var(--text3)">For entertainment and research purposes only. Gamble responsibly. 21+</div>
</footer>

<script>
(function(){
'use strict';

let ALL = [];
let statFilt = '', dirFilt = '', searchFilt = '', sortKey = 'edge';
const charts = {};
let cdTimer;

// ── Fetch (pointer-aware, cache-busted) ────────────────────────────
// A3: latest.json is a pointer. Follow it to the immutable release payload.
//   1. fetch latest.json with cache:'no-store' + ?t=timestamp
//   2. validate pointer.game_date == target date
//   3. fetch pointer.payload_path with cache:'no-store' + ?r=release_id
//   4. reject payload whose game_date differs from pointer
//   5. never display stale cards as LIVE DATA
function _todayETdist() {
  const d = new Date();
  const et = new Date(d.toLocaleString('en-US', {timeZone: 'America/New_York'}));
  return et.toISOString().slice(0, 10);
}
function load() {
  const params = new URLSearchParams(location.search);
  const targetDate = params.get('date') || _todayETdist();
  fetch('latest.json?t=' + Date.now(), {cache:'no-store'})
    .then(r => r.json())
    .then(ptr => {
      if (!ptr.pointer) {
        // Legacy full-payload — accept if game_date matches
        if (ptr.game_date && ptr.game_date !== targetDate) {
          console.warn('[WOO] Stale pointer game_date', ptr.game_date, '!= target', targetDate);
          return;
        }
        ALL = ptr.props || []; updateKPIs(ptr); render(); startCountdown(300); return;
      }
      if (ptr.game_date !== targetDate) {
        console.warn('[WOO] Stale pointer game_date', ptr.game_date, '!= target', targetDate);
        return;
      }
      const payloadUrl = (ptr.payload_path || ptr.release_payload_path) + '?r=' + encodeURIComponent(ptr.release_id || '');
      fetch(payloadUrl, {cache:'no-store'})
        .then(r => r.json())
        .then(data => {
          if (data.game_date !== ptr.game_date) { console.warn('[WOO] Payload/pointer date mismatch'); return; }
          ALL = data.props || []; updateKPIs(data); render(); startCountdown(300);
        })
        .catch(err => console.warn('[WOO] payload fetch failed:', err));
    })
    .catch(err => console.warn('[WOO] pointer fetch failed:', err));
}

function updateKPIs(data) {
  document.getElementById('kTotal').textContent = data.total_props ?? ALL.length;
  const pos = ALL.filter(p => p.edge_pp !== null && p.edge_pp > 0).length;
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
      // Props with no market line have no edge signal — exclude from directional filters
      if (!p.has_market_line || p.edge_pp === null) return false;
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
  const hasEdge = p.has_market_line && p.edge_pp !== null;
  const ep = hasEdge ? p.edge_pp : 0;
  const epCls = !hasEdge ? 'edge-none' : ep >= 0 ? 'edge-pos' : 'edge-neg';
  const epLabel = !hasEdge ? 'No market line' : (ep >= 0 ? '+' : '') + ep.toFixed(1) + '% edge';
  const statLabel = p.stat_label || p.stat;

  const f2 = v => v != null ? (+v).toFixed(2) : '—';
  const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';

  const modelP = pct(p.model_p_over);
  const mktP = hasEdge ? pct(p.market_p_over) : '—';
  const edgePctLabel = !hasEdge ? '—' : (ep >= 0 ? '+' : '') + ep.toFixed(1) + '%';
  const edgePCls = !hasEdge ? 'prob-mkt' : ep >= 0 ? 'prob-edge-pos' : 'prob-edge-neg';

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
  // ── Bug 4: move the slider query line on the chart ──
  const ch = charts['ch_' + i];
  if (ch) { ch.options._sliderLine = n; ch.update('none'); }
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
      const ch = new Chart(canvas.getContext('2d'), {
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
          _sliderLine: null,  // updated by updateSlider on user interaction
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
            const { ctx, chartArea } = chart;
            const meta = chart.getDatasetMeta(0);

            // Helper: resolve pixel x for a given threshold value
            const xPosFor = v => {
              for (let li = 0; li < labels.length; li++) {
                if (labels[li] >= v) {
                  if (meta.data[li]) return meta.data[li].x - (meta.data[li].width || 8) / 2;
                  break;
                }
              }
              return null;
            };

            const drawVLine = (v, color, labelText) => {
              const xPos = xPosFor(v);
              if (xPos === null) return;
              ctx.save();
              ctx.strokeStyle = color;
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 3]);
              ctx.beginPath();
              ctx.moveTo(xPos, chartArea.top);
              ctx.lineTo(xPos, chartArea.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = color;
              ctx.font = '9px JetBrains Mono';
              ctx.fillText(labelText, xPos + 3, chartArea.top + 9);
              ctx.restore();
            };

            // Gold dashed line = fixed market line
            if (line > 0) drawVLine(line, 'rgba(212,175,55,.85)', 'Line ' + line);

            // Blue dashed line = slider query threshold (only when different from market line)
            const sliderLine = chart.options._sliderLine;
            if (sliderLine != null && sliderLine !== Math.ceil(line)) {
              drawVLine(sliderLine, 'rgba(99,179,237,.9)', 'n=' + sliderLine);
            }
          }
        }]
      });
      charts['ch_' + i] = ch;
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
    release_id: str = typer.Option(
        "",
        "--release-id",
        help="Current-run release identifier (GITHUB_RUN_ID). Written to output JSON and "
             "validated against the source PMF-Distributions JSON.",
    ),
    git_commit: str = typer.Option(
        "",
        "--git-commit",
        help="Current git commit SHA. Written to output for traceability.",
    ),
    model_version: str = typer.Option(
        "",
        "--model-version",
        help="Model version string. Written to Distributions output for traceability.",
    ),
    calibration_version: str = typer.Option(
        "",
        "--calibration-version",
        help="Calibration version string. Written to Distributions output for traceability.",
    ),
) -> None:
    """Generate pure PMF visualization page at Pre-Game/Distributions/.

    Fails closed when the source PMF-Distributions JSON is missing, unreadable,
    or from a different release (release_id mismatch).
    """
    import sys as _sys
    if not game_date:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base = Path(base_dir)
    # A3: Prefer date-specific PMF payload over latest.json pointer
    _pmf_date_path  = base / "Pre-Game" / "PMF-Distributions" / f"{game_date}.json"
    _pmf_latest_path = base / "Pre-Game" / "PMF-Distributions" / "latest.json"
    pmf_path = _pmf_date_path if _pmf_date_path.exists() else _pmf_latest_path
    out_dir  = base / "Pre-Game" / "Distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[generate_distributions_page] game_date={game_date}")
    typer.echo(f"  pmf_path : {pmf_path}")
    typer.echo(f"  out_dir  : {out_dir}")

    try:
        payload = _build_json(pmf_path, game_date,
                              release_id=release_id, git_commit=git_commit)
        if model_version:
            payload["model_version"] = model_version
        if calibration_version:
            payload["calibration_version"] = calibration_version
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[FATAL] {exc}", err=True)
        raise typer.Exit(1)

    import hashlib as _hashlib  # noqa: PLC0415
    # A3: Cache-safe payload structure — immutable release + date-specific + pointer latest.json
    _payload_str = json.dumps(_sanitize(payload), separators=(",", ":"))
    _payload_sha256 = _hashlib.sha256(_payload_str.encode()).hexdigest()
    _eff_release_id = release_id or game_date

    (out_dir / f"{game_date}.json").write_text(_payload_str)

    (out_dir / "releases").mkdir(parents=True, exist_ok=True)
    _release_path = out_dir / "releases" / f"{_eff_release_id}.json"
    _release_path.write_text(_payload_str)

    _now_utc = datetime.now(timezone.utc).isoformat()
    _pointer = _sanitize({
        "pointer": True,
        "release_id": _eff_release_id,
        "game_date": game_date,
        "payload_path": f"releases/{_eff_release_id}.json",
        "payload_sha256": _payload_sha256,
        "release_payload_path": f"releases/{_eff_release_id}.json",
        "date_payload_path": f"{game_date}.json",
        "git_commit": git_commit,
        "model_version": model_version,
        "calibration_version": calibration_version,
        "generated_at_utc": _now_utc,
        "row_count": payload["total_props"],
        "total_props": payload["total_props"],  # backward compat alias
    })
    # latest.json is SELF-CONTAINED (full payload incl. props) — the deployed
    # Distributions shell fetches latest.json directly and reads `data.props`;
    # it does not follow a payload_path pointer. Payload carries release_id /
    # game_date / model_version for lineage. Cache-busting is via the shell's
    # `?t=<timestamp>` query param.
    (out_dir / "latest.json").write_text(_payload_str)

    if not json_only:
        (out_dir / "index.html").write_text(_HTML)

    typer.echo(f"  → releases/{_eff_release_id}.json ({payload['total_props']} props) sha256={_payload_sha256[:12]}")
    typer.echo(f"  → {game_date}.json")
    typer.echo(f"  → latest.json (self-contained payload, release_id={_eff_release_id!r}, row_count={payload['total_props']})")
    if not json_only:
        typer.echo(f"  → index.html")
    typer.echo("[generate_distributions_page] Done.")


if __name__ == "__main__":
    app()
