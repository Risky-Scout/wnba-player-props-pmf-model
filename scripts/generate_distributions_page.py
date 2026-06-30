#!/usr/bin/env python3
"""
Generate the WNBA Pre-Game Distributions page — elite combined Edges + PMF view.

Reads:
  Pre-Game/Edge/latest.json             (v2 edge data, model vs. market)
  Pre-Game/PMF-Distributions/latest.json (v2 PMF data with full distributions)
  Pre-Game-Edge/latest.json             (v1 blueprint — bookmaker deep_links, optional)

Outputs:
  Pre-Game/Distributions/latest.json   — schema v3.0 with edges + PMF + bookmaker links
  Pre-Game/Distributions/{date}.json   — dated archive copy
  Pre-Game/Distributions/index.html    — premium combined tabbed page

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

# ---------------------------------------------------------------------------
# Stat and bookmaker metadata
# ---------------------------------------------------------------------------

_STAT_LABELS: dict[str, str] = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists",
    "FG3M": "3-Pointers Made", "STL": "Steals", "BLK": "Blocks",
    "TURNOVER": "Turnovers", "STOCKS": "Steals + Blocks",
    "PA": "Pts + Ast", "PR": "Pts + Reb",
    "RA": "Reb + Ast", "PRA": "Pts + Reb + Ast",
}

_BOOK_META: dict[str, dict] = {
    "fanduel":    {"label": "FanDuel",     "color": "#1977f3"},
    "draftkings": {"label": "DraftKings",  "color": "#53d337"},
    "betmgm":     {"label": "BetMGM",      "color": "#d4af37"},
    "caesars":    {"label": "Caesars",     "color": "#003087"},
    "bovada":     {"label": "Bovada",      "color": "#e84d4d"},
    "pointsbet":  {"label": "PointsBet",   "color": "#ef1c25"},
    "bet365":     {"label": "Bet365",      "color": "#009b3a"},
    "espnbet":    {"label": "ESPN BET",    "color": "#e51937"},
    "fanatics":   {"label": "Fanatics",    "color": "#ff5f00"},
}


def _american_ev(american: float | int | None, model_p: float) -> float | None:
    """EV per $100 at given American odds and model probability."""
    if american is None or model_p <= 0:
        return None
    try:
        payout = american / 100.0 if american >= 0 else 100.0 / abs(american)
        return round((model_p * payout - (1 - model_p)) * 100, 2)
    except Exception:
        return None


def _best_book(books: list[dict]) -> tuple[str | None, int | None, str | None]:
    """Return (name, odds, link) for the most favourable odds (highest payout)."""
    if not books:
        return None, None, None

    def payout_score(b: dict) -> float:
        o = b.get("odds")
        if o is None:
            return -1e9
        return float(o) if o >= 0 else -100.0 / abs(o)

    best = max(books, key=payout_score)
    return best.get("name"), best.get("odds"), best.get("deep_link")


def _extract_books_v1(v1: dict, player: str, stat_raw: str, direction: str) -> list[dict]:
    """Pull per-bookmaker deep links from the v1 blueprint JSON."""
    books: list[dict] = []
    dir_key = direction.lower()
    for game in v1.get("games", []):
        for pl in game.get("players", []):
            if pl.get("player_name", "").strip().lower() != player.strip().lower():
                continue
            sp = pl.get("stat_projections", {}).get(stat_raw.lower(), {})
            for link_key, val in sp.get("deep_links", {}).items():
                if not isinstance(val, dict):
                    continue
                if dir_key not in link_key:
                    continue
                book_slug = link_key.split("_")[0]
                meta = _BOOK_META.get(book_slug, {"label": book_slug.capitalize(), "color": "#888"})
                odds_val = val.get("odds")
                if odds_val is None:
                    continue
                books.append({
                    "name": meta["label"],
                    "color": meta["color"],
                    "odds": odds_val,
                    "deep_link": val.get("url") or val.get("link"),
                })
    # de-dup by name, keep best odds per bookmaker
    best: dict[str, dict] = {}
    for b in books:
        nm = b["name"]
        if nm not in best or (b.get("odds") or -999) > (best[nm].get("odds") or -999):
            best[nm] = b
    return list(best.values())


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------

def _build_json(edge_path: Path, pmf_path: Path, v1_path: Path, game_date: str) -> dict:
    try:
        edge_data = json.loads(edge_path.read_text())
    except Exception as exc:
        typer.echo(f"[WARN] Edge JSON not found: {exc}")
        edge_data = {"props": []}

    try:
        pmf_data = json.loads(pmf_path.read_text())
    except Exception as exc:
        typer.echo(f"[WARN] PMF JSON not found: {exc}")
        pmf_data = {"props": []}

    v1: dict = {}
    try:
        if v1_path.exists():
            v1 = json.loads(v1_path.read_text())
    except Exception:
        pass

    # Index PMF by (player.lower, stat_raw.lower, line)
    pmf_idx: dict = {}
    for p in pmf_data.get("props", []):
        k = (p["player"].lower(), p.get("stat_raw", p["stat"]).lower(), float(p.get("line", 0)))
        pmf_idx[k] = p

    props: list[dict] = []
    pos_edges = 0
    best_edge_pp = 0.0
    best_edge_label = ""
    pos_sum = 0.0
    pos_count = 0

    for ep in edge_data.get("props", []):
        player = ep.get("player", "")
        stat_up = ep.get("stat", "PTS").upper()
        stat_raw = stat_up.lower()
        direction = ep.get("direction", "OVER")
        line = float(ep.get("market_line", 0))
        model_p = float(ep.get("model_p_over") or 0)
        market_p = float(ep.get("market_p_over") or 0)
        edge_frac = float(ep.get("edge") or 0)
        edge_pp = round(edge_frac * 100, 2)
        kelly = float(ep.get("kelly_pct") or 0)

        # Merge PMF moments
        pmf_rec = pmf_idx.get((player.lower(), stat_raw, line), {})
        pmf_pairs = pmf_rec.get("pmf", [])
        pmf_std = pmf_rec.get("std_dev")
        pmf_skew = pmf_rec.get("skewness")
        pmf_kurt = pmf_rec.get("excess_kurtosis")

        # Bookmaker data from v1
        books = _extract_books_v1(v1, player, stat_raw, direction)
        best_name, best_odds, best_link = _best_book(books)

        ev = _american_ev(best_odds, model_p) if best_odds else None

        abs_e = abs(edge_pp)
        conf = "A" if abs_e >= 10 else ("B" if abs_e >= 6 else ("C" if abs_e >= 3 else "D"))

        if kelly >= 8:
            action = "BET"
        elif kelly >= 4:
            action = "SMALL BET"
        elif kelly >= 1:
            action = "LEAN"
        else:
            action = "PASS"

        if edge_pp > 0:
            pos_edges += 1
            pos_sum += edge_pp
            pos_count += 1
        if abs_e > abs(best_edge_pp):
            best_edge_pp = edge_pp
            best_edge_label = f"{player} {stat_up} {direction} {line}"

        props.append({
            "player_name": player,
            "stat": stat_up,
            "stat_label": _STAT_LABELS.get(stat_up, stat_up),
            "stat_raw": stat_raw,
            "direction": direction,
            "market_line": line,
            "model_mean": round(float(ep.get("model_mean") or pmf_rec.get("mean") or 0), 2),
            "model_median": round(float(ep.get("median") or pmf_rec.get("median") or 0), 1),
            "model_mode": pmf_rec.get("mode"),
            "model_p_over": round(model_p, 4),
            "market_p_over": round(market_p, 4),
            "edge_pp": edge_pp,
            "ev_per_100": ev,
            "kelly_pct": round(kelly, 2),
            "clv_adj_edge_pp": round(float(ep.get("clv_adj_edge") or edge_frac) * 100, 2),
            "reverse_line_movement": bool(ep.get("reverse_line_movement", False)),
            "confidence_tier": conf,
            "action": action,
            "best_odds_american": best_odds,
            "best_bookmaker": best_name,
            "best_deep_link": best_link,
            "all_bookmakers": books,
            "pmf": pmf_pairs,
            "pmf_std_dev": pmf_std,
            "pmf_skewness": pmf_skew,
            "pmf_excess_kurtosis": pmf_kurt,
        })

    props.sort(key=lambda x: -abs(x["edge_pp"]))

    return {
        "schema_version": "3.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "kpis": {
            "total_props": len(props),
            "positive_edges": pos_edges,
            "best_edge_pp": round(best_edge_pp, 2),
            "best_edge_label": best_edge_label,
            "avg_positive_edge_pp": round(pos_sum / max(pos_count, 1), 2),
        },
        "props": props,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA Pre-Game Distributions — WizardOfOdds</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0A0E1A;--surface:#0F1529;--surface2:#141B30;--border:#1E2640;--border2:#2A3350;
  --gold:#D4AF37;--gold2:#B8941E;--text:#E8EAF2;--text2:#9BA3BC;--text3:#5A6380;
  --green:#2ECC71;--red:#E74C3C;--blue:#3B82F6;--amber:#F59E0B;
  --bet-bg:#16341A;--bet-border:#2ECC71;
  --small-bg:#2D2610;--small-border:#F59E0B;
  --lean-bg:#0F1E38;--lean-border:#3B82F6;
}
html{font-size:14px}
body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5}

/* ── Header ───────────────────────────────────────────────── */
header{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}
.brand{font-family:'Fraunces',serif;font-size:1.2rem;font-weight:700;color:var(--text);letter-spacing:.5px}
.brand .accent{color:var(--gold)}
.nav-sep{width:1px;height:18px;background:var(--border2)}
.nav-links{display:flex;gap:4px}
.nav-link{font-size:.72rem;color:var(--text3);padding:3px 10px;border-radius:4px;text-decoration:none;transition:color .15s}
.nav-link:hover,.nav-link.active{color:var(--gold)}
.header-right{margin-left:auto;display:flex;align-items:center;gap:12px;font-size:.72rem;color:var(--text3)}
.live-badge{display:flex;align-items:center;gap:5px;color:var(--green);font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.refresh-cd{color:var(--text3);font-size:.7rem}

/* ── Main ─────────────────────────────────────────────────── */
main{max-width:1400px;margin:20px auto;padding:0 20px}

/* ── KPI strip ────────────────────────────────────────────── */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 18px}
.kpi .val{font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;line-height:1.1}
.kpi .lbl{font-size:.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}
.kpi.gold .val{color:var(--gold)}
.kpi.green .val{color:var(--green)}
.kpi.blue .val{color:var(--blue)}

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:18px}
.tab-btn{font-family:'JetBrains Mono',monospace;font-size:.78rem;padding:10px 22px;border:none;background:none;color:var(--text3);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;letter-spacing:.3px}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--gold);border-bottom-color:var(--gold)}
.tab-pane{display:none}
.tab-pane.active{display:block}

/* ── Filters ──────────────────────────────────────────────── */
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
.filter-group{display:flex;gap:4px;flex-wrap:wrap}
.pill{background:var(--surface2);border:1px solid var(--border2);color:var(--text3);border-radius:20px;padding:3px 12px;font-size:.7rem;cursor:pointer;transition:all .15s;font-family:inherit;white-space:nowrap}
.pill:hover{border-color:var(--gold);color:var(--text)}
.pill.active{background:rgba(212,175,55,.15);border-color:var(--gold);color:var(--gold)}
.search-box{background:var(--surface2);border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:4px 12px;font-size:.78rem;font-family:inherit;outline:none;width:200px}
.search-box:focus{border-color:var(--gold)}
.filter-label{font-size:.7rem;color:var(--text3);white-space:nowrap}
.edge-slider{accent-color:var(--gold);width:120px}
.filter-spacer{flex:1}
.sort-hint{font-size:.7rem;color:var(--text3)}

/* ── Table ────────────────────────────────────────────────── */
.table-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.78rem}
thead{position:sticky;top:57px;z-index:10}
th{background:var(--surface);color:var(--text3);padding:9px 12px;text-align:left;font-weight:600;font-size:.65rem;text-transform:uppercase;letter-spacing:.7px;border-bottom:1px solid var(--border);cursor:pointer;white-space:nowrap;user-select:none}
th:hover{color:var(--text)}
th.sorted-asc::after{content:' ↑';color:var(--gold)}
th.sorted-desc::after{content:' ↓';color:var(--gold)}
th.num{text-align:right}
td{padding:8px 12px;border-bottom:1px solid rgba(30,38,64,.7);white-space:nowrap;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(212,175,55,.025)}
.player{font-weight:600;color:var(--text)}
.stat-chip{display:inline-block;background:var(--surface2);border:1px solid var(--border2);border-radius:4px;padding:1px 7px;font-size:.68rem;font-weight:600;color:var(--text3);letter-spacing:.3px}
.dir-over{color:var(--green);font-weight:700}
.dir-under{color:var(--red);font-weight:700}
.edge-pos{color:var(--green);font-weight:700}
.edge-neg{color:var(--red);font-weight:700}
.kelly-val{color:var(--blue)}
.rlm-badge{display:inline-block;background:rgba(123,79,207,.18);border:1px solid rgba(123,79,207,.35);color:#b4a6f5;border-radius:3px;font-size:.6rem;padding:1px 5px;margin-left:4px}
.action-badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:.68rem;font-weight:700;letter-spacing:.3px}
.ab-bet{background:var(--bet-bg);border:1px solid var(--bet-border);color:var(--green)}
.ab-small{background:var(--small-bg);border:1px solid var(--small-border);color:var(--amber)}
.ab-lean{background:var(--lean-bg);border:1px solid var(--lean-border);color:var(--blue)}
.ab-pass{background:var(--surface2);border:1px solid var(--border2);color:var(--text3)}
.conf-a{color:var(--green);font-weight:700}
.conf-b{color:var(--amber);font-weight:700}
.conf-c{color:var(--blue)}
.conf-d{color:var(--text3)}
.odds-btn{display:inline-block;border-radius:5px;padding:3px 10px;font-size:.7rem;font-weight:600;text-decoration:none;white-space:nowrap;transition:opacity .15s;font-family:inherit}
.odds-btn:hover{opacity:.8}
.no-odds{color:var(--text3);font-size:.72rem}
.num{text-align:right;font-variant-numeric:tabular-nums}
.table-empty{text-align:center;padding:60px;color:var(--text3);font-size:.85rem}

/* ── PMF grid ─────────────────────────────────────────────── */
.pmf-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.pmf-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px;transition:border-color .2s}
.pmf-card:hover{border-color:var(--border2)}
.pmf-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
.pmf-player{font-family:'Fraunces',serif;font-size:.95rem;font-weight:600;color:var(--text)}
.pmf-meta{font-size:.68rem;color:var(--text3);margin-top:2px}
.pmf-edge-badge{font-size:.72rem;font-weight:700;padding:3px 9px;border-radius:4px}
.pmf-edge-pos{background:rgba(46,204,113,.12);color:var(--green)}
.pmf-edge-neg{background:rgba(231,76,60,.12);color:var(--red)}
.pmf-chart-wrap{position:relative;height:110px;margin-bottom:10px}
.pmf-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}
.pmf-stat{text-align:center;background:var(--surface2);border-radius:6px;padding:5px 4px}
.pmf-stat .sv{font-size:.8rem;font-weight:700;color:var(--text)}
.pmf-stat .sk{font-size:.6rem;color:var(--text3);margin-top:1px}
.pmf-prob-row{display:flex;justify-content:space-between;font-size:.72rem;padding:6px 8px;background:var(--surface2);border-radius:6px}
.pmf-prob-row .model-p{color:var(--green)}
.pmf-prob-row .mkt-p{color:var(--text3)}
.pmf-prob-row .edge-p{font-weight:700}
.prob-calc{margin-top:8px}
.prob-calc-label{font-size:.68rem;color:var(--text3);margin-bottom:4px}
.prob-calc input[type=range]{width:100%;accent-color:var(--gold)}
.prob-calc-result{font-size:.75rem;color:var(--text);margin-top:3px}

/* ── No-data state ────────────────────────────────────────── */
.no-data{text-align:center;padding:80px 20px;color:var(--text3)}
.no-data h2{font-family:'Fraunces',serif;font-size:1.4rem;color:var(--text2);margin-bottom:8px}

/* ── Footer ───────────────────────────────────────────────── */
footer{text-align:center;font-size:.68rem;color:var(--text3);padding:28px 0 20px;border-top:1px solid var(--border);margin-top:28px;line-height:1.8}

/* ── Responsive ───────────────────────────────────────────── */
@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px){.kpis{grid-template-columns:repeat(2,1fr)}.search-box{width:140px}.pmf-grid{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <div class="brand">WizardOf<span class="accent">Odds</span></div>
  <div class="nav-sep"></div>
  <nav class="nav-links">
    <a href="../Edge/" class="nav-link">Edge Board</a>
    <a href="" class="nav-link active">Distributions</a>
    <a href="../PMF-Distributions/" class="nav-link">PMF Raw</a>
    <a href="../../Inplay/Edges/" class="nav-link">Live In-Play</a>
  </nav>
  <div class="header-right">
    <span id="dateLabel" style="color:var(--text2)">—</span>
    <div class="live-badge"><div class="dot"></div>LIVE DATA</div>
    <span class="refresh-cd">Refresh in <span id="cdSec">300</span>s</span>
  </div>
</header>

<main>
  <!-- KPI strip -->
  <div class="kpis">
    <div class="kpi"><div class="val" id="kTotal">—</div><div class="lbl">Props Analyzed</div></div>
    <div class="kpi green"><div class="val" id="kPos">—</div><div class="lbl">Positive Edges</div></div>
    <div class="kpi gold"><div class="val" id="kBest">—</div><div class="lbl">Best Edge</div></div>
    <div class="kpi blue"><div class="val" id="kAvg">—</div><div class="lbl">Avg Pos. Edge</div></div>
    <div class="kpi"><div class="val" id="kUpdated" style="font-size:.85rem">—</div><div class="lbl">Last Updated</div></div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" data-tab="edges">Prop Edges</button>
    <button class="tab-btn" data-tab="pmf">PMF Distributions</button>
  </div>

  <!-- ═══════════════ PROP EDGES TAB ═══════════════ -->
  <div id="tab-edges" class="tab-pane active">
    <div class="filters">
      <div class="filter-group" id="statPills">
        <button class="pill active" data-stat="">All Stats</button>
        <button class="pill" data-stat="PTS">PTS</button>
        <button class="pill" data-stat="REB">REB</button>
        <button class="pill" data-stat="AST">AST</button>
        <button class="pill" data-stat="FG3M">3PM</button>
        <button class="pill" data-stat="STL">STL</button>
        <button class="pill" data-stat="BLK">BLK</button>
        <button class="pill" data-stat="PRA">PRA</button>
      </div>
      <div class="filter-group" id="dirPills">
        <button class="pill active" data-dir="">Both</button>
        <button class="pill" data-dir="OVER">Over</button>
        <button class="pill" data-dir="UNDER">Under</button>
      </div>
      <div class="filter-group" id="actionPills">
        <button class="pill active" data-action="">All</button>
        <button class="pill" data-action="BET">BET</button>
        <button class="pill" data-action="SMALL BET">SMALL BET</button>
        <button class="pill" data-action="LEAN">LEAN</button>
      </div>
      <input class="search-box" type="text" id="searchInput" placeholder="Search player…">
      <span class="filter-label">Min edge:</span>
      <input class="edge-slider" type="range" id="edgeSlider" min="0" max="30" step="0.5" value="0">
      <span id="edgeSliderVal" style="font-size:.7rem;color:var(--gold);min-width:28px">0%</span>
      <div class="filter-spacer"></div>
      <span class="sort-hint" id="edgeCount">—</span>
    </div>

    <div class="table-wrap" id="edgeTableWrap">
      <table id="edgeTable">
        <thead>
          <tr>
            <th data-sort="player_name">Player</th>
            <th data-sort="stat">Stat</th>
            <th data-sort="direction">Dir</th>
            <th data-sort="market_line" class="num">Line</th>
            <th data-sort="model_mean" class="num">Mdl Mean</th>
            <th data-sort="model_p_over" class="num">Mdl %</th>
            <th data-sort="market_p_over" class="num">Mkt %</th>
            <th data-sort="edge_pp" class="num sorted-desc">Edge</th>
            <th data-sort="ev_per_100" class="num">EV/$100</th>
            <th data-sort="kelly_pct" class="num">Kelly %</th>
            <th data-sort="confidence_tier">Conf</th>
            <th data-sort="action">Action</th>
            <th>Best Odds</th>
          </tr>
        </thead>
        <tbody id="edgeTbody">
          <tr><td colspan="13" class="table-empty">Loading edges…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ═══════════════ PMF DISTRIBUTIONS TAB ═══════════════ -->
  <div id="tab-pmf" class="tab-pane">
    <div class="filters">
      <div class="filter-group" id="pmfStatPills">
        <button class="pill active" data-pmfstat="">All Stats</button>
        <button class="pill" data-pmfstat="PTS">PTS</button>
        <button class="pill" data-pmfstat="REB">REB</button>
        <button class="pill" data-pmfstat="AST">AST</button>
        <button class="pill" data-pmfstat="FG3M">3PM</button>
        <button class="pill" data-pmfstat="STL">STL</button>
        <button class="pill" data-pmfstat="BLK">BLK</button>
      </div>
      <input class="search-box" type="text" id="pmfSearch" placeholder="Search player…">
      <div class="filter-spacer"></div>
      <span class="sort-hint" id="pmfCount">—</span>
    </div>
    <div class="pmf-grid" id="pmfGrid">
      <div class="no-data"><h2>Loading distributions…</h2></div>
    </div>
  </div>
</main>

<footer>
  <div>WNBA Pre-Game Prop Model — WizardOfOdds Sports Analytics</div>
  <div>Edge = Model P(over) − Market P(over, no-vig) · Kelly criterion fractional sizing · PMF computed from Bayesian hierarchical model with IDR calibration</div>
  <div style="margin-top:4px;color:var(--text3)">For entertainment and research purposes only. Gamble responsibly. 21+</div>
</footer>

<script>
(function(){
'use strict';

// ── State ──────────────────────────────────────────────────────────
let ALL_PROPS = [];
let edgeSortCol = 'edge_pp';
let edgeSortDir = -1; // -1 = desc
let statFilt = '', dirFilt = '', actionFilt = '', searchFilt = '', edgeMin = 0;
let pmfStatFilt = '', pmfSearchFilt = '';
const charts = {};
let cdInterval;

// ── Fetch ──────────────────────────────────────────────────────────
function load() {
  fetch('latest.json?t=' + Date.now())
    .then(r => r.json())
    .then(data => {
      ALL_PROPS = data.props || [];
      updateKPIs(data);
      renderEdge();
      renderPMF();
      startCountdown(300);
    })
    .catch(err => console.warn('Fetch failed:', err));
}

function updateKPIs(data) {
  const k = data.kpis || {};
  document.getElementById('kTotal').textContent = k.total_props ?? ALL_PROPS.length;
  document.getElementById('kPos').textContent = k.positive_edges ?? '—';
  document.getElementById('kBest').textContent = k.best_edge_pp != null ? '+' + k.best_edge_pp.toFixed(1) + '%' : '—';
  document.getElementById('kAvg').textContent = k.avg_positive_edge_pp != null ? '+' + k.avg_positive_edge_pp.toFixed(1) + '%' : '—';
  const gen = data.generated_at ? new Date(data.generated_at) : null;
  document.getElementById('kUpdated').textContent = gen ? gen.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '—';
  document.getElementById('dateLabel').textContent = data.game_date || '—';
}

// ── EDGE TABLE ─────────────────────────────────────────────────────
function filteredEdge() {
  return ALL_PROPS.filter(p => {
    if (statFilt && p.stat !== statFilt) return false;
    if (dirFilt && p.direction !== dirFilt) return false;
    if (actionFilt && p.action !== actionFilt) return false;
    if (searchFilt && !p.player_name.toLowerCase().includes(searchFilt)) return false;
    if (Math.abs(p.edge_pp || 0) < edgeMin) return false;
    return true;
  });
}

function sortedEdge(rows) {
  return [...rows].sort((a, b) => {
    let va = a[edgeSortCol], vb = b[edgeSortCol];
    if (va == null) va = edgeSortDir < 0 ? -Infinity : Infinity;
    if (vb == null) vb = edgeSortDir < 0 ? -Infinity : Infinity;
    if (typeof va === 'string') return edgeSortDir * va.localeCompare(vb);
    return edgeSortDir * (va - vb);
  });
}

function actionClass(a) {
  if (a === 'BET') return 'ab-bet';
  if (a === 'SMALL BET') return 'ab-small';
  if (a === 'LEAN') return 'ab-lean';
  return 'ab-pass';
}
function confClass(c) {
  return {A:'conf-a',B:'conf-b',C:'conf-c',D:'conf-d'}[c] || 'conf-d';
}
function pct(v) { return v != null ? (v*100).toFixed(1)+'%' : '—'; }
function f2(v)  { return v != null ? (+v).toFixed(2) : '—'; }
function f1(v)  { return v != null ? (+v).toFixed(1) : '—'; }
function signPct(v) {
  if (v == null) return '—';
  const s = v >= 0 ? '+' : '';
  return s + (+v).toFixed(1) + '%';
}
function americanFmt(o) {
  if (o == null) return null;
  return o >= 0 ? '+' + o : '' + o;
}

function renderEdge() {
  const rows = sortedEdge(filteredEdge());
  document.getElementById('edgeCount').textContent = rows.length + ' props shown';
  const tbody = document.getElementById('edgeTbody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="13" class="table-empty">No props match current filters.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(p => {
    const ep = p.edge_pp || 0;
    const isOver = p.direction === 'OVER';
    const dirCls = isOver ? 'dir-over' : 'dir-under';
    const edgeCls = ep >= 0 ? 'edge-pos' : 'edge-neg';
    const rlm = p.reverse_line_movement ? '<span class="rlm-badge">RLM</span>' : '';
    const oddsAm = americanFmt(p.best_odds_american);
    const oddsBtn = (oddsAm && p.best_bookmaker)
      ? `<a class="odds-btn" href="${p.best_deep_link || '#'}" target="_blank" rel="noopener"
            style="background:rgba(212,175,55,.12);border:1px solid rgba(212,175,55,.35);color:var(--gold)">
           ${p.best_bookmaker} ${oddsAm}
         </a>`
      : `<span class="no-odds">—</span>`;
    const ev = p.ev_per_100 != null ? (p.ev_per_100 >= 0 ? '+' : '') + p.ev_per_100.toFixed(2) : '—';
    return `<tr>
      <td class="player">${p.player_name}${rlm}</td>
      <td><span class="stat-chip">${p.stat}</span></td>
      <td><span class="${dirCls}">${p.direction}</span></td>
      <td class="num">${p.market_line}</td>
      <td class="num">${f2(p.model_mean)}</td>
      <td class="num">${pct(p.model_p_over)}</td>
      <td class="num" style="color:var(--text3)">${pct(p.market_p_over)}</td>
      <td class="num ${edgeCls}">${signPct(ep)}</td>
      <td class="num" style="color:${p.ev_per_100>0?'var(--green)':p.ev_per_100<0?'var(--red)':'var(--text3)'}">${ev}</td>
      <td class="num ${p.kelly_pct>0?'kelly-val':''}">${p.kelly_pct > 0 ? f1(p.kelly_pct)+'%' : '—'}</td>
      <td class="${confClass(p.confidence_tier)}">${p.confidence_tier}</td>
      <td><span class="action-badge ${actionClass(p.action)}">${p.action}</span></td>
      <td>${oddsBtn}</td>
    </tr>`;
  }).join('');
}

// ── Sort headers ───────────────────────────────────────────────────
document.querySelectorAll('#edgeTable th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.sort;
    if (edgeSortCol === col) {
      edgeSortDir *= -1;
    } else {
      edgeSortCol = col;
      edgeSortDir = -1;
    }
    document.querySelectorAll('#edgeTable th[data-sort]').forEach(h => {
      h.classList.remove('sorted-asc','sorted-desc');
    });
    th.classList.add(edgeSortDir < 0 ? 'sorted-desc' : 'sorted-asc');
    renderEdge();
  });
});

// ── Filters ────────────────────────────────────────────────────────
function pillGroup(containerId, stateKey, renderFn) {
  document.getElementById(containerId).addEventListener('click', e => {
    const btn = e.target.closest('[data-' + stateKey + ']');
    if (!btn) return;
    window['_filt_' + stateKey] = btn.dataset[stateKey];
    if (stateKey === 'stat') statFilt = btn.dataset[stateKey];
    else if (stateKey === 'dir') dirFilt = btn.dataset[stateKey];
    else if (stateKey === 'action') actionFilt = btn.dataset[stateKey];
    else if (stateKey === 'pmfstat') { pmfStatFilt = btn.dataset[stateKey]; renderPMF(); return; }
    document.querySelectorAll('#' + containerId + ' .pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderFn();
  });
}
pillGroup('statPills', 'stat', renderEdge);
pillGroup('dirPills', 'dir', renderEdge);
pillGroup('actionPills', 'action', renderEdge);
pillGroup('pmfStatPills', 'pmfstat', renderPMF);

document.getElementById('searchInput').addEventListener('input', e => {
  searchFilt = e.target.value.trim().toLowerCase();
  renderEdge();
});
document.getElementById('pmfSearch').addEventListener('input', e => {
  pmfSearchFilt = e.target.value.trim().toLowerCase();
  renderPMF();
});
document.getElementById('edgeSlider').addEventListener('input', e => {
  edgeMin = parseFloat(e.target.value);
  document.getElementById('edgeSliderVal').textContent = edgeMin.toFixed(1) + '%';
  renderEdge();
});

// ── Tabs ───────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'pmf') renderPMF();
  });
});

// ── PMF CHARTS ─────────────────────────────────────────────────────
function filteredPMF() {
  return ALL_PROPS.filter(p => {
    if (!p.pmf || p.pmf.length === 0) return false;
    if (pmfStatFilt && p.stat !== pmfStatFilt) return false;
    if (pmfSearchFilt && !p.player_name.toLowerCase().includes(pmfSearchFilt)) return false;
    return true;
  });
}

function renderPMF() {
  const rows = filteredPMF();
  const grid = document.getElementById('pmfGrid');
  document.getElementById('pmfCount').textContent = rows.length + ' distributions';

  if (!rows.length) {
    grid.innerHTML = '<div class="no-data"><h2>No distributions match filters</h2><p>PMF data requires a full pipeline run with player projections.</p></div>';
    return;
  }

  // Destroy existing charts
  Object.keys(charts).forEach(id => { try { charts[id].destroy(); } catch(e){} delete charts[id]; });

  grid.innerHTML = rows.map((p, i) => {
    const ep = p.edge_pp || 0;
    const epCls = ep >= 0 ? 'pmf-edge-pos' : 'pmf-edge-neg';
    const epLabel = (ep >= 0 ? '+' : '') + ep.toFixed(1) + '% ' + p.direction;
    const statLabel = p.stat_label || p.stat;
    const ev_str = p.ev_per_100 != null ? ' · EV $' + (p.ev_per_100>=0?'+':'') + p.ev_per_100.toFixed(2) + '/100' : '';
    return `
    <div class="pmf-card">
      <div class="pmf-header">
        <div>
          <div class="pmf-player">${p.player_name}</div>
          <div class="pmf-meta">${statLabel} · Line: ${p.market_line}${ev_str}</div>
        </div>
        <span class="pmf-edge-badge ${epCls}">${epLabel}</span>
      </div>
      <div class="pmf-chart-wrap"><canvas id="pmfc_${i}"></canvas></div>
      <div class="pmf-stats">
        <div class="pmf-stat"><div class="sv">${f2(p.model_mean)}</div><div class="sk">Mean</div></div>
        <div class="pmf-stat"><div class="sv">${f2(p.model_median)}</div><div class="sk">Median</div></div>
        <div class="pmf-stat"><div class="sv">${p.pmf_std_dev != null ? f2(p.pmf_std_dev) : '—'}</div><div class="sk">Std Dev</div></div>
      </div>
      <div class="pmf-prob-row">
        <span>Model P(Over ${p.market_line}): <span class="model-p">${pct(p.model_p_over)}</span></span>
        <span>Market: <span class="mkt-p">${pct(p.market_p_over)}</span></span>
        <span>Edge: <span class="edge-p ${ep>=0?'dir-over':'dir-under'}">${signPct(ep)}</span></span>
      </div>
      <div class="prob-calc">
        <div class="prob-calc-label">P(X ≥ n): drag to explore</div>
        <input type="range" min="${p.pmf[0]?.[0]??0}" max="${p.pmf[p.pmf.length-1]?.[0]??30}" step="1" value="${Math.round(p.market_line)}" data-pidx="${i}">
        <div class="prob-calc-result" id="pcr_${i}">P(X ≥ ${Math.round(p.market_line)}) = ${pct(p.model_p_over)}</div>
      </div>
    </div>`;
  }).join('');

  // Wire probability calculators
  document.querySelectorAll('.prob-calc input[type=range]').forEach(inp => {
    inp.addEventListener('input', () => {
      const idx = parseInt(inp.dataset.pidx);
      const p = rows[idx];
      const n = parseInt(inp.value);
      const prob = p.pmf.filter(([k]) => k >= n).reduce((s, [, v]) => s + v, 0);
      document.getElementById('pcr_' + idx).textContent = `P(X ≥ ${n}) = ${(prob*100).toFixed(1)}%`;
    });
  });

  // Draw charts
  requestAnimationFrame(() => {
    rows.forEach((p, i) => {
      const canvas = document.getElementById('pmfc_' + i);
      if (!canvas || !p.pmf.length) return;
      const line = p.market_line;
      const labels = p.pmf.map(([k]) => k);
      const vals = p.pmf.map(([, v]) => +(v * 100).toFixed(3));
      const colors = labels.map(k => k > line ? 'rgba(46,204,113,.7)' : 'rgba(231,76,60,.6)');
      const borderColors = labels.map(k => k > line ? 'rgba(46,204,113,.9)' : 'rgba(231,76,60,.8)');
      try {
        charts['pmfc_' + i] = new Chart(canvas.getContext('2d'), {
          type: 'bar',
          data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderColor: borderColors, borderWidth: 1, borderRadius: 1 }] },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
              legend: { display: false },
              tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(2) + '%' } }
            },
            scales: {
              x: { grid: { color: 'rgba(30,38,64,.6)' }, ticks: { color: '#5A6380', font: { size: 9 }, maxRotation: 0, maxTicksLimit: 8 } },
              y: { grid: { color: 'rgba(30,38,64,.6)' }, ticks: { color: '#5A6380', font: { size: 9 } }, title: { display: false } }
            },
            annotation: {}
          }
        });
      } catch(e) { console.warn('Chart error', e); }
    });
  });
}

// ── Countdown ──────────────────────────────────────────────────────
function startCountdown(sec) {
  clearInterval(cdInterval);
  let t = sec;
  document.getElementById('cdSec').textContent = t;
  cdInterval = setInterval(() => {
    t--;
    document.getElementById('cdSec').textContent = Math.max(t, 0);
    if (t <= 0) { clearInterval(cdInterval); load(); }
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
) -> None:
    """Merge v2 Edge + PMF + v1 deep_links → Pre-Game/Distributions/."""
    if not game_date:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    base = Path(base_dir)
    edge_path = base / "Pre-Game" / "Edge" / "latest.json"
    pmf_path  = base / "Pre-Game" / "PMF-Distributions" / "latest.json"
    v1_path   = base / "Pre-Game-Edge" / "latest.json"
    out_dir   = base / "Pre-Game" / "Distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[generate_distributions_page] game_date={game_date}")
    typer.echo(f"  edge_path : {edge_path}")
    typer.echo(f"  pmf_path  : {pmf_path}")
    typer.echo(f"  v1_path   : {v1_path}")
    typer.echo(f"  out_dir   : {out_dir}")

    payload = _build_json(edge_path, pmf_path, v1_path, game_date)

    (out_dir / "latest.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out_dir / f"{game_date}.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out_dir / "index.html").write_text(_HTML)

    typer.echo(f"  → latest.json ({payload['kpis']['total_props']} props)")
    typer.echo(f"  → {game_date}.json")
    typer.echo(f"  → index.html")
    typer.echo("[generate_distributions_page] Done.")


if __name__ == "__main__":
    app()
