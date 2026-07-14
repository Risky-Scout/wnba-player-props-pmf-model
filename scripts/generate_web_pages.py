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


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


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

def _american_to_decimal(american: float | None) -> float | None:
    """Convert American odds to decimal (for EV calc). Returns None if invalid."""
    if american is None:
        return None
    try:
        a = float(american)
        return (a / 100 + 1) if a >= 0 else (100 / abs(a) + 1)
    except (TypeError, ZeroDivisionError):
        return None


def _build_edge_json(
    edges_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    game_date: str,
    v1_path: str | None = None,
    release_id: str = "",
    git_commit: str = "",
    model_version: str = "",
    calibration_version: str = "",
    market_status: str = "",
    raw_quote_count: int | None = None,
    reconciled_quote_count: int | None = None,
    market_request_timestamp_utc: str = "",
) -> dict:
    """Build the payload for Pre-Game/Edge/latest.json.

    Merges best bookmaker odds from two sources (in priority order):
      1. edges_df columns: ``deep_link``, ``bookmaker``, ``over_odds``, ``under_odds``
         (populated when the pipeline ran with --props-source odds_api)
      2. ``Pre-Game-Edge/latest.json`` v1 blueprint ``deep_links`` dict
         (populated once format_pregame_output writes them)

    Falls back gracefully to None values when odds data is unavailable.
    """
    # ── Merge projection moments ──────────────────────────────────────
    proj_cols = [c for c in ["player_id", "stat", "pmf_mean", "median"] if c in proj_df.columns]
    merge_keys = [k for k in ["player_id", "stat"] if k in edges_df.columns and k in proj_df.columns]
    if len(proj_cols) >= 2 and len(merge_keys) == 2:
        merged = edges_df.merge(
            proj_df[proj_cols],
            on=merge_keys,
            how="left",
            suffixes=("", "_proj"),
        )
    else:
        merged = edges_df.copy()

    # ── Load v1 blueprint deep-links index ────────────────────────────
    # key: (player_name_lower, stat_slug_lower, direction_upper) → dict
    v1_idx: dict[tuple, dict] = {}
    _v1_path = v1_path or ""
    if not _v1_path:
        # Relative to CWD; resolve from the out_dir path if needed
        _v1_path = "tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json"
    try:
        v1 = json.loads(Path(_v1_path).read_text())
        for game in v1.get("games", []):
            for pl in game.get("players", []):
                pname = (pl.get("player_name") or "").lower()
                for stat_key, sv in pl.get("stat_projections", {}).items():
                    for link_key, val in (sv.get("deep_links") or {}).items():
                        # link_key format: "{book}_{stat}_{over|under}"
                        parts = link_key.rsplit("_", 1)
                        direction = "OVER" if (parts[-1] if len(parts) > 1 else "").lower() == "over" else "UNDER"
                        v1_idx[(pname, stat_key.lower(), direction)] = {
                            "deep_link": val.get("url") or val if isinstance(val, str) else None,
                            "bookmaker": val.get("bookmaker") or link_key.split("_")[0] if isinstance(val, dict) else link_key.split("_")[0],
                            "odds_american": val.get("odds_american") if isinstance(val, dict) else None,
                        }
    except Exception:
        pass  # No v1 blueprint or missing deep_links — safe fallback

    # ── Build rows ────────────────────────────────────────────────────
    rows = []
    for _, r in merged.iterrows():
        edge = float(r.get("edge_over", 0) or 0)
        kelly = float(r.get("kelly_fraction", 0) or 0)
        direction = "OVER" if edge >= 0 else "UNDER"
        abs_edge = abs(edge)

        # Action badge
        kelly_pct = kelly * 100
        if kelly_pct >= 8:
            action = "BET"
        elif kelly_pct >= 4:
            action = "SMALL BET"
        elif kelly_pct >= 1 or abs_edge * 100 >= 3:
            action = "LEAN"
        else:
            action = "PASS"

        # Confidence tier
        abs_ep = abs_edge * 100
        if abs_ep >= 10:
            conf = "A"
        elif abs_ep >= 6:
            conf = "B"
        elif abs_ep >= 3:
            conf = "C"
        else:
            conf = "D"

        # Best bookmaker odds — primary: edges_df columns
        best_link: str | None = None
        best_book: str | None = None
        best_odds: float | None = None

        # Vendor display name map
        _VENDOR_DISPLAY = {
            "draftkings": "DraftKings",
            "fanduel": "FanDuel",
            "betmgm": "BetMGM",
            "caesars": "Caesars",
            "pinnacle": "Pinnacle",
            "betonlineag": "BetOnline",
            "circasports": "Circa",
            "pointsbetus": "PointsBet",
            "superbook": "SuperBook",
            "wynnbet": "WynnBet",
        }

        # Try edges DataFrame (Odds API source); check both 'bookmaker' and 'vendor'
        raw_vendor = None
        if "bookmaker" in r.index:
            raw_vendor = r.get("bookmaker") or None
        if not raw_vendor and "vendor" in r.index:
            raw_vendor = r.get("vendor") or None
        if raw_vendor:
            best_book = _VENDOR_DISPLAY.get(str(raw_vendor).lower(), str(raw_vendor).title())
        if "deep_link" in r.index:
            best_link = r.get("deep_link") or None
        if direction == "OVER" and "over_odds" in r.index:
            val = r.get("over_odds")
            best_odds = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None
        elif direction == "UNDER" and "under_odds" in r.index:
            val = r.get("under_odds")
            best_odds = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None

        # Fallback: v1 blueprint
        if best_link is None and v1_idx:
            pname = str(r.get("player_name", "")).lower()
            stat_slug = str(r.get("stat", "")).lower()
            v1_rec = v1_idx.get((pname, stat_slug, direction))
            if v1_rec:
                best_link = best_link or v1_rec.get("deep_link")
                best_book = best_book or v1_rec.get("bookmaker")
                best_odds = best_odds if best_odds is not None else v1_rec.get("odds_american")

        # Model vs Market plain-English signal text
        model_p = float(r.get("model_prob_over", 0) or 0)
        market_p_val = float(r.get("market_prob_over_no_vig", 0) or 0)
        model_pct = round(model_p * 100)
        market_pct = round(market_p_val * 100)
        edge_abs_pp = abs_edge * 100
        if edge_abs_pp >= 15:
            strength = "Strong signal"
        elif edge_abs_pp >= 8:
            strength = "Signal"
        else:
            strength = "Lean"
        direction_word = direction  # "OVER" or "UNDER"
        signal_text = (
            f"{strength}: Model {model_pct}% vs Market {market_pct}% \u2192 {direction_word}"
        )

        # Best Odds display: "DraftKings -108" format
        best_odds_display: str | None = None
        if best_book and best_odds is not None:
            odds_str = (
                f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds))
            )
            best_odds_display = f"{best_book} {odds_str}"
        elif best_odds is not None:
            odds_str = (
                f"+{int(best_odds)}" if best_odds > 0 else str(int(best_odds))
            )
            best_odds_display = odds_str

        rows.append({
            "player": r["player_name"],
            "stat": str(r["stat"]).upper(),
            "direction": direction,
            "action": action,
            "confidence": conf,
            "model_mean": round(float(r.get("pmf_mean", 0) or 0), 2),
            "median": round(float(r.get("median", 0) or 0), 1),
            "market_line": round(float(r.get("line", 0) or 0), 1),
            "model_p_over": round(float(r.get("model_prob_over", 0) or 0), 4),
            # Compute push-aware under from PMF when available; fallback is 1 - p_over
            "model_p_under": round(
                (1.0 - float(r.get("model_prob_over", 0) or 0)
                 - float(r.get("model_prob_push", 0) or 0)), 4
            ),
            "model_p_push": round(float(r.get("model_prob_push", 0) or 0), 4),
            "market_p_over": round(float(r.get("market_prob_over_no_vig", 0) or 0), 4),
            "no_vig_over_prob": round(float(r.get("no_vig_over_prob", r.get("market_prob_over_no_vig", 0)) or 0), 4),
            "no_vig_under_prob": round(float(r.get("no_vig_under_prob", 1.0 - float(r.get("market_prob_over_no_vig", 0) or 0)) or 0), 4),
            "edge": round(edge, 4),
            # edge_pp mirrors Distributions page format: percentage points (e.g. -37.44 means -37.44pp).
            # This allows downstream consumers to compare edge values across both pages using the same unit.
            "edge_pp": round(edge * 100, 2),
            "kelly_pct": round(kelly_pct, 2),
            "kelly_units": round(kelly_pct, 2),
            "abs_edge": abs_edge,
            # time_decay_adjusted_edge: model edge multiplied by a time-decay factor.
            # This is NOT CLV. Renamed from the legacy 'clv_adj_edge' output key.
            # Dashboard consumers should read 'time_decay_adjusted_edge'.
            # The legacy key 'clv_adj_edge' is no longer written for new files.
            "time_decay_adjusted_edge": round(float(r.get("time_decay_adjusted_edge", r.get("clv_decay_adjusted_edge", edge)) or edge), 4),
            "line_moved_toward_over": bool(r.get("line_moved_toward_over", False)),
            "reverse_line_movement": bool(r.get("reverse_line_movement_flag", False)),
            "model_market_ratio": round(float(r.get("model_market_ratio", 1) or 1), 3),
            "best_bookmaker": best_book,
            "best_odds_american": best_odds,
            "best_odds_display": best_odds_display,
            "best_deep_link": best_link,
            "signal_text": signal_text,
        })

    rows.sort(key=lambda x: -x["abs_edge"])
    payload: dict = {
        "schema_version": "2.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(rows),
        "over_signals": sum(1 for r in rows if r["direction"] == "OVER"),
        "under_signals": sum(1 for r in rows if r["direction"] == "UNDER"),
        "props": rows,
    }
    if release_id:
        payload["release_id"] = release_id
    if git_commit:
        payload["git_commit"] = git_commit
    if model_version:
        payload["model_version"] = model_version
    if calibration_version:
        payload["calibration_version"] = calibration_version

    # Market status — evidence-based: read from audit JSON or infer from row count.
    # Allowed values: SUCCESS_WITH_MARKETS, LIVE_MARKETS_NOT_YET_AVAILABLE, FAILURE.
    # Consistency rules enforced here:
    #   SUCCESS_WITH_MARKETS requires total_props > 0
    #   LIVE_MARKETS_NOT_YET_AVAILABLE requires total_props == 0
    #   FAILURE requires total_props == 0
    n_rows = len(rows)
    inferred_status = "SUCCESS_WITH_MARKETS" if n_rows > 0 else "LIVE_MARKETS_NOT_YET_AVAILABLE"
    if market_status:
        # Validate consistency between supplied status and actual row count
        if market_status == "SUCCESS_WITH_MARKETS" and n_rows == 0:
            typer.echo(
                "[WARN] --market-status=SUCCESS_WITH_MARKETS but 0 edge rows — "
                "overriding to LIVE_MARKETS_NOT_YET_AVAILABLE", err=True
            )
            market_status = "LIVE_MARKETS_NOT_YET_AVAILABLE"
        elif market_status == "LIVE_MARKETS_NOT_YET_AVAILABLE" and n_rows > 0:
            typer.echo(
                f"[WARN] --market-status=LIVE_MARKETS_NOT_YET_AVAILABLE but {n_rows} rows exist — "
                "overriding to SUCCESS_WITH_MARKETS", err=True
            )
            market_status = "SUCCESS_WITH_MARKETS"
    else:
        market_status = inferred_status

    payload["market_status"] = market_status
    payload["raw_quote_count"] = raw_quote_count if raw_quote_count is not None else (
        len(edges_df) if edges_df is not None else 0
    )
    payload["reconciled_quote_count"] = reconciled_quote_count if reconciled_quote_count is not None else n_rows
    if market_request_timestamp_utc:
        payload["market_request_timestamp_utc"] = market_request_timestamp_utc
    return payload


def _build_pmf_json(
    edges_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    game_date: str,
    release_id: str = "",
    git_commit: str = "",
    model_version: str = "",
    calibration_version: str = "",
) -> dict:
    """Build the payload for Pre-Game/PMF-Distributions/latest.json.

    Shows ALL players × ALL stats (except suppressed stats). Edge columns are
    left-joined from edges_df so players with no market line still appear with
    their full PMF distribution (edge/kelly/market columns default to 0/null).
    """
    # No stats are suppressed: all valid modeled PMFs appear on the public page.
    # If a stat has weak calibration, it carries a calibration_status/warning field
    # but is not deleted from the page.

    # Start from the full projection universe (all players × all stats).
    base_df = proj_df.copy()

    # Columns to pull from the edge report (market line + edge signal).
    _edge_payload_cols = [
        "edge_over", "kelly_fraction", "model_prob_over",
        "market_prob_over_no_vig", "no_vig_over_prob", "no_vig_under_prob",
        "line", "bookmaker", "over_odds", "under_odds",
    ]
    edge_join_cols = [c for c in _edge_payload_cols if c in edges_df.columns]
    merge_keys = [k for k in ["player_id", "stat"] if k in base_df.columns and k in edges_df.columns]

    if merge_keys and edge_join_cols:
        merged = base_df.merge(
            edges_df[merge_keys + edge_join_cols].drop_duplicates(subset=merge_keys),
            on=merge_keys,
            how="left",
            suffixes=("", "_edge"),
        )
    else:
        merged = base_df.copy()

    # Try to load last-5 actual averages from the features parquet for the actuals overlay
    _last5_lookup: dict[tuple, float] = {}
    _STAT_TO_L5_COL = {
        "pts": "player_pts_mean_l5",
        "reb": "player_reb_mean_l5",
        "ast": "player_ast_mean_l5",
        "fg3m": "player_fg3m_mean_l5",
        "stl": "player_stl_mean_l5",
        "blk": "player_blk_mean_l5",
        "turnover": "player_turnover_mean_l5",
    }
    try:
        import glob as _glob
        import pathlib as _pathlib
        _feat_path = _pathlib.Path("data/processed/wnba_player_game_features_wide.parquet")
        if _feat_path.exists():
            _feat_df = pd.read_parquet(_feat_path)
            for _stat, _l5_col in _STAT_TO_L5_COL.items():
                if _l5_col in _feat_df.columns and "player_id" in _feat_df.columns:
                    _latest = (
                        _feat_df.sort_values("game_date")
                        .groupby("player_id")[_l5_col]
                        .last()
                    )
                    for _pid, _val in _latest.items():
                        if pd.notna(_val):
                            _last5_lookup[(_pid, _stat)] = round(float(_val), 2)
    except Exception:
        pass

    # Load calibration 30-day hit rate if available
    _cal_over_hit_rate: float | None = None
    _cal_under_hit_rate: float | None = None
    try:
        _results_path = _pathlib.Path("data/clv_tracking/results.parquet")
        if _results_path.exists():
            _res = pd.read_parquet(_results_path)
            if not _res.empty and "actual_outcome" in _res.columns and "line" in _res.columns:
                _res["date"] = pd.to_datetime(_res.get("game_date", pd.Series(dtype="object")), errors="coerce")
                _cutoff = pd.Timestamp(game_date) - pd.Timedelta(days=30) if game_date else None
                if _cutoff is not None:
                    _res_30 = _res[_res["date"] >= _cutoff]
                else:
                    _res_30 = _res
                if len(_res_30) >= 10:
                    _res_30 = _res_30.copy()
                    _res_30["_went_over"] = _res_30["actual_outcome"] > _res_30["line"]
                    _cal_over_hit_rate = round(float(_res_30["_went_over"].mean()), 4)
                    _cal_under_hit_rate = round(1.0 - _cal_over_hit_rate, 4)
    except Exception:
        pass

    props = []
    for _, r in merged.iterrows():
        raw_pmf = r.get("pmf_json", None)
        pmf_str = raw_pmf if isinstance(raw_pmf, str) and raw_pmf.strip() else "{}"
        pairs, mu, var, std, skew, kurt, mode_k, median_k = _parse_pmf(pmf_str)
        if not pairs:
            print(f"  [WARN] Skipping {r.get('player_name','?')} {r.get('stat','?')} — no PMF data after merge (pmf_json={repr(raw_pmf)[:40]})")
            continue
        # Compute pmf_full from raw JSON so it sums to 1.0 (no threshold filtering).
        # _parse_pmf filters at 0.001 which causes pmf_full mass < 1; fix here.
        try:
            import json as _j
            _d = _j.loads(pmf_str) if pmf_str and pmf_str != "{}" else {}
            _k_raw = np.array([int(x) for x in _d.keys()], dtype=float)
            _p_raw = np.array(list(_d.values()), dtype=float)
            _total = _p_raw.sum()
            if _total > 0:
                _p_raw = _p_raw / _total
            _pairs_full = [[int(kk), round(float(pp), 7)] for kk, pp in zip(_k_raw, _p_raw) if pp > 0]
        except Exception:
            _pairs_full = pairs
        edge = float(r.get("edge_over", 0) or 0)
        _cal_mean = r.get("pmf_mean") or r.get("pmf_mean_proj")
        _display_mean = round(float(_cal_mean), 2) if _cal_mean is not None and float(_cal_mean) > 0 else round(mu, 2)
        market_line = round(float(r.get("line", 0) or 0), 1)

        # Compute push-aware model probabilities from the full PMF
        # For integer lines: p_push > 0; p_under = P(X < line) not 1-p_over
        _pmf_arr_raw = {}
        try:
            import json as _j
            _pmf_arr_raw = _j.loads(pmf_str) if pmf_str and pmf_str != "{}" else {}
        except Exception:
            pass
        if _pmf_arr_raw and market_line > 0:
            _k = np.array([int(kk) for kk in _pmf_arr_raw.keys()], dtype=float)
            _p = np.array(list(_pmf_arr_raw.values()), dtype=float)
            _tot = _p.sum()
            if _tot > 0:
                _p = _p / _tot
            model_p_over = round(float(_p[_k > float(market_line)].sum()), 6)
            _is_int_line = (float(market_line) == math.floor(float(market_line)))
            model_p_push = round(float(_p[_k == float(market_line)].sum()), 6) if _is_int_line else 0.0
            model_p_under = round(max(0.0, 1.0 - model_p_over - model_p_push), 6)
        else:
            model_p_over = round(float(r.get("model_prob_over", 0) or 0), 4)
            model_p_push = 0.0
            model_p_under = round(1.0 - model_p_over, 4)

        market_p_over = round(float(r.get("market_prob_over_no_vig", 0) or 0), 4)
        no_vig_over = round(float(r.get("no_vig_over_prob", market_p_over) or market_p_over), 4)
        no_vig_under = round(float(r.get("no_vig_under_prob", 1.0 - market_p_over) or (1.0 - market_p_over)), 4)
        _pid = r.get("player_id")
        _stat_raw = str(r.get("stat", ""))
        last_5_avg = _last5_lookup.get((_pid, _stat_raw)) if _pid is not None else None

        # pmf_full: ALL probability pairs from raw JSON (normalized, sums to 1)
        # pmf_chart: filtered for chart rendering (omit tiny masses for performance)
        _CHART_THRESHOLD = 0.001
        pmf_chart = [[k, v] for k, v in _pairs_full if v >= _CHART_THRESHOLD]
        omitted_mass = round(
            sum(v for _, v in _pairs_full) - sum(v for _, v in pmf_chart), 8
        )

        props.append({
            "player": r["player_name"],
            "stat": str(r["stat"]).upper(),
            "stat_raw": _stat_raw,
            "line": market_line,
            "mean": _display_mean,
            "median": round(median_k, 1),
            "mode": mode_k,
            "median_vs_line": round(median_k - market_line, 2) if market_line > 0 else None,
            "variance": round(var, 3),
            "std_dev": round(std, 3),
            "skewness": round(skew, 3),
            "excess_kurtosis": round(kurt, 3),
            "model_p_over": model_p_over,
            "model_p_under": model_p_under,
            "model_p_push": model_p_push,
            "model_p_over_pct": round(model_p_over * 100, 1),
            "model_p_under_pct": round(model_p_under * 100, 1),
            "market_p_over": market_p_over,
            "no_vig_over_prob": no_vig_over,
            "no_vig_under_prob": no_vig_under,
            "edge": round(edge, 4),
            "kelly_pct": round(float(r.get("kelly_fraction", 0) or 0) * 100, 2),
            "last_5_avg": last_5_avg,
            # pmf_full: complete mass used for all probability calculations (sums to 1)
            "pmf_full": _pairs_full,
            # pmf_chart: filtered for rendering performance
            "pmf_chart": pmf_chart,
            "omitted_chart_mass": omitted_mass if omitted_mass > 0 else None,
            # Legacy alias (also full, not filtered)
            "pmf": _pairs_full,
        })
    props.sort(key=lambda x: -abs(x["edge"]))
    result = {
        "schema_version": "2.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_date": game_date,
        "total_props": len(props),
        "props": props,
    }
    if release_id:
        result["release_id"] = release_id
    if git_commit:
        result["git_commit"] = git_commit
    if model_version:
        result["model_version"] = model_version
    if calibration_version:
        result["calibration_version"] = calibration_version
    if _cal_over_hit_rate is not None:
        result["calibration_30d"] = {
            "over_hit_rate": _cal_over_hit_rate,
            "under_hit_rate": _cal_under_hit_rate,
            "note": "30-day empirical hit rate from results tracking",
        }
    return result


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_EDGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WNBA Prop Edges — WizardOfOdds</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0A0E1A;--surface:#0F1529;--surface2:#141B30;--surface3:#1A2238;
  --border:#1E2640;--border2:#2A3350;
  --gold:#D4AF37;--gold-dim:rgba(212,175,55,.12);--gold-border:rgba(212,175,55,.3);
  --text:#E8EAF2;--text2:#9BA3BC;--text3:#5A6380;
  --green:#2ECC71;--green-dim:rgba(46,204,113,.12);--green-border:rgba(46,204,113,.3);
  --red:#E74C3C;--red-dim:rgba(231,76,60,.12);--red-border:rgba(231,76,60,.3);
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
main{max-width:1500px;margin:20px auto;padding:0 20px}

/* KPI strip */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 18px}
.kpi .val{font-family:'Fraunces',serif;font-size:1.55rem;font-weight:700;line-height:1.1;color:var(--text)}
.kpi .val.gold{color:var(--gold)}
.kpi .val.green{color:var(--green)}
.kpi .val.red{color:var(--red)}
.kpi .val.amber{color:var(--amber)}
.kpi .lbl{font-size:.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}

/* Filters */
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
.pill{background:var(--surface2);border:1px solid var(--border2);color:var(--text3);border-radius:20px;padding:3px 13px;font-size:.7rem;cursor:pointer;transition:all .15s;font-family:inherit;white-space:nowrap}
.pill:hover{border-color:var(--gold);color:var(--text)}
.pill.active{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}
.search-box{background:var(--surface2);border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:4px 12px;font-size:.76rem;font-family:inherit;outline:none;width:200px}
.search-box:focus{border-color:var(--gold)}
.flt-spacer{flex:1}
.count-lbl{font-size:.7rem;color:var(--text3)}

/* Table */
.tbl-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:.76rem}
thead{position:sticky;top:45px;z-index:5}
th{background:var(--surface);color:var(--text3);padding:9px 12px;text-align:left;font-weight:600;font-size:.63rem;text-transform:uppercase;letter-spacing:.7px;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
th:hover{color:var(--text)}
th.sorted{color:var(--gold)}
th.sorted::after{content:attr(data-arrow);color:var(--gold);margin-left:4px}
th.numeric{text-align:right}
td{padding:8px 12px;border-bottom:1px solid rgba(30,38,64,.5);white-space:nowrap;vertical-align:middle}
td.numeric{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(212,175,55,.025)}

/* Row tiers */
tr.row-a td{background:rgba(46,204,113,.03)}
tr.row-a:hover td{background:rgba(46,204,113,.055)}
tr.row-b td{background:rgba(212,175,55,.02)}

/* Cell styles */
.player-cell{display:flex;align-items:center;gap:6px}
.player-name{font-weight:600;color:var(--text)}
.rlm-badge{background:rgba(123,79,207,.12);border:1px solid rgba(123,79,207,.35);color:#9386f2;border-radius:3px;font-size:.6rem;padding:1px 5px}
.stat-chip{background:var(--surface3);border:1px solid var(--border2);border-radius:4px;padding:1px 7px;font-size:.67rem;font-weight:600;color:var(--text2)}
.dir-over{color:var(--green);font-weight:700}
.dir-under{color:var(--amber);font-weight:700}
.edge-pos{color:var(--green);font-weight:700}
.edge-neg{color:var(--amber);font-weight:700}
.prob-val{color:var(--text2)}

/* Action badges */
.ab{display:inline-flex;align-items:center;border-radius:4px;padding:2px 9px;font-size:.65rem;font-weight:700;letter-spacing:.3px;white-space:nowrap}
.ab-bet{background:rgba(46,204,113,.12);border:1px solid rgba(46,204,113,.35);color:var(--green)}
.ab-small{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:var(--amber)}
.ab-lean{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:var(--blue)}
.ab-pass{background:var(--surface2);border:1px solid var(--border2);color:var(--text3)}

/* Confidence tier */
.tier{display:inline-block;width:18px;height:18px;line-height:18px;text-align:center;border-radius:3px;font-size:.65rem;font-weight:700}
.tier-a{background:rgba(46,204,113,.2);border:1px solid rgba(46,204,113,.5);color:var(--green)}
.tier-b{background:var(--gold-dim);border:1px solid var(--gold-border);color:var(--gold)}
.tier-c{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:var(--blue)}
.tier-d{background:var(--surface2);border:1px solid var(--border2);color:var(--text3)}

/* Best Odds button */
.odds-btn{display:inline-block;padding:3px 10px;border-radius:5px;font-size:.7rem;font-weight:600;text-decoration:none;white-space:nowrap;transition:all .15s;background:var(--gold-dim);border:1px solid var(--gold-border);color:var(--gold)}
.odds-btn:hover{background:rgba(212,175,55,.22);border-color:var(--gold)}
.no-odds{color:var(--text3);font-size:.7rem}
.odds-pos{color:var(--green)}
.odds-neg{color:var(--text2)}

/* Empty / loading */
.empty-row td{text-align:center;padding:50px;color:var(--text3);font-size:.82rem}

/* Footer */
footer{text-align:center;font-size:.67rem;color:var(--text3);padding:26px 0 18px;border-top:1px solid var(--border);margin-top:28px;line-height:1.9}

/* Responsive */
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}}
@media(max-width:768px){.kpis{grid-template-columns:repeat(2,1fr)}.search-box{width:140px}}
</style>
</head>
<body>

<header>
  <div class="brand">WizardOf<span class="acc">Odds</span></div>
  <div class="sep"></div>
  <nav class="nav-links">
    <a href="" class="nav-link active">Prop Edges</a>
    <a href="../Distributions/" class="nav-link">Distributions</a>
    <a href="../Pricer/" class="nav-link">Market X-Ray</a>
    <a href="../../In-Play/Edges/" class="nav-link">Live In-Play</a>
  </nav>
  <div class="hdr-right">
    <span id="dateLabel">—</span>
    <span><span class="dot"></span>WNBA Pre-Game · Bayesian PMF Model</span>
    <span id="genTime" style="color:var(--text3)">—</span>
  </div>
</header>

<main>
  <!-- KPIs -->
  <div class="kpis">
    <div class="kpi"><div class="val" id="kTotal">—</div><div class="lbl">Props Analyzed</div></div>
    <div class="kpi"><div class="val green" id="kOver">—</div><div class="lbl">Over Signals</div></div>
    <div class="kpi"><div class="val amber" id="kUnder">—</div><div class="lbl">Under Signals</div></div>
    <div class="kpi"><div class="val gold" id="kBet">—</div><div class="lbl">BET Signals</div></div>
    <div class="kpi"><div class="val" id="kTopEdge">—</div><div class="lbl">Best Edge</div></div>
  </div>

  <!-- Filters -->
  <div class="filters">
    <div id="statPills">
      <button class="pill active" data-stat="">All Stats</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
      <button class="pill" data-stat="FG3M">3PM</button>
      <button class="pill" data-stat="STL">STL</button>
      <button class="pill" data-stat="BLK">BLK</button>
      <button class="pill" data-stat="PTS_REB_AST">PRA</button>
    </div>
    <div id="dirPills">
      <button class="pill active" data-dir="">Both</button>
      <button class="pill" data-dir="OVER">Over</button>
      <button class="pill" data-dir="UNDER">Under</button>
    </div>
    <div id="actionPills">
      <button class="pill active" data-action="">All</button>
      <button class="pill" data-action="BET">BET</button>
      <button class="pill" data-action="SMALL BET">SMALL BET</button>
      <button class="pill" data-action="LEAN">LEAN</button>
    </div>
    <input class="search-box" type="text" id="searchInput" placeholder="Search player…">
    <div class="flt-spacer"></div>
    <span class="count-lbl" id="countLbl">—</span>
  </div>

  <!-- Table -->
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th data-col="player">Player</th>
          <th data-col="stat">Stat</th>
          <th data-col="direction">Dir</th>
          <th data-col="action">Action</th>
          <th data-col="confidence">Tier</th>
          <th data-col="market_line" class="numeric">Line</th>
          <th data-col="model_mean" class="numeric">Mdl Mean</th>
          <th data-col="model_p_over" class="numeric">Mdl P%</th>
          <th data-col="market_p_over" class="numeric">Mkt P%</th>
          <th data-col="edge" class="numeric sorted" data-arrow="↓">Edge%</th>
          <th data-col="kelly_pct" class="numeric">Kelly%</th>
          <th data-col="best_odds_american" class="numeric">Best Odds</th>
        </tr>
      </thead>
      <tbody id="tableBody">
        <tr class="empty-row"><td colspan="12">Loading prop edges…</td></tr>
      </tbody>
    </table>
  </div>
</main>

<footer>
  <div>WNBA Pre-Game Prop Edges — WizardOfOdds Sports Analytics</div>
  <div>Edge = Model P(over) − Market P(no-vig) · Kelly criterion fractional sizing · BET ≥ 8% Kelly · SMALL BET ≥ 4% · LEAN ≥ 1%</div>
  <div>Confidence A ≥ 10pp edge · B ≥ 6pp · C ≥ 3pp · D &lt; 3pp · RLM = Reverse Line Movement detected</div>
  <div style="margin-top:4px;color:var(--text3)">For entertainment and research purposes only. Gamble responsibly. 21+</div>
</footer>

<script>
(function(){
'use strict';

let ALL = [];
let sortCol = 'edge', sortAsc = false;
let statFilt = '', dirFilt = '', actionFilt = '', searchFilt = '';

const params = new URLSearchParams(location.search);
const dataUrl = params.get('date') ? params.get('date') + '.json' : 'latest.json';

fetch(dataUrl)
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(data => {
    ALL = data.props || [];
    updateKPIs(data);
    render();
  })
  .catch(err => {
    document.getElementById('tableBody').innerHTML =
      `<tr class="empty-row"><td colspan="12">Failed to load: ${err.message}</td></tr>`;
  });

function updateKPIs(data) {
  document.getElementById('kTotal').textContent = data.total_props || 0;
  document.getElementById('kOver').textContent = data.over_signals || 0;
  document.getElementById('kUnder').textContent = data.under_signals || 0;
  document.getElementById('kBet').textContent = ALL.filter(p => p.action === 'BET').length;
  const best = ALL.length ? Math.max(...ALL.map(p => Math.abs(p.edge) * 100)) : null;
  document.getElementById('kTopEdge').textContent = best != null ? '+' + best.toFixed(1) + '%' : '—';
  document.getElementById('dateLabel').textContent = data.game_date || '—';
  document.getElementById('genTime').textContent = data.generated_at
    ? 'Updated ' + new Date(data.generated_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
    : '';
}

const _STAT_DISPLAY = {
  'FG3M':'3PM','PTS_REB':'Pts+Reb','PTS_AST':'Pts+Ast',
  'REB_AST':'Reb+Ast','PTS_REB_AST':'Pts+Reb+Ast','STOCKS':'Stl+Blk','TURNOVER':'TO'
};

function filtered() {
  return ALL.filter(p => {
    if (statFilt && p.stat !== statFilt) return false;
    if (dirFilt && p.direction !== dirFilt) return false;
    if (actionFilt && p.action !== actionFilt) return false;
    if (searchFilt && !p.player.toLowerCase().includes(searchFilt)) return false;
    return true;
  });
}

function sortedRows(rows) {
  return [...rows].sort((a, b) => {
    let av = a[sortCol] ?? 0, bv = b[sortCol] ?? 0;
    if (sortCol === 'edge') { av = Math.abs(av); bv = Math.abs(bv); }
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  });
}

function pct(v) { return (+(v) * 100).toFixed(1) + '%'; }

function render() {
  const rows = sortedRows(filtered());
  const tbody = document.getElementById('tableBody');
  document.getElementById('countLbl').textContent = rows.length + ' props shown';

  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="12">No props match the current filters.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(p => {
    const isOver = p.direction === 'OVER';
    const ep = p.edge * 100;
    const edgeFmt = (ep >= 0 ? '+' : '') + ep.toFixed(1) + '%';
    const edgeCls = isOver ? 'edge-pos' : 'edge-neg';
    const rowCls = p.confidence === 'A' ? 'row-a' : p.confidence === 'B' ? 'row-b' : '';

    const acCls = p.action === 'BET' ? 'ab-bet' : p.action === 'SMALL BET' ? 'ab-small' : p.action === 'LEAN' ? 'ab-lean' : 'ab-pass';
    const tierCls = p.confidence === 'A' ? 'tier-a' : p.confidence === 'B' ? 'tier-b' : p.confidence === 'C' ? 'tier-c' : 'tier-d';

    const rlm = p.reverse_line_movement ? '<span class="rlm-badge">RLM</span>' : '';
    const st = _STAT_DISPLAY[p.stat] || p.stat;

    // Best odds cell
    let oddsCell;
    if (p.best_bookmaker && p.best_odds_american != null) {
      const sign = p.best_odds_american >= 0 ? '+' : '';
      const oddsLabel = `${p.best_bookmaker} ${sign}${p.best_odds_american}`;
      const href = p.best_deep_link || '#';
      const oddsCls = p.best_odds_american >= 0 ? 'odds-pos' : 'odds-neg';
      oddsCell = `<a href="${href}" target="_blank" rel="noopener" class="odds-btn">
        <span class="${oddsCls}">${oddsLabel}</span></a>`;
    } else {
      oddsCell = '<span class="no-odds">—</span>';
    }

    const kelly = p.kelly_pct > 0 ? p.kelly_pct.toFixed(1) + '%' : '—';
    const kellyCls = p.kelly_pct > 0 ? 'style="color:var(--blue)"' : 'style="color:var(--text3)"';

    return `<tr class="${rowCls}">
      <td><div class="player-cell"><span class="player-name">${p.player}</span>${rlm}</div></td>
      <td><span class="stat-chip">${st}</span></td>
      <td><span class="${isOver ? 'dir-over' : 'dir-under'}">${p.direction}</span></td>
      <td><span class="ab ${acCls}">${p.action || '—'}</span></td>
      <td><span class="tier ${tierCls}">${p.confidence || '—'}</span></td>
      <td class="numeric">${p.market_line}</td>
      <td class="numeric">${p.model_mean ? p.model_mean.toFixed(2) : '—'}</td>
      <td class="numeric prob-val">${pct(p.model_p_over)}</td>
      <td class="numeric prob-val">${pct(p.market_p_over)}</td>
      <td class="numeric ${edgeCls}">${edgeFmt}</td>
      <td class="numeric" ${kellyCls}>${kelly}</td>
      <td class="numeric">${oddsCell}</td>
    </tr>`;
  }).join('');
}

// Filters
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
document.getElementById('actionPills').addEventListener('click', e => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  actionFilt = btn.dataset.action;
  document.querySelectorAll('#actionPills .pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
});
document.getElementById('searchInput').addEventListener('input', e => {
  searchFilt = e.target.value.trim().toLowerCase();
  render();
});

// Column sort
document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = col === 'player' || col === 'stat'; }
    document.querySelectorAll('th[data-col]').forEach(t => {
      t.classList.remove('sorted');
      delete t.dataset.arrow;
    });
    th.classList.add('sorted');
    th.dataset.arrow = sortAsc ? '↑' : '↓';
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
.edge-under{background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44}
.card-body{display:flex;gap:0;padding:12px 14px 14px}
.chart-area{flex:0 0 auto;position:relative}
.stats-panel{flex:1 1 0;min-width:0;padding-left:14px;border-left:1px solid #1e2130}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:.78rem}
.stat-label{color:#888}
.stat-value{font-weight:500;color:#e4e4e4;font-variant-numeric:tabular-nums}
.stat-value.green{color:#3fa266}
.stat-value.red{color:#e05a6a}
.stat-value.amber-edge{color:#f59e0b}
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
      <button class="pill" data-stat="FG3M">3PM</button>
      <button class="pill" data-stat="PTS_REB_AST">PRA</button>
    </div>
    <input class="search" type="text" placeholder="Filter player..." id="searchInput">
  </div>

  <div class="legend">
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#e05a6a" stroke-width="2" stroke-dasharray="5 3"/></svg>Mkt Line</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#599ce7" stroke-width="2" stroke-dasharray="4 2"/></svg>Mean</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#f1b467" stroke-width="2" stroke-dasharray="4 2"/></svg>Median</div>
    <div class="legend-item"><svg width="22" height="3"><line x1="0" y1="1.5" x2="22" y2="1.5" stroke="#81a1c1" stroke-width="2" stroke-dasharray="4 2"/></svg>Mode</div>
    <div class="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="#a3e635" fill-opacity="0.85"/></svg>Last 5 Avg</div>
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
  // A3: Cache-bust fetch — pointer → immutable release payload
  function _todayETpmf() {
    const d = new Date();
    const et = new Date(d.toLocaleString('en-US', {timeZone: 'America/New_York'}));
    return et.toISOString().slice(0, 10);
  }
  const targetDate = dateParam || _todayETpmf();

  function _loadPMFPayload(url) {
    return fetch(url, {cache:'no-store'}).then(r => { if(!r.ok) throw new Error(r.status); return r.json(); });
  }

  // Fetch latest.json (pointer) with cache-bust, then follow to immutable payload
  _loadPMFPayload('latest.json?t=' + Date.now())
    .then(ptr => {
      if (!ptr.pointer) {
        // Legacy full payload — validate date
        if (ptr.game_date && ptr.game_date !== targetDate) {
          document.getElementById('pmfGrid').innerHTML = '<div class="loading" style="grid-column:1/-1;color:#e05a6a">Stale data (' + ptr.game_date + ') — waiting for current update.</div>';
          return;
        }
        allProps = ptr.props || [];
        document.getElementById('hdrDate').textContent = ptr.game_date || '';
        document.getElementById('genTime').textContent = ptr.generated_at ? new Date(ptr.generated_at).toLocaleString() : '—';
        render(); return;
      }
      if (ptr.game_date !== targetDate) {
        document.getElementById('pmfGrid').innerHTML = '<div class="loading" style="grid-column:1/-1;color:#e05a6a">Stale data (' + ptr.game_date + ') — waiting for current update.</div>';
        return;
      }
      const payloadUrl = (ptr.payload_path || ptr.release_payload_path) + '?r=' + encodeURIComponent(ptr.release_id || '');
      _loadPMFPayload(payloadUrl)
        .then(data => {
          if (data.game_date !== ptr.game_date) {
            document.getElementById('pmfGrid').innerHTML = '<div class="loading" style="grid-column:1/-1;color:#e05a6a">Payload date mismatch — refreshing.</div>';
            return;
          }
          allProps = data.props || [];
          document.getElementById('hdrDate').textContent = data.game_date || '';
          document.getElementById('genTime').textContent = data.generated_at ? new Date(data.generated_at).toLocaleString() : '—';
          render();
        })
        .catch(err => { document.getElementById('pmfGrid').innerHTML = '<div class="loading" style="grid-column:1/-1;color:#e05a6a">Failed to load: ' + err.message + '</div>'; });
    })
    .catch(err => {
      document.getElementById('pmfGrid').innerHTML = `<div class="loading" style="grid-column:1/-1;color:#e05a6a">Failed to load: ${err.message}</div>`;
    });

  const _PMF_STAT_DISPLAY = {
    'FG3M':'3PM','PTS_REB':'Pts+Reb','PTS_AST':'Pts+Ast',
    'REB_AST':'Reb+Ast','PTS_REB_AST':'Pts+Reb+Ast','STOCKS':'Stl+Blk','TURNOVER':'TO'
  };

  function filtered() {
    return allProps.filter(p => {
      if (statFilter !== 'ALL' && p.stat !== statFilter) return false;
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

    // Market line (bold red dashed)
    const line = prop.line > 0 ? `<line x1="${vx(prop.line).toFixed(1)}" y1="${PT}" x2="${vx(prop.line).toFixed(1)}" y2="${PT+ph}" stroke="#e05a6a" stroke-width="2" stroke-dasharray="5 3"/>` : '';
    const meanX = (vx(prop.mean)+bw/2).toFixed(1);
    const meanL = `<line x1="${meanX}" y1="${PT}" x2="${meanX}" y2="${PT+ph}" stroke="#599ce7" stroke-width="1.5" stroke-dasharray="4 2"/>`;
    const medX = (vx(prop.median)+bw/2).toFixed(1);
    const medL = `<line x1="${medX}" y1="${PT}" x2="${medX}" y2="${PT+ph}" stroke="#f1b467" stroke-width="1.5" stroke-dasharray="4 2"/>`;
    const modeX = (vx(prop.mode)+bw/2).toFixed(1);
    const modeL = `<line x1="${modeX}" y1="${PT}" x2="${modeX}" y2="${PT+ph}" stroke="#81a1c1" stroke-width="1.5" stroke-dasharray="4 2"/>`;

    // Last 5 avg dot (green diamond)
    let l5dot = '';
    if (prop.last_5_avg != null && prop.last_5_avg >= kMin && prop.last_5_avg <= kMax + 1) {
      const lx = (vx(prop.last_5_avg)+bw/2).toFixed(1);
      const ly = (PT + 10).toFixed(1);
      l5dot = `<circle cx="${lx}" cy="${ly}" r="4" fill="#a3e635" fill-opacity="0.85"/><text x="${lx}" y="${(PT+7).toFixed(1)}" text-anchor="middle" font-size="7" fill="#a3e635" font-family="system-ui">L5</text>`;
    }

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
    if (prop.line > 0) addLabel(prop.line, prop.line);
    addLabel(kMax, kMax);

    return `<svg width="${W}" height="${H}" style="display:block">${bars}${line}${meanL}${medL}${modeL}${l5dot}${axis}${labels.join('')}</svg>`;
  }

  function pct(v) { return (v*100).toFixed(1)+'%'; }
  function sign(v) { return v>=0 ? '+' : ''; }

  function buildCard(p) {
    const isOver = p.edge >= 0;
    const edgePct = (Math.abs(p.edge)*100).toFixed(1);
    const st = _PMF_STAT_DISPLAY[p.stat] || p.stat;
    const skewFlagged = Math.abs(p.skewness) > 1;
    const kurtFlagged = Math.abs(p.excess_kurtosis) > 3;
    const svg = buildSVG(p);
    const medVsLine = p.median_vs_line != null ? `${p.median_vs_line >= 0 ? '+' : ''}${p.median_vs_line.toFixed(2)}` : '—';
    const l5str = p.last_5_avg != null ? p.last_5_avg.toFixed(1) : '—';

    return `<div class="card">
      <div class="card-header">
        <div><span class="card-title">${p.player}</span><span class="stat-tag">${st} ${p.line > 0 ? p.line : '—'}</span></div>
        <span class="edge-badge ${isOver?'edge-over':'edge-under'}">${isOver?'OVER':'UNDER'} ${sign(p.edge)}${edgePct}%</span>
      </div>
      <div class="card-body">
        <div class="chart-area">${svg}</div>
        <div class="stats-panel">
          <div class="stat-row"><span class="stat-label">EV (mean)</span><span class="stat-value">${p.mean.toFixed(2)}</span></div>
          <div class="stat-row"><span class="stat-label">Median</span><span class="stat-value">${p.median.toFixed(1)}</span></div>
          <div class="stat-row"><span class="stat-label">Median vs Line</span><span class="stat-value ${p.median_vs_line > 0 ? 'green' : p.median_vs_line < 0 ? 'red' : ''}">${medVsLine}</span></div>
          <div class="stat-row"><span class="stat-label">Last 5 avg</span><span class="stat-value blue">${l5str}</span></div>
          <div class="divider"></div>
          <div class="stat-row"><span class="stat-label">P(over) mdl</span><span class="stat-value ${isOver?'green':'amber-edge'}">${p.model_p_over_pct != null ? p.model_p_over_pct.toFixed(1)+'%' : pct(p.model_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">P(under) mdl</span><span class="stat-value ${!isOver?'green':'amber-edge'}">${p.model_p_under_pct != null ? p.model_p_under_pct.toFixed(1)+'%' : pct(1-p.model_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">Mkt no-vig P(O)</span><span class="stat-value">${p.no_vig_over_prob != null ? (p.no_vig_over_prob*100).toFixed(1)+'%' : pct(p.market_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">Mkt no-vig P(U)</span><span class="stat-value">${p.no_vig_under_prob != null ? (p.no_vig_under_prob*100).toFixed(1)+'%' : pct(1-p.market_p_over)}</span></div>
          <div class="stat-row"><span class="stat-label">Edge</span><span class="stat-value ${isOver?'green':'amber-edge'}">${sign(p.edge)}${(p.edge*100).toFixed(1)}%</span></div>
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0A0E1A;--surface:#0F1529;--surface2:#141B30;--border:#1E2640;--border2:#2A3350;
  --gold:#D4AF37;--text:#E8EAF2;--text2:#9BA3BC;--text3:#5A6380;
  --green:#2ECC71;--red:#E74C3C;--blue:#3B82F6;--amber:#F59E0B;
  --bet-bg:#16341A;--bet-border:#2ECC71;
  --small-bg:#2D2610;--small-border:#F59E0B;
  --lean-bg:#0F1E38;--lean-border:#3B82F6;
}
body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5}
/* Header */
header{background:var(--surface);border-bottom:1px solid var(--border);padding:11px 22px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100}
.brand{font-family:'Fraunces',serif;font-size:1.1rem;font-weight:700;color:var(--text)}
.brand .acc{color:var(--gold)}
.nav-sep{width:1px;height:16px;background:var(--border2)}
.nav-links{display:flex;gap:4px}
.nav-link{font-size:.7rem;color:var(--text3);padding:3px 10px;border-radius:4px;text-decoration:none;transition:color .15s}
.nav-link:hover,.nav-link.active{color:var(--gold)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:14px;font-size:.7rem;color:var(--text3)}
.live-badge{display:flex;align-items:center;gap:5px;color:var(--green);font-weight:700;font-size:.78rem}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 1.5s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.stale-banner{background:rgba(231,76,60,.12);border:1px solid rgba(231,76,60,.35);color:#f08080;font-size:.75rem;padding:6px 16px;display:flex;align-items:center;gap:8px;border-radius:6px;margin-bottom:14px}
/* Main layout */
main{max-width:1400px;margin:18px auto;padding:0 18px}
/* KPI strip */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:13px 16px}
.kpi .val{font-family:'Fraunces',serif;font-size:1.55rem;font-weight:700;line-height:1.1}
.kpi .lbl{font-size:.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}
.kpi.green .val{color:var(--green)}
.kpi.red .val{color:var(--red)}
.kpi.gold .val{color:var(--gold)}
.kpi.amber .val{color:var(--amber)}
/* Filters */
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
.pill{background:var(--surface2);border:1px solid var(--border2);color:var(--text3);border-radius:20px;padding:3px 12px;font-size:.7rem;cursor:pointer;transition:all .15s;font-family:inherit}
.pill:hover{border-color:var(--gold);color:var(--text)}
.pill.active{background:rgba(212,175,55,.15);border-color:var(--gold);color:var(--gold)}
.search-box{background:var(--surface2);border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:4px 11px;font-size:.76rem;font-family:inherit;outline:none;width:190px}
.search-box:focus{border-color:var(--gold)}
.flt-spacer{flex:1}
.count-label{font-size:.7rem;color:var(--text3)}
/* Per-game panels */
.game-panel{border:1px solid var(--border);border-radius:10px;margin-bottom:18px;overflow:hidden}
.game-panel-hdr{background:var(--surface);padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}
.game-matchup{font-family:'Fraunces',serif;font-size:1.05rem;font-weight:600;color:var(--text)}
.game-score{font-size:1.1rem;font-weight:700;color:var(--gold);padding:2px 10px;background:rgba(212,175,55,.1);border:1px solid rgba(212,175,55,.25);border-radius:6px}
.game-clock{font-size:.75rem;color:var(--text3);background:var(--surface2);border:1px solid var(--border2);border-radius:4px;padding:2px 8px}
.game-status{font-size:.7rem;color:var(--green);font-weight:600;margin-left:auto}
/* Table inside panel */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.78rem}
thead{position:sticky;top:46px;z-index:5}
th{background:var(--surface);color:var(--text3);padding:8px 12px;text-align:left;font-weight:600;font-size:.63rem;text-transform:uppercase;letter-spacing:.7px;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
th:hover{color:var(--text)}
th.sorted-desc::after{content:' ↓';color:var(--gold)}
th.sorted-asc::after{content:' ↑';color:var(--gold)}
td{padding:7px 12px;border-bottom:1px solid rgba(30,38,64,.6);white-space:nowrap;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(212,175,55,.02)}
/* Row color by action */
tr.row-bet td{background:rgba(46,204,113,.04)}
tr.row-bet:hover td{background:rgba(46,204,113,.07)}
tr.row-small td{background:rgba(245,158,11,.03)}
.player-name{font-weight:600;color:var(--text)}
.oncourt-yes{color:var(--green);font-size:.8rem}
.oncourt-no{color:var(--text3);font-size:.8rem}
.stat-chip{background:var(--surface2);border:1px solid var(--border2);border-radius:4px;padding:1px 7px;font-size:.68rem;font-weight:600;color:var(--text3)}
.prog-wrap{display:flex;align-items:center;gap:6px;min-width:110px}
.prog-bar{flex:1;height:5px;background:var(--border2);border-radius:3px;overflow:hidden}
.prog-fill{height:100%;border-radius:3px;background:var(--gold);transition:width .3s}
.prog-fill.over-line{background:var(--green)}
.prog-text{font-size:.7rem;color:var(--text3);white-space:nowrap}
.dir-over{color:var(--green);font-weight:700}
.dir-under{color:var(--red);font-weight:700}
.edge-pos{color:var(--green);font-weight:700}
.edge-neg{color:var(--red);font-weight:700}
.kelly-val{color:var(--blue)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.action-badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:.68rem;font-weight:700;letter-spacing:.3px}
.ab-bet{background:var(--bet-bg);border:1px solid var(--bet-border);color:var(--green)}
.ab-small{background:var(--small-bg);border:1px solid var(--small-border);color:var(--amber)}
.ab-lean{background:var(--lean-bg);border:1px solid var(--lean-border);color:var(--blue)}
.ab-pass{background:var(--surface2);border:1px solid var(--border2);color:var(--text3)}
.ab-wait{background:rgba(90,99,128,.12);border:1px solid rgba(90,99,128,.35);color:var(--text3)}
/* Empty / no-games states */
.no-games{text-align:center;padding:70px 20px;color:var(--text3)}
.no-games h2{font-family:'Fraunces',serif;font-size:1.5rem;color:var(--text2);margin-bottom:8px}
.no-games p{font-size:.82rem;line-height:1.7}
.next-games{margin-top:20px;display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.next-game-pill{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:.78rem}
.next-game-pill .ng-teams{font-weight:600;color:var(--text)}
.next-game-pill .ng-time{font-size:.68rem;color:var(--text3);margin-top:2px}
/* Footer */
footer{text-align:center;font-size:.68rem;color:var(--text3);padding:24px 0 18px;border-top:1px solid var(--border);margin-top:22px;line-height:1.8}
/* Responsive */
@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px){.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<header>
  <div class="brand">WizardOf<span class="acc">Odds</span></div>
  <div class="nav-sep"></div>
  <nav class="nav-links">
    <a href="../Pre-Game/Distributions/" class="nav-link">Pre-Game</a>
    <a href="" class="nav-link active">Live In-Play</a>
  </nav>
  <div class="hdr-right">
    <div class="live-badge"><div class="dot"></div>LIVE</div>
    <span>Gamma-Poisson Bayesian · Pace-adjusted</span>
    <span>Refreshes every 30 s · <span id="cdSec" style="color:var(--gold)">30</span>s</span>
  </div>
</header>

<main>
  <div id="staleBanner" class="stale-banner" style="display:none">
    ⚠ Data may be stale — last update <span id="staleAge">?</span> ago. Next update is scheduled via pipeline.
  </div>

  <!-- KPI strip -->
  <div class="kpis">
    <div class="kpi"><div class="val" id="k-games">—</div><div class="lbl">Active Games</div></div>
    <div class="kpi green"><div class="val" id="k-edges">—</div><div class="lbl">Live Edges</div></div>
    <div class="kpi gold"><div class="val" id="k-best">—</div><div class="lbl">Best Edge</div></div>
    <div class="kpi amber"><div class="val" id="k-bet">—</div><div class="lbl">BET Signals</div></div>
    <div class="kpi"><div class="val" id="k-updated" style="font-size:.82rem">—</div><div class="lbl">Data Age</div></div>
  </div>

  <!-- Filters -->
  <div class="filters">
    <div id="statPills">
      <button class="pill active" data-stat="">All</button>
      <button class="pill" data-stat="PTS">PTS</button>
      <button class="pill" data-stat="REB">REB</button>
      <button class="pill" data-stat="AST">AST</button>
      <button class="pill" data-stat="FG3M">3PM</button>
      <button class="pill" data-stat="STL">STL</button>
      <button class="pill" data-stat="BLK">BLK</button>
    </div>
    <div id="actionPills">
      <button class="pill active" data-action="">All</button>
      <button class="pill" data-action="BET">BET</button>
      <button class="pill" data-action="SMALL BET">SMALL</button>
      <button class="pill" data-action="LEAN">LEAN</button>
    </div>
    <input class="search-box" type="text" id="searchInput" placeholder="Search player…">
    <div class="flt-spacer"></div>
    <span class="count-label" id="totalEdgeCount">—</span>
  </div>

  <!-- Game panels container -->
  <div id="gamePanels"></div>
</main>

<footer>
  <div>WNBA Live In-Play Model — WizardOfOdds Sports Analytics</div>
  <div>Gamma-Poisson Bayesian online updating · Pace-adjusted posterior projections · Updates ~every 7 min during games</div>
  <div style="margin-top:4px;color:var(--text3)">For entertainment and research purposes only. Gamble responsibly. 21+</div>
</footer>

<script>
(function(){
'use strict';

let DATA = {};
let ALL_ROWS = [];
let statFilt = '', actionFilt = '', searchFilt = '';
let cdTimer;
const STALE_THRESHOLD_MS = 8 * 60 * 1000; // 8 minutes
const POLL_INTERVAL_MS = 30 * 1000;

// ── Fetch (cache-bust pointer, then fetch immutable payload) ────────
// Phase A3: latest.json is a pointer. Page JS must:
//   1. fetch latest.json with cache:'no-store' + timestamp
//   2. validate pointer fields (release_id, game_date, payload_path)
//   3. reject pointer whose game_date != target WNBA game date
//   4. fetch immutable payload (releases/<release_id>.json) with cache:'no-store'
//   5. reject payload if game_date or release_id mismatch
//   6. never display stale cards as LIVE DATA
function _todayET() {
  const d = new Date();
  const et = new Date(d.toLocaleString('en-US', {timeZone: 'America/New_York'}));
  return et.toISOString().slice(0, 10);
}
function load() {
  const targetDate = new URLSearchParams(window.location.search).get('date') || _todayET();
  fetch('latest.json?t=' + Date.now(), {cache: 'no-store'})
    .then(r => r.json())
    .then(ptr => {
      // Pointer validation
      if (!ptr.pointer) {
        // Legacy full-payload format — accept but warn
        DATA = ptr;
        if (ptr.game_date && ptr.game_date !== targetDate) {
          _showStaleAlert(ptr.game_date);
          return;
        }
        buildRows(ptr); updateKPIs(ptr); renderPanels();
        updateStaleCheck(ptr.generated_at || ptr.generated_at_utc);
        startCountdown(POLL_INTERVAL_MS / 1000);
        return;
      }
      if (ptr.game_date !== targetDate) {
        _showStaleAlert(ptr.game_date); return;
      }
      const payloadUrl = (ptr.payload_path || ptr.release_payload_path) + '?r=' + encodeURIComponent(ptr.release_id || '');
      fetch(payloadUrl, {cache: 'no-store'})
        .then(r => r.json())
        .then(data => {
          if (data.game_date !== ptr.game_date) { _showStaleAlert(data.game_date); return; }
          DATA = data;
          buildRows(data); updateKPIs(data); renderPanels();
          updateStaleCheck(data.generated_at || data.generated_at_utc || ptr.generated_at_utc);
          startCountdown(POLL_INTERVAL_MS / 1000);
        })
        .catch(() => _showStaleAlert(null));
    })
    .catch(() => { /* keep old data, countdown continues */ });
}
function _showStaleAlert(foundDate) {
  const el = document.getElementById('staleBanner');
  if (el) {
    el.style.display = '';
    el.innerHTML = foundDate
      ? '⚠ Page data is for ' + foundDate + ' — not the current game date. Refresh when updated.'
      : '⚠ Could not load current page data. Check pipeline status.';
  }
}

// ── Parse games → flat rows ───────────────────────────────────────
function buildRows(data) {
  ALL_ROWS = [];
  const games = data.games || [];
  games.forEach(game => {
    (game.players || []).forEach(pl => {
      const pp = pl.posterior_predictive || pl.live_props || {};
      const liveStats = pl.live_stats || {};
      Object.entries(pp).forEach(([statKey, pd]) => {
        if (!pd || !pd.market_line) return;
        const edge = (pd.p_over || pd.live_p_over || pd.model_p_over || 0) -
                     (pd.market_p_over || pd.open_p_over || 0);
        const kelly = (pd.live_kelly || pd.kelly_fraction || 0) * 100;
        const line = pd.market_line;
        const liveStat = liveStats[statKey.replace('posterior_','').replace('_','').toLowerCase()] ?? null;
        ALL_ROWS.push({
          game_id: game.game_id,
          game_label: `${game.away_team||'?'} @ ${game.home_team||'?'}`,
          score: game.home_score != null ? `${game.away_score}-${game.home_score}` : null,
          period: game.current_period || game.period || null,
          clock: game.current_clock || null,
          player: pl.player_name,
          on_court: pl.currently_on_court,
          stat: statKey.toUpperCase().replace('POSTERIOR_PREDICTIVE.',''),
          direction: edge >= 0 ? 'OVER' : 'UNDER',
          live_stat: liveStat,
          line,
          prior_mean: pd.prior_mean || 0,
          posterior_mean: pd.posterior_mean || pd.mean || 0,
          model_p: pd.p_over || pd.live_p_over || pd.model_p_over || 0,
          market_p: pd.market_p_over || pd.open_p_over || 0,
          edge,
          kelly,
          action: edgeAction(edge, kelly),
          game_status: game.game_status || 'in_play',
        });
      });
    });
  });
  ALL_ROWS.sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge));
}

function edgeAction(edge, kelly) {
  const ep = Math.abs(edge) * 100;
  if (kelly >= 8) return 'BET';
  if (kelly >= 4) return 'SMALL BET';
  if (kelly >= 1) return 'LEAN';
  if (ep < 2) return 'PASS';
  return 'PASS';
}

// ── KPIs ──────────────────────────────────────────────────────────
function updateKPIs(data) {
  const activeGames = (data.games || []).filter(g => g.game_status === 'in_play' || (data.active_games || 0) > 0).length || data.active_games || (data.games||[]).length;
  document.getElementById('k-games').textContent = activeGames;
  const edges = ALL_ROWS.filter(r => Math.abs(r.edge)*100 >= 2);
  document.getElementById('k-edges').textContent = edges.length;
  const bets = ALL_ROWS.filter(r => r.action === 'BET');
  document.getElementById('k-bet').textContent = bets.length;
  const bestEdge = ALL_ROWS.length ? Math.max(...ALL_ROWS.map(r => Math.abs(r.edge)*100)) : null;
  document.getElementById('k-best').textContent = bestEdge != null ? '+' + bestEdge.toFixed(1) + '%' : '—';
}

function updateStaleCheck(genAt) {
  const banner = document.getElementById('staleBanner');
  if (!genAt) return;
  const ageMs = Date.now() - new Date(genAt).getTime();
  const ageMins = Math.floor(ageMs / 60000);
  document.getElementById('k-updated').textContent = ageMins < 2 ? 'Just now' : ageMins + ' min ago';
  if (ageMs > STALE_THRESHOLD_MS) {
    banner.style.display = 'flex';
    document.getElementById('staleAge').textContent = ageMins + ' min';
  } else {
    banner.style.display = 'none';
  }
}

// ── Render per-game panels ────────────────────────────────────────
function filteredRows() {
  return ALL_ROWS.filter(r => {
    if (statFilt && r.stat !== statFilt) return false;
    if (actionFilt && r.action !== actionFilt) return false;
    if (searchFilt && !r.player.toLowerCase().includes(searchFilt)) return false;
    return true;
  });
}

function renderPanels() {
  const rows = filteredRows();
  const container = document.getElementById('gamePanels');
  document.getElementById('totalEdgeCount').textContent = rows.length + ' props';

  if (!rows.length && !(DATA.games||[]).length) {
    container.innerHTML = buildNoGamesHTML();
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text3)">No props match current filters.</div>';
    return;
  }

  // Group by game
  const byGame = {};
  rows.forEach(r => {
    if (!byGame[r.game_id]) byGame[r.game_id] = { label: r.game_label, score: r.score, period: r.period, clock: r.clock, status: r.game_status, rows: [] };
    byGame[r.game_id].rows.push(r);
  });

  container.innerHTML = Object.values(byGame).map(g => buildGamePanel(g)).join('');
}

function buildGamePanel(g) {
  const scoreHtml = g.score ? `<span class="game-score">${g.score}</span>` : '';
  const clockHtml = (g.period || g.clock) ? `<span class="game-clock">${g.period ? 'Q' + g.period : ''} ${g.clock || ''}</span>` : '';
  const statusHtml = `<span class="game-status">${g.status === 'in_play' ? '● LIVE' : 'FINAL'}</span>`;

  const rows = g.rows.map(r => buildPropRow(r)).join('');
  return `
  <div class="game-panel">
    <div class="game-panel-hdr">
      <span class="game-matchup">${g.label}</span>
      ${scoreHtml}${clockHtml}${statusHtml}
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>On</th>
            <th>Stat</th>
            <th>So Far</th>
            <th>Progress</th>
            <th>Line</th>
            <th>Post. Mean</th>
            <th>Dir</th>
            <th class="num">Live P%</th>
            <th class="num">Open P%</th>
            <th class="num">Edge</th>
            <th class="num">Kelly%</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

function buildPropRow(r) {
  const ep = r.edge * 100;
  const isOver = r.direction === 'OVER';
  const edgeCls = ep >= 0 ? 'edge-pos' : 'edge-neg';
  const dirCls = isOver ? 'dir-over' : 'dir-under';
  const rowCls = r.action === 'BET' ? 'row-bet' : r.action === 'SMALL BET' ? 'row-small' : '';
  const acCls = r.action === 'BET' ? 'ab-bet' : r.action === 'SMALL BET' ? 'ab-small' : r.action === 'LEAN' ? 'ab-lean' : r.action === 'WAIT' ? 'ab-wait' : 'ab-pass';
  const onCls = r.on_court ? 'oncourt-yes' : 'oncourt-no';
  const onTxt = r.on_court ? '●' : '○';

  // Progress bar
  let progHtml = '—';
  if (r.live_stat != null && r.line > 0) {
    const pct = Math.min(100, Math.round((r.live_stat / r.line) * 100));
    const fillCls = r.live_stat >= r.line ? 'prog-fill over-line' : 'prog-fill';
    progHtml = `<div class="prog-wrap">
      <div class="prog-bar"><div class="${fillCls}" style="width:${pct}%"></div></div>
      <span class="prog-text">${r.live_stat}/${r.line}</span>
    </div>`;
  }

  const soFar = r.live_stat != null ? r.live_stat : '—';
  const postMean = r.posterior_mean ? (+r.posterior_mean).toFixed(1) : '—';
  const liveP = (r.model_p * 100).toFixed(1) + '%';
  const openP = (r.market_p * 100).toFixed(1) + '%';
  const edgeFmt = (ep >= 0 ? '+' : '') + ep.toFixed(1) + '%';
  const kellyFmt = r.kelly > 0 ? r.kelly.toFixed(1) + '%' : '—';

  return `<tr class="${rowCls}">
    <td class="player-name">${r.player}</td>
    <td><span class="${onCls}">${onTxt}</span></td>
    <td><span class="stat-chip">${r.stat}</span></td>
    <td class="num">${soFar}</td>
    <td>${progHtml}</td>
    <td class="num">${r.line}</td>
    <td class="num">${postMean}</td>
    <td><span class="${dirCls}">${r.direction}</span></td>
    <td class="num">${liveP}</td>
    <td class="num" style="color:var(--text3)">${openP}</td>
    <td class="num ${edgeCls}">${edgeFmt}</td>
    <td class="num ${r.kelly>0?'kelly-val':''}">${kellyFmt}</td>
    <td><span class="action-badge ${acCls}">${r.action}</span></td>
  </tr>`;
}

function buildNoGamesHTML() {
  return `<div class="no-games">
    <h2>No Live Games Right Now</h2>
    <p>This platform activates automatically when WNBA games tip off.<br>
       Live Bayesian posterior odds and pace-adjusted projections appear here in real time.<br>
       Typical game windows: 7 PM – 11 PM ET (Thu–Sun).</p>
    <div class="next-games" id="nextGames"></div>
  </div>`;
}

// ── Filters ────────────────────────────────────────────────────────
document.getElementById('statPills').addEventListener('click', e => {
  const btn = e.target.closest('[data-stat]');
  if (!btn) return;
  statFilt = btn.dataset.stat;
  document.querySelectorAll('#statPills .pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderPanels();
});
document.getElementById('actionPills').addEventListener('click', e => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  actionFilt = btn.dataset.action;
  document.querySelectorAll('#actionPills .pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderPanels();
});
document.getElementById('searchInput').addEventListener('input', e => {
  searchFilt = e.target.value.trim().toLowerCase();
  renderPanels();
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
    json_only: bool = typer.Option(False, "--json-only", help="Write only JSON data files, skip index.html regeneration."),
    release_id: str = typer.Option(
        "",
        "--release-id",
        help=(
            "Current-run release identifier (GITHUB_RUN_ID or equivalent). "
            "Written to both page JSON outputs so consumers can verify both pages "
            "come from the same run. Required for release lineage validation."
        ),
    ),
    git_commit: str = typer.Option(
        "",
        "--git-commit",
        help="Current git commit SHA. Written to both page JSON outputs for traceability.",
    ),
    slate_manifest: str = typer.Option(
        "",
        "--slate-manifest",
        help=(
            "Path to slate manifest JSON containing scheduled_game_count. "
            "When provided and scheduled_game_count > 0, missing or empty projections "
            "cause a fatal nonzero exit (fail-closed). "
            "When scheduled_game_count == 0, exits 0 with VERIFIED_NO_GAMES status."
        ),
    ),
    model_version: str = typer.Option(
        "",
        "--model-version",
        help="Model version string written to both page JSON outputs for traceability.",
    ),
    calibration_version: str = typer.Option(
        "",
        "--calibration-version",
        help="Calibration version string written to both page JSON outputs for traceability.",
    ),
    market_status: str = typer.Option(
        "",
        "--market-status",
        help=(
            "Market status from build_edge_report audit. "
            "Allowed: SUCCESS_WITH_MARKETS, LIVE_MARKETS_NOT_YET_AVAILABLE, FAILURE. "
            "Consistency-validated against edge row count. Inferred from row count when absent."
        ),
    ),
    market_audit_json: str = typer.Option(
        "",
        "--market-audit-json",
        help=(
            "Path to build_edge_report output JSON. When provided, market_status, "
            "raw_quote_count, and market_request_timestamp_utc are read from this file."
        ),
    ),
) -> None:
    """Generate all three web page directories (Edge, PMF-Distributions, Inplay/Edges)."""
    out = Path(out_dir)
    edge_dir = out / "Edge"
    pmf_dir = out / "PMF-Distributions"
    live_out = Path(live_dir)

    edge_dir.mkdir(parents=True, exist_ok=True)
    pmf_dir.mkdir(parents=True, exist_ok=True)
    live_out.mkdir(parents=True, exist_ok=True)

    # --- Load and validate slate manifest (fail-closed) ---
    scheduled_game_count: int | None = None
    if slate_manifest:
        sm_path = Path(slate_manifest)
        if sm_path.exists():
            try:
                sm_data = json.loads(sm_path.read_text())
                scheduled_game_count = int(sm_data.get("scheduled_game_count", 0))
            except Exception as exc:
                typer.echo(f"[FATAL] Cannot read slate manifest {slate_manifest}: {exc}", err=True)
                raise typer.Exit(1)
        else:
            typer.echo(f"[FATAL] Slate manifest not found: {slate_manifest}", err=True)
            raise typer.Exit(1)

        if scheduled_game_count == 0:
            typer.echo(f"[generate_web_pages] VERIFIED_NO_GAMES: slate has 0 scheduled games for {game_date}")
            raise typer.Exit(0)

    # --- Load source data ---
    proj_path = projections or f"deliveries/tonight/player_projections_{game_date}.parquet"
    edges_path = edges or "deliveries/tonight/publishable_edges.parquet"

    typer.echo(f"[generate_web_pages] game_date={game_date}")
    typer.echo(f"  projections : {proj_path}")
    typer.echo(f"  edges       : {edges_path}")

    # Fail closed when scheduled games exist but projections are missing/empty
    proj_df_loaded = True
    try:
        proj_df = pd.read_parquet(proj_path)
        typer.echo(f"  Loaded projections: {len(proj_df)} rows")
        if proj_df.empty and scheduled_game_count is not None and scheduled_game_count > 0:
            typer.echo(
                f"[FATAL] Projections file at {proj_path} is empty but slate has "
                f"{scheduled_game_count} scheduled game(s). Status: FAILURE", err=True
            )
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        if scheduled_game_count is not None and scheduled_game_count > 0:
            typer.echo(
                f"[FATAL] Could not load projections from {proj_path}: {exc}. "
                f"Slate has {scheduled_game_count} scheduled game(s) — cannot continue. "
                f"Status: FAILURE", err=True
            )
            raise typer.Exit(1)
        typer.echo(f"  [WARN] Could not load projections: {exc} — using empty DataFrame")
        proj_df = pd.DataFrame(columns=["player_id", "stat", "pmf_json", "pmf_mean",
                                         "pmf_variance", "median", "mode"])
        proj_df_loaded = False

    edges_df_loaded = True
    try:
        edges_df = pd.read_parquet(edges_path)
        typer.echo(f"  Loaded edges: {len(edges_df)} rows")
        if edges_df.empty:
            if scheduled_game_count is not None and scheduled_game_count > 0:
                typer.echo(
                    f"[generate_web_pages] LIVE_MARKETS_NOT_YET_AVAILABLE: "
                    f"slate has {scheduled_game_count} game(s) but market file is empty"
                )
    except Exception as exc:
        if scheduled_game_count is not None and scheduled_game_count > 0:
            typer.echo(
                f"[FATAL] Could not load market props from {edges_path}: {exc}. "
                f"Status: FAILURE", err=True
            )
            raise typer.Exit(1)
        typer.echo(f"  [WARN] Could not load edges: {exc} — using empty DataFrame")
        edges_df = pd.DataFrame(columns=["player_name", "player_id", "stat", "line", "edge_over",
                                          "kelly_fraction", "model_prob_over", "market_prob_over_no_vig"])
        edges_df_loaded = False

    # --- Apply canonical player identity resolution before building JSON ---
    # Resolves duplicate BDL player IDs (e.g. two records for the same physical player)
    # using config/player_identity_aliases.json. Must run on both proj_df and edges_df.
    try:
        from wnba_props_model.data.identity import (  # noqa: PLC0415
            apply_canonical_ids, deduplicate_pmfs, validate_no_duplicate_identities
        )
        if "player_id" in proj_df.columns:
            proj_df = apply_canonical_ids(proj_df, "player_id")
            proj_df = deduplicate_pmfs(proj_df, key_cols=["game_id", "player_id", "stat"])
        if "player_id" in edges_df.columns:
            edges_df = apply_canonical_ids(edges_df, "player_id")
    except Exception as _id_exc:
        typer.echo(f"  [WARN] Identity resolution failed (non-fatal): {_id_exc}", err=True)

    # --- Read market audit JSON for evidence-based market status ---
    _raw_quote_count: int | None = None
    _reconciled_quote_count: int | None = None
    _market_req_ts: str = ""
    if market_audit_json and Path(market_audit_json).exists():
        try:
            _audit = json.loads(Path(market_audit_json).read_text())
            if not market_status:
                market_status = _audit.get("market_status", "")
            _raw_quote_count = int(_audit.get("total_market_rows", 0))
            _reconciled_quote_count = int(_audit.get("total_market_rows", 0))
            _market_req_ts = _audit.get("generated_at", "")
            typer.echo(f"  Market audit: status={market_status} raw={_raw_quote_count}")
        except Exception as exc:
            typer.echo(f"  [WARN] Could not read market audit JSON {market_audit_json}: {exc}", err=True)

    # --- Build JSON ---
    edge_json = _build_edge_json(edges_df, proj_df, game_date,
                                 release_id=release_id, git_commit=git_commit,
                                 model_version=model_version,
                                 calibration_version=calibration_version,
                                 market_status=market_status,
                                 raw_quote_count=_raw_quote_count,
                                 reconciled_quote_count=_reconciled_quote_count,
                                 market_request_timestamp_utc=_market_req_ts)
    pmf_json  = _build_pmf_json(edges_df, proj_df, game_date,
                                 release_id=release_id, git_commit=git_commit,
                                 model_version=model_version,
                                 calibration_version=calibration_version)

    # --- Write immutable release payloads and cache-safe latest.json pointer ---
    # A3: Immutable release files at releases/<release_id>.json
    # A3: Date-specific files at <game_date>.json
    # A3: latest.json is a pointer only (contains release_id, game_date, path, versions)
    #     Page JS must fetch the date-specific or immutable payload with cache: 'no-store'.

    import hashlib as _hashlib  # noqa: PLC0415
    _edge_payload_str = json.dumps(_sanitize(edge_json), separators=(",", ":"))
    _pmf_payload_str  = json.dumps(_sanitize(pmf_json),  separators=(",", ":"))
    _edge_sha256 = _hashlib.sha256(_edge_payload_str.encode()).hexdigest()
    _pmf_sha256  = _hashlib.sha256(_pmf_payload_str.encode()).hexdigest()

    # Date-specific payloads (overwritten each run for that date)
    (edge_dir / f"{game_date}.json").write_text(_edge_payload_str)
    (pmf_dir  / f"{game_date}.json").write_text(_pmf_payload_str)

    # Immutable release payloads (keyed by release_id — never overwrite an existing one)
    _eff_release_id = release_id or game_date
    (edge_dir / "releases").mkdir(parents=True, exist_ok=True)
    (pmf_dir  / "releases").mkdir(parents=True, exist_ok=True)
    _edge_release_path = edge_dir / "releases" / f"{_eff_release_id}.json"
    _pmf_release_path  = pmf_dir  / "releases" / f"{_eff_release_id}.json"
    _edge_release_path.write_text(_edge_payload_str)
    _pmf_release_path.write_text(_pmf_payload_str)

    # latest.json = pointer only — contains:
    #   release_id, game_date, payload_path, payload_sha256, git_commit,
    #   model_version, calibration_version, generated_at_utc, row_count
    # Page JS must:
    #   1. fetch latest.json with cache: "no-store" + ?t=<timestamp>
    #   2. validate pointer fields
    #   3. fetch releases/<release_id>.json with cache: "no-store" + ?r=<release_id>
    #   4. verify payload_sha256 against fetched content
    #   5. reject if game_date != target date or release_id mismatch
    _now_utc = datetime.now(timezone.utc).isoformat()
    _edge_pointer = _sanitize({
        "pointer": True,
        "release_id": _eff_release_id,
        "game_date": game_date,
        "payload_path": f"releases/{_eff_release_id}.json",
        "payload_sha256": _edge_sha256,
        "release_payload_path": f"releases/{_eff_release_id}.json",
        "date_payload_path": f"{game_date}.json",
        "git_commit": git_commit,
        "model_version": model_version,
        "calibration_version": calibration_version,
        "generated_at_utc": _now_utc,
        "row_count": edge_json["total_props"],
        "total_props": edge_json["total_props"],  # backward compat alias
    })
    _pmf_pointer = _sanitize({
        "pointer": True,
        "release_id": _eff_release_id,
        "game_date": game_date,
        "payload_path": f"releases/{_eff_release_id}.json",
        "payload_sha256": _pmf_sha256,
        "release_payload_path": f"releases/{_eff_release_id}.json",
        "date_payload_path": f"{game_date}.json",
        "git_commit": git_commit,
        "model_version": model_version,
        "calibration_version": calibration_version,
        "generated_at_utc": _now_utc,
        "row_count": pmf_json["total_props"],
        "total_props": pmf_json["total_props"],  # backward compat alias
    })
    (edge_dir / "latest.json").write_text(json.dumps(_edge_pointer, separators=(",", ":")))
    (pmf_dir  / "latest.json").write_text(json.dumps(_pmf_pointer,  separators=(",", ":")))
    typer.echo(f"  Edge: releases/{_eff_release_id}.json ({edge_json['total_props']} props) sha256={_edge_sha256[:12]}")
    typer.echo(f"  PMF:  releases/{_eff_release_id}.json ({pmf_json['total_props']} props) sha256={_pmf_sha256[:12]}")
    typer.echo(f"  Both latest.json → pointer release_id={_eff_release_id!r}")

    # --- Write HTML templates (skip when --json-only) ---
    if not json_only:
        (edge_dir / "index.html").write_text(_EDGE_HTML)
        typer.echo(f"  Edge HTML → {edge_dir}/index.html")

        (pmf_dir / "index.html").write_text(_PMF_HTML)
        typer.echo(f"  PMF HTML → {pmf_dir}/index.html")

    # --- Live page: JSON is written by live_inplay.yml; HTML skipped when --json-only ---
    if not skip_live_html and not json_only:
        (live_out / "index.html").write_text(_LIVE_HTML)
        typer.echo(f"  Live HTML → {live_out}/index.html")

    typer.echo("[generate_web_pages] Done.")


if __name__ == "__main__":
    app()
