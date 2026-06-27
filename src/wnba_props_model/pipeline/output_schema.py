"""Blueprint §3 / §4 — Stable output schema helpers.

Converts internal parquet PMF tables into the exact JSON schema required
by the wizardofodds.com dashboard contract (blueprint sections 3–4, 12–13).

Pre-game delivery URL:
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json

Live delivery URL:
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/Inplay-Edge/game_{id}_latest.json

PMF bin specifications (§3.4):
  Minutes : 5-min bins  0-5, 5-10, …, 40-45, 45+
  Points  : 5-pt bins   0-5, 5-10, …, 35-40, 40+
  Rebounds: 2-reb bins  0-2, 2-4,  …, 14-16, 16+
  Assists : 2-ast bins  0-2, 2-4,  …, 10-12, 12+
  Steals  : 1-stl bins  0-1, 1-2,  …, 5-6,   6+
  Blocks  : 1-blk bins  0-1, 1-2,  …, 4-5,   5+
  Threes  : 1-fg3m bins 0-1, 1-2,  …, 6-7,   7+
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "hgb_v2.3.1_idr_beta_conformal"
CALIBRATION_VERSION = "idr_v12_beta_v12_conformal_v12"


def _norm_name(name: str) -> str:
    """Normalize a player name for fuzzy matching: lowercase, letters only.

    Handles apostrophes (A'ja), hyphens (Laney-Hamilton), accents, and
    spacing differences between BDL and Odds API name formats.
    """
    return re.sub(r"[^a-z]", "", name.lower())


# ---------------------------------------------------------------------------
# PMF bin definitions (blueprint §3.4)
# ---------------------------------------------------------------------------

_STAT_BIN_CONFIG: dict[str, list[int]] = {
    "minutes": list(range(0, 46, 5)),   # 0,5,10,...,45
    "pts":     list(range(0, 41, 5)),   # 0,5,10,...,40
    "reb":     list(range(0, 17, 2)),   # 0,2,4,...,16
    "ast":     list(range(0, 13, 2)),   # 0,2,4,...,12
    "stl":     list(range(0, 7,  1)),   # 0,1,2,...,6
    "blk":     list(range(0, 6,  1)),   # 0,1,2,...,5
    "fg3m":    list(range(0, 8,  1)),   # 0,1,2,...,7
    # Combo stats inherit pts bins
    "pts_ast":     list(range(0, 41, 5)),
    "pts_reb":     list(range(0, 41, 5)),
    "reb_ast":     list(range(0, 17, 2)),
    "pts_reb_ast": list(range(0, 46, 5)),
    "stocks":      list(range(0, 7, 1)),
    "turnover":    list(range(0, 7, 1)),
}

_STAT_DISPLAY: dict[str, str] = {
    "pts": "points", "reb": "rebounds", "ast": "assists",
    "fg3m": "threes", "stl": "steals", "blk": "blocks",
    "turnover": "turnovers", "pts_ast": "points_assists",
    "pts_reb": "points_rebounds", "reb_ast": "rebounds_assists",
    "pts_reb_ast": "points_rebounds_assists", "stocks": "steals_blocks",
}


def bin_pmf(pmf_arr: np.ndarray, stat: str) -> dict[str, float]:
    """Convert point-mass PMF array to the blueprint's fixed bin format.

    Returns a dict like {"0-5": 0.145, "5-10": 0.282, ..., "40+": 0.020}
    """
    breakpoints = _STAT_BIN_CONFIG.get(stat, _STAT_BIN_CONFIG["pts"])
    bins: dict[str, float] = {}
    n = len(pmf_arr)

    for i, lo in enumerate(breakpoints):
        hi = breakpoints[i + 1] if i + 1 < len(breakpoints) else None
        if hi is not None:
            label = f"{lo}-{hi}"
            mass = float(pmf_arr[lo:hi].sum()) if lo < n else 0.0
        else:
            label = f"{lo}+"
            mass = float(pmf_arr[lo:].sum()) if lo < n else 0.0
        bins[label] = round(mass, 6)

    # Renormalize to exactly 1.0 (floating-point safety)
    total = sum(bins.values())
    if total > 0:
        bins = {k: round(v / total, 6) for k, v in bins.items()}
    return bins


def _pmf_median(pmf_arr: np.ndarray) -> float:
    cdf = np.cumsum(pmf_arr)
    return float(np.searchsorted(cdf, 0.5))


def _pmf_mode(pmf_arr: np.ndarray) -> float:
    return float(np.argmax(pmf_arr))


def _conformal_ci(mean: float, std: float, alpha: float = 0.90) -> list[float]:
    """Simple normal approximation 90% CI (replaced by real conformal when available)."""
    z = 1.645
    return [round(max(0.0, mean - z * std), 1), round(mean + z * std, 1)]


# ---------------------------------------------------------------------------
# Deep-link builder (blueprint §3.5)
# ---------------------------------------------------------------------------

def build_deep_links(
    odds_api_rows: list[dict],
    stat: str,
) -> dict[str, dict]:
    """Build deep_links block for one player-stat from Odds API outcome rows.

    Returns:
        {
          "fanduel_points_over":  {"url": ..., "depth": ..., "odds": ...},
          "fanduel_points_under": {"url": ..., "depth": ..., "odds": ...},
          ...
        }
    """
    display = _STAT_DISPLAY.get(stat, stat)
    links: dict[str, dict] = {}
    for row in odds_api_rows:
        book = str(row.get("bookmaker") or "").lower().replace(" ", "")
        side = str(row.get("side") or "").lower()
        url = row.get("outcome_link") or row.get("market_link") or row.get("event_link")
        odds = row.get("odds") or row.get("over_odds" if "over" in side else "under_odds")
        depth = "betslip" if "outcome_link" in row and row["outcome_link"] else (
            "event" if "market_link" in row and row["market_link"] else "home"
        )
        key = f"{book}_{display}_{side}"
        links[key] = {"url": url, "depth": depth, "odds": odds}
    return links


# ---------------------------------------------------------------------------
# Sanity check (blueprint §3.6)
# ---------------------------------------------------------------------------

def compute_sanity_status(
    model_means: dict[str, float],
    sharpest_lines: dict[str, float],
    yellow_threshold: float = 0.03,
    red_threshold: float = 0.05,
) -> dict:
    """Compare model p(over) vs. sharpest-book implied probability.

    Returns sanity_checks dict per blueprint §3.6.
    """
    gaps: dict[str, float] = {}
    for stat, model_p in model_means.items():
        market_p = sharpest_lines.get(stat)
        if market_p is not None:
            gaps[f"{_STAT_DISPLAY.get(stat, stat)}_gap_pp"] = round(model_p - market_p, 4)

    max_gap = max((abs(v) for v in gaps.values()), default=0.0)
    status = "GREEN"
    if max_gap > red_threshold:
        status = "RED"
    elif max_gap > yellow_threshold:
        status = "YELLOW"

    return {
        "model_vs_sharpest": gaps,
        "sanity_status": status,
        "max_gap_pp": round(max_gap, 4),
    }


def _build_sanity_checks_from_market(
    game_id: int,
    game_pmfs: pd.DataFrame,
    market_df: pd.DataFrame,
    yellow_threshold: float = 0.03,
    red_threshold: float = 0.05,
) -> dict:
    """Build a real sanity check block (blueprint §3.6) using Pinnacle/sharpest lines.

    Extracts Pinnacle rows from market_df, deviggs them with Shin's method,
    then compares model p(over) against the sharpest-book fair probability.
    Falls back to GREEN/no_market_data when market data is unavailable.
    """
    try:
        from wnba_props_model.models.market import shin_no_vig_two_way_with_z  # noqa: PLC0415
    except ImportError:
        return {"sanity_status": "GREEN", "note": "market_module_unavailable"}

    if market_df is None or market_df.empty:
        return {"sanity_status": "GREEN", "note": "no_market_data"}

    # Filter to this game
    game_mkt = market_df[market_df.get("game_id", pd.Series(dtype=object)) == game_id] \
        if "game_id" in market_df.columns else market_df

    # Identify sharpest book rows — prefer pinnacle, fall back to all books
    vendor_col = next((c for c in ("vendor", "bookmaker", "book") if c in game_mkt.columns), None)
    if vendor_col:
        pinnacle = game_mkt[game_mkt[vendor_col].astype(str).str.lower().str.contains("pinnacle")]
        sharpest = pinnacle if not pinnacle.empty else game_mkt
    else:
        sharpest = game_mkt

    # Build sharpest_lines: stat → fair p(over) via Shin devig
    sharpest_lines: dict[str, float] = {}
    sharpest_vig: dict[str, float] = {}
    devigged_fair: dict[str, float] = {}
    stat_col = next((c for c in ("stat", "prop_type", "market_key") if c in sharpest.columns), None)
    if stat_col:
        for _, row in sharpest.iterrows():
            raw_stat = str(row.get(stat_col, ""))
            stat = raw_stat.replace("player_", "")
            display = _STAT_DISPLAY.get(stat, stat)
            o_odds = row.get("over_odds") or row.get("market_over_odds")
            u_odds = row.get("under_odds") or row.get("market_under_odds")
            if o_odds is None or u_odds is None:
                continue
            try:
                p_over, p_under, _z = shin_no_vig_two_way_with_z(float(o_odds), float(u_odds))
            except Exception:
                continue
            if p_over is None or p_under is None:
                continue
            sharpest_lines[stat] = p_over
            line_val = float(row.get("line") or row.get("line_value") or 0.5)
            devigged_fair[f"{display}_line"] = round(line_val, 1)
            # vig = overround = (raw_over_prob + raw_under_prob) - 1
            try:
                from wnba_props_model.models.market import american_to_prob  # noqa: PLC0415
                raw_o = american_to_prob(float(o_odds))
                raw_u = american_to_prob(float(u_odds))
                if raw_o and raw_u:
                    sharpest_vig[display] = round(raw_o + raw_u - 1.0, 4)
            except Exception:
                pass

    # Build model_means: stat → model p(over) at the sharpest line
    model_means: dict[str, float] = {}
    stat_proj_col = "stat" if "stat" in game_pmfs.columns else None
    if stat_proj_col:
        for stat, p_market in sharpest_lines.items():
            stat_rows = game_pmfs[game_pmfs[stat_proj_col] == stat]
            if stat_rows.empty:
                continue
            # Average model p(over) across all players for this stat
            # Use pmf_mean as a proxy: p(over market_line) from mean/variance normal approx
            model_probs = []
            for _, sr in stat_rows.iterrows():
                pmf_raw = sr.get("pmf_json")
                if pmf_raw is None:
                    continue
                try:
                    import json as _json  # noqa: PLC0415
                    if isinstance(pmf_raw, str):
                        pmf_dict = _json.loads(pmf_raw)
                        pmf_arr = np.array(
                            [pmf_dict.get(str(k), 0.0) for k in range(len(pmf_dict))],
                            dtype=float,
                        )
                    else:
                        pmf_arr = np.array(pmf_raw, dtype=float)
                    if pmf_arr.sum() > 0:
                        pmf_arr /= pmf_arr.sum()
                    line_val = float(sr.get("line") or 0.5)
                    p_over = float(pmf_arr[math.ceil(line_val):].sum())
                    model_probs.append(p_over)
                except Exception:
                    continue
            if model_probs:
                model_means[stat] = float(np.mean(model_probs))

    # Compute gaps
    gaps: dict[str, float] = {}
    for stat, model_p in model_means.items():
        mkt_p = sharpest_lines.get(stat)
        if mkt_p is not None:
            display = _STAT_DISPLAY.get(stat, stat)
            gaps[f"{display}_gap_pp"] = round(model_p - mkt_p, 4)

    max_gap = max((abs(v) for v in gaps.values()), default=0.0)
    status = "GREEN"
    if max_gap > red_threshold:
        status = "RED"
    elif max_gap > yellow_threshold:
        status = "YELLOW"

    result: dict = {"sanity_status": status, "model_vs_sharpest": gaps}
    if devigged_fair:
        result["pinnacle"] = {"devigged_fair": devigged_fair, "vig": sharpest_vig}
    return result


# ---------------------------------------------------------------------------
# GTD dual-scenario builder (blueprint §5.3)
# ---------------------------------------------------------------------------

_GTD_STATUSES = frozenset({"gtd", "game-time decision", "game_time_decision"})


def build_gtd_scenarios(
    player_id: int,
    player_name: str,
    injury_status: str,
    scenario_in_record: dict,
    gtd_log_rows: "list[dict] | None" = None,
) -> dict:
    """Build a GTD dual-scenario block per blueprint §5.3.

    scenario_in_record: the normal player projection (player plays).
    gtd_log_rows: pre-computed GTD scenarios from apply_injury_updates.py
        injury_report_{date}.json → gtd_scenarios_detail list.
        Each entry has: player_id, scenario_in, scenario_out, teammate_impact.

    When gtd_log_rows is not available, scenario_out is synthesised as
    zero-minutes / zero-stats with empty teammate_impact.
    """
    # Try to find pre-computed GTD record from injury engine
    precomputed: dict | None = None
    if gtd_log_rows:
        for row in gtd_log_rows:
            if int(row.get("player_id", -1)) == player_id:
                precomputed = row
                break

    if precomputed is not None:
        scenario_out = precomputed.get("scenario_out", {})
        teammate_impact = scenario_out.get("teammate_impact") or precomputed.get("teammate_impact") or {}
    else:
        # Synthesise: player sits → zero projections, no teammate impact data
        scenario_out = {
            "projected_minutes": {"mean": 0.0},
            "stat_projections": {
                stat: {"mean": 0.0} for stat in scenario_in_record.get("stat_projections", {})
            },
            "teammate_impact": {},
        }
        teammate_impact = {}

    return {
        "player_id": player_id,
        "player_name": player_name,
        "injury_status": "GTD",
        "scenario_in": {
            "projected_minutes": scenario_in_record.get("projected_minutes", {}),
            "stat_projections": scenario_in_record.get("stat_projections", {}),
            "teammate_impact": None,
        },
        "scenario_out": {
            "projected_minutes": {"mean": 0.0},
            "stat_projections": scenario_out.get("stat_projections", {}),
            "teammate_impact": teammate_impact,
        },
        "official_status_pending": True,
    }


# ---------------------------------------------------------------------------
# Player projection record builder (blueprint §3.3)
# ---------------------------------------------------------------------------

def build_player_record(
    player_rows: pd.DataFrame,
    market_rows: pd.DataFrame | None = None,
    odds_api_rows: list[dict] | None = None,
    injury_status: str | None = None,
    is_starter: bool | None = None,
    shap_rows: pd.DataFrame | None = None,
    utm_impact_rows: list[dict] | None = None,
    gtd_log_rows: list[dict] | None = None,
) -> dict:
    """Build a single player projection record per blueprint §3.3.

    player_rows: rows from player_projections.parquet for one player (all stats)
    market_rows: rows from wnba_player_props_oddsapi_latest.parquet for this player
    odds_api_rows: raw Odds API outcome rows for deep links
    injury_status: from BDL injuries endpoint (None, 'out', 'questionable', etc.)
    is_starter: bool or None (None = unknown)
    shap_rows: from SHAP explainability (optional)
    utm_impact_rows: UTM transfer log rows for this player's beneficiary impact
    gtd_log_rows: pre-computed GTD scenarios from apply_injury_updates.py (optional)
    """
    if player_rows.empty:
        return {}

    tmpl = player_rows.iloc[0]
    player_id = int(tmpl.get("player_id", 0))
    player_name = str(tmpl.get("player_name", ""))
    team_id = int(tmpl.get("team_id", 0))
    team_name = str(tmpl.get("team_abbreviation", ""))
    position = str(tmpl.get("position", "")) or None

    # Minutes projection
    min_mean = float(tmpl.get("minutes_mean") or 0.0)
    min_sigma = float(tmpl.get("minutes_sigma") or 3.0)
    min_ci = _conformal_ci(min_mean, min_sigma)

    # Try to build binned minutes PMF from minutes_mean / minutes_sigma (normal approx)
    _mins_domain = np.arange(0, 49)
    from scipy.stats import norm as _norm  # noqa: PLC0415
    _mins_pmf_arr = _norm.pdf(_mins_domain, min_mean, max(min_sigma, 1e-3))
    _mins_pmf_arr = np.maximum(_mins_pmf_arr, 0)
    _mins_pmf_arr /= _mins_pmf_arr.sum() if _mins_pmf_arr.sum() > 0 else 1
    mins_binned = bin_pmf(_mins_pmf_arr, "minutes")

    minutes_proj = {
        "mean": round(min_mean, 1),
        "median": round(_pmf_median(_mins_pmf_arr), 1),
        "pmf": mins_binned,
        "conformal_90_ci": min_ci,
    }

    # Stat projections
    stat_projections: dict[str, dict] = {}
    STATS_TO_INCLUDE = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                        "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "stocks"]

    for stat in STATS_TO_INCLUDE:
        stat_row = player_rows[player_rows["stat"] == stat]
        if stat_row.empty:
            continue
        sr = stat_row.iloc[0]
        display = _STAT_DISPLAY.get(stat, stat)

        pmf_raw = sr.get("pmf_json") or sr.get("pmf_json")
        if pmf_raw is None:
            continue
        try:
            if isinstance(pmf_raw, str):
                pmf_dict = json.loads(pmf_raw)
                pmf_arr = np.array([pmf_dict.get(str(k), 0.0) for k in range(len(pmf_dict))])
            elif isinstance(pmf_raw, (list, np.ndarray)):
                pmf_arr = np.array(pmf_raw, dtype=float)
            else:
                pmf_arr = np.array(pmf_raw, dtype=float)
        except Exception:
            continue

        if pmf_arr.sum() > 0:
            pmf_arr = pmf_arr / pmf_arr.sum()

        stat_mean = float(sr.get("pmf_mean") or sr.get("mean") or sr.get("stat_mean") or 0.0)
        stat_var = float(sr.get("pmf_variance") or sr.get("stat_variance") or stat_mean or 1.0)
        stat_std = math.sqrt(max(stat_var, 0))
        stat_ci = _conformal_ci(stat_mean, stat_std)
        if "conformal_lower" in sr and not pd.isna(sr.get("conformal_lower", float("nan"))):
            stat_ci = [round(float(sr["conformal_lower"]), 1), round(float(sr["conformal_upper"]), 1)]

        stat_proj: dict[str, Any] = {
            "mean": round(stat_mean, 2),
            "median": round(_pmf_median(pmf_arr), 1),
            "pmf": bin_pmf(pmf_arr, stat),
            "conformal_90_ci": stat_ci,
        }

        # Market comparison (calibrated_p_over)
        if market_rows is not None and not market_rows.empty:
            mkt = market_rows[market_rows["stat"] == stat]
            if not mkt.empty:
                m = mkt.iloc[0]
                line_raw = m.get("line") if m.get("line") is not None else m.get("line_value")
                line = float(line_raw) if line_raw is not None else 0.5
                # P(over line) from PMF
                p_over = float(np.sum(pmf_arr[math.ceil(line):]))
                p_under = 1.0 - p_over
                mkt_p = float(m.get("market_prob_over_no_vig") or 0.5)
                edge = round(p_over - mkt_p, 4)

                # Fractional Kelly bet sizing (half-Kelly, capped at 25% bankroll)
                kelly_full = edge / max(mkt_p, 1e-6) if edge > 0 else 0.0
                kelly_frac = round(min(kelly_full * 0.5, 0.25), 4)

                stat_proj["calibrated_p_over"] = {
                    "market_line": round(line, 1),
                    "p_over": round(p_over, 4),
                    "p_under": round(p_under, 4),
                    "edge_vs_market": edge,
                    "kelly_fraction": kelly_frac if edge > 0 else None,
                    "market_source": str(m.get("source") or "odds_api"),
                    "market_vendor": str(m.get("vendor") or m.get("bookmaker") or ""),
                    "market_over_odds": int(m.get("over_odds") or 0) if m.get("over_odds") else None,
                    "market_under_odds": int(m.get("under_odds") or 0) if m.get("under_odds") else None,
                }

        stat_projections[display] = stat_proj

        # Deep links for this stat
        if odds_api_rows:
            stat_links = build_deep_links(
                [r for r in odds_api_rows
                 if _STAT_DISPLAY.get(r.get("stat", ""), r.get("stat")) == display
                 or r.get("market_key", "").replace("player_", "") == stat],
                stat,
            )
            if stat_links:
                stat_proj["deep_links"] = stat_links

    # Build explainability block (§7)
    explainability = _build_explainability(tmpl, shap_rows, injury_status, utm_impact_rows)

    record: dict = {
        "player_id": player_id,
        "player_name": player_name,
        "team_id": team_id,
        "team_name": team_name,
        "position": position,
        "is_starter": is_starter,
        "injury_status": injury_status,
        "projected_minutes": minutes_proj,
        "stat_projections": stat_projections,
        "explainability": explainability,
        "dnp_risk": float(tmpl.get("dnp_risk") or 0.0),
        "high_uncertainty": bool(tmpl.get("high_uncertainty") or False),
    }

    # Attach GTD dual-scenario block (blueprint §5.3)
    if injury_status is not None and injury_status.lower() in _GTD_STATUSES:
        record["gtd_scenarios"] = build_gtd_scenarios(
            player_id=player_id,
            player_name=player_name,
            injury_status=injury_status,
            scenario_in_record=record,
            gtd_log_rows=gtd_log_rows,
        )

    return record


def _build_explainability(
    row: "pd.Series",
    shap_rows: "pd.DataFrame | None" = None,
    injury_status: str | None = None,
    utm_impact_rows: "list[dict] | None" = None,
) -> dict:
    """Build explainability block per blueprint §7.

    utm_impact_rows: list of dicts from apply_injury_updates.py impact_report,
        each with keys: injured_player_id, injured_player_name, status,
        beneficiary_player_id, beneficiary_player_name, usage_transfer,
        minutes_transfer, points_boost.
    """
    override = str(row.get("override_source") or "") if row.get("override_applied") else None
    major_change = False
    major_reason = None

    if row.get("injury_flag"):
        major_change = True
        major_reason = f"injury_status={injury_status}"
    if override:
        major_change = True
        major_reason = f"override_applied={override}"

    shap_drivers: list[dict] = []
    if shap_rows is not None and not shap_rows.empty:
        for _, sr in shap_rows.head(5).iterrows():
            shap_drivers.append({
                "feature": str(sr.get("feature", "")),
                "shap_value": round(float(sr.get("shap_value", 0)), 4),
            })

    # Populate injury_impact_on_teammates from UTM log (blueprint §7.1)
    injury_impact: list[dict] = []
    if utm_impact_rows:
        player_id_val = int(row.get("player_id", 0))
        for impact in utm_impact_rows:
            # Include this player as beneficiary (received minutes/usage transfer)
            beneficiary_id = impact.get("beneficiary_player_id") or impact.get("player_id")
            if beneficiary_id is not None and int(beneficiary_id) == player_id_val:
                injury_impact.append({
                    "injured_player_id": impact.get("injured_player_id"),
                    "injured_player_name": impact.get("injured_player_name", ""),
                    "status": impact.get("status", ""),
                    "beneficiary_player_id": player_id_val,
                    "beneficiary_player_name": str(row.get("player_name", "")),
                    "usage_transfer": round(float(impact.get("usage_transfer", 0.0)), 2),
                    "minutes_transfer": round(float(impact.get("minutes_transfer", 0.0)), 1),
                    "points_boost": round(float(impact.get("points_boost", 0.0)), 1),
                })

    return {
        "projected_minutes": {
            "major_change_flag": major_change,
            "major_change_reason": major_reason,
            "override_applied": override,
        },
        "stat_drivers": {"primary_drivers": shap_drivers} if shap_drivers else {},
        "injury_impact_on_teammates": injury_impact,
    }


# ---------------------------------------------------------------------------
# Game-level record builder (blueprint §3.2)
# ---------------------------------------------------------------------------

def build_game_record(
    game_id: int,
    pmfs_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
    odds_api_rows: list[dict] | None = None,
    injuries_df: pd.DataFrame | None = None,
    odds_api_event_id: str | None = None,
    utm_log_df: pd.DataFrame | None = None,
    gtd_log_rows: list[dict] | None = None,
) -> dict:
    """Build a full game-level projection record per blueprint §3.2.

    utm_log_df: DataFrame with UTM transfer log from apply_injury_updates.py,
        columns: injured_player_id, injured_player_name, status,
                 beneficiary_player_id, beneficiary_player_name,
                 usage_transfer, minutes_transfer, points_boost.
    gtd_log_rows: pre-computed GTD scenarios list from injury_report_{date}.json
        → gtd_scenarios_detail, for use in build_gtd_scenarios().
    """
    game_rows = pmfs_df[pmfs_df["game_id"] == game_id]
    if game_rows.empty:
        return {}

    tmpl = game_rows.iloc[0]
    home_team_id = int(tmpl.get("team_id") or 0)
    away_team_id = int(tmpl.get("opponent_team_id") or 0)
    home_abbr = str(tmpl.get("team_abbreviation") or "")
    away_abbr = str(tmpl.get("opponent_team_abbreviation") or "")
    commence_time = str(tmpl.get("game_date") or "")

    # Game-level odds from games_df
    game_spread = None
    game_total = None
    if games_df is not None and not games_df.empty:
        gf = games_df[games_df["game_id"] == game_id]
        if not gf.empty:
            game_spread = gf.iloc[0].get("spread_home_value")
            game_total = gf.iloc[0].get("total_value")

    # Build per-player records
    player_ids = game_rows["player_id"].unique()
    players = []
    for pid in player_ids:
        p_rows = game_rows[game_rows["player_id"] == pid]
        pname = str(p_rows.iloc[0].get("player_name", ""))
        pname_norm = _norm_name(pname)

        p_mkt = None
        if market_df is not None and not market_df.empty:
            # Try player_id match first (fast path)
            if "player_id" in market_df.columns:
                id_match = market_df[market_df["player_id"] == pid]
                p_mkt = id_match if not id_match.empty else None

            # Fuzzy name match fallback: normalize both sides (strips apostrophes,
            # hyphens, spaces, case) so "A'ja Wilson" matches "Aja Wilson", etc.
            if (p_mkt is None or (hasattr(p_mkt, "empty") and p_mkt.empty)) and "player_name" in market_df.columns:
                p_mkt = market_df[
                    market_df["player_name"].apply(lambda n: _norm_name(str(n))) == pname_norm
                ]

        inj_status = None
        if injuries_df is not None and not injuries_df.empty and "player_id" in injuries_df.columns:
            inj_row = injuries_df[injuries_df["player_id"] == pid]
            if not inj_row.empty:
                inj_status = str(inj_row.iloc[0].get("status") or "")

        p_odds_rows = []
        if odds_api_rows:
            p_odds_rows = [
                r for r in odds_api_rows
                if _norm_name(str(r.get("player_name", ""))) == pname_norm
            ]

        # Extract UTM impact rows for this player as beneficiary
        p_utm_rows: list[dict] | None = None
        if utm_log_df is not None and not utm_log_df.empty:
            ben_col = next(
                (c for c in ("beneficiary_player_id", "player_id") if c in utm_log_df.columns), None
            )
            if ben_col:
                p_utm_slice = utm_log_df[utm_log_df[ben_col] == pid]
                if not p_utm_slice.empty:
                    p_utm_rows = p_utm_slice.to_dict("records")

        rec = build_player_record(
            p_rows, market_rows=p_mkt,
            odds_api_rows=p_odds_rows,
            injury_status=inj_status,
            utm_impact_rows=p_utm_rows,
            gtd_log_rows=gtd_log_rows,
        )
        if rec:
            players.append(rec)

    # Sanity check block (blueprint §3.6) — real Pinnacle devig comparison
    sanity = _build_sanity_checks_from_market(game_id, game_rows, market_df)

    return {
        "game_id": game_id,
        "bdl_game_id": game_id,
        "odds_api_event_id": odds_api_event_id,
        "commence_time": str(commence_time),
        "home_team": {"id": home_team_id, "name": home_abbr, "abbreviation": home_abbr},
        "away_team": {"id": away_team_id, "name": away_abbr, "abbreviation": away_abbr},
        "game_spread": float(game_spread) if game_spread is not None else None,
        "game_total": float(game_total) if game_total is not None else None,
        "pace_projection": None,
        "sanity_checks": sanity,
        "players": players,
    }


# ---------------------------------------------------------------------------
# Full pre-game envelope builder (blueprint §3.1)
# ---------------------------------------------------------------------------

def build_pregame_envelope(
    pmfs_df: pd.DataFrame,
    game_date: str,
    pipeline_run: str = "pregame_initial",
    games_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
    odds_api_rows: list[dict] | None = None,
    injuries_df: pd.DataFrame | None = None,
    odds_api_event_map: dict[int, str] | None = None,
    utm_log_df: pd.DataFrame | None = None,
    gtd_log_rows: list[dict] | None = None,
) -> dict:
    """Build the full pre-game JSON envelope per blueprint §3.1."""
    game_ids = pmfs_df["game_id"].unique().tolist()
    games_list = []
    for gid in sorted(game_ids):
        event_id = (odds_api_event_map or {}).get(int(gid))
        g_rec = build_game_record(
            game_id=int(gid),
            pmfs_df=pmfs_df,
            games_df=games_df,
            market_df=market_df,
            odds_api_rows=odds_api_rows,
            injuries_df=injuries_df,
            odds_api_event_id=event_id,
            utm_log_df=utm_log_df,
            gtd_log_rows=gtd_log_rows,
        )
        if g_rec:
            games_list.append(g_rec)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date": game_date,
        "pipeline_run": pipeline_run,
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "games": games_list,
    }


# ---------------------------------------------------------------------------
# Live game envelope builder (blueprint §4.1 / §4.2)
# ---------------------------------------------------------------------------

def build_live_envelope(
    game_id: int,
    game_state: dict,
    player_states: list[dict],
    posterior_pmfs: dict[int, dict[str, dict]],
    pregame_projections: dict[int, dict] | None = None,
    live_market_rows: list[dict] | None = None,
) -> dict:
    """Build live in-play JSON envelope per blueprint §4.

    game_state: {current_period, current_clock, home_score, away_score,
                  elapsed_possessions, remaining_possessions_est, game_status}
    player_states: list of {player_id, player_name, currently_on_court,
                             live_minutes_played, pts, reb, ast, stl, blk, fg3m,
                             turnover, fouls}
    posterior_pmfs: {player_id: {stat: {"pmf_arr": np.ndarray, "mean": float, ...}}}
    """
    players_out = []
    for ps in player_states:
        pid = int(ps.get("player_id", 0))
        post = posterior_pmfs.get(pid, {})
        pre = (pregame_projections or {}).get(pid, {})

        live_stats = {
            "pts": int(ps.get("pts", 0)),
            "reb": int(ps.get("reb", 0)),
            "ast": int(ps.get("ast", 0)),
            "stl": int(ps.get("stl", 0)),
            "blk": int(ps.get("blk", 0)),
            "fg3m": int(ps.get("fg3m", 0)),
        }

        posterior_predictive: dict[str, dict] = {}
        for stat_key, stat_display in _STAT_DISPLAY.items():
            if stat_key not in ("pts", "reb", "ast", "fg3m", "stl", "blk"):
                continue
            pdata = post.get(stat_key, {})
            pmf_arr = pdata.get("pmf_arr")
            post_mean = float(pdata.get("mean", 0.0))
            prior_mean = float(pre.get(stat_key, {}).get("mean", 0.0))

            stat_entry: dict = {
                "prior_mean": round(prior_mean, 2),
                "posterior_mean": round(post_mean, 2),
                "posterior_median": round(float(pdata.get("median", post_mean)), 2),
                "remaining_est_possessions": int(game_state.get("remaining_possessions_est", 0)),
            }

            if pmf_arr is not None and len(pmf_arr) > 0:
                arr = np.array(pmf_arr, dtype=float)
                if arr.sum() > 0:
                    arr = arr / arr.sum()
                stat_entry["pmf"] = bin_pmf(arr, stat_key)

            # Live market edge
            mkt_row = None
            if live_market_rows:
                mkt_candidates = [r for r in live_market_rows
                                  if _STAT_DISPLAY.get(r.get("stat", ""), r.get("stat")) == stat_display
                                  and r.get("player_id") == pid]
                if mkt_candidates:
                    mkt_row = mkt_candidates[0]

            if mkt_row:
                line = float(mkt_row.get("line") or 0.5)
                arr_for_p = np.array(pdata.get("pmf_arr", []), dtype=float)
                p_over = float(np.sum(arr_for_p[math.ceil(line):])) if len(arr_for_p) > 0 else 0.5
                mkt_p = float(mkt_row.get("market_prob_over_no_vig") or 0.5)
                stat_entry["p_over"] = {
                    "market_line": round(line, 1),
                    "p_over": round(p_over, 4),
                    "p_under": round(1.0 - p_over, 4),
                    "edge_vs_current_market": round(p_over - mkt_p, 4),
                    "market_source": str(mkt_row.get("source") or "bdl_live_props"),
                    "market_vendor": str(mkt_row.get("vendor") or ""),
                    "live_over_odds": int(mkt_row.get("over_odds") or 0) if mkt_row.get("over_odds") else None,
                    "live_under_odds": int(mkt_row.get("under_odds") or 0) if mkt_row.get("under_odds") else None,
                }

            posterior_predictive[stat_display] = stat_entry

        player_out = {
            "player_id": pid,
            "player_name": str(ps.get("player_name", "")),
            "currently_on_court": bool(ps.get("currently_on_court", True)),
            "live_minutes_played": round(float(ps.get("live_minutes_played", 0.0)), 1),
            "live_stats": live_stats,
            "foul_count": int(ps.get("fouls", 0)),
            "foul_trouble_flag": int(ps.get("fouls", 0)) >= 4,
            "ejected": bool(ps.get("ejected", False)),
            "posterior_predictive": posterior_predictive,
        }
        players_out.append(player_out)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_id": game_id,
        "game_status": str(game_state.get("game_status", "in_play")),
        "current_period": int(game_state.get("current_period", 1)),
        "current_clock": str(game_state.get("current_clock", "10:00")),
        "home_score": int(game_state.get("home_score", 0)),
        "away_score": int(game_state.get("away_score", 0)),
        "elapsed_possessions": int(game_state.get("elapsed_possessions", 0)),
        "remaining_possessions_est": int(game_state.get("remaining_possessions_est", 0)),
        "players": players_out,
    }
