#!/usr/bin/env python3
"""Build the pregame verification audit package against a staged forecast.

Read-only w.r.t. model code. Writes audit CSVs/JSON/MD under --audit-dir.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
# Float accumulation over many PMF atoms routinely lands in the 1e-8..1e-7 band.
# Keep this far tighter than page-level mass checks (0.98..1.02) while avoiding
# false FAIL_DO_NOT_PUBLISH on numerically healthy distributions.
MASS_TOL = 1e-6
PROB_TOL = 1e-8
DECIMAL_TOL = 1e-6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes()) if p.exists() else ""


def _parse_pmf(raw: Any) -> list[tuple[float, float]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out: list[tuple[float, float]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out.append((float(k), float(v)))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((float(item[0]), float(item[1])))
            elif isinstance(item, dict):
                k = item.get("k", item.get("value", item.get("x")))
                p = item.get("p", item.get("prob", item.get("probability")))
                if k is not None and p is not None:
                    out.append((float(k), float(p)))
        return sorted(out, key=lambda x: x[0])
    return out


def _american_to_decimal(american: float) -> float:
    a = float(american)
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / abs(a)
    raise ValueError(f"invalid american odds {american}")


def _prob_to_american(p: float) -> float | None:
    if p <= 0 or p >= 1 or not math.isfinite(p):
        return None
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def _decimal_to_american(d: float) -> float | None:
    if d <= 1.0 or not math.isfinite(d):
        return None
    if d >= 2.0:
        return 100.0 * (d - 1.0)
    return -100.0 / (d - 1.0)


def _no_vig(over_american: float, under_american: float) -> tuple[float, float]:
    do = 1.0 / _american_to_decimal(over_american)
    du = 1.0 / _american_to_decimal(under_american)
    s = do + du
    return do / s, du / s


def _settle_from_pmf(atoms: list[tuple[float, float]], line: float) -> dict[str, float]:
    p_over = sum(p for k, p in atoms if k > line)
    p_under = sum(p for k, p in atoms if k < line)
    p_push = sum(p for k, p in atoms if abs(k - line) < 1e-12)
    denom = p_over + p_under
    if denom <= 0:
        return {
            "p_over_win": p_over,
            "p_under_win": p_under,
            "p_push": p_push,
            "p_over_settled": float("nan"),
            "p_under_settled": float("nan"),
            "fair_decimal_over": float("nan"),
            "fair_decimal_under": float("nan"),
        }
    po = p_over / denom
    pu = p_under / denom
    return {
        "p_over_win": p_over,
        "p_under_win": p_under,
        "p_push": p_push,
        "p_over_settled": po,
        "p_under_settled": pu,
        "fair_decimal_over": (1.0 / po) if po > 0 else float("inf"),
        "fair_decimal_under": (1.0 / pu) if pu > 0 else float("inf"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _team_abbr_from_odds_name(name: str) -> str:
    aliases = {
        "indiana fever": "IND",
        "portland fire": "POR",
        "las vegas aces": "LV",
        "chicago sky": "CHI",
        "new york liberty": "NY",
        "phoenix mercury": "PHX",
        "seattle storm": "SEA",
        "atlanta dream": "ATL",
        "dallas wings": "DAL",
        "washington mystics": "WSH",
        "minnesota lynx": "MIN",
        "connecticut sun": "CON",
        "los angeles sparks": "LA",
        "golden state valkyries": "GS",
    }
    return aliases.get((name or "").strip().lower(), name or "")


def audit_games(game_date: str, audit_dir: Path, blocking: list[str], warnings: list[str]) -> list[dict]:
    bdl_key = os.environ.get("BDL_API_KEY", "")
    odds_key = os.environ.get("ODDS_API_KEY", "")
    games_path = Path("data/processed/wnba_games.parquet")
    games_tbl = pd.read_parquet(games_path)
    games_tbl["tip_utc"] = pd.to_datetime(games_tbl["game_date"], utc=True, errors="coerce")
    games_tbl["et_date"] = games_tbl["tip_utc"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    prod = games_tbl[
        (games_tbl["et_date"] == game_date)
        & (games_tbl["status_normalized"].isin(["scheduled", "unknown", "in_progress", "final", "completed"]))
    ].copy()

    bdl_games = []
    bdl_fetched = _utc_now()
    if bdl_key:
        r = requests.get(
            "https://api.balldontlie.io/wnba/v1/games",
            headers={"Authorization": bdl_key},
            params={"dates[]": game_date, "per_page": 100},
            timeout=30,
        )
        r.raise_for_status()
        bdl_games = r.json().get("data", [])
        bdl_fetched = _utc_now()
    (audit_dir / "bdl_games_raw.json").write_text(
        json.dumps({"fetched_at_utc": bdl_fetched, "games": bdl_games}, indent=2)
    )
    bdl_by_id = {int(g["id"]): g for g in bdl_games}

    odds_events = []
    odds_fetched = None
    if odds_key:
        orr = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_wnba/events",
            params={"apiKey": odds_key, "dateFormat": "iso"},
            timeout=30,
        )
        orr.raise_for_status()
        odds_events = orr.json()
        odds_fetched = _utc_now()
    (audit_dir / "odds_events_raw.json").write_text(
        json.dumps({"fetched_at_utc": odds_fetched, "events": odds_events}, indent=2)
    )
    odds_rows = []
    for e in odds_events:
        tip = pd.to_datetime(e["commence_time"], utc=True)
        odds_rows.append(
            {
                "odds_event_id": e["id"],
                "away_abbr": _team_abbr_from_odds_name(e.get("away_team")),
                "home_abbr": _team_abbr_from_odds_name(e.get("home_team")),
                "commence_utc": tip.isoformat().replace("+00:00", "Z"),
                "et_date": tip.tz_convert(ET).strftime("%Y-%m-%d"),
            }
        )
    odds_df = pd.DataFrame(odds_rows) if odds_rows else pd.DataFrame(
        columns=["odds_event_id", "away_abbr", "home_abbr", "commence_utc", "et_date"]
    )

    rows = []
    used_odds: set[str] = set()
    for _, g in prod.iterrows():
        gid = int(g["game_id"])
        tip = g["tip_utc"]
        tip_s = tip.isoformat().replace("+00:00", "Z") if pd.notna(tip) else ""
        home = g["home_team_abbreviation"]
        away = g["visitor_team_abbreviation"]
        bdl_g = bdl_by_id.get(gid)
        bdl_match = "MATCH" if bdl_g else "NOT_IN_BDL_DATE_QUERY"
        bdl_tip_s = ""
        venue = ""
        bdl_status = ""
        if bdl_g:
            bdl_tip = pd.to_datetime(bdl_g.get("datetime") or bdl_g.get("date"), utc=True)
            bdl_tip_s = bdl_tip.isoformat().replace("+00:00", "Z")
            venue = bdl_g.get("arena") or bdl_g.get("venue") or ""
            bdl_status = bdl_g.get("status") or ""
            if tip_s and bdl_tip_s and tip_s[:19] != bdl_tip_s[:19]:
                bdl_match = "TIP_MISMATCH"
                blocking.append(f"game {gid}: tip mismatch prod={tip_s} bdl={bdl_tip_s}")

        cands = odds_df[(odds_df["home_abbr"] == home) & (odds_df["away_abbr"] == away)]
        identity = "UNRESOLVED"
        odds_event_id = ""
        odds_tip = ""
        if len(cands) == 0:
            # Prefer ET-date match soft warning if tip still pregame scheduled
            if str(g.get("status_normalized")) in {"scheduled", "unknown"}:
                identity = "NO_ODDS_EVENT"
                blocking.append(f"game {gid} {away}@{home}: no Odds API event")
            else:
                identity = "NO_ODDS_EVENT_POST_STATUS"
                warnings.append(f"game {gid} {away}@{home}: no Odds event (status={g.get('status_normalized')})")
        else:
            c2 = cands.copy()
            c2["skew"] = (pd.to_datetime(c2["commence_utc"], utc=True) - tip).abs()
            c2 = c2.sort_values("skew")
            best = c2.iloc[0]
            odds_event_id = best["odds_event_id"]
            odds_tip = best["commence_utc"]
            if odds_event_id in used_odds:
                identity = "ODDS_EVENT_REUSED"
                blocking.append(f"game {gid}: odds event reused")
            else:
                used_odds.add(odds_event_id)
                skew = float(best["skew"].total_seconds())
                identity = "TIP_SKEW" if skew > 3600 else "MATCH"
                if identity == "TIP_SKEW":
                    blocking.append(f"game {gid}: tip skew {skew}s")
        if not tip_s:
            identity = "MISSING_TIP"
            blocking.append(f"game {gid}: missing tip")

        rows.append(
            {
                "game_date_et": game_date,
                "bdl_game_id": gid,
                "odds_api_event_id": odds_event_id,
                "away_team": away,
                "home_team": home,
                "scheduled_tip_utc": tip_s,
                "odds_commence_utc": odds_tip,
                "bdl_tip_utc": bdl_tip_s,
                "venue": venue,
                "game_status": g.get("status_normalized") or g.get("status"),
                "bdl_status": bdl_status,
                "source_timestamp_utc": g.get("pull_timestamp_utc"),
                "bdl_source_timestamp_utc": bdl_fetched,
                "odds_source_timestamp_utc": odds_fetched,
                "identity_match_status": identity,
                "bdl_api_match_status": bdl_match,
            }
        )

    for _, e in odds_df[odds_df["et_date"] == game_date].iterrows():
        if e["odds_event_id"] not in used_odds:
            warnings.append(
                f"orphan odds event {e['odds_event_id']} {e['away_abbr']}@{e['home_abbr']} on ET date"
            )

    for gid, g in bdl_by_id.items():
        tip = pd.to_datetime(g.get("datetime") or g.get("date"), utc=True)
        et_date = tip.tz_convert(ET).strftime("%Y-%m-%d")
        if et_date != game_date:
            warnings.append(
                f"BDL dates[]={game_date} includes game {gid} tip ET {et_date} (excluded from ET slate)"
            )

    _write_csv(audit_dir / "GAME_AUDIT.csv", rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-date", required=True)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke-status", required=True)
    args = ap.parse_args()

    game_date = args.game_date
    stage = Path(args.stage_dir)
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    blocking: list[str] = []
    warnings: list[str] = []

    if args.smoke_status not in {"PASS", "NO_GAMES"}:
        blocking.append(f"smoke_status={args.smoke_status}")

    game_rows = audit_games(game_date, audit_dir, blocking, warnings)

    slate_path = stage / f"slate_{game_date}.parquet"
    pmf_path = stage / "full_pmfs_wide.parquet"
    proj_path = stage / f"player_projections_{game_date}.parquet"
    inj_path = stage / f"injuries_{game_date}.json"
    if not inj_path.exists():
        inj_path = Path(f"data/injuries/{game_date}.json")
    mc_path = stage / "market_comparison.parquet"
    edges_path = stage / "publishable_edges.parquet"
    odds_path = Path("data/processed/wnba_player_props_oddsapi_latest.parquet")
    if not odds_path.exists():
        odds_path = stage / "wnba_player_props_oddsapi_latest.parquet"
    avail_path = stage / f"availability_table_{game_date}.parquet"
    meta_path = stage / "run_metadata.json"

    for req, label in [
        (slate_path, "slate"),
        (pmf_path, "full_pmfs_wide"),
        (proj_path, "player_projections"),
    ]:
        if not req.exists():
            blocking.append(f"missing required staged file: {label} ({req})")

    slate = pd.read_parquet(slate_path) if slate_path.exists() else pd.DataFrame()
    pmfs = pd.read_parquet(pmf_path) if pmf_path.exists() else pd.DataFrame()
    proj = pd.read_parquet(proj_path) if proj_path.exists() else pd.DataFrame()
    mc = pd.read_parquet(mc_path) if mc_path.exists() else pd.DataFrame()
    edges = pd.read_parquet(edges_path) if edges_path.exists() else pd.DataFrame()
    odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
    avail = pd.read_parquet(avail_path) if avail_path.exists() else pd.DataFrame()
    injuries = json.loads(inj_path.read_text()) if inj_path.exists() else []
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pred_ts = run_meta.get("prediction_timestamp") or run_meta.get("prediction_timestamp_utc") or ""

    # Tip cutoff check
    tips = [r["scheduled_tip_utc"] for r in game_rows if r.get("scheduled_tip_utc")]
    now = datetime.now(timezone.utc)
    for tip_s in tips:
        tip_dt = pd.to_datetime(tip_s, utc=True).to_pydatetime()
        if tip_dt <= now:
            warnings.append(f"scheduled tip {tip_s} is not strictly after verification now={now.isoformat()}")
            # If prediction evidence is post-tip for that game, block
            if pred_ts:
                pts = pd.to_datetime(pred_ts, utc=True, errors="coerce")
                if pd.notna(pts) and pts.to_pydatetime() >= tip_dt:
                    blocking.append(f"prediction_timestamp {pred_ts} at/after tip {tip_s}")

    # ---------- Player identity ----------
    players = pd.read_parquet("data/processed/wnba_players.parquet") if Path("data/processed/wnba_players.parquet").exists() else pd.DataFrame()
    player_rows = []
    rejected = unresolved = duplicates = team_mismatch = 0
    slate_keys = set()
    if not slate.empty:
        for _, r in slate.iterrows():
            pid = int(r["player_id"])
            gid = int(r["game_id"])
            key = (gid, pid)
            dup = key in slate_keys
            if dup:
                duplicates += 1
                blocking.append(f"duplicate player-game in slate: game={gid} player={pid}")
            slate_keys.add(key)
            team = r.get("team_abbreviation")
            opp = r.get("opponent_abbreviation") or r.get("opponent_team_abbreviation")
            name = r.get("player_name")
            roster_ok = True
            src = "slate_from_features_latest_row"
            reason = ""
            if players is not None and not players.empty and "player_id" in players.columns:
                prow = players[players["player_id"] == pid]
                if prow.empty:
                    unresolved += 1
                    roster_ok = False
                    reason = "player_id_not_in_players_table"
                else:
                    # team check when column exists
                    pteam = None
                    for c in ("team_abbreviation", "team_abbr", "current_team_abbreviation"):
                        if c in prow.columns:
                            pteam = prow.iloc[0][c]
                            break
                    if pteam is not None and pd.notna(pteam) and str(pteam) != str(team):
                        team_mismatch += 1
                        warnings.append(f"player {pid} slate_team={team} players_table_team={pteam}")
                        reason = "current_team_mismatch_warning"
            status = "ACCEPTED" if roster_ok and not dup else "REJECTED"
            if status == "REJECTED":
                rejected += 1
            player_rows.append(
                {
                    "canonical_player_id": pid,
                    "player_name": name,
                    "current_team": team,
                    "opponent": opp,
                    "game_id": gid,
                    "roster_membership": "YES" if roster_ok else "NO",
                    "expected_active_status": r.get("dnp_risk"),
                    "injury_flag": r.get("injury_flag"),
                    "source_of_roster_identity": src,
                    "source_timestamp": r.get("game_date"),
                    "duplicate_player_status": "DUPLICATE" if dup else "UNIQUE",
                    "audit_status": status,
                    "reject_reason": reason,
                }
            )
    _write_csv(audit_dir / "PLAYER_IDENTITY_AUDIT.csv", player_rows)

    # ---------- Availability ----------
    inj_by_id = {int(x["player_id"]): x for x in injuries if x.get("player_id") is not None}
    avail_rows = []
    out_priced = 0
    for _, r in (pmfs.drop_duplicates(["game_id", "player_id"]) if not pmfs.empty else pd.DataFrame()).iterrows():
        pid = int(r["player_id"])
        inj = inj_by_id.get(pid)
        status = (inj or {}).get("status", "available_absent_from_injury_feed")
        src_ts = (inj or {}).get("fetched_at_utc", "")
        p_dnp = r.get("p_dnp")
        role = r.get("role") if "role" in r else r.get("minutes_mean")
        flag = ""
        if status == "out":
            # Confirm whether priced as active in edges/mc
            priced = False
            if not edges.empty and "player_id" in edges.columns:
                priced = bool((edges["player_id"].astype(int) == pid).any())
            if not mc.empty and "player_id" in mc.columns and not priced:
                sub = mc[mc["player_id"].astype(int) == pid]
                if not sub.empty and "status" in sub.columns:
                    priced = any("PRICED" in str(x) for x in sub["status"].astype(str))
                elif not sub.empty:
                    priced = True
            if priced:
                out_priced += 1
                blocking.append(f"confirmed OUT player priced: {pid} {r.get('player_name')}")
                flag = "OUT_PRICED"
            else:
                flag = "OUT_NOT_PRICED_OK"
        if p_dnp is not None and pd.notna(p_dnp) and abs(float(p_dnp) - 0.5) < 1e-15:
            warnings.append(f"player {pid}: p_dnp exactly 0.5 (possible silent default)")
        avail_rows.append(
            {
                "player_id": pid,
                "player_name": r.get("player_name"),
                "game_id": int(r["game_id"]),
                "availability_status": status,
                "source": "bdl_player_injuries" if inj else "none",
                "source_timestamp": src_ts,
                "ingestion_timestamp": src_ts,
                "forecast_timestamp": pred_ts,
                "freshness": "current_feed" if inj else "missing",
                "participation_probability": p_dnp,
                "role_minutes_status": role,
                "missing_data_indicators": flag,
                "injury_comment": (inj or {}).get("comment"),
            }
        )
    _write_csv(audit_dir / "AVAILABILITY_AUDIT.csv", avail_rows)

    # ---------- Feature sanity ----------
    feat_rows = []
    feat_cols_interest = [
        c
        for c in [
            "player_minutes_mean_l5",
            "minutes_mean",
            "minutes_sigma",
            "player_usage_percentage_l5",
            "player_pts_mean_l5",
            "rest_days",
            "is_back_to_back",
            "home_away",
            "opponent_abbreviation",
            "opponent_team_abbreviation",
            "team_abbreviation",
            "player_zero_minute_rate_l5",
        ]
        if (not slate.empty and c in slate.columns)
    ]
    vectors = {}
    if not slate.empty:
        for _, r in slate.iterrows():
            pid = int(r["player_id"])
            gid = int(r["game_id"])
            issues = []
            mins = r.get("player_minutes_mean_l5")
            if mins is not None and pd.notna(mins) and (float(mins) < 0 or float(mins) > 48):
                issues.append("impossible_minutes")
                blocking.append(f"impossible minutes for player {pid}: {mins}")
            vec = tuple(
                float(r[c]) if c in r and pd.notna(r[c]) and isinstance(r[c], (int, float, np.floating)) else None
                for c in feat_cols_interest
            )
            if all(v == 0 or v is None for v in vec):
                issues.append("all_zero_or_missing_key_features")
                warnings.append(f"player {pid}: all-zero/missing key features")
            vectors.setdefault(vec, []).append(pid)
            # future date?
            gd = str(r.get("game_date"))
            if gd and gd > game_date and game_date not in gd:
                issues.append("future_date")
            feat_rows.append(
                {
                    "player_id": pid,
                    "player_name": r.get("player_name"),
                    "game_id": gid,
                    "team": r.get("team_abbreviation"),
                    "opponent": r.get("opponent_abbreviation") or r.get("opponent_team_abbreviation"),
                    "home_away": r.get("home_away"),
                    "recent_minutes": r.get("player_minutes_mean_l5"),
                    "projected_minutes": r.get("minutes_mean"),
                    "minutes_sigma": r.get("minutes_sigma"),
                    "recent_usage": r.get("player_usage_percentage_l5"),
                    "rest_days": r.get("rest_days"),
                    "issues": "|".join(issues),
                    **{c: r.get(c) for c in feat_cols_interest},
                }
            )
        for vec, pids in vectors.items():
            if vec and len(pids) > 1 and any(v is not None for v in vec):
                # identical vectors for unrelated players
                if len(set(pids)) > 1:
                    warnings.append(f"identical feature vectors for players {pids[:8]}")
    _write_csv(audit_dir / "FEATURE_SANITY_AUDIT.csv", feat_rows)

    # ---------- Quote pairs ----------
    quote_rows = []
    rejected_quotes = Counter()
    if not odds.empty:
        # normalize columns
        cols = {c.lower(): c for c in odds.columns}
        def col(*names):
            for n in names:
                if n in odds.columns:
                    return n
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        c_event = col("event_id", "odds_event_id")
        c_game = col("game_id")
        c_player = col("player_id")
        c_book = col("bookmaker", "vendor", "sportsbook")
        c_mkt = col("market_key", "market", "stat")
        c_line = col("line", "point")
        c_side = col("side", "name")
        c_american = col("price", "american_odds", "odds")
        c_ts = col("provider_timestamp", "last_update", "market_last_update", "updated_at")
        # If already paired over/under wide format:
        c_over = col("over_odds")
        c_under = col("under_odds")
        if c_over and c_under and c_player and c_line:
            for _, r in odds.iterrows():
                try:
                    oa, ua = float(r[c_over]), float(r[c_under])
                    nv_o, nv_u = _no_vig(oa, ua)
                    ok = True
                    reason = ""
                except Exception as exc:
                    ok = False
                    reason = f"odds_parse:{exc}"
                    nv_o = nv_u = float("nan")
                    rejected_quotes[reason] += 1
                quote_rows.append(
                    {
                        "odds_api_event_id": r.get(c_event) if c_event else "",
                        "canonical_game_id": r.get(c_game) if c_game else "",
                        "canonical_player_id": r.get(c_player),
                        "sportsbook": r.get(c_book) if c_book else "",
                        "market_key": r.get(c_mkt) if c_mkt else "",
                        "period": r.get("period", "game"),
                        "line": r.get(c_line),
                        "over_american": r.get(c_over),
                        "under_american": r.get(c_under),
                        "no_vig_over": nv_o,
                        "no_vig_under": nv_u,
                        "provider_quote_timestamp": r.get(c_ts) if c_ts else "",
                        "pair_status": "VALID_PAIR" if ok else "REJECTED",
                        "reject_reason": reason,
                    }
                )
        elif c_side and c_american and c_player and c_line and c_book:
            # long format — pair
            key_cols = [c for c in [c_book, c_event, c_player, c_mkt, c_line] if c]
            grouped = odds.groupby(key_cols, dropna=False)
            for key, g in grouped:
                sides = {str(x).lower(): x for x in g[c_side].astype(str)}
                over = g[g[c_side].astype(str).str.lower().isin(["over", "o"])]
                under = g[g[c_side].astype(str).str.lower().isin(["under", "u"])]
                if len(over) != 1 or len(under) != 1:
                    rejected_quotes["one_sided_or_ambiguous"] += 1
                    quote_rows.append(
                        {
                            "canonical_player_id": key[key_cols.index(c_player)] if c_player in key_cols else "",
                            "sportsbook": key[key_cols.index(c_book)] if c_book in key_cols else "",
                            "line": key[key_cols.index(c_line)] if c_line in key_cols else "",
                            "pair_status": "REJECTED",
                            "reject_reason": "one_sided_or_ambiguous",
                            "raw_quote_sides": len(g),
                        }
                    )
                    continue
                oa = float(over.iloc[0][c_american])
                ua = float(under.iloc[0][c_american])
                nv_o, nv_u = _no_vig(oa, ua)
                quote_rows.append(
                    {
                        "odds_api_event_id": over.iloc[0].get(c_event) if c_event else "",
                        "canonical_game_id": over.iloc[0].get(c_game) if c_game else "",
                        "canonical_player_id": over.iloc[0].get(c_player),
                        "sportsbook": over.iloc[0].get(c_book),
                        "market_key": over.iloc[0].get(c_mkt) if c_mkt else "",
                        "period": "game",
                        "line": over.iloc[0].get(c_line),
                        "over_american": oa,
                        "under_american": ua,
                        "no_vig_over": nv_o,
                        "no_vig_under": nv_u,
                        "provider_quote_timestamp": over.iloc[0].get(c_ts) if c_ts else "",
                        "pair_status": "VALID_PAIR",
                        "reject_reason": "",
                        "raw_quote_sides": 2,
                    }
                )
        else:
            warnings.append("odds parquet schema not recognized for pairing")
    else:
        warnings.append("no odds quotes available for QUOTE_PAIR_AUDIT")
    _write_csv(audit_dir / "QUOTE_PAIR_AUDIT.csv", quote_rows)

    # ---------- PMF audit ----------
    pmf_rows = []
    mono_rows = []
    pricing_rows = []
    status_rows = []
    traces = []
    pmf_hash_by_key = {}

    pmf_col = "active_pmf_json" if (not pmfs.empty and "active_pmf_json" in pmfs.columns) else "pmf_json"
    for idx, r in (pmfs.iterrows() if not pmfs.empty else []):
        atoms = _parse_pmf(r.get(pmf_col) or r.get("pmf_json"))
        probs = [p for _, p in atoms]
        support = [k for k, _ in atoms]
        stored_mass = float(sum(probs)) if probs else 0.0
        overflow = float(r["overflow_probability"]) if "overflow_probability" in r and pd.notna(r.get("overflow_probability")) else 0.0
        # sometimes overflow stored under other names
        for alt in ("tail_mass", "overflow_prob", "p_overflow"):
            if overflow == 0.0 and alt in r and pd.notna(r.get(alt)):
                overflow = float(r[alt])
        total = stored_mass + overflow
        norm_err = abs(total - 1.0)
        finite = all(math.isfinite(p) for p in probs)
        nonnegative = all(p >= -1e-15 for p in probs)
        mean = sum(k * p for k, p in atoms) if atoms else float("nan")
        var = sum(((k - mean) ** 2) * p for k, p in atoms) if atoms and math.isfinite(mean) else float("nan")
        zero_p = sum(p for k, p in atoms if k == 0)
        issues = []
        if not finite:
            issues.append("nonfinite")
            blocking.append(f"nonfinite PMF atom player={r.get('player_id')} stat={r.get('stat')}")
        if not nonnegative:
            issues.append("negative")
            blocking.append(f"negative PMF atom player={r.get('player_id')} stat={r.get('stat')}")
        if norm_err > MASS_TOL:
            issues.append("mass_tolerance")
            blocking.append(
                f"PMF mass outside tol player={r.get('player_id')} stat={r.get('stat')} "
                f"stored={stored_mass} overflow={overflow} err={norm_err}"
            )
        if math.isfinite(var) and var < -1e-12:
            issues.append("neg_variance")
            blocking.append(f"negative variance player={r.get('player_id')} stat={r.get('stat')}")
        dnp = r.get("p_dnp")
        # DNP mixed into zero heuristic: if p_dnp large and zero_p ~= p_dnp within active pmf, warn
        if dnp is not None and pd.notna(dnp) and float(dnp) > 0.05 and abs(zero_p - float(dnp)) < 1e-6 and pmf_col == "pmf_json":
            warnings.append(
                f"possible DNP mixed into zero mass player={r.get('player_id')} stat={r.get('stat')}"
            )
            issues.append("possible_dnp_in_zero")

        blob = json.dumps(atoms, separators=(",", ":")).encode()
        phash = _sha256_bytes(blob)
        key = (int(r["game_id"]), int(r["player_id"]), str(r["stat"]))
        pmf_hash_by_key[key] = phash

        pmf_rows.append(
            {
                "game_id": int(r["game_id"]),
                "player_id": int(r["player_id"]),
                "player_name": r.get("player_name"),
                "stat": r.get("stat"),
                "pmf_column": pmf_col,
                "n_atoms": len(atoms),
                "all_finite": finite,
                "all_nonnegative": nonnegative,
                "stored_atom_mass": stored_mass,
                "overflow_probability": overflow,
                "total_mass": total,
                "normalization_error": norm_err,
                "support_min": min(support) if support else None,
                "support_max": max(support) if support else None,
                "predictive_mean": mean,
                "predictive_variance": var,
                "zero_probability": zero_p,
                "p_dnp": dnp,
                "pmf_hash": phash,
                "issues": "|".join(issues),
                "model_status": r.get("model_status", ""),
                "calibration_status": r.get("calibration_status", ""),
            }
        )

        # Monotonicity across offered lines for this player-stat
        lines = []
        if not mc.empty:
            sub = mc[
                (mc["player_id"].astype(int) == int(r["player_id"]))
                & (mc["stat"].astype(str) == str(r["stat"]))
            ] if {"player_id", "stat"}.issubset(mc.columns) else pd.DataFrame()
            if not sub.empty and "line" in sub.columns:
                lines = sorted({float(x) for x in sub["line"].dropna().tolist()})
        if len(lines) >= 2 and atoms:
            p_gt = {L: sum(p for k, p in atoms if k > L) for L in lines}
            for a, b in zip(lines, lines[1:]):
                ok = p_gt[a] + 1e-12 >= p_gt[b]
                if not ok:
                    blocking.append(
                        f"monotonicity violation player={r.get('player_id')} stat={r.get('stat')} "
                        f"L1={a} P={p_gt[a]} L2={b} P={p_gt[b]}"
                    )
                mono_rows.append(
                    {
                        "game_id": int(r["game_id"]),
                        "player_id": int(r["player_id"]),
                        "stat": r.get("stat"),
                        "L1": a,
                        "L2": b,
                        "P_gt_L1": p_gt[a],
                        "P_gt_L2": p_gt[b],
                        "ok": ok,
                    }
                )

    _write_csv(audit_dir / "PMF_AUDIT.csv", pmf_rows)
    _write_csv(audit_dir / "MONOTONICITY_AUDIT.csv", mono_rows)

    # ---------- Pricing reproducibility from market_comparison ----------
    if not mc.empty:
        for _, r in mc.iterrows():
            try:
                pid = int(r["player_id"])
                gid = int(r["game_id"])
                stat = str(r["stat"])
                line = float(r["line"])
            except Exception:
                continue
            pmf_r = pmfs[
                (pmfs["player_id"].astype(int) == pid)
                & (pmfs["game_id"].astype(int) == gid)
                & (pmfs["stat"].astype(str) == stat)
            ]
            atoms = _parse_pmf(pmf_r.iloc[0].get(pmf_col) or pmf_r.iloc[0].get("pmf_json")) if len(pmf_r) else []
            indep = _settle_from_pmf(atoms, line) if atoms else None
            # delivered fields — try common names
            def getf(*names):
                for n in names:
                    if n in r and pd.notna(r[n]):
                        try:
                            return float(r[n])
                        except Exception:
                            continue
                return None

            d_po = getf("p_over_settled", "p_over", "model_p_over")
            d_pu = getf("p_under_settled", "p_under", "model_p_under")
            d_fo = getf("fair_decimal_over", "fair_odds_over", "model_fair_decimal_over")
            d_fu = getf("fair_decimal_under", "fair_odds_under", "model_fair_decimal_under")
            status = str(r.get("status", r.get("edge_status", r.get("recommendation_status", ""))))
            diffs_ok = True
            detail = ""
            if indep is None:
                diffs_ok = False
                detail = "missing_pmf"
                blocking.append(f"pricing missing PMF game={gid} player={pid} stat={stat}")
            else:
                if d_po is not None and abs(d_po - indep["p_over_settled"]) > PROB_TOL:
                    diffs_ok = False
                    detail = f"p_over_delta={abs(d_po - indep['p_over_settled'])}"
                    blocking.append(
                        f"pricing p_over not reproducible game={gid} player={pid} stat={stat} line={line} {detail}"
                    )
                if d_pu is not None and abs(d_pu - indep["p_under_settled"]) > PROB_TOL:
                    diffs_ok = False
                    detail = f"p_under_delta={abs(d_pu - indep['p_under_settled'])}"
                    blocking.append(
                        f"pricing p_under not reproducible game={gid} player={pid} stat={stat} line={line} {detail}"
                    )
                if d_fo is not None and math.isfinite(indep["fair_decimal_over"]) and abs(d_fo - indep["fair_decimal_over"]) > DECIMAL_TOL:
                    diffs_ok = False
                    detail = f"fair_over_delta={abs(d_fo - indep['fair_decimal_over'])}"
                    blocking.append(
                        f"fair decimal over not reproducible game={gid} player={pid} stat={stat} {detail}"
                    )
                if d_fu is not None and math.isfinite(indep["fair_decimal_under"]) and abs(d_fu - indep["fair_decimal_under"]) > DECIMAL_TOL:
                    diffs_ok = False
                    detail = f"fair_under_delta={abs(d_fu - indep['fair_decimal_under'])}"
                    blocking.append(
                        f"fair decimal under not reproducible game={gid} player={pid} stat={stat} {detail}"
                    )
                # half-point push
                if abs(line - round(line)) > 1e-9 and indep["p_push"] > 1e-12:
                    warnings.append(f"half-point line with push mass game={gid} player={pid} stat={stat} line={line}")

            pricing_rows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "player_name": r.get("player_name", ""),
                    "stat": stat,
                    "line": line,
                    "status": status,
                    "delivered_p_over_settled": d_po,
                    "delivered_p_under_settled": d_pu,
                    "indep_p_over_win": None if indep is None else indep["p_over_win"],
                    "indep_p_under_win": None if indep is None else indep["p_under_win"],
                    "indep_p_push": None if indep is None else indep["p_push"],
                    "indep_p_over_settled": None if indep is None else indep["p_over_settled"],
                    "indep_p_under_settled": None if indep is None else indep["p_under_settled"],
                    "indep_fair_decimal_over": None if indep is None else indep["fair_decimal_over"],
                    "indep_fair_decimal_under": None if indep is None else indep["fair_decimal_under"],
                    "reproducible": diffs_ok,
                    "detail": detail,
                }
            )
            status_rows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "stat": stat,
                    "line": line,
                    "status": status or "UNKNOWN",
                    "vendor": r.get("vendor", r.get("bookmaker", "")),
                }
            )
    _write_csv(audit_dir / "PRICING_AUDIT.csv", pricing_rows)
    _write_csv(audit_dir / "STATUS_AUDIT.csv", status_rows)

    # ---------- Traces (up to 20+) ----------
    if not pricing_rows:
        warnings.append("no market_comparison rows for pricing traces")
    else:
        # diversify by stat
        by_stat = defaultdict(list)
        for row in pricing_rows:
            by_stat[row["stat"]].append(row)
        selected = []
        for st, rows in by_stat.items():
            selected.extend(rows[: max(1, 20 // max(1, len(by_stat)))])
        selected = selected[:24]
        for row in selected:
            pmf_r = pmfs[
                (pmfs["player_id"].astype(int) == row["player_id"])
                & (pmfs["game_id"].astype(int) == row["game_id"])
                & (pmfs["stat"].astype(str) == row["stat"])
            ]
            atoms = _parse_pmf(pmf_r.iloc[0].get(pmf_col) or pmf_r.iloc[0].get("pmf_json")) if len(pmf_r) else []
            line = float(row["line"])
            around = [(k, p) for k, p in atoms if abs(k - line) <= 3]
            indep = _settle_from_pmf(atoms, line) if atoms else {}
            # quote odds if available
            q = next(
                (
                    q
                    for q in quote_rows
                    if str(q.get("canonical_player_id")) == str(row["player_id"])
                    and abs(float(q.get("line") or -999) - line) < 1e-9
                ),
                {},
            )
            traces.append(
                {
                    "game_id": row["game_id"],
                    "player_id": row["player_id"],
                    "stat": row["stat"],
                    "line": line,
                    "raw_over_odds": q.get("over_american"),
                    "raw_under_odds": q.get("under_american"),
                    "no_vig_over": q.get("no_vig_over"),
                    "no_vig_under": q.get("no_vig_under"),
                    "pmf_hash": pmf_hash_by_key.get((row["game_id"], row["player_id"], row["stat"]), ""),
                    "atoms_around_line": json.dumps(around),
                    "tail_probability": next(
                        (p["overflow_probability"] for p in pmf_rows if p["player_id"] == row["player_id"] and p["stat"] == row["stat"] and p["game_id"] == row["game_id"]),
                        None,
                    ),
                    "over_win_mass": indep.get("p_over_win"),
                    "under_win_mass": indep.get("p_under_win"),
                    "push_mass": indep.get("p_push"),
                    "p_over_settled": indep.get("p_over_settled"),
                    "p_under_settled": indep.get("p_under_settled"),
                    "fair_decimal_over": indep.get("fair_decimal_over"),
                    "fair_decimal_under": indep.get("fair_decimal_under"),
                    "fair_american_over": _decimal_to_american(indep["fair_decimal_over"])
                    if indep.get("fair_decimal_over")
                    else None,
                    "fair_american_under": _decimal_to_american(indep["fair_decimal_under"])
                    if indep.get("fair_decimal_under")
                    else None,
                    "model_status": pmf_r.iloc[0].get("model_status", "") if len(pmf_r) else "",
                    "calibration_status": pmf_r.iloc[0].get("calibration_status", "") if len(pmf_r) else "",
                    "row_status": row.get("status"),
                    "reproducible": row.get("reproducible"),
                }
            )
    _write_csv(audit_dir / "PRICING_TRACE.csv", traces)

    # ---------- Human-readable PMF samples ----------
    sample_md = ["# PMF Samples", ""]
    if not pmfs.empty:
        # pick archetypes using minutes_mean when available
        base = pmfs[pmfs["stat"].astype(str) == "pts"].copy() if (pmfs["stat"].astype(str) == "pts").any() else pmfs.copy()
        if "minutes_mean" in base.columns:
            base = base.sort_values("minutes_mean", ascending=False)
        picks = []
        if not base.empty:
            picks.append(("high_minute_star", base.iloc[0]))
            picks.append(("average_starter", base.iloc[len(base) // 2]))
            picks.append(("bench_player", base.iloc[-1]))
        stl = pmfs[pmfs["stat"].astype(str) == "stl"]
        if not stl.empty:
            stl2 = stl.copy()
            stl2["_z"] = stl2.apply(lambda r: sum(p for k, p in _parse_pmf(r.get(pmf_col) or r.get("pmf_json")) if k == 0), axis=1)
            picks.append(("zero_heavy_steals", stl2.sort_values("_z", ascending=False).iloc[0]))
        for label, r in picks:
            atoms = _parse_pmf(r.get(pmf_col) or r.get("pmf_json"))
            sample_md.append(f"## {label}: {r.get('player_name')} ({r.get('stat')})")
            sample_md.append(f"- game_id={r.get('game_id')} player_id={r.get('player_id')}")
            sample_md.append(f"- minutes_mean={r.get('minutes_mean')} p_dnp={r.get('p_dnp')}")
            sample_md.append("```")
            for k, p in atoms[:40]:
                sample_md.append(f"{k}: {p:.8f}")
            if len(atoms) > 40:
                sample_md.append(f"... ({len(atoms)-40} more atoms)")
            sample_md.append("```")
            sample_md.append("")
    (audit_dir / "PMF_SAMPLES.md").write_text("\n".join(sample_md))

    # ---------- Frozen manifest ----------
    frozen_rows = []
    registry = []
    model_hash = _sha256_file(Path("artifacts/models/stage4_baseline/artifact_manifest_model.json"))
    cal_hash = _sha256_file(Path("artifacts/models/calibration/artifact_manifest_calibrator.json"))
    feat_hash = _sha256_file(Path("data/processed/feature_schema_manifest.json"))
    data_hash = _sha256_file(Path("data/processed/schema_manifest.json"))
    quote_hash = _sha256_file(odds_path) if odds_path.exists() else ""
    avail_hash = _sha256_file(inj_path) if inj_path.exists() else ""

    # Store full atom distributions immutably under audit_dir/frozen_pmfs/
    frozen_pmf_dir = audit_dir / "frozen_pmfs"
    frozen_pmf_dir.mkdir(parents=True, exist_ok=True)
    for p in pmf_rows:
        key = f"{p['game_id']}_{p['player_id']}_{p['stat']}"
        atoms = None
        match = pmfs[
            (pmfs["game_id"].astype(int) == p["game_id"])
            & (pmfs["player_id"].astype(int) == p["player_id"])
            & (pmfs["stat"].astype(str) == p["stat"])
        ]
        if len(match):
            atoms = _parse_pmf(match.iloc[0].get(pmf_col) or match.iloc[0].get("pmf_json"))
        ident = {
            "game_id": p["game_id"],
            "player_id": p["player_id"],
            "target": p["stat"],
            "prediction_timestamp": pred_ts,
            "forecast_horizon": game_date,
            "model_hash": model_hash,
            "distribution_hash": p["pmf_hash"],
        }
        outp = frozen_pmf_dir / f"{key}.json"
        if outp.exists():
            blocking.append(f"frozen pmf overwrite attempted: {outp}")
        else:
            outp.write_text(json.dumps({"identity": ident, "atoms": atoms}, indent=2))
        frozen_rows.append(ident)
        registry.append(
            {
                **ident,
                "actual_scheduled_tip_utc": next(
                    (g["scheduled_tip_utc"] for g in game_rows if int(g["bdl_game_id"]) == p["game_id"]),
                    "",
                ),
                "forecast_timestamp": pred_ts,
                "code_sha": args.code_sha,
                "data_hash": data_hash,
                "feature_hash": feat_hash,
                "calibrator_hash": cal_hash,
                "quote_hash": quote_hash,
                "availability_hash": avail_hash,
                "pmf_hash": p["pmf_hash"],
                "delivery_path": str(stage),
            }
        )

    # Append-only registry
    reg_path = audit_dir / "FROZEN_PREDICTION_REGISTRY.jsonl"
    existing_keys = set()
    if reg_path.exists():
        for line in reg_path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            existing_keys.add(
                (obj.get("game_id"), obj.get("player_id"), obj.get("target"), obj.get("prediction_timestamp"))
            )
    with reg_path.open("a") as f:
        for row in registry:
            k = (row["game_id"], row["player_id"], row["target"], row["prediction_timestamp"])
            if k in existing_keys:
                blocking.append(f"duplicate frozen registry append: {k}")
                continue
            f.write(json.dumps(row) + "\n")
            existing_keys.add(k)

    manifest = {
        "game_date": game_date,
        "prediction_timestamp": pred_ts,
        "code_sha": args.code_sha,
        "run_id": args.run_id,
        "stage_dir": str(stage),
        "n_frozen_identities": len(frozen_rows),
        "model_hash": model_hash,
        "calibrator_hash": cal_hash,
        "feature_hash": feat_hash,
        "data_hash": data_hash,
        "quote_hash": quote_hash,
        "availability_hash": avail_hash,
        "smoke_status": args.smoke_status,
        "generated_at_utc": _utc_now(),
    }
    (audit_dir / "FROZEN_PREDICTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # ---------- Verdict ----------
    status_counts = Counter(r.get("status") or "UNKNOWN" for r in status_rows)
    if args.smoke_status == "FAIL" or blocking:
        verdict = "FAIL_DO_NOT_PUBLISH"
    elif warnings:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS_FOR_PROSPECTIVE_EVALUATION"

    summary = {
        "verdict": verdict,
        "game_date": game_date,
        "prediction_timestamp": pred_ts,
        "code_sha": args.code_sha,
        "run_id": args.run_id,
        "smoke_status": args.smoke_status,
        "games": game_rows,
        "players_discovered": len(player_rows),
        "players_accepted": sum(1 for r in player_rows if r["audit_status"] == "ACCEPTED"),
        "players_rejected": rejected,
        "unresolved_identities": unresolved,
        "duplicate_identities": duplicates,
        "current_team_mismatches": team_mismatch,
        "pmf_rows": len(pmf_rows),
        "quote_pairs_valid": sum(1 for r in quote_rows if r.get("pair_status") == "VALID_PAIR"),
        "quote_pairs_rejected": sum(1 for r in quote_rows if r.get("pair_status") == "REJECTED"),
        "rejected_quote_reasons": dict(rejected_quotes),
        "monotonicity_violations": sum(1 for r in mono_rows if not r.get("ok")),
        "pricing_failures": sum(1 for r in pricing_rows if not r.get("reproducible")),
        "status_counts": dict(status_counts),
        "blocking_defects": blocking,
        "warnings": warnings,
        "stage_dir": str(stage),
        "publish": False,
        "dry_run": True,
    }
    (audit_dir / "PREGAME_VERIFICATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    report = [
        f"# Pregame Verification Report — {game_date}",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"- code_sha: `{args.code_sha}`",
        f"- prediction_timestamp: `{pred_ts}`",
        f"- stage_dir: `{stage}`",
        f"- smoke_status: `{args.smoke_status}`",
        f"- publish: false (enforced)",
        "",
        "## Games",
    ]
    for g in game_rows:
        report.append(
            f"- {g['away_team']}@{g['home_team']} tip={g['scheduled_tip_utc']} "
            f"bdl={g['bdl_game_id']} odds={g['odds_api_event_id']} identity={g['identity_match_status']}"
        )
    report += [
        "",
        "## Players",
        f"- discovered: {len(player_rows)}",
        f"- accepted: {sum(1 for r in player_rows if r['audit_status']=='ACCEPTED')}",
        f"- rejected: {rejected}",
        f"- unresolved: {unresolved}",
        f"- duplicates: {duplicates}",
        "",
        "## Status counts",
        json.dumps(dict(status_counts), indent=2),
        "",
        "## Blocking defects",
    ]
    report.extend([f"- {b}" for b in blocking] or ["- none"])
    report += ["", "## Warnings"]
    report.extend([f"- {w}" for w in warnings] or ["- none"])
    report += [
        "",
        "## Files for independent review",
        f"- `{audit_dir}/GAME_AUDIT.csv`",
        f"- `{audit_dir}/PLAYER_IDENTITY_AUDIT.csv`",
        f"- `{audit_dir}/AVAILABILITY_AUDIT.csv`",
        f"- `{audit_dir}/FEATURE_SANITY_AUDIT.csv`",
        f"- `{audit_dir}/QUOTE_PAIR_AUDIT.csv`",
        f"- `{audit_dir}/PMF_AUDIT.csv`",
        f"- `{audit_dir}/PMF_SAMPLES.md`",
        f"- `{audit_dir}/MONOTONICITY_AUDIT.csv`",
        f"- `{audit_dir}/PRICING_AUDIT.csv`",
        f"- `{audit_dir}/PRICING_TRACE.csv`",
        f"- `{audit_dir}/STATUS_AUDIT.csv`",
        f"- `{audit_dir}/FROZEN_PREDICTION_MANIFEST.json`",
        f"- `{audit_dir}/FROZEN_PREDICTION_REGISTRY.jsonl`",
        f"- `{audit_dir}/PREGAME_VERIFICATION_SUMMARY.json`",
        f"- staged delivery: `{stage}`",
    ]
    (audit_dir / "PREGAME_VERIFICATION_REPORT.md").write_text("\n".join(report))
    print(json.dumps({"verdict": verdict, "blocking": len(blocking), "warnings": len(warnings)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
