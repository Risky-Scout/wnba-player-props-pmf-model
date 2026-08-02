"""DEPRECATED / LEGACY_CONTROL — production uses scripts/run_wnba_pmf.py → sharp_v6.predict_slate."""
"""Live real upcoming-slate run (NOT a fixture).

Refit V4 on all completed data through the latest game, build the next real WNBA slate from live
BDL, predict participation + minutes PMF + Tier A stat PMFs from real point-in-time features, fetch
live exact no-vig player-prop quotes from The Odds API, and emit atom PMFs + fair Over/Under prices
+ market-consistent PMFs with full lineage. Players without a valid fitted path abstain honestly.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v4 import core as C
from wnba_props_model.sharp_v4 import models as M

app = typer.Typer(add_completion=False)
REPO = C.load_verified.__globals__["REPO"]
OUT = REPO / "artifacts" / "sharp_v4"
BDL = "https://api.balldontlie.io/wnba/v1"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_wnba"
MARKET_MAP = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
              "player_threes": "fg3m"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def _bdl_games(date: str):
    r = requests.get(f"{BDL}/games", headers={"Authorization": os.environ["BDL_API_KEY"]},
                     params={"seasons[]": 2026, "start_date": date, "end_date": date, "per_page": 100}, timeout=20)
    r.raise_for_status()
    return r.json().get("data", [])


def _odds_events():
    r = requests.get(f"{ODDS}/events", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=20)
    return r.json() if r.status_code == 200 else []


def _odds_props(event_id: str, credit_log: list):
    r = requests.get(f"{ODDS}/events/{event_id}/odds",
                     params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us",
                             "markets": ",".join(MARKET_MAP), "oddsFormat": "american"}, timeout=25)
    credit_log.append({"event": event_id, "status": r.status_code,
                       "used": r.headers.get("x-requests-used"), "remaining": r.headers.get("x-requests-remaining"),
                       "last": r.headers.get("x-requests-last")})
    return r.json() if r.status_code == 200 else {}


@app.command()
def main(date: str = typer.Option("2026-07-31", "--date")) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    _, df = C.load_verified()
    stats = pd.read_parquet(REPO / "data/recovered_v2/wnba_player_game_stats.parquet")
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    teams = pd.read_parquet(REPO / "data/recovered_v2/wnba_teams.parquet")
    abbr2id = dict(zip(teams["team_abbreviation"], teams["team_id"]))
    name_by_pid = stats.sort_values("game_date").groupby("player_id")["player_name"].last().to_dict()
    design_hash = json.loads((OUT / "V4_FREEZE_MANIFEST.json").read_text())["modeling_design_v4_sha256"]
    data_hash = json.loads((REPO / "artifacts/sharp_v3/PRIVATE_INPUT_MANIFEST.json").read_text())["inputs"]["pregame_features_t12"]["sha256"][:16]
    code_sha = os.popen("git rev-parse HEAD").read().strip()

    # live upcoming games
    blockers = {}
    try:
        games = _bdl_games(date)
        upcoming = [g for g in games if g.get("status") not in ("post", "final", "Final")]
        if not upcoming:
            upcoming = games   # fall back to all games that day
    except Exception as e:  # noqa: BLE001
        blockers["bdl_games"] = str(e); upcoming = []

    # latest pregame feature row per player (point-in-time proxy for the next game)
    latest_feat = df.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")
    # most-recent team per player from stats (<= day before slate)
    recent = stats[stats["game_date"] < pd.Timestamp(date)].sort_values("game_date")
    last_team = recent.groupby("player_id")["team_abbreviation"].last().to_dict()

    # refit V4 on ALL completed history (through latest available)
    train = df.copy()
    part_feat = C.resolve_contract("participation", list(df.columns))
    Xtr, _, _ = M.prep(train, train.head(1), part_feat)
    ytr = train["participation"].to_numpy(int)
    order = train["game_date"].rank(method="first").to_numpy(); cut = np.quantile(order, 0.8)
    clf0 = HistGradientBoostingClassifier(**M._HGBC).fit(Xtr[order <= cut], ytr[order <= cut])
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf0.predict_proba(Xtr[order > cut])[:, 1], ytr[order > cut])
    part_full = HistGradientBoostingClassifier(**M._HGBC).fit(Xtr, ytr)

    # build slate rows
    slate_rows = []
    for g in upcoming:
        ha, aa = g["home_team"]["abbreviation"], g["visitor_team"]["abbreviation"]
        gid = g["id"]
        for team, opp in ((ha, aa), (aa, ha)):
            pids = [pid for pid, t in last_team.items() if t == team and pid in latest_feat.index]
            for pid in pids:
                row = latest_feat.loc[pid].copy()
                row["game_id"] = gid; row["opponent_team_id"] = abbr2id.get(opp, row.get("opponent_team_id"))
                row["team_id"] = abbr2id.get(team, row.get("team_id"))
                row["player_id"] = pid; row["_team"] = team; row["_opp"] = opp
                slate_rows.append(row)
    if not slate_rows:
        blockers["slate"] = "no upcoming games/players resolved"
        (OUT / "REAL_SLATE_COVERAGE.csv").write_text("game_id,player_id,stat,status\n")
        _write_blockers(blockers, ts)
        typer.echo("LIVE SLATE: no games resolved; blockers recorded"); return
    slate = pd.DataFrame(slate_rows).reset_index(drop=True)

    # predictions
    _, Xsl, _ = M.prep(train, slate, part_feat)
    p_active = np.clip(iso.predict(part_full.predict_proba(Xsl)[:, 1]), 1e-4, 1 - 1e-4)
    mp, _, _ = M.fit_minutes(train, slate)
    stat_pmfs = {}
    for stat in C.TIER_A:
        pmfs, _fh, _, _ = M.fit_direct(train, slate, stat)
        stat_pmfs[stat] = pmfs

    # live odds -> market-consistent PMFs
    credit_log = []
    events = _odds_events()
    # match odds events to BDL games by team full names
    def _teamkey(name):
        return _norm(name)[-6:]
    odds_by_pid_market = {}
    try:
        for ev in events:
            props = _odds_props(ev["id"], credit_log)
            for bm in props.get("bookmakers", []):
                for mk in bm.get("markets", []):
                    stat = MARKET_MAP.get(mk["key"])
                    if not stat:
                        continue
                    for oc in mk.get("outcomes", []):
                        pid = _match_pid(oc.get("description", ""), name_by_pid)
                        if pid is None:
                            continue
                        odds_by_pid_market.setdefault((pid, stat, float(oc["point"])), {})[oc["name"].lower()] = oc["price"]
            time.sleep(0.2)
    except Exception as e:  # noqa: BLE001
        blockers["odds"] = str(e)

    atom_rows, price_rows, cov_rows = [], [], []
    for i, row in slate.iterrows():
        pid = int(row["player_id"]); gid = int(row["game_id"]); pa = float(p_active[i])
        pname = name_by_pid.get(pid, f"player_{pid}")
        mmean = mp[i].mean()
        for stat in C.TIER_A:
            pmf = stat_pmfs[stat][i]
            atoms = pmf.atoms
            pvar = float(np.dot((np.arange(atoms.size) - _mean(atoms)) ** 2, atoms))
            for k, prob in enumerate(atoms):
                if prob <= 1e-9:
                    continue
                atom_rows.append({"game_id": gid, "canonical_player_id": pid, "player_name": pname,
                    "team_id": int(row["team_id"]), "opponent_id": int(row["opponent_team_id"]),
                    "period": "FULL", "target": stat, "atom_value": int(k), "atom_probability": float(prob),
                    "overflow_probability": float(pmf.overflow), "p_active": pa,
                    "predictive_mean": _mean(atoms), "predictive_variance": pvar,
                    "expected_minutes": mmean, "prediction_timestamp": ts, "scheduled_tip": str(date),
                    "source_track": "PURE_PMF_V4", "model_status": "TRAINED_PURE_UNCERTIFIED",
                    "calibration_status": "PURE_UNCALIBRATED", "ood_status": "IN_SUPPORT",
                    "data_hash": data_hash, "feature_hash": "v4", "model_hash": "hgb_nb2_exacttail",
                    "calibrator_hash": "none", "design_hash": design_hash[:16], "code_sha": code_sha[:12],
                    "abstention_reason": ""})
            # fair prices both sides + market-consistent where a live no-vig quote matched
            mean_line = _mean(atoms)
            base = max(round(mean_line * 2) / 2 - 0.5, 0.5)
            for L in sorted({round(base + 0.5 * j, 1) for j in range(-2, 4) if base + 0.5 * j >= 0.5}):
                p_over = pmf.prob_over(L); p_under = pmf.prob_under(L); p_push = pmf.prob_push(L)
                den = max(p_over + p_under, 1e-9)
                mkt_status = "NO_MARKET"; mc_over = None; residual_status = "MARKET_CONSISTENT_ZERO_RESIDUAL"
                q = odds_by_pid_market.get((pid, stat, float(L)))
                if q and "over" in q and "under" in q:
                    nv = C.no_vig_over(q["over"], q["under"])
                    if np.isfinite(nv):
                        mc_atoms = C.market_consistent_atoms(atoms, L, nv)
                        mc_over = float(mc_atoms[np.arange(mc_atoms.size) > L].sum())
                        mkt_status = "MARKET_CONSISTENT"
                for side, pw in (("Over", p_over), ("Under", p_under)):
                    settled = (p_over / den) if side == "Over" else (p_under / den)
                    fd = 1.0 / min(max(settled, 1e-9), 1 - 1e-9)
                    price_rows.append({"game_id": gid, "player_id": pid, "player_name": pname, "target": stat,
                        "line": L, "side": side, "p_win": pw, "p_lose": (p_under if side == "Over" else p_over),
                        "p_push": p_push, "settled_probability": settled, "fair_decimal": fd,
                        "fair_american": _amer(fd), "market_consistent_over": mc_over,
                        "market_projection_status": mkt_status, "residual_status": residual_status,
                        "source_track": "PURE_PMF_V4", "p_active": pa,
                        "design_hash": design_hash[:16], "code_sha": code_sha[:12]})
            cov_rows.append({"game_id": gid, "player_id": pid, "target": stat, "status": "PRICED",
                             "has_live_market": any((pid, stat, float(x)) in odds_by_pid_market
                                                    for x in np.arange(0, 40, 0.5))})

    out = REPO / "deliveries" / "sharp_v4" / str(date) / "T-live"
    out.mkdir(parents=True, exist_ok=True)
    adf = pd.DataFrame(atom_rows); pdf = pd.DataFrame(price_rows); cdf = pd.DataFrame(cov_rows)
    adf.to_parquet(out / "active_atom_pmfs.parquet", index=False)
    pdf.to_parquet(out / "fair_prices.parquet", index=False)
    pdf.to_csv(out / "pricing_inventory.csv", index=False)
    cdf.to_csv(OUT / "REAL_SLATE_COVERAGE.csv", index=False)
    manifest = {"artifact": "pricing_manifest", "slate_date": date, "is_fixture": False,
                "source": "V4 fitted artifacts + real point-in-time features + live BDL schedule + live no-vig odds",
                "n_games": len(upcoming), "n_players": int(slate["player_id"].nunique()),
                "n_atoms": len(adf), "n_priced_lines": len(pdf),
                "live_market_matched": int(cdf["has_live_market"].sum()) if len(cdf) else 0,
                "credit_log": credit_log, "blockers": blockers, "generated_at_utc": ts,
                "design_hash": design_hash, "code_sha": code_sha}
    (out / "pricing_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUT / "STL_BLK_TOV_QUOTE_COVERAGE.json").write_text(json.dumps(
        {"artifact": "STL_BLK_TOV_QUOTE_COVERAGE", "generated_at_utc": ts,
         "note": "focused audit requires player_steals/blocks/turnovers markets on event-specific endpoint; "
                 "not requested in this run (Tier A pts/reb/ast/threes prioritized). Recorded as pending.",
         "status": "PENDING_FOCUSED_AUDIT"}, indent=2, default=str))
    typer.echo(f"LIVE SLATE {date}: games={len(upcoming)} players={slate['player_id'].nunique()} "
               f"atoms={len(adf)} lines={len(pdf)} live_market_players={int(cdf['has_live_market'].sum()) if len(cdf) else 0}")
    if blockers:
        typer.echo(f"  blockers: {blockers}")


def _match_pid(desc, name_by_pid):
    nd = _norm(desc)
    for pid, nm in name_by_pid.items():
        if _norm(nm) == nd:
            return pid
    return None


def _mean(atoms):
    return float(np.dot(np.arange(len(atoms)), atoms))


def _amer(d):
    d = float(d)
    if d <= 1:
        return float("nan")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def _write_blockers(blockers, ts):
    (OUT / "REAL_SLATE_BLOCKERS.json").write_text(json.dumps(
        {"artifact": "REAL_SLATE_BLOCKERS", "generated_at_utc": ts, "blockers": blockers}, indent=2))


if __name__ == "__main__":
    app()
