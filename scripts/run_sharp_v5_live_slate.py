"""DEPRECATED / LEGACY_CONTROL — production uses scripts/run_wnba_pmf.py → sharp_v6.predict_slate.

Research-only V5 live runner. Pass --allow-research to execute.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v4 import core as V4C
from wnba_props_model.sharp_v5 import market_projection as MP
from wnba_props_model.sharp_v5 import models as M5

app = typer.Typer(add_completion=False)
REPO = V4C.load_verified.__globals__["REPO"]
OUT = REPO / "artifacts" / "sharp_v5"
BDL = "https://api.balldontlie.io/wnba/v1"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_wnba"
MARKET_MAP = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


@app.command()
def main(
    date: str = typer.Option(None, "--date"),
    allow_research: bool = typer.Option(False, "--allow-research"),
) -> None:
    if not allow_research:
        raise SystemExit(
            "DEPRECATED: production inference is scripts/run_wnba_pmf.py → "
            "sharp_v6.predict_slate. Pass --allow-research for LEGACY_CONTROL only."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    _, df = V4C.load_verified()
    stats = pd.read_parquet(REPO / "data/recovered_v2/wnba_player_game_stats.parquet")
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    teams = pd.read_parquet(REPO / "data/recovered_v2/wnba_teams.parquet")
    abbr2id = dict(zip(teams["team_abbreviation"], teams["team_id"]))
    name_by_pid = stats.sort_values("game_date").groupby("player_id")["player_name"].last().to_dict()
    design_hash = json.loads((OUT / "V5_FREEZE_MANIFEST.json").read_text())["modeling_design_v5_sha256"]
    code_sha = os.popen("git rev-parse HEAD").read().strip()
    blockers = {}

    # pick next upcoming date if not given
    try:
        g = requests.get(f"{BDL}/games", headers={"Authorization": os.environ["BDL_API_KEY"]},
                         params={"seasons[]": 2026, "start_date": date or "2026-07-30",
                                 "end_date": "2026-08-05", "per_page": 100}, timeout=20).json().get("data", [])
    except Exception as e:  # noqa: BLE001
        g = []; blockers["bdl"] = str(e)
    upcoming = [x for x in g if x.get("status") not in ("post", "final", "Final")]
    if not upcoming:
        blockers["slate"] = "no upcoming games"; _blk(blockers, ts); typer.echo("no upcoming slate"); return
    slate_date = min({x["date"][:10] for x in upcoming})
    games = [x for x in upcoming if x["date"][:10] == slate_date]

    latest_feat = df.sort_values("game_date").groupby("player_id").tail(1).set_index("player_id")
    last_team = stats[stats["game_date"] < pd.Timestamp(slate_date)].sort_values("game_date") \
        .groupby("player_id")["team_abbreviation"].last().to_dict()
    train = df.copy()
    minutes_contract = V4C.resolve_contract("minutes", list(df.columns))

    slate_rows = []
    for gm in games:
        ha, aa = gm["home_team"]["abbreviation"], gm["visitor_team"]["abbreviation"]
        for team, opp in ((ha, aa), (aa, ha)):
            for pid in [p for p, t in last_team.items() if t == team and p in latest_feat.index]:
                row = latest_feat.loc[pid].copy()
                row["game_id"] = gm["id"]; row["team_id"] = abbr2id.get(team, row.get("team_id"))
                row["opponent_team_id"] = abbr2id.get(opp, row.get("opponent_team_id")); row["player_id"] = pid
                slate_rows.append(row)
    slate = pd.DataFrame(slate_rows).reset_index(drop=True)

    matoms, _, _ = M5.minutes_pmf_rows(train, slate, minutes_contract)
    stat_dists = {}
    for stat in V4C.TIER_A:
        lam, r_rows, _, _, _ = M5.stat_mixture_rows(train, slate, stat, matoms)
        stat_dists[stat] = [(lam[i], None if np.isnan(r_rows[i]) else float(r_rows[i])) for i in range(len(lam))]

    # live odds -> per (pid, stat) multi-line constraints
    credit = []
    quotes = {}
    try:
        events = requests.get(f"{ODDS}/events", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=20).json()
        for ev in events:
            r = requests.get(f"{ODDS}/events/{ev['id']}/odds",
                             params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us",
                                     "markets": ",".join(MARKET_MAP), "oddsFormat": "american"}, timeout=25)
            credit.append({"event": ev["id"], "status": r.status_code, "remaining": r.headers.get("x-requests-remaining")})
            if r.status_code != 200:
                continue
            for bm in r.json().get("bookmakers", []):
                for mk in bm.get("markets", []):
                    stat = MARKET_MAP.get(mk["key"])
                    if not stat:
                        continue
                    byline = {}
                    for oc in mk.get("outcomes", []):
                        pid = _match(oc.get("description", ""), name_by_pid)
                        if pid is None:
                            continue
                        byline.setdefault((pid, float(oc["point"])), {})[oc["name"].lower()] = oc["price"]
                    for (pid, line), od in byline.items():
                        if "over" in od and "under" in od:
                            nv = MP.no_vig_settled_over(od["over"], od["under"])
                            if np.isfinite(nv):
                                quotes.setdefault((pid, stat), []).append({"line": line, "q_over": nv, "weight": 1.0})
            time.sleep(0.15)
    except Exception as e:  # noqa: BLE001
        blockers["odds"] = str(e)

    atom_rows, price_rows, cov_rows = [], [], []
    n_proj, n_infeasible = 0, 0
    for i, row in slate.iterrows():
        pid = int(row["player_id"]); gid = int(row["game_id"])
        pname = name_by_pid.get(pid, f"player_{pid}")
        for stat in V4C.TIER_A:
            lam_i, r = stat_dists[stat][i]
            dist = M5.build_mixture(lam_i, r, matoms[i])
            mat = dist.materialize(required_max=V4C.EMERGENCY_CAP[stat])
            for k, prob in enumerate(mat.atoms):
                if prob <= 1e-9:
                    continue
                atom_rows.append({"game_id": gid, "canonical_player_id": pid, "player_name": pname,
                    "team_id": int(row["team_id"]), "opponent_id": int(row["opponent_team_id"]),
                    "period": "FULL", "target": stat, "atom_value": int(k), "atom_probability": float(prob),
                    "overflow_probability": float(mat.overflow_probability),
                    "predictive_mean": dist.mean(), "predictive_variance": dist.variance(),
                    "prediction_timestamp": ts, "scheduled_tip": slate_date, "source_track": "PURE_PMF_V5",
                    "model_status": "TRAINED_PURE_UNCERTIFIED", "calibration_status": "PURE_UNCALIBRATED",
                    "tail_method": mat.tail_method, "design_hash": design_hash[:16], "code_sha": code_sha[:12]})
            # market-consistent multi-line projection (push-aware)
            mc_dist = None; proj_status = "NO_MARKET"
            cons = quotes.get((pid, stat))
            if cons and stat in MARKET_MAP.values():
                res = MP.project_multiline(dist, cons, required_max=V4C.EMERGENCY_CAP[stat])
                proj_status = res.status
                if res.feasible:
                    mc_dist = res.distribution; n_proj += 1
                else:
                    n_infeasible += 1
            # fair prices both sides + market-consistent settled
            mean = dist.mean()
            base = max(round(mean * 2) / 2 - 0.5, 0.5)
            for L in sorted({round(base + 0.5 * j, 1) for j in range(-2, 4) if base + 0.5 * j >= 0.5}):
                s = dist.settle_over_under(L)
                mc_over = mc_dist.settle_over_under(L).p_over_settled if mc_dist is not None else None
                for side, pw, ps in (("Over", s.p_over_win, s.p_over_settled), ("Under", s.p_under_win, s.p_under_settled)):
                    fd = 1.0 / min(max(ps, 1e-9), 1 - 1e-9)
                    price_rows.append({"game_id": gid, "player_id": pid, "player_name": pname, "target": stat,
                        "line": L, "side": side, "p_win": pw, "p_push": s.p_push, "settled_probability": ps,
                        "fair_decimal": fd, "fair_american": _amer(fd),
                        "market_consistent_over": mc_over, "market_projection_status": proj_status,
                        "residual_status": "MARKET_CONSISTENT_ZERO_RESIDUAL", "source_track": "PURE_PMF_V5"})
            cov_rows.append({"game_id": gid, "player_id": pid, "target": stat, "status": "PRICED",
                             "market_projection_status": proj_status})

    out = REPO / "deliveries" / "sharp_v5" / slate_date / "T-live"
    out.mkdir(parents=True, exist_ok=True)
    adf = pd.DataFrame(atom_rows); pdf = pd.DataFrame(price_rows); cdf = pd.DataFrame(cov_rows)
    adf.to_parquet(out / "active_atom_pmfs.parquet", index=False)
    pdf.to_parquet(out / "fair_prices.parquet", index=False)
    pdf.to_csv(out / "pricing_inventory.csv", index=False)
    cdf.to_csv(OUT / "REAL_SLATE_COVERAGE.csv", index=False)
    (out / "pricing_manifest.json").write_text(json.dumps({
        "artifact": "pricing_manifest", "slate_date": slate_date, "is_fixture": False,
        "source": "V5 minutes-mixture PMFs + real point-in-time features + live BDL schedule + live no-vig odds",
        "n_games": len(games), "n_players": int(slate["player_id"].nunique()), "n_atoms": len(adf),
        "n_priced_lines": len(pdf), "market_projections_feasible": n_proj,
        "market_projections_infeasible": n_infeasible, "credit_log": credit, "blockers": blockers,
        "generated_at_utc": ts, "design_hash": design_hash, "code_sha": code_sha}, indent=2, default=str))
    typer.echo(f"V5 LIVE SLATE {slate_date}: games={len(games)} players={slate['player_id'].nunique()} "
               f"atoms={len(adf)} lines={len(pdf)} market_projected={n_proj} infeasible={n_infeasible}")


def _match(desc, name_by_pid):
    nd = _norm(desc)
    for pid, nm in name_by_pid.items():
        if _norm(nm) == nd:
            return pid
    return None


def _amer(d):
    d = float(d)
    return float("nan") if d <= 1 else (round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1)))


def _blk(b, ts):
    (OUT / "REAL_SLATE_COVERAGE.csv").write_text("game_id,player_id,target,status\n")
    (OUT / "MODEL_LINEAGE.json").write_text(json.dumps({"blockers": b, "generated_at_utc": ts}, indent=2))


if __name__ == "__main__":
    app()
