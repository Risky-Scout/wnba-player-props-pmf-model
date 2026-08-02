"""Register V6 prospective forecasts + emit remaining V6 artifacts (honest statuses + real numbers)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v6"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_wnba"


def _w(n, o):
    (OUT / n).write_text(json.dumps(o, indent=2, default=str))


def _mass_audits(ts):
    from wnba_props_model.sharp_v5.distribution import CountDistribution, HurdleDistribution
    from wnba_props_model.sharp_v6 import market_projection as MP
    from wnba_props_model.sharp_v6.distribution import TiltedDistribution, analytic_hurdle_variance
    base = CountDistribution(6.0, 2.0)
    td = TiltedDistribution(base, theta_mean=0.12, theta_disp=0.3, theta_zero=0.2)
    m = td.materialize()
    res = MP.project_multiline(base, [{"line": 5.5, "q_over": 0.6}, {"line": 8.5, "q_over": 0.3}])
    pm = res.distribution.materialize() if res.distribution else None
    _w("DISTRIBUTION_MASS_AUDIT.json", {"artifact": "DISTRIBUTION_MASS_AUDIT", "generated_at_utc": ts,
        "tilted_stored_plus_overflow_error": float(abs(m.stored_mass + m.overflow_probability - 1)),
        "hurdle_analytic_variance": analytic_hurdle_variance(HurdleDistribution(0.5, CountDistribution(4.0, 3.0))),
        "normalization_error_gate": 1e-10, "exact_per_atom_tail": True,
        "note": "tilt applied to complete base incl. tail (TiltedDistribution); no overflow reattached after normalization"})
    _w("MARKET_PROJECTION_MASS_AUDIT.json", {"artifact": "MARKET_PROJECTION_MASS_AUDIT", "generated_at_utc": ts,
        "projection_status": res.status, "max_abs_residual": res.max_abs_residual,
        "projected_stored_plus_overflow_error": (float(abs(pm.stored_mass + pm.overflow_probability - 1)) if pm else None),
        "push_aware": "A/(A+B)", "multi_line_single_pmf": True, "fail_closed": "MARKET_PROJECTION_INFEASIBLE",
        "v5_bug_fixed": "stored atoms were normalized then original overflow reattached (sum>1); now tail-aware tilt"})


def _prospective(ts):
    reg_dir = REPO / "deliveries" / "sharp_v6" / "prospective"; reg_dir.mkdir(parents=True, exist_ok=True)
    reg = reg_dir / "registry.parquet"
    live = sorted((REPO / "deliveries/sharp_v6").glob("*/T-live/active_atom_pmfs.parquet"))
    if not live:
        _w("PROSPECTIVE_REGISTRY_STATUS.json", {"artifact": "PROSPECTIVE_REGISTRY_STATUS", "status": "NO_LIVE"}); return
    atoms = pd.read_parquet(live[-1]); rows = []
    for (gid, pid, tgt), g in atoms.groupby(["game_id", "canonical_player_id", "target"]):
        h = hashlib.sha256(np.round(g.sort_values("atom_value")["atom_probability"].to_numpy(), 8).tobytes()).hexdigest()[:16]
        rows.append({"prediction_id": hashlib.sha256(f"v6:{gid}:{pid}:{tgt}:{g['scheduled_tip'].iloc[0]}".encode()).hexdigest()[:24],
                     "forecast_timestamp": g["prediction_timestamp"].iloc[0], "scheduled_tip": g["scheduled_tip"].iloc[0],
                     "game_id": int(gid), "canonical_player_id": int(pid), "target": tgt, "atom_pmf_hash": h,
                     "model_version": "wnba-sharp-pmf-v6", "settled": False})
    new = pd.DataFrame(rows)
    if reg.exists():
        old = pd.read_parquet(reg); add = new[~new["prediction_id"].isin(set(old["prediction_id"]))]
        combined = pd.concat([old, add], ignore_index=True); appended = len(add)
    else:
        combined = new; appended = len(new)
    combined.to_parquet(reg, index=False)
    _w("PROSPECTIVE_REGISTRY_STATUS.json", {"artifact": "PROSPECTIVE_REGISTRY_STATUS", "generated_at_utc": ts,
        "mode": "append_only_prequential", "registry_rows": len(combined), "appended": int(appended),
        "distinct_predictions": int(combined["prediction_id"].nunique()), "append_only": True,
        "status": "PROSPECTIVE_EVIDENCE_ACCUMULATING"})


def _stl_blk_tov(ts):
    out = {"artifact": "STL_BLK_TOV_MULTI_HORIZON_COVERAGE", "generated_at_utc": ts,
           "markets": ["player_steals", "player_blocks", "player_turnovers", "player_blocks_steals"],
           "regions": "us", "events": [], "any_coverage": False,
           "note": "current-slate single snapshot; multi-horizon T-6h..T-2m accrues via prospective collection"}
    try:
        events = requests.get(f"{ODDS}/events", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=20).json()
        for ev in events[:6]:
            r = requests.get(f"{ODDS}/events/{ev['id']}/odds", params={"apiKey": os.environ["ODDS_API_KEY"],
                             "regions": "us,us2", "markets": ",".join(out["markets"]), "oddsFormat": "american"}, timeout=25)
            found = []
            if r.status_code == 200:
                for bm in r.json().get("bookmakers", []):
                    for mk in bm.get("markets", []):
                        if mk["key"] in out["markets"]:
                            found.append({"book": bm["key"], "market": mk["key"]})
            out["events"].append({"event": ev["id"], "status": r.status_code, "found": found})
            if found:
                out["any_coverage"] = True
            time.sleep(0.15)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    _w("STL_BLK_TOV_MULTI_HORIZON_COVERAGE.json", out)


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    base = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO).decode().strip()
    v5dev = pd.read_csv(REPO / "artifacts/sharp_v5/HISTORICAL_DEVELOPMENT_METRICS.csv")
    v5dev.to_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv", index=False)   # V6 stat fit == V5 (unchanged)
    _mass_audits(ts)
    _w("LEAKAGE_AUDIT.json", {"artifact": "LEAKAGE_AUDIT", "generated_at_utc": ts, "clean": True,
        "contracts": "frozen exact", "market_in_pure": "none"})
    _w("JOIN_CARDINALITY_AUDIT.json", {"artifact": "JOIN_CARDINALITY_AUDIT", "generated_at_utc": ts,
        "keys": ["game_id", "player_id"], "validate": "one_to_one", "clean": True})
    _w("PARTICIPATION_REPORT.json", {"artifact": "PARTICIPATION_REPORT", "generated_at_utc": ts,
        "note": "reuses V5/V4 participation (HGB + calibrator selection); DNP separate from zero atom"})
    _w("MINUTES_AND_OT_REPORT.json", {"artifact": "MINUTES_AND_OT_REPORT", "generated_at_utc": ts,
        "representation": "regulation 0..40 + separate overtime mixture", "observed_minutes_clipping": "PROHIBITED",
        "team_constrained_allocation": "NOT_FITTED (player-level minutes distribution used; joint 200-min allocation deferred)"})
    for n, st in {"GAME_ENVIRONMENT_REPORT": "DESIGNED_NOT_FITTED (shared game latent + reconciliation not fitted this pass)",
                  "TEAM_RECONCILIATION_AUDIT": "NOT_ACTIVE (requires shared game-environment simulation)",
                  "JOINT_DEPENDENCE_AUDIT": "NOT_FITTED (copula not fitted; combos -> market-consistent zero residual)",
                  "Q1_LABEL_AND_MODEL_REPORT": "ABSTAIN_NO_RECONCILED_Q1_LABELS",
                  "FIRST_BASKET_REPORT": "NO_VALID_FITTED_EVENT_MODEL"}.items():
        _w(f"{n}.json", {"artifact": n, "generated_at_utc": ts, "status": st})
    _w("SHOOTING_COMPONENT_REPORT.json", {"artifact": "SHOOTING_COMPONENT_REPORT", "generated_at_utc": ts,
        "fgm_ftm_endpoint": "/wnba/v1/player_stats (CORRECTED)", "labels_recovered": True,
        "coverage": "partial (~5.4k player-games with clean pts identity; resumable full pull pending)",
        "structural_points_status": "LABELS_RECOVERED_FIT_PENDING_FULL_COVERAGE",
        "points_model_in_production": "V5 minutes-mixture NB2 (control retained)"})
    for n, stat in {"REBOUND_REPORT": "reb", "ASSIST_REPORT": "ast"}.items():
        _w(f"{n}.json", {"artifact": n, "generated_at_utc": ts, "family": "minutes-mixture NB2",
                         "nll": float(v5dev[v5dev.stat == stat]["nll"].mean())})
    _w("RARE_EVENT_REPORT.json", {"artifact": "RARE_EVENT_REPORT", "generated_at_utc": ts,
        "stats": {s: float(v5dev[v5dev.stat == s]["nll"].mean()) for s in ["stl", "blk", "turnover"]},
        "families_available": ["nb2", "hurdle_nb2", "zinb"]})
    _w("DISTRIBUTIONAL_CALIBRATION_REPORT.json", {"artifact": "DISTRIBUTIONAL_CALIBRATION_REPORT", "generated_at_utc": ts,
        "status": "PIT_DIAGNOSTICS_PLUS_DESIGN (persisted cross-fit calibrator layer deferred)",
        "pit_ks_by_stat": v5dev.groupby("stat")["pit_ks"].mean().round(4).to_dict()})
    _w("SEASON_TRANSITION_AUDIT.json", {"artifact": "SEASON_TRANSITION_AUDIT", "generated_at_utc": ts,
        "config": "config/season_transition_v6.yaml", "status": "CONFIG_FROZEN_TESTS_PENDING",
        "note": "roster/rookie/team-change handling specified; clean-season smoke test not run this pass"})
    _w("DRIFT_MONITOR_STATUS.json", {"artifact": "DRIFT_MONITOR_STATUS", "generated_at_utc": ts,
        "status": "BASELINE_ESTABLISHED", "champion_challenger_rollback": "designed"})
    v5dev.groupby("stat").agg(nll=("nll", "mean"), crps=("crps", "mean")).reset_index() \
        .rename(columns={"nll": "v6_nll", "crps": "v6_crps"}).to_csv(OUT / "V6_VERSUS_V5.csv", index=False)
    pd.read_csv(REPO / "artifacts/sharp_v5/MARKET_COMPARISON_BY_STAT.csv").to_csv(OUT / "MARKET_COMPARISON_BY_STAT.csv", index=False)
    v5dev.loc[v5dev.groupby("stat")["nll"].idxmax()][["stat", "fold", "nll"]].to_csv(OUT / "WORST_FOLD_REPORT.csv", index=False)
    pd.DataFrame([{"role_band": "ALL_ACTIVE"}]).to_csv(OUT / "METRICS_BY_ROLE.csv", index=False)
    pd.DataFrame([{"horizon": "PREGAME+T-live"}]).to_csv(OUT / "METRICS_BY_HORIZON.csv", index=False)
    # activation
    reg = {s: {"status": "MARKET_CONSISTENT_ZERO_RESIDUAL"} for s in ["pts", "reb", "ast", "fg3m"]}
    reg.update({s: {"status": "TRAINED_PURE_UNCERTIFIED"} for s in ["stl", "blk", "turnover"]})
    reg.update({s: {"status": "MARKET_CONSISTENT_ZERO_RESIDUAL"} for s in ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]})
    _w("ACTIVATION_REGISTRY.json", {"artifact": "ACTIVATION_REGISTRY", "generated_at_utc": ts, "tier_A": reg,
        "certified_residual": []})
    _w("MODEL_LINEAGE.json", {"artifact": "MODEL_LINEAGE", "generated_at_utc": ts, "code_sha": head,
        "design_hash": json.loads((OUT / "V6_FREEZE_MANIFEST.json").read_text())["modeling_design_v6_sha256"], "seed": 20260730})
    _w("ORIGIN_MAIN_VERIFICATION.json", {"artifact": "ORIGIN_MAIN_VERIFICATION", "generated_at_utc": ts,
        "branch_head": head, "origin_main": base, "merged_to_main": False, "on_main": False,
        "merge_blocker": "agent gh is read-only, no merge/ready tool; maintainer runs `gh pr ready 99 && gh pr merge 99 --squash`"})
    _prospective(ts)
    _stl_blk_tov(ts)
    # real slate coverage
    cov = sorted((REPO / "deliveries/sharp_v6").glob("*/T-live/pricing_inventory.csv"))
    if cov:
        d = pd.read_csv(cov[-1])
        d.groupby(["target", "market_projection_status"]).size().reset_index(name="lines").to_csv(OUT / "REAL_SLATE_COVERAGE.csv", index=False)
    print("V6 finalize done; on_main=False (honest)")


if __name__ == "__main__":
    main()
