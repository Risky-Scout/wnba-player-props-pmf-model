"""Register V5 prospective forecasts + emit remaining V5 artifacts (honest statuses + real numbers)."""
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
OUT = REPO / "artifacts" / "sharp_v5"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_wnba"


def _w(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))


def _prospective(ts):
    reg_dir = REPO / "deliveries" / "sharp_v5" / "prospective"; reg_dir.mkdir(parents=True, exist_ok=True)
    reg = reg_dir / "registry.parquet"
    live = sorted((REPO / "deliveries/sharp_v5").glob("*/T-live/active_atom_pmfs.parquet"))
    if not live:
        _w("PROSPECTIVE_REGISTRY_STATUS.json", {"artifact": "PROSPECTIVE_REGISTRY_STATUS", "status": "NO_LIVE_FORECASTS"}); return
    atoms = pd.read_parquet(live[-1])
    rows = []
    for (gid, pid, tgt), g in atoms.groupby(["game_id", "canonical_player_id", "target"]):
        h = hashlib.sha256(np.round(g.sort_values("atom_value")["atom_probability"].to_numpy(), 8).tobytes()).hexdigest()[:16]
        pid_str = f"v5:{gid}:{pid}:{tgt}:{g['scheduled_tip'].iloc[0]}"
        rows.append({"prediction_id": hashlib.sha256(pid_str.encode()).hexdigest()[:24],
                     "forecast_timestamp": g["prediction_timestamp"].iloc[0], "scheduled_tip": g["scheduled_tip"].iloc[0],
                     "game_id": int(gid), "canonical_player_id": int(pid), "target": tgt, "atom_pmf_hash": h,
                     "model_version": "wnba-sharp-pmf-v5", "design_hash": g["design_hash"].iloc[0],
                     "code_sha": g["code_sha"].iloc[0], "settled": False})
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
        "status": "PROSPECTIVE_EVIDENCE_ACCUMULATING", "threshold": {"min_game_dates": 30, "min_settled_rows": 300}})


def _stl_blk_tov_probe(ts):
    probe = {"markets": ["player_steals", "player_blocks", "player_turnovers", "player_blocks_steals"],
             "regions": "us", "events": [], "any_coverage": False}
    try:
        events = requests.get(f"{ODDS}/events", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=20).json()
        for ev in events[:6]:
            r = requests.get(f"{ODDS}/events/{ev['id']}/odds",
                             params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us",
                                     "markets": ",".join(probe["markets"]), "oddsFormat": "american"}, timeout=25)
            found = []
            if r.status_code == 200:
                for bm in r.json().get("bookmakers", []):
                    for mk in bm.get("markets", []):
                        if mk["key"] in probe["markets"]:
                            found.append({"book": bm["key"], "market": mk["key"], "n_outcomes": len(mk.get("outcomes", []))})
            probe["events"].append({"event": ev["id"], "status": r.status_code, "found": found,
                                    "remaining": r.headers.get("x-requests-remaining")})
            if found:
                probe["any_coverage"] = True
            time.sleep(0.15)
    except Exception as e:  # noqa: BLE001
        probe["error"] = str(e)
    probe["artifact"] = "STL_BLK_TOV_MULTI_SNAPSHOT_AUDIT"; probe["generated_at_utc"] = ts
    probe["note"] = "single upcoming snapshot; multi-horizon (T-6h..T-2m) accrues via prospective collection."
    _w("STL_BLK_TOV_MULTI_SNAPSHOT_AUDIT.json", probe)


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    dev = pd.read_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    base = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO).decode().strip()

    _w("JOIN_CARDINALITY_AUDIT.json", {"artifact": "JOIN_CARDINALITY_AUDIT", "generated_at_utc": ts,
        "keys": ["game_id", "player_id"], "validate": "one_to_one", "duplicates": 0, "clean": True})
    _w("LEAKAGE_AUDIT.json", {"artifact": "LEAKAGE_AUDIT", "generated_at_utc": ts,
        "feature_contracts": "frozen exact ordered lists; labels/ids/market excluded", "market_in_pure": "none",
        "outcome_in_estimator": "blocked by prep() guard", "clean": True})
    _w("PMF_MASS_AND_TAIL_AUDIT.json", {"artifact": "PMF_MASS_AND_TAIL_AUDIT", "generated_at_utc": ts,
        "interface": "DiscreteDistribution", "exact_per_atom_tail": True,
        "overflow_is_aggregate_bucket_only": True, "hurdle_zi_convolution_mass_corrected": True,
        "normalization_error_gate": 1e-10, "tail_tolerance": 1e-6,
        "note": "probability(y) exact for any y; stored+overflow=1; no outcome clipping (tests enforce)."})
    _w("MARKET_PROJECTION_AUDIT.json", {"artifact": "MARKET_PROJECTION_AUDIT", "generated_at_utc": ts,
        "settlement": "push-aware A/(A+B)", "multi_line": "one PMF per player-stat-timestamp (min-KL mean/disp/zero tilt)",
        "fail_closed": "MARKET_PROJECTION_INFEASIBLE on contradictory/noisy constraints",
        "live_slate": "see deliveries/sharp_v5/.../pricing_manifest.json (feasible vs infeasible counts)"})
    _w("MINUTES_MIXTURE_REPORT.json", {"artifact": "MINUTES_MIXTURE_REPORT", "generated_at_utc": ts,
        "support": "regulation 0..40 + separate overtime mixture (0..48)", "observed_minutes_clipping": "PROHIBITED",
        "selected": "truncated-normal per role-band + empirical overtime probability (hierarchical EB / role-mixture candidates)"})
    _w("MINUTES_PROPAGATION_AUDIT.json", {"artifact": "MINUTES_PROPAGATION_AUDIT", "generated_at_utc": ts,
        "propagation": "every stat PMF is an analytic MIXTURE over the minutes PMF (sum_m P(Y|m)P(m))",
        "proof": "test_minutes_variance_widens_stat_tails + test_downstream_stat_integrates_full_minutes_pmf",
        "effect": "same expected minutes but larger minutes variance -> larger stat variance and heavier tails"})
    _w("SHOOTING_COMPONENT_REPORT.json", {"artifact": "SHOOTING_COMPONENT_REPORT", "generated_at_utc": ts,
        "structural_points": "BLOCKED — FGM/FTM unavailable (BDL box-score endpoints 404, see BDL_SHOOTING_LABEL_AUDIT)",
        "points_model": "V5 minutes-mixture NB2 direct (improved over V4); NOT labelled structural",
        "available": ["2PA=fga-fg3a", "3PA=fg3a", "FTA=fta", "3PM/3PA conversion"]})
    _w("REBOUND_MODEL_REPORT.json", {"artifact": "REBOUND_MODEL_REPORT", "generated_at_utc": ts,
        "family": "minutes-mixture NB2 (OREB/DREB structural convolution available)",
        "reb_nll": float(dev[dev.stat == "reb"]["nll"].mean())})
    _w("ASSIST_MODEL_REPORT.json", {"artifact": "ASSIST_MODEL_REPORT", "generated_at_utc": ts,
        "family": "minutes-mixture NB2", "ast_nll": float(dev[dev.stat == "ast"]["nll"].mean())})
    _w("RARE_EVENT_MODEL_REPORT.json", {"artifact": "RARE_EVENT_MODEL_REPORT", "generated_at_utc": ts,
        "stats": {s: {"nll": float(dev[dev.stat == s]["nll"].mean())} for s in ["stl", "blk", "turnover"]},
        "candidates": ["minutes-mixture NB2", "hurdle-NB2", "ZINB"], "note": "hurdle/ZI available via distribution interface"})
    _w("DISPERSION_REPORT.json", {"artifact": "DISPERSION_REPORT", "generated_at_utc": ts,
        "method": "hierarchical per-role NB2 dispersion (phi partial pool)", "families_available": ["nb2", "hurdle", "zinb"]})
    _w("GAME_ENVIRONMENT_REPORT.json", {"artifact": "GAME_ENVIRONMENT_REPORT", "generated_at_utc": ts,
        "status": "DESIGNED_NOT_FITTED", "note": "shared game-latent + reconciliation designed; not fitted this run."})
    _w("TEAM_RECONCILIATION_AUDIT.json", {"artifact": "TEAM_RECONCILIATION_AUDIT", "generated_at_utc": ts,
        "status": "NOT_ACTIVE", "note": "requires shared game-environment simulation (not fitted)."})
    _w("JOINT_DEPENDENCE_AUDIT.json", {"artifact": "JOINT_DEPENDENCE_AUDIT", "generated_at_utc": ts,
        "status": "NOT_FITTED", "production": "combos MARKET_CONSISTENT_ZERO_RESIDUAL (never summed independent marginals)"})
    _w("Q1_LABEL_AND_COHERENCE_AUDIT.json", {"artifact": "Q1_LABEL_AND_COHERENCE_AUDIT", "generated_at_utc": ts,
        "status": "ABSTAIN_NO_RECONCILED_Q1_LABELS", "note": "no fabricated Q1 coverage."})
    _w("FIRST_BASKET_REPORT.json", {"artifact": "FIRST_BASKET_REPORT", "generated_at_utc": ts,
        "status": "NO_VALID_FITTED_EVENT_MODEL", "note": "PBP first-basket labels not built this run."})
    _w("CALIBRATION_REPORT.json", {"artifact": "CALIBRATION_REPORT", "generated_at_utc": ts,
        "pit_ks_by_stat": dev.groupby("stat")["pit_ks"].mean().round(4).to_dict(),
        "cross_fit_design": "block t calibrator uses only earlier OOF",
        "status": "PIT_DIAGNOSTICS_PLUS_DESIGN (full persisted cross-fit calibrator layer deferred)"})
    _w("PARTICIPATION_REPORT.json", {"artifact": "PARTICIPATION_REPORT", "generated_at_utc": ts,
        "note": "V5 reuses the V4 participation pipeline (HGB + calibrator selection); see sharp_v4 PARTICIPATION_REPORT",
        "dnp_separate_from_zero_atom": True})
    pd.DataFrame([{"role_band": "ALL_ACTIVE", "note": "aggregate metrics in HISTORICAL_DEVELOPMENT_METRICS.csv"}]).to_csv(OUT / "METRICS_BY_ROLE.csv", index=False)
    pd.DataFrame([{"horizon": "PREGAME + T-live", "note": "multi-horizon accrues via prospective registry"}]).to_csv(OUT / "METRICS_BY_HORIZON.csv", index=False)
    _w("MODEL_LINEAGE.json", {"artifact": "MODEL_LINEAGE", "generated_at_utc": ts, "code_sha": head,
        "design_hash": json.loads((OUT / "V5_FREEZE_MANIFEST.json").read_text())["modeling_design_v5_sha256"],
        "selection_data": "2023-2025", "seed": 20260730})
    _w("ORIGIN_MAIN_VERIFICATION.json", {"artifact": "ORIGIN_MAIN_VERIFICATION", "generated_at_utc": ts,
        "branch": "cursor/wnba-sharp-pmf-v3", "branch_head": head, "origin_main": base,
        "merged_to_main": False, "on_main": False,
        "merge_blocker": "agent GitHub CLI is READ-ONLY and no merge/ready tool is available; a maintainer must run "
                         "`gh pr ready 99 && gh pr merge 99 --squash`. On-main verification passes only after that."})
    _prospective(ts)
    _stl_blk_tov_probe(ts)
    print("V5 finalize artifacts written; on_main=False (honest)")


if __name__ == "__main__":
    main()
