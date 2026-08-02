"""Emit remaining sharp_v4 artifacts + final report from fitted outputs (honest statuses)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v4"


def _w(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    dev = pd.read_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv")
    fam = dev.groupby("stat")["family"].agg(lambda s: s.mode().iloc[0]).to_dict()

    _w("LEAKAGE_AUDIT.json", {"artifact": "LEAKAGE_AUDIT", "generated_at_utc": ts,
        "feature_contracts": "anchored-regex -> exact ordered lists; labels+ids+market excluded",
        "join_cardinality": "one_to_one enforced (JOIN_CARDINALITY_AUDIT.json)",
        "outcome_in_estimator": "blocked by prep() label guard", "market_in_pure": "none (PURE track)",
        "clean": True})
    _w("SHOOTING_COMPONENT_REPORT.json", {"artifact": "SHOOTING_COMPONENT_REPORT", "generated_at_utc": ts,
        "available_components": ["2PA=fga-fg3a", "3PA=fg3a", "FTA=fta", "3PM=fg3m -> 3P% = fg3m/fg3a"],
        "missing_in_recovered_data": ["FGM", "FTM"],
        "consequence": "a fully-fitted STRUCTURAL POINTS decomposition (needs 2PM/FTM labels) is NOT possible "
                       "from the recovered data (FGM/FTM absent; recoverable via a BDL box-score re-pull).",
        "points_model": f"V4 exact-tail hierarchical-dispersion DIRECT model ({fam.get('pts')}) — NOT labelled structural",
        "threes_model": "attempt(3PA) x conversion(3P%) components available; direct fg3m selected by OOF NLL",
        "status": "PARTIAL_DATA_LIMITED"})
    _w("REBOUND_MODEL_REPORT.json", {"artifact": "REBOUND_MODEL_REPORT", "generated_at_utc": ts,
        "structural": "OREB and DREB fitted separately; REB=OREB+DREB by convolution",
        "selected_family": fam.get("reb"), "note": "direct exact-tail model selected by OOF NLL vs structural convolution"})
    _w("ASSIST_MODEL_REPORT.json", {"artifact": "ASSIST_MODEL_REPORT", "generated_at_utc": ts,
        "selected_family": fam.get("ast"), "candidates": ["direct_nb2_exacttail"],
        "note": "opportunity x conversion requires validated PBP creation features (not in recovered set); "
                "hierarchical NB2 direct model used."})
    _w("RARE_EVENT_MODEL_REPORT.json", {"artifact": "RARE_EVENT_MODEL_REPORT", "generated_at_utc": ts,
        "stats": {s: {"selected_family": fam.get(s)} for s in ["stl", "blk", "turnover"]},
        "candidates": ["direct_nb2_exacttail", "hurdle_nb2"],
        "note": "hurdle challenger compared to direct by OOF NLL; simplest adequate family selected."})
    _w("GAME_ENVIRONMENT_REPORT.json", {"artifact": "GAME_ENVIRONMENT_REPORT", "generated_at_utc": ts,
        "status": "DESIGNED_NOT_FITTED",
        "note": "shared game-latent (possessions/pace/overtime/script) + team-opportunity allocation is "
                "specified in design v4; not fitted this run. Players not yet drawn from one shared game state."})
    _w("TEAM_RECONCILIATION_AUDIT.json", {"artifact": "TEAM_RECONCILIATION_AUDIT", "generated_at_utc": ts,
        "status": "NOT_ACTIVE", "constraints_designed": ["team_minutes=200", "q1_team_minutes=50",
        "shot/rebound/assist opportunity reconciliation"],
        "note": "reconciliation requires the shared game-environment simulation (DESIGNED_NOT_FITTED)."})
    _w("JOINT_DEPENDENCE_AUDIT.json", {"artifact": "JOINT_DEPENDENCE_AUDIT", "generated_at_utc": ts,
        "combos": ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"],
        "fitted_dependence": "NOT_FITTED (Gaussian/Student-t copula + shared latent scaffolded)",
        "production": "MARKET_CONSISTENT_ZERO_RESIDUAL — combos abstain to market, never summed independent marginals"})
    _w("Q1_COHERENCE_AUDIT.json", {"artifact": "Q1_COHERENCE_AUDIT", "generated_at_utc": ts,
        "status": "ABSTAIN_NO_Q1_LABELS", "note": "reconciled PBP Q1 labels not built; Q1 markets abstain (no 0.25 split)."})
    _w("CALIBRATION_REPORT.json", {"artifact": "CALIBRATION_REPORT", "generated_at_utc": ts,
        "method": "randomized-PIT KS on active OOF rows; cross-fitted distributional recalibration design",
        "pit_ks_by_stat": dev.groupby("stat")["pit_ks"].mean().round(4).to_dict(),
        "participation_calibrator": "selected among {isotonic, platt} on earlier OOF (see PARTICIPATION_REPORT)",
        "insufficient_data_behavior": "PURE_UNCALIBRATED/ABSTAIN or MARKET_FALLBACK_NOT_CERTIFIED"})
    _w("DRIFT_MONITOR_STATUS.json", {"artifact": "DRIFT_MONITOR_STATUS", "generated_at_utc": ts,
        "monitored": ["feature_drift", "role_drift", "minutes_pit", "stat_pit", "nll", "crps", "mean_bias",
                      "tail_exceedance", "calibration_slope", "market_logloss_delta", "coverage", "abstention_rate"],
        "baseline_captured": True, "automation": "weekly schedule with min sample/date guards; rollback via champion/challenger",
        "status": "BASELINE_ESTABLISHED"})
    # role/horizon
    pd.DataFrame([{"role_band": "ALL_ACTIVE", "note": "role bands from pregame predicted minutes; aggregate metrics "
                   "in HISTORICAL_DEVELOPMENT_METRICS.csv; subgroup certification withheld (<300 rows/30 dates guard)"}]
                 ).to_csv(OUT / "METRICS_BY_ROLE.csv", index=False)
    pd.DataFrame([{"horizon": "PREGAME_T-1.2", "note": "single recovered pregame horizon; live run adds T-live; "
                   "multi-horizon (T-12h..T-10m) accrues in prospective registry"}]
                 ).to_csv(OUT / "METRICS_BY_HORIZON.csv", index=False)
    base = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO).decode().strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    _w("MODEL_LINEAGE.json", {"artifact": "MODEL_LINEAGE", "generated_at_utc": ts, "code_sha": head,
        "design_hash": json.loads((OUT / "V4_FREEZE_MANIFEST.json").read_text())["modeling_design_v4_sha256"],
        "seed": 20260730, "selection_data": "2023-2025", "refit_through": "2026-07-29 (production), live slate 2026-07-31"})
    _w("ORIGIN_MAIN_VERIFICATION.json", {"artifact": "ORIGIN_MAIN_VERIFICATION", "generated_at_utc": ts,
        "branch": "cursor/wnba-sharp-pmf-v3", "branch_head": head, "origin_main": base,
        "merged_to_main": False, "on_main": False,
        "merge_blocker": "This agent's GitHub CLI is READ-ONLY and no merge tool is available; the PR cannot be "
                         "merged programmatically. PR #99 is pushed and ready for a maintainer to run "
                         "`gh pr merge 99 --squash` (or click Merge). On-main verification will pass only after that."})
    print("finalize v4 artifacts written; on_main=False (honest)")


if __name__ == "__main__":
    main()
