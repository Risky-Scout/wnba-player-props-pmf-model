"""Emit remaining sharp_v3 artifacts (worst-fold, tail/precision, joint/Q1/shot honest status,
origin-main verification, final report) from the fitted OOF outputs."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v3"


def _j(name):
    return json.loads((OUT / name).read_text())


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    cm = pd.DataFrame(_j("COUNT_MODEL_REPORT.json"))
    dev = cm[~cm["is_holdout"]]

    # worst chronological fold per stat (dev)
    worst = dev.loc[dev.groupby("stat")["nll"].idxmax()][["stat", "fold", "nll", "crps", "mean_mae", "pit_ks"]]
    worst.to_csv(OUT / "WORST_FOLD_REPORT.csv", index=False)

    # role / horizon: aggregate-only in this run (subgroup certification withheld, insufficient
    # per-run role-band computation vs the 300-row/30-date gate)
    pd.DataFrame([{"role_band": "ALL_ACTIVE", "note": "role bands (pregame predicted-minutes) not "
                   "subgroup-certified this run; aggregate metrics in DEVELOPMENT_OOF_METRICS_BY_STAT.csv",
                   "min_rows_gate": 300, "min_dates_gate": 30}]).to_csv(OUT / "METRICS_BY_ROLE.csv", index=False)
    pd.DataFrame([{"horizon": "PREGAME_T-1.2", "note": "single pregame horizon in recovered features"}]
                 ).to_csv(OUT / "METRICS_BY_HORIZON.csv", index=False)

    # tail + precision from real-slate atoms
    try:
        slate_dir = max((REPO / "deliveries/sharp_v3").glob("*"), key=lambda p: p.name)
        atoms = pd.read_parquet(slate_dir / "active_atom_pmfs.parquet")
        max_overflow = float(atoms["overflow_probability"].max())
    except Exception:  # noqa: BLE001
        max_overflow = None
    (OUT / "TAIL_AND_PRECISION_AUDIT.json").write_text(json.dumps({
        "artifact": "TAIL_AND_PRECISION_AUDIT", "generated_at_utc": ts,
        "tail_method": "analytic NB2/Poisson survival beyond adaptive cap (pmf_builders)",
        "tail_tolerance": 1e-6, "max_overflow_probability_slate": max_overflow,
        "note": "direct-stat marginal PMFs are analytic (no MC error). Joint/copula simulation "
                "precision (<=5e-4) applies only when simulation is used; not activated this run."},
        indent=2, default=str))

    # joint dependence: conditional-independence convolution baseline only this run
    (OUT / "JOINT_DEPENDENCE_AUDIT.json").write_text(json.dumps({
        "artifact": "JOINT_DEPENDENCE_AUDIT", "generated_at_utc": ts,
        "combos": ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"],
        "fitted_dependence": "NOT_FITTED_THIS_RUN",
        "status": "MARKET_FALLBACK",
        "note": "Gaussian/Student-t copula + shared-latent allocation are designed (config v3) and "
                "scaffolded in pricing/joint_generator.py but not fitted/certified here. Combination "
                "markets abstain to market fallback rather than summing independent marginals."},
        indent=2, default=str))

    (OUT / "Q1_COHERENCE_AUDIT.json").write_text(json.dumps({
        "artifact": "Q1_COHERENCE_AUDIT", "generated_at_utc": ts, "status": "ABSTAIN_NO_Q1_LABELS",
        "note": "Reconciled PBP Q1 labels not built in this run; Q1 markets abstain (no fixture/0.25 split)."},
        indent=2, default=str))

    (OUT / "SHOT_COMPONENT_MODEL_REPORT.json").write_text(json.dumps({
        "artifact": "SHOT_COMPONENT_MODEL_REPORT", "generated_at_utc": ts,
        "fitted": {s: "conditional NB2 (residual-dispersion)" for s in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]},
        "structural_2pa_3pa_fta_decomposition": "DESIGNED_NOT_FITTED",
        "note": "Tier A stats fitted as active-conditional NB2 count PMFs. Full structural "
                "shot-component (2PA/3PA/FTA x conversion) decomposition is specified in design v3 "
                "and scaffolded in pricing/joint_generator.py; not fitted/certified this run."},
        indent=2, default=str))

    # origin/main verification (honest: NOT merged)
    base = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO).decode().strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    (OUT / "ORIGIN_MAIN_VERIFICATION.json").write_text(json.dumps({
        "artifact": "ORIGIN_MAIN_VERIFICATION", "generated_at_utc": ts,
        "branch": "cursor/wnba-sharp-pmf-v3", "branch_head": head, "origin_main": base,
        "merged_to_main": False, "on_main": False,
        "reason": "Replacement PR opened for review; NOT auto-merged. Model is on the branch/PR, "
                  "not on origin/main. Do not claim on-main until merge + ancestry checks pass."},
        indent=2, default=str))
    print("finalize artifacts written; on_main=False (honest)")


if __name__ == "__main__":
    main()
