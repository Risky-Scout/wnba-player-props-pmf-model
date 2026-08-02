"""Emit sharp_v3 leakage / feature-policy / calibration audits from fitted OOF outputs + data."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.sharp_v3 import core as C

OUT = C.REPO / "artifacts" / "sharp_v3"


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    _, df = C.load_verified()
    cols = list(df.columns)

    # feature policy per component
    policy = {}
    for stat in C.TIER_A + ["minutes", "participation"]:
        feat = C.stat_feature_contract(stat, cols)
        policy[stat] = {"n_domain_core": len(feat), "feature_hash": C.feature_schema_hash(feat),
                        "sample": feat[:12], "excludes_labels": True}
    (OUT / "FEATURE_POLICY_BY_COMPONENT.json").write_text(json.dumps(
        {"artifact": "FEATURE_POLICY_BY_COMPONENT", "generated_at_utc": ts,
         "label_columns_excluded": C.LABEL_COLS, "components": policy}, indent=2, default=str))

    # leakage audit: physical check that no feature column equals any label column
    label_equal = []
    for stat in C.TIER_A:
        y = df[stat].to_numpy(float)
        for c in C.stat_feature_contract(stat, cols):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            m = np.isfinite(v) & np.isfinite(y)
            if m.sum() > 100 and np.corrcoef(v[m], y[m])[0, 1] > 0.995:
                label_equal.append({"stat": stat, "feature": c,
                                    "corr": float(np.corrcoef(v[m], y[m])[0, 1])})
    cm = json.loads((OUT / "COUNT_MODEL_REPORT.json").read_text())
    mae = {r["stat"]: r["mean_mae"] for r in cm if not r["is_holdout"]}
    sane = {s: (v > 0.3) for s, v in mae.items()}   # realistic MAE (a leaking model had ~0.01-0.03)
    (OUT / "LEAKAGE_AUDIT.json").write_text(json.dumps({
        "artifact": "LEAKAGE_AUDIT", "generated_at_utc": ts,
        "detected_and_fixed": {
            "issue": "same-game target leakage: bare label columns (pts,reb,...) matched the "
                     "stat-name feature contract after features+targets merge.",
            "symptom": "impossible OOF mean_mae ~0.01-0.03 and model logloss ~0.20 vs market ~0.69.",
            "fix": "core.LABEL_COLS excluded from every contract + fail-closed guard in _prep()."},
        "post_fix_mean_mae": mae,
        "post_fix_mae_realistic": sane,
        "all_mae_realistic": bool(all(sane.values())),
        "feature_equals_label_violations": label_equal,
        "clean": bool(all(sane.values()) and not label_equal)}, indent=2, default=str))

    # calibration report from PIT
    pit = {}
    for r in cm:
        if r["is_holdout"]:
            continue
        pit.setdefault(r["stat"], []).append(r["pit_ks"])
    (OUT / "CALIBRATION_REPORT.json").write_text(json.dumps({
        "artifact": "CALIBRATION_REPORT", "generated_at_utc": ts,
        "method": "randomized-PIT KS uniformity (pre-recalibration) on active OOF rows",
        "pit_ks_by_stat": {s: float(np.mean(v)) for s, v in pit.items()},
        "note": "distributional (monotone-CDF) recalibration hook in pricing/calibration.py; "
                "PURE track abstains where calibrator data insufficient, MARKET track uses "
                "market-consistent PMF."}, indent=2, default=str))
    print("wrote FEATURE_POLICY_BY_COMPONENT / LEAKAGE_AUDIT / CALIBRATION_REPORT")
    print("all_mae_realistic:", all(sane.values()), "feature==label violations:", len(label_equal))


if __name__ == "__main__":
    main()
