"""WNBA Sharp PMF v3 — fitted, chronologically-evaluated, active-conditional player-prop PMFs.

Modules:
- core: hash-verified loading, stat feature contracts, chronological folds, count-PMF + metrics.
- pipeline (script scripts/run_sharp_v3_oof.py): participation + minutes + Tier A stat OOF, market
  comparison, calibration, activation registry.
"""
DESIGN_VERSION = "wnba-sharp-pmf-v3"
SEED = 20260730
