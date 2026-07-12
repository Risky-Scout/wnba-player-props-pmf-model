# Hotfix Baseline 734e6d79 — Artifact Notes

## GitHub Run 29210161447: SHAP Feature Importance Failure

GitHub run 29210161447: SHAP feature importance step failed (non-critical step,
workflow continued). The SHAP computation is optional and does not affect predictions.
Investigation needed: likely missing model artifact or sklearn version incompatibility.

## Contents

- `bias_corrections.json` — per-stat mean bias correction factors applied at inference
- `bias_corrections_by_role.json` — role-stratified bias corrections
- `calibration_metadata.json` — calibration run metadata
- `player_form_corrections_2026.json` — player-level form correction values
- `player_variance_compress.json` — per-player variance compression factors
- `variance_compress.json` — per-stat variance compression factors
