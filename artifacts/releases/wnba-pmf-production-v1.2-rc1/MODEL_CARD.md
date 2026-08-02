# WNBA PMF Production v1.2-rc1 (Phase-3 Challenger)

Challenger bundle for injury-conditioned participation + conditional-active minutes.
Does **not** claim unconditional P(active | all roster-eligible players).
NOT_LISTED is an operational gate, not a historically calibrated probability.
Minutes family: `hgb_residual`.
Participation family: `hgb` / calibrator `platt`.
Downstream gate: `FAIL`.
Rollback: `artifacts/releases/wnba-pmf-production-v1.1`.
Production pointer is not updated by this package step.
