# Calibration Contract

## Non-negotiables

- Never fit calibration on in-sample PMFs.
- Never train on market lines or odds.
- Never join outcomes on a market line; join on `game_id`, `player_id`, `stat`.
- Pushes are excluded from binary over/under scoring.
- Full PMF scoring uses NLL and RPS/CRPS.
- Market comparison uses no-vig over probabilities only; it never reconstructs a market PMF.

## Role-aware PMF calibration

For each stat:

1. Calculate randomized PIT from OOF PMFs.
2. Fit a global isotonic CDF calibrator.
3. Fit role-specific calibrators when row count is high enough.
4. Blend bucket and global calibrators with shrinkage.
5. Route `inactive_risk` to global only.

## Promotion rules

A calibrator is eligible only if:

- It does not materially worsen OOF NLL by stat×role.
- PIT KS and ECE are inside gates or marked REVIEW with guarded fallback.
- Sparse stats preserve p0 calibration.

## Market superiority claim

A stat×role claim is certified only when:

- at least 100 scored market rows exist;
- bootstrap UCB95 of model minus market log-loss delta is below -0.0025;
- bootstrap UCB95 of model minus market Brier delta is below -0.0010.

Negative deltas mean the model is better than the no-vig market baseline.
