# Modeling Design V5 — coherent atom distributions and production pricing

V2/V3/V4 preserved unmodified. V5 frozen (`artifacts/sharp_v5/V5_FREEZE_MANIFEST.json`) before
production OOF. Selection uses **2023-2025 only** (V3/V4 2026 holdout CONSUMED). Prospective
evidence begins with the first immutable forecast after this freeze.

## Corrections over V4
- **One `DiscreteDistribution` interface** with correct mass: exact per-atom analytic tail
  (`probability(y)` for any y), overflow is an aggregate bucket only, `stored+overflow=1`.
- **Correct hurdle / zero-inflated / convolution** mass (positive normalizer includes the analytic
  positive tail; convolution not renormalized-then-overflow-added).
- **Push-aware, multi-line market projection**: `A/(A+B)=q_over`; one PMF fit to all same-time
  no-vig lines via a min-KL mean/dispersion/zero tilt; fails closed `MARKET_PROJECTION_INFEASIBLE`.
- **Minutes as regulation(0..40)+overtime**; observed minutes never clipped.
- **Minutes-PMF propagation**: every stat PMF is an analytic mixture over the minutes PMF, so
  wider minutes uncertainty widens the stat PMF/tails (proven by test).
- **Frozen exact feature contracts** (resolved on training, hashed, reused; drift fails closed).
- **BDL FGM/FTM recovered** to enable structural shooting identities.

## Honest scope
Distribution correctness, push-aware multi-line projection, minutes propagation, frozen contracts,
and BDL label recovery are implemented + fitted (2023-2025, improving on V4). Shared
game-environment reconciliation, copula joint dependence, first-basket, and full Q1 remain
designed/scaffolded and abstain honestly.
