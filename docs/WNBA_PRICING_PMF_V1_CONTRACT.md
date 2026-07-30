# WNBA Pricing PMF v1 — Contract

Release identifier: **`wnba-pricing-pmf-v1.0.0-rc1`**.

A release candidate exists only when all required artifacts, inference paths, tests, and a
pricing run are complete. This RC delivers the **feature-driven pricing engine, coherent joint
generator, market registry, calibration, first-basket competing-risk, event/alternate/combination
markets, tests, a fixture pricing run, and the immutable bundle.** Sportsbook data is never a
model input, offset, prior, target, or selection/calibration signal. Market-superiority
certification is a **separate** gate and is NOT claimed here.

## Primitive active-player variables
Full game: active/DNP, minutes, 2PA/2PM, 3PA/3PM, FTA/FTM, OREB/DREB, AST, STL, BLK, TOV.
First quarter: Q1 minutes + the same primitives via a **separate Q1 layer** (never full×0.25).

## Structural identities (hold in every joint sample)
`fgm = 2pm + 3pm` · `pts = 2·2pm + 3·3pm + ftm` · `reb = oreb + dreb` · `stocks = stl + blk`.

## PMF contract
Every active-player count distribution emits all atoms `P(Y=0..K)`. Support is adaptive (grow
until omitted tail < 1e-8 or emit `TAIL_TOLERANCE_NOT_MET`). `p_dnp` and the active PMF are
separate — DNP mass is never inserted at the zero atom for settlement. Alternates settle from
the same distribution as the base market (monotone by construction).

## Modules
- `pricing/market_registry.py` — canonical registry (families, settlement, push/DNP rules, status).
- `pricing/engine.py` — push-safe fair odds, margin layer (does not modify the PMF), yes/no, categorical.
- `pricing/joint_generator.py` — shared-latent (minutes) deterministic simulation; identities enforced.
- `pricing/first_basket.py` — competing-risk first-basket / first-team / method categorical.
- `pricing/calibration.py` — monotone-CDF distributional calibration (never per-line isotonic).

## Tracks
- PURE_PMF — no market info anywhere (this RC).
- MARKET_ANCHORED_PMF — production track (KL/exp-tilt to same-time no-vig + chronological residual);
  **scoped, not run in this pass** (requires Phase-1 data). Statuses: CERTIFIED_RESIDUAL /
  MARKET_ANCHORED_UNCERTIFIED / PURE_RESEARCH_ONLY / NO_MARKET_AVAILABLE / NO_EVIDENCE /
  ABSTAIN_*.

## Release states
No product is VALIDATED (no prospective evidence). Feature-only pricing is complete and tested;
market comparison + certification are deferred to a separate gate. `player_fantasy_points` is
CONFIG_REQUIRED (needs an operator scoring-rule id).
