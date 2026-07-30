# Model Card — WNBA Pricing PMF v1 (wnba-pricing-pmf-v1.0.0-rc1)

**Feature-driven** coherent player-prop pricing engine. Sportsbook data is NOT used to fit or
select the distribution. A joint shared-latent (minutes) simulation produces every primitive
active-player outcome with structural identities holding in every sample
(`fgm=2pm+3pm`, `pts=2*2pm+3*3pm+ftm`, `reb=oreb+dreb`, `stocks=stl+blk`); combination markets
use joint dependence (not sums of independent marginals). Alternates settle from the same
distribution as the base market (monotone by construction). p_dnp is kept separate from the
zero atom (void-on-DNP settlement).

## Status
- IMPLEMENTED + TESTED: pricing engine, market registry, joint generator, calibration hooks,
  first-basket competing-risk, event markets, odds conversions, margin layer.
- PRICED_TODAY: deterministic FIXTURE slate only (no live slate on this branch).
- NOT_CERTIFIED for market superiority (no market comparison run; separate gate).
- `player_fantasy_points`: CONFIG_REQUIRED (needs an operator scoring-rule id).

## Limitations
- Real "today's pricing run" requires the Phase-1 recovered feature/quote data (blocked here).
- Monte-Carlo precision is tracked (`mc_max_se`); prices exceeding the SE tolerance are flagged
  SIMULATION_PRECISION_NOT_MET.
- No prospective validation → no product is VALIDATED / bettor-ready.
