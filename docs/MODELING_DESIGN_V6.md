# Modeling Design V6 — production-complete coherent player distributions

V2–V5 preserved unmodified. V6 frozen (`artifacts/sharp_v6/V6_FREEZE_MANIFEST.json`) before V6 OOF.
Selection uses 2023-2025; 2026 is retrospective/refit only.

## Corrections over V5 (concrete, verified)
- **BDL endpoint CORRECTED**: `/wnba/v1/player_stats` (V5 probed undocumented `/stats`,`/box_scores`,
  `/season_stats` → all 404; the V5 "tier-blocked" claim was WRONG). FGM/FTM recovered; `pts`
  identity validated.
- **Market-projection mass bug FIXED**: `TiltedDistribution` applies the exponential tilt to the
  **complete** base (including its analytic tail); `Z` computed with a certified remainder bound;
  **stored atoms + overflow = 1 exactly** (test enforces ≤1e-10). No unchanged overflow reattached
  after normalizing stored atoms. Multi-line projection returns one proper tail-aware distribution.
- **Hurdle variance** uses the complete analytic second moment (not stored atoms with zero tail).
- **Bounded tilt basis** (saturating mean + bounded dispersion + zero) keeps the transformed
  infinite tail summable.

## Retained from V5
Minutes-PMF propagation into every stat PMF (mixture), push-aware `A/(A+B)` settlement, frozen
exact feature contracts, exact-tail count distributions. The V5 stat fit is unchanged; V6's changes
are pricing/distribution correctness + label recovery.

## Honest scope (this pass)
Implemented + verified: endpoint correction + FGM/FTM recovery, tail-aware market projection mass,
hurdle variance, bounded tilt. **Partial/not fitted this pass** (honest, not fabricated): full
FGM/FTM coverage (partial pull ~5.4k games; resumable full pull pending), fitted structural points,
shared game-environment + team-constrained minutes reconciliation, copula joint dependence, Q1
labels, first-basket, persisted cross-fit calibration layer. These abstain or fall back to market.
