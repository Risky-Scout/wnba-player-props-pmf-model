# Modeling Design V3 — WNBA Sharp PMF

V2 is preserved unmodified in `config/design_registry/v2_archive/`. V3 is frozen (hashed in
`artifacts/sharp_v3/V3_FREEZE_MANIFEST.json`) **before** any outer-fold prediction. After the
first outer prediction V3 is closed; a semantic change requires quarantining V3 predictions and
freezing V4.

## What V3 fixes (vs V2)
1. Separate **PURE** and **MARKET_ANCHORED** tracks; sportsbook data never enters PURE (features,
   fitting, selection, dispersion, calibration, support, tie-breaks).
2. **Stat-specific compact feature contracts** (domain core + inner-fold stable extensions +
   full-matrix overfitting control), not all 336 columns to every target.
3. **Structural primitives**: model 2PA/3PA/FTA × conversion; `pts`, `fgm`, `reb` derived by
   identity — never a generic count that contradicts its components.
4. **Role-state minutes mixture**, not one clipped normal.
5. **Game-level shared latent** (pace/possessions/overtime/script) allocated coherently.
6. **Analytic tails**, `tail_tolerance = 1e-6`, explicit overflow atom, ≤5e-4 per-line error.
7. **Untouched 2026 holdout**, opened exactly once after candidates lock.
8. **Active-conditional** semantics: `p_active` separate from `P(stat=k|active)`; DNP never adds
   Under mass.
9. **Distributional calibration** (monotone CDF/PIT), cross-fitted on earlier OOF only.
10. **Conservative fallback**: abstain / market-fallback under OOD or missing critical inputs.

## Data (private, gitignored, hash-verified)
23,931 pregame player-games × 336 features; Tier A targets; 50,753 exact same-time settled
no-vig market pairs. Coverage 2023-05-19 → 2026-07-29.

## Folds
Development outer folds over 2024 & 2025 (expanding window, game-date grouped); 2023 for priors;
2026 season-to-date is the single-use final holdout. Boundaries are frozen in
`config/chronological_oof_v3.yaml` from dates/coverage only (never outcomes/scores).

## Tiers
Tier A (`pts reb ast fg3m stl blk turnover` + derived `stocks pts_ast pts_reb reb_ast pts_reb_ast`)
is fitted and evaluated first. Tier B/C must abstain honestly rather than emit default/fixture
probabilities.
