# Modeling Design V4 — WNBA Sharp PMF (atom-reliable)

V2 and V3 are preserved unmodified as historical baselines. V4 is frozen (hashed in
`artifacts/sharp_v4/V4_FREEZE_MANIFEST.json`) **before** any V4 OOF prediction. The **2026 V3
holdout is CONSUMED** (`HOLDOUT_LINEAGE_AUDIT.json`) and is never used for V4 selection: V4
candidate/feature/calibration/hyperparameter selection uses **2023-2025 folds only**. The V4
prospective holdout is append-only and begins with the first forecast after this freeze.

## Core statistical defects repaired (Section 4)
- **A. Explicit feature contracts** — exact ordered per-component feature lists with provenance,
  missingness policy, and train/inference schema hashes. No `if stat in column` substring
  selection; no unexpected column admitted by name; no required feature silently dropped.
- **B. Join cardinality** — every feature/target/quote/settlement join asserts 1:1 on
  `(game_id, canonical_player_id[, prediction_as_of])`; fails on duplicate/many-to-many; row
  counts recorded before/after.
- **C. No outcome clipping** — an outcome above nominal support is scored on its exact analytic
  tail (or a preserved overflow atom), never `min(y, cap)`. NLL/CRPS/PIT/pricing/settlement share
  one support semantics.
- **D. Adaptive support + overflow** — expand until survival < 1e-6; store `support_min/max`,
  `overflow_probability`, `tail_upper_bound`, `tail_method`, `normalization_error`; overflow mass
  is included in Over settlement; `TAIL_TOLERANCE_NOT_MET` when unmet.
- **E. Hierarchical dispersion** — partially-pooled dispersion (`player → role → stat → league`)
  fit inside each training fold, replacing one global scalar; NB2 / Poisson-lognormal / hurdle-NB2
  / ZINB compared by OOF proper scores; dispersion uncertainty persisted.

## Structural shooting (Section 9)
Model `2PA, 3PA, FTA` and conversions `2P%, 3P%, FT%` (hierarchical beta-binomial, shrunk
`player → role → league`), integrate over the **full minutes PMF**, and derive
`FGM = 2PM+3PM`, `PTS = 2·2PM+3·3PM+FTM` with `makes ≤ attempts` in every draw. The points PMF is
the **marginal of the structural simulation**, not a separate inconsistent NB2 (V3 NB2 retained as
a control).

## Minutes as a distribution (Section 7)
`P(regulation_minutes = m | active)` on 0..40 (overtime modeled separately); downstream stats
integrate over the whole minutes PMF rather than a point estimate.

## Tracks, calibration, market anchoring
PURE (no market anywhere) and MARKET_ANCHORED (min-KL/exp-tilt projection of the pure PMF to the
exact same-time no-vig market + a shrink-to-zero hierarchical residual). Distributional
calibration is cross-fitted (block `t` uses only earlier OOF). Where residual superiority is
unproven, `delta = 0` and production returns the market-consistent PMF — never an uncertified
inferior pure forecast.

## Honest scope
Structural shooting, minutes PMF, hierarchical dispersion, exact-tail scoring, explicit contracts,
join guards, market-consistent projection, and distributional calibration are implemented and
fit on 2023-2025. Full game-environment reconciliation, role-state minutes mixture, copula joint
dependence, first-basket competing-risk, drift automation, and season-transition are specified
here and scaffolded; uncertified components abstain or fall back to market rather than emitting
fabricated probabilities.
