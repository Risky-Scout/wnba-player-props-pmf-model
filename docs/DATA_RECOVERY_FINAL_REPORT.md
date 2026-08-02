# WNBA Player-Prop Data Recovery — Final Report

Branch: `cursor/per-stat-compact-pmf-v1`. All numbers below are from executed runs and tracked
audit artifacts, not estimates.

## Status by stage (staged, fail-closed contract)

| Stage | Scope | Status |
|-------|-------|--------|
| 1 | `MODEL_PROP_MARKETS` (12) + 12-market runtime assertion | **COMPLETE** |
| 2 | Isolate rebuilt artifacts to `data/recovered_v2` (frozen v1 preserved) | **COMPLETE** |
| 3 | Offline recovery of 382 raw files with retained eligibility | **COMPLETE** |
| 4 | Durable/resumable backfill + resumption tests | **COMPLETE** |
| 5 | 25-event durability pilot (all gates) | **COMPLETE** |
| 6 | Complete historical quote recovery | **COMPLETE** |
| 7 | Deterministic player-identity resolution | **COMPLETE** |
| 8 | Exact pairs + BDL settlement (decision vs closing separate) | **COMPLETE** |
| 9 | Pregame feature registry + leakage audit | **COMPLETE** |
| 10–22 | Feature-driven minutes + PMF models, chronological OOF, distributional evaluation, model selection, push-safe market comparison, uncertainty/abstention, freeze, prospective ledger, pricing CLI, static package | **NOT BUILT** (see "Honest limitations") |

## BDL coverage (recovered_v2, isolated namespace)

Final (completed) games by season: **2023: 263, 2024: 264, 2025: 312, 2026: 210** (1,049 total,
2023-05-19 → 2026-07-29). Player box scores: 23,931 rows / 334 players. Wide feature matrix:
23,931 rows × 561 cols (394 model features), all build-time leakage/forbidden guards PASS.
`foundation_lock_status = DEFERRED_MISSING_FROZEN_ARTIFACT`; recovered_v2 is **not** frozen v1.

## Odds API coverage + credits

- Total API credits used (all sessions): **693,790**; remaining: **4,306,210**.
- Stage 6 delta: **57,485** credits (≤ 260k budget); ran to completion (not budget/interrupt).
- 1,159 raw responses cached; 580 unique matched events; 10 US sportsbooks.
- Every request: `basketball_wnba` / `us` / `american` / `iso` / exactly the 12 model markets
  (fail-closed runtime assertion). Deterministic 404 tombstones; no secret logged.

## Atomic store + identity

- 256,160 normalized atomic side rows; **244,644 ELIGIBLE / 11,516 BLOCKED** (all retained with
  explicit reasons); 0 duplicate primary keys; eligible post-cutoff/post-tip = 0.
- Identity: **248,057 exact** (diacritic-folded), **0 collisions** (never forced), 8,103 unmatched
  (predominantly DNP players absent from the appeared-roster — correct).

## Exact pairs + settlement (decision role)

50,753 decision + 67,635 closing EXACT pairs. Settled decision: OVER 23,264 / UNDER 26,781 /
VOID 708. **Market-evidence-ready (≥300 settled non-push decision pairs, ≥30 unique dates):**

| Prop | decision pairs | settled non-push | unique dates | market evidence |
|------|---------------:|-----------------:|-------------:|-----------------|
| pts | 14,928 | 14,752 | 300 | **READY** |
| reb | 11,390 | 11,238 | 296 | **READY** |
| ast | 6,458 | 6,368 | 295 | **READY** |
| fg3m | 6,779 | 6,682 | 281 | **READY** |
| pts_ast | 2,267 | 2,227 | 219 | **READY** |
| pts_reb | 3,321 | 3,268 | 219 | **READY** |
| reb_ast | 1,645 | 1,611 | 217 | **READY** |
| pts_reb_ast | 3,965 | 3,899 | 227 | **READY** |
| stl / blk / turnover / stocks | 0 | 0 | 0 | INSUFFICIENT (no US-book coverage) |

## Release states (honest)

No feature-driven PMF model has yet been trained or validated (Stages 10–22 not built).
Therefore **no product is VALIDATED**, and **no market-superiority claim is made** for any prop.
Per the product definition, the market consensus is **never** substituted for the model.

| Product | Data foundation | Release state |
|---------|-----------------|---------------|
| pts, reb, ast, fg3m, pts_ast, pts_reb, reb_ast, pts_reb_ast | quotes READY + BDL outcomes | **RESEARCH** (model not yet built) |
| stl, blk, turnover, stocks | BDL outcomes only (no US-book quotes) | **RESEARCH** (outcome-only; market evidence unattainable) |

## Honest limitations (Stages 10–22 remaining)

Stages 10–22 constitute an entire feature-driven modeling/evaluation/pricing/publishing
platform (participation + distributional minutes model; 7 direct distributional PMFs with a
minutes mixture; 5 coherent combination PMFs; genuine chronological OOF across all seasons with
in-fold selection/tuning/calibration; NLL/RPS/CRPS/PIT/calibration/tail evaluation; predeclared
model-selection gates; push-safe model-vs-market comparison with clustered bootstrap; aleatoric
+ epistemic uncertainty; freeze ceremony + prospective ledger; pricing CLI; and a static
package). These are **not** built in this pass. They are unblocked by the completed data
foundation and the settled decision pairs above, but building and validating them to the
required quality is a substantial further effort. No metrics for these stages are reported
because none were produced — no benchmark is mislabeled as the predictive model.
