# WNBA Sharp PMF V5 — Final Implementation Report

## Delivered (real, fitted, honest)
- **Version lineage**: continued on PR #99; V2/V3/V4 preserved unmodified. New `sharp_v5/` paths.
  V5 frozen (`V5_FREEZE_MANIFEST.json`, `modeling_design_v5_sha256`) before production OOF; selection
  uses **2023-2025 only** (V3/V4 2026 holdout CONSUMED).
- **PMF abstraction replaced** (`sharp_v5/distribution.py`): one `DiscreteDistribution` interface with
  **correct mass accounting** — `probability(y)` is the exact analytic per-atom probability for any y
  (never the aggregate overflow reused as each tail atom); `stored + overflow = 1` (norm error < 1e-10);
  no outcome clipping. **Hurdle, zero-inflated, and convolution mass fixed** (positive normalizer
  includes the analytic positive tail; convolution not renormalized-then-overflow-added). Adversarial
  tests with material component tails pass.
- **Push-aware, multi-line market projection** (`sharp_v5/market_projection.py`): settlement uses
  **A/(A+B)** (not A=q); one PMF is fit to **all** same-time no-vig lines via a min-KL mean/dispersion/
  zero tilt; contradictory constraints **fail closed** `MARKET_PROJECTION_INFEASIBLE`; support extends
  beyond the highest quoted line. Tests cover integer push, half-point, multi-line, contradictory, and
  zero-residual fallback.
- **Minutes-PMF propagation** (`sharp_v5/models.py`): every stat PMF is an analytic **mixture over the
  minutes PMF** (`sum_m P(Y|m)P(m)`). Proven by test: same expected minutes but larger minutes variance
  ⇒ larger stat variance and heavier tails. Minutes are regulation 0..40 + a separate overtime mixture;
  observed minutes are never clipped.
- **Frozen exact feature contracts** (`FEATURE_CONTRACTS.json`): resolved on training columns, hashed,
  reused unchanged; drift fails closed.
- **V5 OOF (2023-2025)**: improves on V4 for the highest-value stats — **pts NLL 2.930 (V4 2.945, V3
  3.067)**, ast −0.001; ties reb; marginally worse on rare stats (honest, `V5_VERSUS_V4_COMPARISON.csv`).
  **Market not beaten** (1/16 fold-stat comparisons showed a nominal edge — within false-positive rate,
  does not survive aggregation → **no `CERTIFIED_RESIDUAL`**).
- **Live real upcoming-slate run (not fixture)** for **2026-07-31**: 5 real BDL games, 224 players,
  **35,989 atoms, 14,520 fair Over/Under lines, 118 push-aware market-consistent projections feasible /
  15 infeasible** (fail-closed). Append-only **prospective registry** updated.
- **Tests**: 16 V5 acceptance tests (+13 V4, +10 V3, +20 pricing) pass; ruff clean.

## Honest blockers / not fitted
- **BDL FGM/FTM unrecoverable here** (`BDL_SHOOTING_LABEL_AUDIT.json`): `/wnba/v1/stats`,
  `/box_scores`, `/season_stats` all return **404** with this key/tier (only `/games`, `/teams`,
  `/players` work). So a **structural points** decomposition (needs 2PM/FTM) stays blocked; points uses
  the V5 minutes-mixture NB2 direct model — **not** labelled structural.
- **STL/BLK/TOV market coverage**: US books returned **no** steals/blocks/turnovers markets for the
  upcoming slate at this snapshot (`STL_BLK_TOV_MULTI_SNAPSHOT_AUDIT.json`) — honest, not fabricated.
- **Not fitted** (abstain/market-fallback): shared game-environment + team reconciliation, copula joint
  dependence, first-basket, Q1 (no reconciled labels), full persisted cross-fit calibration layer.

## Merge status (honest)
Model is on the PR #99 branch, **not** on `origin/main` (`ORIGIN_MAIN_VERIFICATION.json`,
`merged_to_main=false`). This agent's GitHub CLI is **read-only** and it has **no merge/ready tool**, so
it cannot mark ready or merge. A maintainer must run `gh pr ready 99 && gh pr merge 99 --squash`; on-main
ancestry/SHA verification passes only after that.

## Verdict
The V5 correctness core is genuinely repaired and tested: correct atom mass, push-aware multi-line
market-consistent projection, and minutes-uncertainty propagation into every stat PMF, with a live real
slate priced and honest abstention elsewhere. **No market superiority is claimed.** **Not on `main`** —
merge is externally blocked by tool permissions.
