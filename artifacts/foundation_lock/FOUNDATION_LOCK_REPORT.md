# Foundation Lock Report

- Foundation version: **v1**
- Generated from commit: `a66baa7f60403ed4e90dafd537104c6a2f70b936` (base commit; NOT the commit containing this manifest)
- Generated (manifest): 2026-07-23T03:34:05.963541+00:00
- Overall: **PASS_WITH_DECLARED_DEFERRALS**  (failures: 0, deferrals: 2)

This report classifies each locked component by what it does and does not prove.
No component below is promotion-eligible; the market edge is NOT proven by any of them.

| # | Component | Status | Promotion-eligible | Paths | Required tests |
|---|---|---|---|---:|---:|
| 1 | Market-superiority evaluator | `locked` | NO | 1 | 1 |
| 2 | G0 baseline proof (historical, pre-Phase-0) | `locked` | NO | 4 | 1 |
| 3 | Signed-CLV gate | `locked` | NO | 3 | 1 |
| 4 | Feature-ablation maps | `locked` | NO | 5 | 2 |
| 5 | Tracking/hustle collector (code + schema contract) | `locked` | NO | 2 | 1 |
| 6 | Tracking/hustle raw data | `not_landed` | NO | 2 | 0 |
| 7 | Tracking/hustle canonical ingestion | `not_landed` | NO | 0 | 0 |
| 8 | Tracking-derived model features | `not_landed` | NO | 0 | 0 |
| 9 | Current feature-matrix snapshot | `not_landed` | NO | 3 | 1 |
| 10 | Prop-ablation runner (surrogate, fail-closed); prior verdict invalidated | `locked` | NO | 2 | 1 |

## Per-component limitations

### Market-superiority evaluator (`market_superiority_evaluator`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: The synthetic --self-test validates evaluator MECHANICS only; it is NOT real market evidence.
- Limitation: A PASS is evidence only for the exact frozen candidate, book, timestamps, dates, and population.

### G0 baseline proof (historical, pre-Phase-0) (`g0_baseline_proof`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Historical pre-Phase-0 baseline. Not promotion-eligible.
- Limitation: Push-safe integer-line handling, exact quote identity, and evaluated-versus-deployed parity are NOT yet fixed.
- Limitation: Books were averaged in the input, so exact quote identity was not preserved.

### Signed-CLV gate (`signed_clv_gate`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Requires post-game closing quotes to evaluate; without them the gate fail-closes (skips), it does not pass.
- Limitation: CLV thresholds are unchanged by Foundation Lock v1.

### Feature-ablation maps (`feature_ablation_maps`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Resolves feature lists only; it does not train models or prove anything about performance.

### Tracking/hustle collector (code + schema contract) (`tracking_collector`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Collector CODE only (mocked, no-network tests). Runs on a LOCAL residential IP; never in CI/agent VMs.
- Limitation: This is not Phase B; it proves nothing about discrimination or market edge.

### Tracking/hustle raw data (`tracking_data`)
- Status: `not_landed`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: NOT landed. No tracking parquet exists in-repo or in a CI-retrievable store yet.
- Limitation: Absence is DEFERRED by the verifier, never silently passed.

### Tracking/hustle canonical ingestion (`tracking_canonical_ingestion`)
- Status: `not_landed`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: NOT started. No canonical_player_tracking_game / canonical_player_hustle_game tables exist.

### Tracking-derived model features (`tracking_features`)
- Status: `not_landed`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: NOT started. No tracking features enter any model; this is not Phase B.

### Current feature-matrix snapshot (`feature_matrix_snapshot`)
- Status: `not_landed`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: The snapshot descriptor is locked, but the wide parquet is LOCAL-ONLY (git-ignored, not in a CI-retrievable store), so the real feature matrix is NOT reproducible from CI. Status is not_landed; ci_verifiable=false.
- Limitation: Do not call the feature matrix reproducible from CI until the parquet is uploaded to an immutable, CI-retrievable artifact store.
- Limitation: A snapshot describes the matrix; it does not validate model performance.

### Prop-ablation runner (surrogate, fail-closed); prior verdict invalidated (`prop_ablation_runner`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Prior verdict INVALIDATED: it was produced by silently discarding 76/128 feature-contract columns absent from the wide parquet. Removed; see ABLATION_STATUS.json.
- Limitation: No real-data surrogate verdict is produced until an immutable full-feature matrix is landed.
- Limitation: Surrogate model (HGB Poisson -> NB PMF), NOT the signed champion pipeline. Development-only; not proof/certification/promotion evidence.

## Deferred (declared data artifacts absent from this checkout)

- tracking_data: wnba_tracking_2021_2026.parquet (data artifact) not present -> DEFERRED
- tracking_data: wnba_hustle_2021_2026.parquet (data artifact) not present -> DEFERRED

