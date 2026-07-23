# Foundation Lock Report

- Foundation version: **v1**
- Source commit: `a66baa7f60403ed4e90dafd537104c6a2f70b936`
- Generated (manifest): 2026-07-23T02:14:02.389445+00:00
- Overall: **PASS**  (failures: 0, deferrals: 2)

This report classifies each locked component by what it does and does not prove.
No component below is promotion-eligible; the market edge is NOT proven by any of them.

| # | Component | Status | Promotion-eligible | Paths | Required tests |
|---|---|---|---|---:|---:|
| 1 | Market-superiority evaluator | `locked` | NO | 1 | 1 |
| 2 | G0 baseline proof (historical, pre-Phase-0) | `locked` | NO | 4 | 1 |
| 3 | Signed-CLV gate | `locked` | NO | 3 | 1 |
| 4 | Feature-ablation maps | `locked` | NO | 5 | 2 |
| 5 | Tracking/hustle collector (infrastructure only) | `not_landed` | NO | 4 | 1 |
| 6 | Current feature-matrix snapshot | `locked` | NO | 3 | 1 |
| 7 | Prop-ablation runner and verdict (corrected, surrogate/dev-only) | `exploratory_locked` | NO | 4 | 1 |

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

### Tracking/hustle collector (infrastructure only) (`tracking_hustle_collector`)
- Status: `not_landed`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Collector code only. Tracking DATA is NOT landed and tracking FEATURES are NOT built.
- Limitation: Runs on a LOCAL residential IP only; never in CI/agent VMs (datacenter IPs are blocked).
- Limitation: This is not Phase B; it proves nothing about discrimination or market edge.

### Current feature-matrix snapshot (`feature_matrix_snapshot`)
- Status: `locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: The wide parquet is git-ignored (large, pipeline-built); the live-parquet drift check is deferred in clean CI checkouts.
- Limitation: A snapshot describes the matrix; it does not validate model performance.

### Prop-ablation runner and verdict (corrected, surrogate/dev-only) (`prop_ablation_runner_verdict`)
- Status: `exploratory_locked`; promotion-eligible: NO; required CI job: `foundation-lock`
- Limitation: Surrogate model (HGB Poisson -> NB PMF), NOT the signed champion OOF/PMF/calibration/delivery pipeline.
- Limitation: Development-only exploration; conditional-on-played (DNP excluded). NOT proof, certification, or promotion evidence.
- Limitation: Conclusion: no tested subset improved the surrogate under this protocol; tracking is a high-priority next source, not proven to be the only one.

## Deferred (declared data artifacts absent from this checkout)

- tracking_hustle_collector: wnba_tracking_2021_2026.parquet (data artifact) not present -> DEFERRED
- tracking_hustle_collector: wnba_hustle_2021_2026.parquet (data artifact) not present -> DEFERRED

