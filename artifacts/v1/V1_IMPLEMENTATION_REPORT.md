# V1 Implementation Report

## Delivered (implemented, tested, run on a fixture slate)
- Feature-driven **pricing engine** (`pricing/engine.py`): P(over/under/push), push-safe settled
  probabilities, fair decimal/American, margin layer that never mutates the PMF, yes/no + full
  normalized categorical vectors. Odds conversions verified.
- Canonical **market registry** covering every Section-2 market key; alternates settle from the
  base distribution; fantasy requires an explicit scoring-rule id.
- Coherent **joint generator**: shared-latent (minutes) deterministic simulation producing all
  primitive active-player outcomes with the structural identities holding in every sample;
  combination markets use joint dependence (verified different from independent convolution);
  separate Q1 layer (not full×0.25); p_dnp kept separate from the zero atom.
- **First-basket** competing-risk (event sums to 1; per-team sums to 1; method categorical).
- **Monotone-CDF calibration** hooks (never per-line isotonic).
- BDL client fixes (`prop_type`; `plays` non-paginated).
- End-to-end **fixture pricing run**: `deliveries/pricing_v1/<date>/` (atom PMFs, joint metadata,
  fair prices, priced inventory, manifest) + audits (atom completeness, structural identity,
  monotonicity, tail mass) + coverage.
- Immutable **release-candidate bundle** `artifacts/releases/wnba-pricing-pmf-v1.0.0-rc1/`
  (MANIFEST, MODEL_CARD, MARKET_REGISTRY, SUPPORT_AND_TAIL, TEST_REPORT, DATA_LINEAGE, SHA256SUMS).
- ~20 pricing tests (identities, monotonic alternates, push/half-point, categorical, odds,
  margin-does-not-change-PMF, DNP-separate, joint-dependence).

## Not done / blocked (honest)
- Live "today's pricing run" is **BLOCKED**: the real slate needs the Phase-1 recovered
  feature/quote data, which is not on this branch (it lives on `cursor/per-stat-compact-pmf-v1`).
- Full Odds-API atomic-schema rework + historical backfill are **Phase-1** and not re-done here.
- Chronological market evaluation + **market-superiority certification** are a separate gate and
  are NOT claimed. No product is VALIDATED / bettor-ready.
- Market-anchored production track (exp-tilt/KL to no-vig + chronological residual) is scoped but
  not run.
