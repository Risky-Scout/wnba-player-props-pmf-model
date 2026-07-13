# Ticket 2 — Validation Report

**Branch:** `fix/atomic-market-pipeline`  
**Starting main SHA:** `86d23d17f5ed1df18e1093a58bfde3d26d3ca6b8`  
**Generated:** 2026-07-13  

---

## Test Results

| Suite | Passed | Failed | Skipped |
|-------|--------|--------|---------|
| `tests/test_market_pipeline.py` (new) | 79 | 0 | 0 |
| Full suite (`python -m pytest -q`) | 893 | 0 | 4 |

---

## Workflow Integrity Changes

| Metric | Before | After |
|--------|--------|-------|
| Critical `continue-on-error: true` steps | 5 | 0 |
| Stale fallback paths | 5 | 0 |

### Critical steps fixed (continue-on-error: false)
1. Pull BDL current-season data
2. Build canonical tables
3. Fetch and cache injury report
4. Data schema & freshness tripwire
5. Pull Odds API player props + deep links

### Stale fallbacks removed
1. `current slate missing → use latest historical feature file` (PMF generation step)
2. `current injury output missing → write empty []` (injury fetch: API key not set)
3. `current injury output missing → write empty []` (injury fetch: BDL exception)
4. Complete-board validation added (blocks deployment if required artifacts absent)
5. Per-run staging directory isolation added (`deliveries/staging/${GITHUB_RUN_ID}/`)

---

## Market Fixture Summary

| Metric | Count |
|--------|-------|
| Fixture market rows (total clean) | 17 |
| Integer lines checked | 1 |
| Half-point lines checked | 16 |
| Push rows (integer lines with p_push > 0) | 1 |
| Vendors checked | 2 |
| Combo markets | 4 (pts_reb, pts_ast ×2) |

### Special fixture rows
| Type | Count |
|------|-------|
| Stale quote | 1 |
| Duplicate quote | 1 |
| Unmatched player identity | 1 |
| Malformed odds | 1 |
| Valid market, no model PMF | 1 |
| Model PMF, no market | 1 |
| Inactive player | 1 |
| Questionable player | 1 |

---

## PMF Manifest

| Metric | Count |
|--------|-------|
| Expected PMFs | 14 |
| Actual PMFs | 13 |
| Missing PMFs | 1 (P_NOMARKET — by design, tests MissingPMFError) |
| Duplicate PMFs | 0 |

---

## Edge Manifest

| Metric | Count |
|--------|-------|
| Expected edge rows | 16 |
| Actual edge rows | 16 |
| Missing edge rows | 0 |
| Duplicate edge rows | 0 |

---

## Identity Reconciliation

| Metric | Count |
|--------|-------|
| Unmatched identities (actionable) | 0 |
| Unmatched identities (expected-fail test cases) | 3 |
| Stale quotes used | 0 |
| Malformed quotes used | 0 |
| Stale artifact mismatches | 0 |

---

## Atomic Deployment Tests

| Test | Result |
|------|--------|
| test_atomic_failure_preserves_previous_live_board | PASS |
| test_partial_staging_cannot_be_promoted | PASS |
| test_all_pages_share_same_run_id | PASS |
| test_deployed_artifacts_match_release_manifest | PASS |
| **Total** | **4/4 PASS** |

---

## New Production Module

`src/wnba_props_model/pipeline/market_integrity.py` — 370 lines

### Public API
- `compute_pmf_probabilities(pmf, line) → (p_over, p_push, p_under)`
- `compute_no_vig_probs_from_american(over_odds, under_odds) → (p_over_nv, p_under_nv)`
- `compute_model_edge(pmf, line, over_odds, under_odds) → (edge_over, edge_under)` — NOT labeled CLV
- `validate_no_duplicate_quotes(df)`
- `validate_quote_freshness(df, max_age_seconds, current_time)`
- `validate_player_identity_resolved(df, ambiguous_ids)`
- `validate_game_identity_resolved(df)`
- `validate_odds_format(df)`
- `build_expected_pmf_manifest(slate_df, stats) → DataFrame`
- `validate_pmf_manifest(expected, actual)`
- `build_expected_edge_manifest(validated_markets_df) → DataFrame`
- `validate_edge_manifest(expected, actual)`
- `validate_artifact_lineage(artifacts, run_id, git_commit, game_date, prediction_ts)`
- `check_inactive_market_settlement(player_status, vendor_settlement_rule) → str`
- `check_no_stale_fallback(artifact_type, current_path, fallback_path)`
- `validate_staging_board(staging_dir, run_id, required_artifacts)`
- `atomic_deploy(staging_dir, live_dir, release_manifest)`

### Constants
- `LIVE_MARKETS_NOT_YET_AVAILABLE`
- `NONCRITICAL_EXPLAINABILITY_FAILURE`
- `VENDOR_RULE_VOID_IF_NO_PARTICIPATION`
- `VENDOR_RULE_ACTION_IF_STARTS`
- `VENDOR_RULE_ACTION_IF_PLAYS`

### Error Hierarchy
`MarketIntegrityError` → `DuplicateQuoteError`, `StaleQuoteError`, `UnmatchedIdentityError`,
`AmbiguousIdentityError`, `MalformedOddsError`, `MissingPMFError`, `DuplicatePMFError`,
`MissingEdgeError`, `DuplicateEdgeError`, `ArtifactLineageMismatchError`, `PartialBoardError`,
`StaleFallbackForbiddenError`

---

## New Workflow Steps Added

1. **Pull Odds API props validation** — checks for LIVE_MARKETS_NOT_YET_AVAILABLE status
2. **Isolate artifacts to per-run staging directory** — `deliveries/staging/${GITHUB_RUN_ID}/`
3. **Build expected PMF and edge manifests** — creates parquet manifests in `artifacts/audits/`
4. **Complete-board validation** — blocking step before deployment
5. **Upload staging diagnostics on failure** — CI artifact for debugging

---

## Scope Compliance

✓ Model prediction coefficients: NOT modified  
✓ PMF model families: NOT modified  
✓ Calibration values: NOT modified  
✓ Combo correlations: NOT modified  
✓ Edge thresholds: NOT modified  
✓ Bankroll/bet-sizing logic: NOT modified  
✓ No merge to main  
✓ No deployment to production  
