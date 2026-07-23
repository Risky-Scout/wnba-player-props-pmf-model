# PR 1A - Probability call-site audit

Inventory command:

```
rg -n 'prob_over_from_pmf|settled_probabilities_from_pmf|model_prob_over|pmf_json' src scripts tests
```

## Classification

### Lineage creator (must be exactly 1)
- `src/wnba_props_model/models/probability_lineage.py` - the **only** creator of
  `model_prob_over_final` (`build_probability_lineage`). Verified by
  `tests/test_probability_source_of_truth.py` and
  `tests/test_decision_grade_probability_source_guard.py` (no `ProbabilityLineage(` construction
  elsewhere). **creators = 1.**

### Legacy output alias (output-only, deprecated)
- `src/wnba_props_model/pipeline/deliver.py` - writes `model_prob_over = model_prob_over_final`
  (Store only) with `probability_alias_version="v1"`; invariant `model_prob_over == final`
  within 1e-12. Deliver reads only `model_prob_over_final` internally (edge, fair odds, Kelly).
- `src/wnba_props_model/models/probability_contract.py` - defines the constants +
  `require_final_probability`/`validate_final_probability`/`assert_alias_invariant`.

### Decision-grade final-probability consumer (MIGRATED)
- `src/wnba_props_model/pipeline/deliver.py` (live delivery) - **migrated**; AST guard green.

### Decision-grade consumers NOT YET migrated (still read `model_prob_over` / reconstruct)
These remain to reach the 0/0/1 acceptance target and are the remaining PR 1A work:

| Consumer | Foundation-Lock hash-pinned? | Notes |
|---|---|---|
| `scripts/build_scored_candidates.py` | no (but Phase-0 scope-protected; PR 1B/C surface) | reconstructs via `_p_over`/`ceil` |
| `scripts/evaluate_market_superiority.py` | **YES (pinned)** | default `--model-prob-col model_prob_over` |
| `scripts/generate_clv_report.py` | **YES (pinned)** | reads `model_prob_over` |
| `scripts/score_daily_predictions.py` | **YES (pinned)** | reads `model_prob_over` |
| `scripts/build_edge_report.py` | no | reads `model_prob_over` |
| `scripts/export_betting_sheet.py` | no | reads `model_prob_over` |
| `scripts/generate_web_pages.py` | no | reads `model_prob_over` |
| `src/wnba_props_model/evaluation/oof_scoring.py` | no | full-PMF scoring helpers (line reconstruction) |
| `src/wnba_props_model/evaluation/historical_market.py` | no | replay uses `model_prob_over` |

`src/wnba_props_model/pipeline/recommendation.py` consumes a probability **parameter**
(`model_prob_over` arg), not a DataFrame column; migration is done by having callers pass
`model_prob_over_final`.

### Permitted PMF diagnostics (not decision-grade)
- `src/wnba_props_model/evaluation/diagnostics.py` - uses `settled_probabilities_from_pmf(...)
  .p_over_unconditional` via `_pmf_over_unconditional`; emits `pmf_prob_over_unconditional`.
- `src/wnba_props_model/models/pmf_grid.py`, `pmf_utils.py`, `evaluation/forecasting.py`,
  `visualization/*` - fair-odds ladders / full-PMF scoring / plots.

### Test fixtures
- `tests/test_*` reference the columns/functions for coverage.

## Honest totals (current)
- **creators of `model_prob_over_final` = 1** (met).
- **decision-grade consumers reading `model_prob_over` = 8** (target 0; only delivery migrated).
- **decision-grade consumers reconstructing from PMF = 2+** (`build_scored_candidates`,
  `oof_scoring`; target 0).

## Blocker requiring an owner decision
Three remaining consumers (`evaluate_market_superiority.py`, `generate_clv_report.py`,
`score_daily_predictions.py`) are **hash-pinned in `config/foundation_lock_v1.json`**.
Migrating them changes their bytes and will fail the `foundation-lock` CI check unless the
Foundation Lock manifest is re-pinned (`python scripts/verify_foundation_lock.py --update`).
Re-pinning is a Foundation Lock governance action; the execution contract said "Do not redo
Foundation Lock except where a regression test requires it." Additionally
`build_scored_candidates.py` is a Phase-0 scope-protected, PR 1B/1C quote-path surface.
The full 0/0/1 migration is therefore gated on an owner decision to re-pin Foundation Lock
and to authorize editing the PR 1B surface within PR 1A.
