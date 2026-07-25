# G0-v2 critical-path report (existing model, exact-quote evaluation)

Scope: **evaluate / calibrate / certify** the existing frozen model against **exact,
decision-time** market quotes. No new model architecture, no feature/training changes.
Every number below is measured from committed artifacts (no fabrication).

Reproduce:

```bash
python3 scripts/build_g0v2_exact.py --test-dates 30           # G0-v2 exact scored table + metrics
python3 scripts/build_lowcost_candidates.py                   # C0-C6 (primary book, cross-fit selection)
python3 scripts/build_lowcost_candidates.py --primary-book all --out-dir artifacts/market_feature_proof/G0_v2_allbooks
python3 scripts/freeze_and_prove_closest.py --prop ast --candidate C4_blend --primary-book all --out-dir artifacts/market_feature_proof/G0_v2_proof_pooled_ast
python3 scripts/build_critical_path_status.py                 # consolidates deliverables 1-7
```

## 1. Data preservation
- Recovery bundle built (`data/recovery/recovery_bundle_v1.tar.gz`, ~11 MB) of the real
  committed evaluation artifacts (7-prop OOF PMFs, atomic exact-quote store, closing
  consensus, prequential ledger, G0 input, manifests/registry). Manifest:
  `artifacts/data_bootstrap/RECOVERY_BUNDLE_MANIFEST.json`.
- **BLOCKED_PRIVATE_DATA_REPO_MISSING**: the private data repository does not exist (404)
  and the fine-grained writer token cannot create it (403). **Owner action:** create the
  empty private repo and grant the token `Contents: read/write`, then
  `python3 scripts/publish_data.py --name recovery_bundle` and verify with
  `python3 scripts/fetch_data.py recovery_bundle --check`.
- The five frozen baseline feature assets are **UNRECOVERABLE_ON_THIS_VM** (gitignored,
  never released, no `BDL_API_KEY` to rebuild). Target hashes preserved for a future
  authenticated rebuild.

## 2. Existing-model OOF status — COMPLETE (one full run)
`artifacts/models/calibration/oof_predictions.parquet`: 32,802 rows, **all 7 props**
(4,686 each), **11 rolling-origin folds**, `oof_prediction_type = model_oof` for 100% of
rows (no failed-model fallback, no prior-only rows), `minutes_prediction_type = model`
100%, train-end < validation-start for every row, 0 null PMFs, PMFs valid. Window
2026-05-08 .. 2026-07-19.

## 3. Exact market comparison data (decision-time snapshot, per book, no averaging)
Atomic store `artifacts/p1/p1_quotes.parquet` (76,620 rows, 5 books, open/decision/close,
both sides, `odds_api_v4_historical`, exact ids). Exact Over/Under decision-time pairs
joined to OOF actuals (non-push):

| prop | draftkings rows | dates | all-books rows | status |
|---|---:|---:|---:|---|
| pts | 824 | 55 | 2,924 | OK |
| reb | 695 | 55 | 2,401 | OK |
| ast | 458 | 55 | 1,466 | OK |
| fg3m | 475 | 55 | 1,632 | OK |
| stl / blk / turnover | 0 | 0 | 0 | **NO_EXACT_QUOTES** |

## 4. G0-v2 metrics (C0 = raw existing model vs exact market, primary book)

| prop | n | model LL | market LL | ΔLL | model Brier | market Brier | ΔBrier | model AUC | market AUC | ΔAUC | model ECE | market ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pts | 824 | 0.802 | 0.693 | +0.110 | 0.284 | 0.250 | +0.034 | 0.485 | 0.522 | −0.036 | 0.159 | 0.019 |
| reb | 695 | 0.771 | 0.681 | +0.090 | 0.272 | 0.244 | +0.028 | 0.551 | 0.595 | −0.045 | 0.138 | 0.040 |
| ast | 458 | 0.781 | 0.694 | +0.087 | 0.271 | 0.250 | +0.020 | 0.534 | 0.525 | +0.009 | 0.136 | 0.064 |
| fg3m | 475 | 0.779 | 0.671 | +0.108 | 0.268 | 0.239 | +0.029 | 0.552 | 0.603 | −0.051 | 0.128 | 0.043 |

Raw model **loses to the exact market on all four props**; model ECE is far worse than
market's (calibration is a large part of the raw gap).

## 5. Low-cost candidates C0-C6 (cross-fit on selection window only)
Advancement rule: candidate LL < market LL **and** candidate Brier < market Brier, with
acceptable ECE. On the selection window several candidates advance (e.g. fg3m `C2_beta`
pure recalibration ΔLL −0.022; ast `C4_blend` ΔLL −0.006). **But these edges do not
generalize:** on the untouched forward window the pure-model recalibration edges **reverse**
(fg3m `C2_beta` ΔLL +0.036, ast `C2_beta` +0.011). Full tables:
`artifacts/market_feature_proof/G0_v2/LOWCOST_CANDIDATE_METRICS.*`.

## 6. Untouched-window proof (frozen contract: ≥300 rows, ≥30 clusters, cluster bootstrap, Holm)

| prop | candidate | n | clusters | ΔLL | ΔBrier | ΔAUC | status |
|---|---|---:|---:|---:|---:|---:|---|
| ast | C4_blend (pooled) | 914 | 30 | −0.0054 | −0.0026 | −0.008 | **FAIL** (not significant) |
| reb | C4_blend (pooled) | 1487 | 30 | −0.0024 | −0.0011 | −0.003 | **FAIL** |
| fg3m | C2_beta (pooled) | 1029 | 30 | +0.036 | +0.017 | −0.099 | **FAIL** |
| pts | C1_platt (pooled) | 1796 | 30 | +0.004 | +0.002 | +0.025 | **FAIL** |

**No prop certifies.** Closest proof-ready prop = **ast** (only prop whose discrimination
matches the market, so recalibration reaches the correct forward sign; margin not yet
significant).

## 7. Per-prop failure diagnosis (owner step 8) & the exact blocker
- **pts** → discrimination (model AUC 0.485 < 0.50 at market lines).
- **reb** → discrimination (market AUC materially higher).
- **fg3m** → discrimination (largest AUC gap; selection calibration edge overfit).
- **ast** → calibration + insufficient forward exact-quote volume (**closest to passing**).
- **stl / blk / turnover** → no exact quotes collected (not evaluable).

**Exact remaining blocker to the first certified Edge Board row:** no prop's probabilities
(existing model ± low-cost correction) certifiably beat the exact decision-time market on
the untouched forward window; the exact-quote history is a single partial season (~55
game-dates) and the only discrimination-competitive prop (ast) is not yet statistically
significant. Targeted model-signal (discrimination) repair for pts/reb/fg3m is **deferred**
pending this evidence, per owner instruction (no new architecture without measured need).
The Edge Board therefore has **zero certified props** and cannot publish a certified row yet.

---

## Phase 0–3 update (corrected evidence contract)

**0.1 Historical window relabeled** `DEVELOPMENT_SELECTION_EVIDENCE / HISTORICAL_HOLDOUT_DIAGNOSTIC /
NOT_FUTURE_PROOF / NOT_PROMOTION_ELIGIBLE` (`HISTORICAL_WINDOW_CLASSIFICATION.json`). All prior
P14 results preserved unchanged. Every new candidate opens a **new prospective** proof period.

**0.2 Deterministic one-quote policy** `config/book_quote_priority_v1.json` (frozen priority
draftkings > fanduel > williamhill_us > betrivers > betonlineag). Primary = one exact quote per
`game_id+player_id+prop`; all-books pooled is **SENSITIVITY only**. G0-v2, candidates, and the
closest-prop ranking were regenerated on this primary. Under one-quote, all four props clear the
300-row / 30-date floor.

**0.3 Probability/target semantics** (`PROBABILITY_TARGET_SEMANTICS_AUDIT.json`): **PTS =
SIGN_INVERSION_DIAGNOSTIC** (mean model P(over) 0.379 vs empirical over-rate 0.479, Platt slope
−0.009, Spearman −0.024, AUC 0.486; cross-fit AUC 0.470 with slope −2.66). No gross target/line
defect found (under = 1 − over by construction; prob computed at the exact quote line; pushes
excluded; identities clean) → the deficit is a genuine near-null/inverted signal, not a bug.
reb/ast/fg3m are oriented correctly.

**0.4 Calibrator monotonicity** (`CALIBRATOR_MONOTONICITY_AUDIT.json`): Platt/Beta by prop+fold;
PTS has negative Platt slope in 3/5 folds (SIGN_INVERSION_DIAGNOSTIC); reb/ast/fg3m monotone. The
pooled-AUC change from cross-fitted calibration is a cross-fold intercept artifact (within-fold
ranking preserved), not added discrimination.

**Phase 2 — AST first edge:** candidate **A1 (monotone-calibrated existing model) is FROZEN**
(`artifacts/candidate_freeze/AST_FIRST_EDGE_FREEZE.json`). On the full development window it beats
the exact one-quote market on both proper scores (ΔLL −0.0037, ΔBrier −0.0019, ECE 0.016, no
catastrophic fold). A2/A3 collapse to market (β→0) under the required strong shrinkage; A4
(feature residual) is **BLOCKED_NO_FEATURE_MATRIX**. Certification is deferred to a **prospective**
proof (new dates after the freeze timestamp); the historical diagnostic is not significant (FAIL),
as expected.

**Phase 3 — PTS/REB/FG3M:** no existing-output challenger beats the market. PTS shows out-of-fold
sign inversion; reb/fg3m have a real market AUC advantage (0.587/0.617 vs 0.545/0.546). The only
repair that could add genuine discrimination (A4 feature residual) is **BLOCKED_NO_FEATURE_MATRIX**.

**Phase 1/4 — forward collection:** the collector supports all seven market keys (incl.
steals/blocks/turnovers); forward collection and stl/blk/turnover backfill are
**BLOCKED_NO_ODDS_API_KEY**. `BDL_API_KEY` is absent (blocks fresh daily predictions, not historical G0).

**Backup:** the private-repo authenticated 404 is **BLOCKED_TOKEN_REPOSITORY_ACCESS** (the token
lists 20 Risky-Scout repos but not the data repo; name matches, owner correct) — a token-scope
issue, not a missing repo. Publish the recovery bundle once access is granted.

---

## Correction pass (leaky-CV defect fix)

A valid defect was found: the candidate cross-validation was **leave-block-out** (folds trained on
future date blocks), not rolling-origin. Corrections:

- **AST A1 freeze INVALIDATED** (`AST_FIRST_EDGE_FREEZE_INVALIDATION.json`): status
  `INVALIDATED_TEMPORAL_CV_LEAKAGE / NOT_PROOF_ELIGIBLE / NOT_DEPLOYABLE / NOT_CERTIFIED`; prior
  evidence preserved. No proof rows collected under it.
- **CV repaired** → `src/wnba_props_model/evaluation/rolling_origin.py`: grouped **expanding-window**
  folds with the invariant `max(train_date) < min(validation_date)`, machine-verifiable fold
  manifest (`ROLLING_ORIGIN_FOLD_MANIFEST.json`, all `chronology_pass=true`), and **nested** inner
  selection. Tests (`tests/test_rolling_origin_cv.py`) reproduce the old leave-block-out folds and
  **require** the chronology check to flag them.
- **One canonical scored-row universe** `PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet` (key
  `game_id+player_id+prop`, one row max) drives all metrics. Row-count reconciliation: primary
  deterministic one-quote = **pts 851 / reb 742 / ast 522 / fg3m 504** (draftkings first, backfilled
  by lower-priority books when absent); the legacy **824/695/458/475** were draftkings-only and are
  superseded.
- **Quote-policy hash resolved**: authoritative = raw file SHA-256
  `962db96af3cceb31eb0e2efc08ca5f069e517e131e10d1a76619de4f8a20c780`; the stale freeze value
  `4b39ee8f…` was a label string-hash (invalidated). Tie-break audit: exactly one eligible pair per
  observation at the selected book → tie-break unused; policy v1 retained.
- **Corrected C0–C6 (nested rolling-origin):** ast advances on proper score (C4_blend ΔLL −0.0048,
  C5 −0.0016); reb advances (C4/C6); fg3m/pts none. **Strict-AUC gate:** no ast candidate beats
  market AUC (best −0.0076) → `strict_auc_selection_eligible=False`.
- **AST re-freeze v2** (`AST_FIRST_EDGE_FREEZE_V2.json`): candidate **C4_blend**, **proper-score
  track**, `proper_score_selection_eligible=True`, `strict_auc_selection_eligible=False` (explicitly
  not strict-gate-ready); raw file policy SHA + canonical + fold-manifest + code hashes recorded.
- **Calibrator monotonicity (expanding-window):** PTS has a **negative Platt slope in all 14 folds**
  (SIGN_INVERSION_DIAGNOSTIC); reb/ast/fg3m monotone.
- **Phase 3:** no existing-output challenger beats market for pts/reb/fg3m; A4 feature residual
  `BLOCKED_NO_FEATURE_MATRIX`.
