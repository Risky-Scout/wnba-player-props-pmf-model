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
