---
name: market-superiority-validation
description: >-
  Canonical offline, leakage-free path for proving (or refusing to claim) that the WNBA
  player-prop PMF model beats the no-vig closing market. Run this whenever validating market
  superiority, running the backfill → OOF → assemble → evaluate_market_superiority pipeline,
  interpreting p1/market_feature_proof artifacts, reasoning about calibration/market/CLV gate
  thresholds, or before promoting any prop. Use it as the source of truth for the command
  sequence, exact gate thresholds, guardrails, and rollback so cold-start agents run the same
  correct path instead of re-deriving it.
---

# Market-Superiority Validation

## Mission

This repo is a production-grade WNBA player-prop PMF model. The **market-superiority
validation loop** is the one path that proves — offline and leakage-free — whether the model
reliably beats the **no-vig closing market**, before any prop is promoted. Every cold-start
agent must execute this exact path instead of re-deriving it; re-derivation is how bugs and
lost hours crept in (stale-table silent empty outputs, leaky OOF folds, scoring the wrong
probability column, claiming an invalidated verdict).

**Golden rule:** a PASS is evidence only for the exact frozen candidate, books, line
timestamps, date range, and population in the input. It is never a guarantee of future
profitability, and no locked component is promotion-eligible on its own.

## Command sequence (run in this order)

Do not skip, reorder, or substitute steps. Each consumes the previous stage's artifact.

### 1. Backfill — historical market quotes (`p1_historical_backfill.py`)

Pulls opening/decision/closing line snapshots for games in the OOF data, canonicalizes
provider IDs to model IDs (exact, roster-constrained — never fuzzy), pairs Over/Under for
no-vig probabilities. Offline evaluation only.

```bash
python scripts/p1_historical_backfill.py \
  --oof data/oof/oof_player_stat_pmfs.parquet \
  --games data/processed/wnba_games.parquet \
  --roster data/processed/wnba_player_game_stats.parquet \
  --out-dir artifacts/p1 --cache-dir artifacts/p1_cache
```

Outputs: `artifacts/p1/p1_quotes.parquet`, `p1_opening_consensus.parquet`,
`p1_closing_consensus.parquet`, `p1_unmatched_audit.parquet`, `p1_coverage_summary.json`.
Requires `ODDS_API_KEY` in env. Idempotent/resumable via `--cache-dir`. Use `--pilot-dates N`
for a cheap first pass, `--preflight-only` to verify entitlement without spending credits.

### 2. OOF — walk-forward out-of-fold PMFs (`build_oof_pmfs.py`)

Strict expanding-window chronological splits: each validation fold is predicted by a model
trained exclusively on `game_date < fold_validation_start_date` (no lookahead). If a fresh
`data/oof/oof_player_stat_pmfs.parquet` already exists (e.g. downloaded from the `oof-latest`
artifact), reuse it rather than regenerating.

```bash
python scripts/build_oof_pmfs.py \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --features-long data/processed/wnba_player_game_features_long.parquet \
  --manifest data/processed/feature_schema_manifest.json \
  --config config/model/stage5_oof.yaml \
  --out-dir data/oof \
  --audit-out artifacts/audits/stage5_oof_audit.json
```

Output: `data/oof/oof_player_stat_pmfs.parquet` (long, raw fold-safe) + `_wide` + fold
manifest. This is the MODEL + OUTCOME source for the next step.

### 3. Assemble — build the evaluator input (`build_market_superiority_input.py`)

Bridges MARKET (closing consensus no-vig P(over) at the modal line), OUTCOME (`actual_outcome`
from OOF), and MODEL (P(over) at the closing line from the OOF PMF, then **fold-safe
walk-forward calibrated** so it is production-equivalent). Emits the evaluator contract
columns.

```bash
python3 scripts/build_market_superiority_input.py \
  --closing artifacts/p1/p1_closing_consensus.parquet \
  --oof data/oof/oof_player_stat_pmfs.parquet \
  --out artifacts/p1/market_superiority_input.parquet \
  --run-eval
```

`--run-eval` invokes step 4 directly in prove mode. Keep `--calibrate` on (default). A
chronological split (`--split-date` or `--test-frac 0.4`) reserves later games as the
untouched forward test. Output columns: `game_date, prop, candidate, split, actual, line,
model_prob_over_final, market_prob_over_no_vig`.

### 4. Evaluate — the market-superiority verdict (`evaluate_market_superiority.py`)

Separates **selection** (choose one frozen candidate per prop on the selection split) from
**proof** (score only the frozen candidates on the untouched test split). Date-clustered
bootstrap with Holm adjustment.

```bash
python3 scripts/evaluate_market_superiority.py \
  --input artifacts/p1/market_superiority_input.parquet \
  --output-dir artifacts/market_feature_proof/from_archive \
  --mode prove
```

Modes: `audit` (exploratory per-candidate metrics), `select` (freeze candidate map on the
selection split, writes `selected_candidates.json`), `prove` (score frozen candidates on the
test split). If multiple candidates per prop exist in prove mode, supply the frozen
`--selected-candidates` map. Outputs `market_superiority_proof.{csv,json}`,
`MARKET_SUPERIORITY_REPORT.md`.

## Exact gate thresholds

### Market-superiority proof gate (`evaluate_market_superiority.py`, prove mode)

Defaults: `--bootstrap 5000`, `--seed 20260720`, `--min-rows 300`, `--alpha 0.05`,
`--min-logloss-delta 0.0`, `--min-brier-delta 0.0`, `--min-auc-delta 0.0`.

- Deltas are **challenger minus market**. Log loss / Brier **negative is better**; AUC
  **positive is better**. Pushes (`actual == line`) are excluded from binary metrics.
- Per prop, `market_superiority_gate` is `PASS` only when ALL three pass:
  - `logloss_delta_ci_high < -|min_logloss_delta|` AND `logloss_delta_p_holm <= alpha`
  - `brier_delta_ci_high  < -|min_brier_delta|`  AND `brier_delta_p_holm  <= alpha`
  - `auc_delta_ci_low     >  |min_auc_delta|`     AND `auc_delta_p_holm    <= alpha`
- Requires `n_settled >= min_rows` and `>= 2` date clusters, else `INSUFFICIENT` (never PASS).
- `all_props_pass` is true only if every scored prop is PASS.

### Calibration gates (`verify_gates.py calibration`, config `config/model/stage6_calibration.yaml`)

- Strict CLI defaults (post-cal): ECE < 0.03, PIT KS < 0.075, |mean_error| < 0.15.
- Mid-season config (relaxed, current production): `ece_threshold 0.25`, `pit_ks_threshold
  0.25`, `mean_error_threshold 3.60`.
- Pre-cal sanity (`--pre-cal`, catches broken models): ECE < 0.12, PIT KS < 0.20,
  |mean_error| < 1.0.
- `min_rows_for_calibration: 50`. Combo stats use `combo_ece_threshold: 0.06`.
- Excluded from the strict gate (calibrators still fit + applied): stats `blk, turnover,
  fg3m`; role buckets `inactive_risk, fringe, bench, rotation`.
- Always pass `--config config/model/stage6_calibration.yaml` so thresholds/exclusions match
  production; never hand-tighten in CLI.

### Market gate (`verify_gates.py market`, UCB95 certified pass)

- `event_logloss_delta` UCB95 < **-0.0025** AND `brier_delta` UCB95 < **-0.0010**.
- `--min-rows` default 100; informational until 300+ samples per stat, then `--hard`.

### CLV tracking gate (`verify_gates.py clv_tracking`)

- `positive_clv_pct >= 0.52` AND `mean_clv > 0.0` over a rolling 30-day window.
- `min_rows_per_stat 100`; hard-fails at `hard_fail_rows 300`.
- CLV is signed (`price_clv`/`line_clv`, close minus open no-vig for the selected side).
  Fail-closed: never a nonnegative `max(|edge_over|,|edge_under|)` proxy.

### Production gates (`verify_gates.py production-gates`, all 6)

ECCE-MAD < 0.05; PIT KS p-value > 0.05; coherence divergence < 5.0 pts; edge distribution
10–35% of props with edge ≥ 4pp; rolling ROI > 0 on last 200 games; live rate corrections in
[0.7, 1.3].

## Guardrails (do not violate)

- **Offline & production-safe.** The P1 workflow (`.github/workflows/p1_historical_validation.yml`)
  must never deploy, publish, commit calibration, run `fit_calibrators`, touch `pregame*`, run
  `generate_web_pages`/`ftp_deploy`, `git commit`/`git push`, or `gh pages`. Enforced by
  `tests/test_p1_workflow_contract.py`.
- **Fail-closed identity/coverage.** Backfill refuses to write empty outputs. `--min-event-match-rate`
  default 0.5; events found but 0 usable quotes = FATAL stale `games`/`roster` table. Fix by
  regenerating canonical tables (`build_canonical_tables.py`) or fetching current ones
  (`fetch_data.py`); inspect `p1_unmatched_audit.parquet`. Only `--allow-empty` bypasses, and
  only when the gap is expected.
- **No lookahead / no dedup masking.** OOF must pass `assert_no_lookahead`; duplicate
  `(game_id, player_id, stat)` keys are FATAL — never `drop_duplicates` them away.
- **Score the delivered probability.** Always `model_prob_over_final`. The legacy
  `model_prob_over` column is forbidden in real proof mode; do not reconstruct P(over) outside
  the production lineage (`probability_lineage.py` is the sole creator).
- **Freeze before proving.** In prove mode with multiple candidates per prop, load a frozen
  `selected_candidates.json` from a separate selection split. Never select and prove on the
  same rows.
- **Secrets.** Never print, log, or persist `ODDS_API_KEY` / `BDL_API_KEY`. Raw responses go to
  a local/artifact cache dir, never committed.
- **Foundation Lock.** Hash-pinned artifacts in `config/foundation_lock_v1.json`;
  `python scripts/verify_foundation_lock.py` fails on any drift.
  `scripts/check_phase0_scope.py` blocks edits to live-delivery surfaces (`models/market.py`,
  `pipeline/deliver.py`, `pipeline/calibrate.py`, `build_scored_candidates.py`,
  `recommendation_policy.yaml`, `stat_registry.json`, live/pregame/daily workflows) unless
  listed in an approved `config/phase0_scope_exception.json`.
- **Honesty about the baseline.** Do NOT claim "G0 wins every prop" — that surrogate verdict is
  INVALIDATED (silent 76/128 feature drop). The locked G0 baseline proves the current model
  FAILS vs the market on every quoted prop. Do not reuse the invalidated result
  (`artifacts/feature_ablation/ABLATION_STATUS.json`).
- **New-branch protocol.** Start from current `main`; run
  `git diff --name-status foundation-lock-v1.1...HEAD` before opening a PR. Consult
  `artifacts/mission_state/MISSION_STATE_V1.json` `do_not_repeat` list before starting.

## Rollback

- **Never `--update` the lock in CI.** `python scripts/verify_foundation_lock.py --update` is a
  maintainer-only, explicitly-reviewed action to re-pin hashes after an intentional change.
- **Revert to a known-good foundation.** Locked tags: `foundation-lock-v1`
  (`d8c9681d6a6334ad8b0897b53009fc28d28dd342`, immutable) and `foundation-lock-v1.1`
  (`574a87a6b71cef6dc2b66b74643f20cd6d7a6420`). Reset the working state to a tag when a change
  breaks lock verification and cannot be quickly fixed.
- **Calibrators promote only on gate pass.** Weekly calibration replaces `calibrators-latest`
  only when the calibration gate passes. To roll back a bad promotion, stop promoting and
  restore the prior `calibrators-latest` artifact; delivery keeps using the last good one.
- **Backfill is resumable, not destructive.** A failed/partial run resumes from `--cache-dir`;
  no partial writes on the fatal coverage gate. Re-run after fixing stale canonical tables.
- **Discard, don't overwrite, evaluator outputs.** If a proof run used the wrong split, wrong
  candidate map, or leaky OOF, delete the run's `--output-dir` and re-run; never edit
  `market_superiority_proof.*` by hand — the pinned G0 baseline hashes must stay intact.

## Quick self-test

Validate evaluator mechanics (synthetic, not market evidence) before trusting a real run:

```bash
python3 scripts/evaluate_market_superiority.py --self-test \
  --output-dir artifacts/market_feature_proof/self_test
```

Expect `selection_chose_expected_candidate: true` and `all_synthetic_props_pass: true`.
