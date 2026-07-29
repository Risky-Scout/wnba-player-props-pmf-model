# Per-Stat Compact PMF Rebuild — Corrected Experiment Design & Status

Branch: `cursor/per-stat-compact-pmf-v1`
Baseline SHA frozen in `artifacts/per_stat_compact/PRE_REBUILD_BASELINE.json`.

This document is the corrected experiment design the directive requires **before** a long
OOF is launched. It also records the immediate-fix results and an honest status. It does
**not** claim market superiority and does **not** authorize production activation.

---

## 0. Scope delivered in this pass

The directive's *Immediate first action* names four issues to fix before any retrain, plus
the audit and corrected design. All four are fixed, tested, and committed:

| # | Issue | Fix | Test |
|---|-------|-----|------|
| 1 | `<8`-column map falls back to full matrix | Floor removed from `stat_feature_subset`; explicit N-feature map stays N features | `test_prop_feature_selection.py::test_two_feature_map_no_floor_fallback` |
| 2 | Empty selected set → full matrix | Empty explicit list → zero-column **base-rate** frame | `...::test_empty_map_is_base_rate_not_full_matrix` |
| 3 | `game_total` leaks into the "pure" ablation | Regex-only classification replaced by enumerated `feature_provenance`; current-game Vegas columns excluded from pure study | `test_feature_ablation_study.py::test_game_total_excluded_from_pure_study` |
| 4 | Binary Over selection treated as PMF selection | `ablation/pmf_selection.py` scores the actual PMF; binary head gated by `assert_cdf_coherent` | `test_pmf_selection.py` |

Data-dependent stages (long chronological OOF, minutes retrain, component-PMF fitting) are
**data-blocked** in this environment: the license-restricted feature/quote parquets
(`data/processed/…`, atomic quote store) are not present. `EXACT_QUOTE_READINESS.json`
confirms there is still **no atomic player-prop quote store** (0 exact pairs for every prop).
Those stages are specified below and left as reproducible drivers, not fabricated results.

---

## 1. Production feature-usage audit (result)

`artifacts/per_stat_compact/PRODUCTION_FEATURE_USAGE_AUDIT.json`. Key proven facts:

1. **Every stat receives the full shared matrix** — Stage 5 (`stage5_oof.yaml`) ships **no**
   `prop_feature_map`, so `stat_feature_subset` returns the matrix unchanged. Production
   trains each stat on the full **128-column** `MODEL_FEATURES` allowlist. (The directive's
   "379" is the larger `recovered_v2` *ablation* wide matrix, not this allowlist.)
2. `prop_feature_map` configured: **False**.
3. `<8`-column floor: **removed** (and moot in prod because no map is configured).
4. Empty-set → full-matrix fallback: **fixed** (now a base-rate candidate).
5. Map/artifact hash parity: **N/A** in prod (no map, no per-stat artifact here); enforced by
   `FittedFeatureSpec` when a policy is used.
6. **PBP features never reach the fitted production artifact** — no `pbp_*` column is in
   `MODEL_FEATURES`; PBP opportunity features exist only in the ablation candidate space.
7. **Current-game market reaches the "pure" candidates** — `prop_feature_map_candidate_v1.json`
   lists `game_total` / `game_spread_home` / `implied_team_total` (and `close_game_indicator`)
   in **every** prop, and the legacy ablation regex failed to exclude them.

---

## 2. Corrected information contract (provenance)

`src/wnba_props_model/features/feature_provenance.py` assigns every column exactly one label:
`PURE_LAGGED`, `IDENTITY_ONLY`, `INTERNAL_GAME_MODEL`, `EXTERNAL_MARKET_CURRENT_GAME`,
`EXTERNAL_MARKET_LAGGED`, `FORWARD_PREGAME_CONTEXT`, `TARGET_GAME_OUTCOME`. The current-game
market set is sourced from `feature_contract.SAME_GAME_MARKET_FEATURES` (single source of
truth). Two separately-labeled studies replace the single "market-excluded" study:

* **PURE_COMPACT** (`study_contract="pure_compact"`): no market, no internal-game-model, no
  forward context. This is the only study allowed to be called *pure*.
* **GAME_CONTEXT_STACKED** (`study_contract="game_context_stacked"`): may add the timestamped
  current-game total/spread and internal game context. **Never** described as pure.

Internal game context must come from a separately-trained internal game model
(`internal_game_total_mean`, `internal_home_margin_mean`, …), not copied from the sportsbook;
the external market context remains a separately-identified challenger (candidate B7).

---

## 3. PMF selection ≠ binary selection

The legacy harness fit a binary Over classifier with the line as an input and scored
`P(Y>L)`. The production target is `P(Y=y|X)` with every line derived from one coherent PMF.
`ablation/pmf_selection.py` selects on **count log score, CRPS, mean bias, variance/zero/tail
calibration, and line-level log loss/Brier** derived from the PMF via push-safe settlement.
The sportsbook line may enter the evaluator or a distributional correction layer — **never**
the PMF generator. An optional line-aware binary residual head must pass
`assert_cdf_coherent` (`P(Y>L2) ≤ P(Y>L1)` for `L2>L1`) before it can replace the PMF.

---

## 4. Fail-closed feature policy

`src/wnba_props_model/models/prop_feature_policy.py`:

* `explicit` — exactly required (+ available optional); missing required **raises**; never the
  full matrix.
* `base_rate` — intentionally empty; an intercept / structured base rate, **not** all features.
* `legacy_full_diagnostic` — the full matrix, `certifiable = False` (frozen comparison only).

`FittedFeatureSpec` stores `feature_set_id`, ordered feature names, schema hash,
information contract, training cutoff, and training-row hash, and fails inference on any
schema/order/hash drift.

`prop_feature_policies_v1.py` defines one contract per prop (Tier-A pure core + Tier-B
optional) plus B0/B4/B6/B7. Every pure core is verified market-free and manifest-resolvable
(`test_prop_feature_policies_v1.py`).

---

## 5. Candidate matrix (frozen families)

For each quote-covered prop: `B0_BASE_RATE`, `B1_MINUTES_ONLY`, `B2_COMPACT_CAUSAL_CORE`,
`B3_COMPACT_PLUS_STABLE_LAGGED`, `B4_FULL_379_CONTROL` (non-certifiable), `B5_PBP_TRACKING`,
`B6_INTERNAL_GAME_CONTEXT`, `B7_EXTERNAL_MARKET_CONTEXT`. Selection happens **inside inner
folds**; one final candidate policy is frozen before the sealed proof. B3/B5 are produced by
`scripts/build_per_stat_feature_policies.py` when data is present.

Frozen regularization grid (S7): `max_leaf_nodes ∈ {7,15,31}`, `min_samples_leaf ∈ {40,80,120}`,
`learning_rate ∈ {0.02,0.05}`, `l2_regularization ∈ {1,5,20}`, `max_iter ∈ {100,200,350}`.

---

## 6. Statistical architecture (specified; fitting is data-blocked here)

* **Minutes foundation (S8)** — factor `P(active) · P(role|active) · P(M|active,role,context)`;
  train conditional minutes on appearances only; replace row-wise quantile **sorting** with a
  minimum-adjustment monotone projection; replace the fixed `0.10/0.15/0.50/0.15/0.10` median
  weighting with equal-probability inverse-CDF nodes; enforce team-minute coherence
  (`Σ E[M_i] = 200` regulation) with a separate overtime mixture.
* **Median-vs-mean (S9)** — PTS/REB/AST currently train a **conditional median** (HGB
  `quantile=0.5`) via `StatRateModel.predict_mean`, then pass it to a count PMF as the expected
  count. This is quarantined: the median path lives only in the **non-certifiable** B4 control.
  Certified candidates must train `E[Y|X]` (mean/exposure-aware count objective) or a
  distributional model, with a test asserting `Σ_y y·P(Y=y)` equals the declared mean.
  *Status: the median-as-mean code in `rate_model.py` is NOT removed (P0 must not be
  modified); it is quarantined by contract and superseded by the component PMFs below once
  they are fit on real data.*
* **PTS (S10)** `PTS=2·2PM+3·3PM+FTM` via `FGA~NB2`, `3PA|FGA~BetaBinomial`,
  `2PM/3PM/FTM|attempts~BetaBinomial`, convolved.
* **REB (S11)** `REB=OREB+DREB` from opportunity (expected team/opponent misses) × chance
  share × conversion; convolve OREB/DREB.
* **AST (S12)** `AST = potential assists × teammate conversion`, `AST|PA~BetaBinomial`.
* **FG3M (S13)** `FG3M = 3PA × 3P%` with a real hurdle on `3PA=0`; conversion via hierarchical
  shrinkage (league→role→position→player). Empty selection ⇒ `FG3M_BASE_RATE_STRUCTURED`,
  never the full matrix.
* **STL/BLK/TOV (S14–16)** opportunity × capture/turnover hurdle models; outcome-only research
  candidates until exact quotes exist (`EXACT_QUOTE_READINESS`: 0 quotes today).
* **Dispersion (S17)** residual, hierarchically shrunk; compare Poisson / global-NB2 /
  role-shrunk / player-shrunk / hurdle-NB2; adaptive support until omitted tail < tolerance
  (never truncate-and-renormalize material mass).

---

## 7. Evaluation & advancement contract

Same exact rows (`game_id, player_id, stat, book, line, over/under price, quote_timestamp,
prediction_cutoff`) with `prediction_cutoff ≤ quote_timestamp`. Push-safe settlement at
integer line `L`: `P_push=P(Y=L)`, `P_over,settled=P(Y>L)/(1-P_push)`. Report log loss, Brier,
AUC, calibration intercept/slope, ECE, PMF log score, CRPS, mean bias, variance/zero/tail
calibration, interval coverage. Paired bootstrap by game date + Holm across the frozen claim
family. A family advances only when Δlog loss<0, ΔBrier<0, count log score / CRPS not
materially worse, and no temporal/contract/fallback violation. Formal market superiority
requires ≥300 settled exact-quote obs, ≥30 date clusters, both deltas' 95% date-cluster CI
entirely below zero, Holm passes, and the candidate frozen before proof.

---

## 8. Status

```
PER-STAT FEATURE CONTRACT COMPLETE
MINUTES FOUNDATION NOT STARTED (data-blocked in this environment)
PROP PMF MATHEMATICS SPECIFIED, NOT YET FIT/CERTIFIED (data-blocked)
CHRONOLOGICAL OOF NOT RUN (feature/quote parquets absent)
MARKET SUPERIORITY NOT PROVEN
PRODUCTION ACTIVATION NOT AUTHORIZED
```

### Remaining information gaps
* No license-restricted feature parquet or atomic quote store in this environment ⇒ minutes
  retrain, component-PMF fitting, and the long chronological OOF cannot be run or proven here.
* STL/BLK/TOV have **zero** exact sportsbook quotes (`EXACT_QUOTE_READINESS`), so they remain
  outcome-only research candidates; no market claim is permitted.
* The component-PMF builders (S10–S16) are specified and partially available in
  `opportunity/pmf_builders.py`; wiring each stat's certified candidate to those builders and
  proving it against the market is the next data-present step.

### Historical quote synthesis (Odds API) — empirical coverage finding
`scripts/synthesize_historical_quotes.py` backfills the atomic quote store from The Odds API
**historical** endpoints (player props back to May 2023, 5-minute snapshots) using the exact
`atomic_quotes.ATOMIC_QUOTE_COLUMNS` schema, then optionally builds validated pairs and
regenerates `EXACT_QUOTE_READINESS.json`. Verified live against the Enterprise key
(2024–2025, `us` + `us2`, tip−1h snapshots):

| Market | Odds API key | Historical book coverage |
|--------|--------------|--------------------------|
| pts / reb / ast / fg3m | player_points/rebounds/assists/threes | **YES** — 7 US books (DraftKings, FanDuel, William Hill, BetRivers, Bovada, BetOnline, BetMGM) |
| stl / blk / tov | player_steals/blocks/turnovers | **NONE** — 0 books post these WNBA props |

Consequence: STL/BLK/TOV **cannot** be synthesized from historical market data (the market
does not exist), confirming they stay outcome-only. pts/reb/ast/fg3m **can** be synthesized;
that is the path that unblocks the market-superiority proof. Turning synthesized single-side
rows into EXACT_PAIR + settled outcomes still requires the BDL canonical tables
(`build_canonical_tables.py`) for player/game-id resolution and actual outcomes; without them
pairs resolve to AMBIGUOUS_PLAYER (fail-closed), as demonstrated in the smoke run.
