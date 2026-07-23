# Beat-Market Blueprint (Self-Contained Execution Spec)

> Purpose: this is the single authoritative spec for driving the WNBA player-prop model
> to **lower log loss and lower Brier than the no-vig closing market, with AUC >= market,
> per target**, and certifying it honestly. An agent starting cold should need only this
> file plus the repository. Every path, function, endpoint, threshold, command, test, and
> guardrail is stated explicitly. Do not rely on external chat context.
>
> ASCII only. Use `->` not arrows, `<=`/`>=` not glyphs, `x` not the times sign, "Delta"
> not the Greek letter. Non-ASCII characters have previously been flagged in review.

---

## 0. How to use this document

1. Read Sections 1-7 fully before editing anything.
2. Execute phases strictly in order: **Phase 0 -> A -> B -> C**. Phase 0 is blocking; no
   performance claim is valid until Phase 0 exit criteria are met.
3. One branch per phase (Section 13). Commit per logical change. Push and open/update a PR
   at the end of every working turn.
4. Never touch the seven live-validated markets' calibrations, registry entries, or Edge
   behavior until a candidate earns a PASS under the frozen contract (Section 5, 11).
5. When a phase's Definition of Done (Section 12) is met, re-run the certification command
   block (Section 11) and record results in the phase PR.

---

## 1. Mission and definition of success

### 1.1 Objective
For each target prop, the model's binary `P(actual > line)` must beat the no-vig market
implied probability on a frozen forward-test window, measured by:

- `logloss_delta = model_logloss - market_logloss` must be **< 0**
- `brier_delta   = model_brier   - market_brier`   must be **< 0**
- `auc_delta     = model_auc     - market_auc`     must be **> 0**

### 1.2 The gate that defines success (authoritative)
Implemented in `scripts/evaluate_market_superiority.py`, function `_prove()`. A prop
`PASS`es only when ALL of:

- `logloss_delta_ci_high < -abs(min_logloss_delta)` AND `logloss_delta_p_holm <= alpha`
- `brier_delta_ci_high   < -abs(min_brier_delta)`   AND `brier_delta_p_holm   <= alpha`
- `auc_delta_ci_low      >  abs(min_auc_delta)`      AND `auc_delta_p_holm     <= alpha`
- `n_settled >= min_rows` (else verdict is `INSUFFICIENT`, not `PASS`)

Frozen contract defaults (Section 11): `alpha = 0.05`, `min_rows = 300`,
`bootstrap = 5000`, cluster = `game_date` (date-cluster bootstrap), pushes excluded,
`min_logloss_delta = min_brier_delta = min_auc_delta = 0.0`.

A PASS is evidence only for the exact frozen candidate, book, line timestamps, date range,
and population represented by the input. It is not a guarantee of future profit.

### 1.3 Target props (canonical keys)
Primary: `pts`, `reb`, `ast`, `fg3m`. Secondary combos: `stocks`, `pts_ast`, `pts_reb`,
`reb_ast`, `pts_reb_ast`. Other singles: `stl`, `blk`, `turnover`.
Stat key definitions live in `src/wnba_props_model/constants.py`:
`DIRECT_STATS = ("pts","reb","ast","fg3m","turnover","stl","blk")`;
`COMBO_STATS = ("stocks","pa","pr","ra","pra")`; canonical combo names via
`CANONICAL_STAT_NAMES` (`pa->pts_ast`, `pr->pts_reb`, `ra->reb_ast`, `pra->pts_reb_ast`).

---

## 2. Environment and repository bootstrap

- Interpreter: `python3` (3.12.3 on this VM). `python` is NOT on PATH; always call `python3`.
- Package import: `wnba_props_model` imports after `sys.path.insert(0, "src")` or editable
  install. Quick check:
  `python3 -c "import sys; sys.path.insert(0,'src'); import wnba_props_model; print('ok')"`.
- Install (if needed): `pip install -e ".[dev]"` (adds pytest, ruff, requests-mock).
- Tests: `python3 -m pytest tests/ -q`. Lint: `ruff check src scripts tests`
  (line-length 100, target py310).
- Env setup status file: `/tmp/cursor/async-install/install-user.status` (exit code when
  finished). If absent and no live installer process, there is no background setup to wait on.
- Secrets/env vars: `ODDS_API_KEY` (The Odds API), BDL key (balldontlie GOAT tier) - required
  for live pulls; not needed for offline scoring against existing parquet. If missing, direct
  the user to Cursor Dashboard -> Cloud Agents -> Secrets.

### 2.1 Data assets already present (offline-runnable)
Under `data/processed/`:
- `wnba_player_game_features_wide.parquet` (~36 MB, 2022-2026, ~29k rows x ~577 cols) - the
  model feature matrix used by ablation and training.
- `wnba_player_game_features_long.parquet`, `wnba_player_game_stats.parquet`,
  `wnba_player_props.parquet`, `wnba_odds.parquet`, `wnba_games.parquet`, plus advanced/shot
  location/standings tables.
Under `artifacts/models/calibration/`: `oof_predictions.parquet` and fitted calibrator pkls.
Under `artifacts/market_feature_proof/G0_baseline/`: the current honest baseline proof.

---

## 3. Current baseline (the exact gap to erase, then beat)

Source: `artifacts/market_feature_proof/G0_baseline/market_superiority_proof.json`
(forward test 2026-06-20..2026-07-15, 21 date clusters). Model = current champion (G0).
Signs: positive logloss/brier delta = worse than market; negative auc delta = worse.

| Prop | n | model LL | market LL | Delta LL | model Brier | market Brier | Delta Brier | model AUC | market AUC | Delta AUC | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pts  | 545 | 0.7504 | 0.6930 | +0.0574 | 0.2715 | 0.2499 | +0.0215 | 0.5182 | 0.5117 | +0.0065 | FAIL |
| reb  | 434 | 0.7424 | 0.6798 | +0.0626 | 0.2668 | 0.2434 | +0.0234 | 0.5378 | 0.5927 | -0.0549 | FAIL |
| ast  | 291 | 0.7098 | 0.6792 | +0.0307 | 0.2545 | 0.2431 | +0.0114 | 0.5479 | 0.5761 | -0.0282 | INSUFFICIENT (n<300) |
| fg3m | 296 | 0.7034 | 0.6557 | +0.0476 | 0.2495 | 0.2319 | +0.0176 | 0.5988 | 0.6529 | -0.0542 | INSUFFICIENT |

**Interpretation:** the model is ~0.03-0.06 nats worse in log loss and ~0.01-0.02 worse in
Brier on every prop. AUC is also below market on reb/fg3m/ast. Therefore two independent
deficits exist: (a) calibration of the betting probability (fixable by recalibration), and
(b) discrimination/ranking (only fixable by adding information). Both workstreams are required.

---

## 4. Root-cause diagnosis (why we lose today)

1. **Measurement point mismatch (evaluated != deployed).**
   `scripts/build_scored_candidates.py::_p_over` uses `k = ceil(line); a[k:].sum()`.
   Delivery (`src/wnba_props_model/pipeline/deliver.py:340`) uses
   `market.prob_over_from_pmf` = `sum(p for k>line)`
   (`src/wnba_props_model/models/market.py:207`). On **integer lines** these disagree and the
   under side silently absorbs the push atom `P(Y=L)`. Any proof built on one and shipped on
   the other is invalid.

2. **Binary-probability calibrators are fitted but not guaranteed in the shipped number.**
   `src/wnba_props_model/pipeline/calibrate.py` fits Beta (`beta_cal_{stat}.pkl`), Venn-Abers
   (`venn_abers_{stat}_{role}.pkl`), and per-line isotonic (`per_line_calibrators.pkl`), but
   `deliver.py:341` sets `model_prob_over` straight from the raw PMF and never applies them in
   the delivery path. The calibrators that most directly reduce log loss are not wired in.

3. **We optimize PMF shape, not the betting objective.** All heavy machinery in
   `calibrate.py` (isotonic on PIT, `bias_corrections.json`, variance compression) targets
   distribution calibration. Log loss/Brier are computed on the binary `P(Y>line)`. Different
   objective; minimizing PIT non-uniformity does not minimize binary log loss.

4. **Discrimination ceiling.** Below-market AUC on reb/fg3m/ast means our features rank
   over/under worse than the book. No post-hoc calibration can raise AUC; only new signal can.

---

## 5. Non-negotiable rules and guardrails

- **Frozen live markets.** The seven live-validated markets are frozen. Do not alter their
  models, calibrations, `config/champion_manifest.json` entries, registry hashes, Edge
  behavior, or page UX until a candidate PASSes under the frozen contract.
- **Leakage control.** Only feature families declared in
  `src/wnba_props_model/features/feature_contract.py` may enter training. Every training
  feature list must pass `feature_contract.assert_no_forbidden_features()`. All new features
  must be strictly lagged (prior-games-only rolling windows); never use same-game or future
  data. Market columns are catalogued but forbidden as model inputs (`FORBIDDEN_MARKET_COLUMNS`).
- **Evaluated == deployed.** The `P(over)` used for proof must be produced by the same
  function and calibration chain that delivery uses.
- **Honesty.** Report signed deltas and CIs verbatim. A prop that is `INSUFFICIENT` or `FAIL`
  is never presented as market-beating. Do not average books for the proof (destroys the sharp
  line).
- **No forced pushes / no amend / stay on branch.** Standard git only. One commit per logical
  change. Create/update the phase PR each turn.

---

## 6. Data sources and endpoints (exact)

### 6.1 balldontlie (BDL) - historical box scores, games, injuries, props
Client: `src/wnba_props_model/data/bdl_client.py`. Base host from
`config/wnba_model.yaml` -> `bdl.base_url = https://api.balldontlie.io`. WNBA endpoints:
- `GET /wnba/v1/games`, `/wnba/v1/players`, `/wnba/v1/players/active`, `/wnba/v1/teams`,
  `/wnba/v1/standings`, `/wnba/v1/player_injuries`.
- GOAT-tier (requires key): `/wnba/v1/player_stats`, `/wnba/v1/player_game_advanced_stats`,
  `/wnba/v1/player_shot_locations`, `/wnba/v1/odds`, `/wnba/v1/odds/player_props`,
  `/wnba/v1/plays`.
Pull scripts: `scripts/pull_bdl_history.py` (history), `scripts/build_canonical_tables.py`
(canonicalize), `scripts/build_features.py` (feature matrix).

### 6.2 The Odds API v4 - market prices, opening/closing lines
Client: `src/wnba_props_model/data/odds_api_client.py`.
`BASE_URL = https://api.the-odds-api.com`, `SPORT_KEY = basketball_wnba`,
API key from `ODDS_API_KEY`. Reference: https://the-odds-api.com/liveapi/guides/v4/.
- `GET /v4/sports/` - 0 credits.
- `GET /v4/sports/basketball_wnba/events` - 0 credits (event IDs). `list_events_for_date()`.
- `GET /v4/sports/basketball_wnba/odds` - 3 credits (bulk game odds).
- `GET /v4/sports/basketball_wnba/events/{event_id}/odds` - 1 credit per market
  (player props). `get_event_player_props()`.
- `GET /v4/sports/basketball_wnba/scores` - 1 credit.
- `GET /v4/historical/sports/basketball_wnba/events?date=ISO` - 10x surcharge (past event IDs).
Market keys -> stat map: `ODDS_API_TO_STAT` (e.g. `player_points->pts`,
`player_rebounds->reb`, `player_assists->ast`, `player_threes->fg3m`,
`player_points_rebounds_assists->pts_reb_ast`). Default pull set: `CORE_PROP_MARKETS`.
Weekly quota budget in `QUOTA_BUDGET` (total_weekly = 7322). Pull scripts:
`scripts/pull_odds_api_props.py`, `scripts/pull_odds_api_closing_lines.py`,
`scripts/pull_closing_lines.py`.

### 6.3 nba_api tracking + hustle (LOCAL ONLY - datacenter IPs blocked)
Script: `scripts/pull_wnba_tracking_local.py`. Runs on a residential IP (user's machine).
`WNBA_LEAGUE_ID = "10"`, seasons 2021-2026, Regular Season + Playoffs. Endpoints:
- `nba_api.stats.endpoints.boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=...)` ->
  touches, passes, secondary assists, rebound chances, contested/uncontested FG, speed,
  distance (playmaking/usage signal).
- `nba_api.stats.endpoints.hustlestatsboxscore.HustleStatsBoxScore(game_id=...)` ->
  deflections, contested 2s/3s, loose balls, charges, screen assists, box-outs
  (defensive-intensity / rebounding signal).
- Game IDs via `nba_api.stats.endpoints.leaguegamelog.LeagueGameLog(league_id="10", ...)`.
Outputs (uploaded back by the user): `wnba_tracking_2021_2026.parquet`,
`wnba_hustle_2021_2026.parquet`, keyed on `(GAME_ID, PLAYER_ID/personId)`. Resumable.

---

## 7. Key files map (with line anchors)

- `src/wnba_props_model/models/market.py`
  - `prob_over_from_pmf(pmf, line)` (line 207) - the P(over) function to unify (Phase 0.1).
  - `no_vig_two_way` / `shin_no_vig_two_way` (line 112) - no-vig extraction (Shin).
  - `fair_american` (213), `binary_logloss` (220), `ignorance_score_binary` (225),
    `market_implied_mean` (Brentq inversion, ~315).
- `src/wnba_props_model/pipeline/deliver.py`
  - `model_prob_over` computed at line 340-341 - insertion point for the calibration chain
    (Phase 0.2). Kelly sizing at 431+. LinePredictor CLV hook at 411 (env `LINE_PREDICTOR_PATH`).
- `src/wnba_props_model/pipeline/calibrate.py`
  - `fit_calibrators` (363): isotonic + bias + variance; `_MIN_CAL_MINUTES = 10` (404).
  - `fit_beta_calibrators` (106), push-exclusion pattern at 160-164.
  - `apply_beta_calibrators` (179), `apply_venn_abers_calibration` (1175),
    `apply_per_line_calibration` (1261, buckets very_low/low/mid/high/very_high).
- `src/wnba_props_model/pipeline/forecast_publication.py`
  - `apply_multistat_forecast` (65) - builds certified Distributions rows for the Edge Board;
    combos built in steps 2a/2b (do not emit combos from raw rows in step 1).
- `scripts/evaluate_market_superiority.py` - the gate. `_metrics` (55), `_select_candidates`
  (169), `_bootstrap_deltas` date-cluster (198), `_prove` (263), gate logic (338-355). CLI
  modes `select|audit|prove`; cols default `model_prob_over` / `market_prob_over_no_vig`.
- `scripts/build_scored_candidates.py` - produces the evaluator input; `_p_over` (27) uses
  `ceil(line)`; groups/averages books at 51 (Phase 0.4 must stop averaging).
- `scripts/run_prop_ablation.py` - CLI opts: `--features
  data/processed/wnba_player_game_features_wide.parquet`, `--maps
  config/feature_ablation_maps_v1.json`, `--out-dir artifacts/feature_ablation`,
  `--holdout-dates 25`.
- `scripts/build_oof_pmfs.py` - `--features-wide`, `--features-long`, `--manifest-path`,
  `--config-path`, `--out-dir data/oof`, `--max-folds`, `--svd-bridge`.
- `scripts/fit_calibrators.py` - `--oof-pmfs` (required), `--out-dir
  artifacts/models/calibration`, `--props-parquet`, `--oof-with-lines`.
- `scripts/verify_gates.py` - subcommands `calibration` (ECE/PIT KS/mean-error) and `market`
  (`--hard` to fail once 300+ rows/stat), plus `clv_tracking` (signed price_clv, fail-closed).
- `scripts/generate_clv_report.py` - `--results data/clv_tracking/results.parquet`,
  `--lookback-days 30`, `--out-dir artifacts/audits`; emits signed price_clv, date-cluster 95%
  CI, `positive_clv_established`.
- `config/wnba_model.yaml` - calibration gate thresholds; `config/champion_manifest.json` -
  certified markets + hashes; `config/certified_forecast_calibration.json` - per-market
  forecast methods.

---

## 8. PHASE 0 - Correctness and proof isolation (BLOCKING)

Branch: `cursor/phase0-correctness-proof-06e5`. No performance claims until 8.7 passes.

### 8.1 Single push-safe P(over)
File: `src/wnba_props_model/models/market.py`, function `prob_over_from_pmf` (line 207).
Add an explicit three-outcome helper and make it the sole implementation:
- Half-integer line `L`: `p_over = sum(p_k for k >= ceil(L))`, `p_under = 1 - p_over`, `p_push = 0`.
- Integer line `L`: `p_push = p_L`; `p_over = sum(p_k for k > L)`; `p_under = sum(p_k for k < L)`;
  two-sided renormalized `p_over_2way = p_over / (p_over + p_under)`.
Expose `prob_over_under_push(pmf, line) -> (p_over, p_under, p_push)` and have
`prob_over_from_pmf` return the two-way renormalized `p_over` for integer lines.
Update callers: `deliver.py:340` and `build_scored_candidates.py:_p_over` to call this one
function. Grep for other call sites: `rg "prob_over_from_pmf|_p_over\(" src scripts`.
Tests (new `tests/test_prob_over_push.py`): assert correct over/under/push at lines
`0.5, 1.0, 1.5, 10.0, 10.5` for a known PMF; assert two-way sums to 1 on integer lines;
assert deliver and scored-candidate paths return identical p_over for the same (pmf, line).

### 8.2 Wire the binary calibrator into delivery, fail-closed
File: `deliver.py` after line 341. Apply, in order, to `model_prob_over`:
1. `apply_per_line_calibration(p_over, stat, line_z)` where
   `line_z = (line - pmf_mean) / pmf_std`.
2. `apply_venn_abers_calibration(comp_df, model_prob_col="model_prob_over")` (per stat, role)
   OR `apply_beta_calibrators(series, stat)` - select per Phase A.1 winner.
Emit new columns: `p_over_raw`, `p_over_calibrated`, `calib_method`, `calib_scope`
(stat|role|line-bucket), `calib_hash`. **Fail closed:** if a required calibrator artifact is
missing for a (stat, scope), drop that row from the Edge Board rather than shipping the raw
prob. Log a WARN with the missing artifact path.
Tests: `tests/test_deliver_calibration_wiring.py` - given fixture PMFs + fitted calibrators,
assert `p_over_calibrated != p_over_raw` where a calibrator exists, and row dropped where it
does not.

### 8.3 Settlement-based calibration population
File: `calibrate.py`. Keep `_MIN_CAL_MINUTES = 10` for the PMF-shape isotonic fit only.
For the **binary P(over)** calibrators (Beta / Venn-Abers / per-line and Phase A.1), train on
the **settlement population**: every offered line that settled (including DNP/low-minute
outcomes), because the market prices and settles those. Add a separate loader path so the
binary calibrators do not inherit the `actual_minutes >= 10` filter.

### 8.4 Preserve exact quote identity (no book averaging)
File: `scripts/build_scored_candidates.py:51`. Replace the
`groupby([...]).agg(mean)` over books with per-book rows: keep `book`, `line`,
`over_odds`, `under_odds`, `line_timestamp`, `settlement`. Produce
`data/processed/closing_quotes_by_book.parquet`. Score against a single named sharp book per
(player, prop, line); prove per book. Choose the sharpest available book present in the data
(document the choice in the contract).

### 8.5 Market-isolated forecasts
Emit two probabilities in the delivery schema:
- `pure_forecast_p_over` - model only; never sees the line/market.
- `market_anchored_p_over` - model blended toward no-vig market (Phase A.2).
The Section 11 proof runs on `pure_forecast_p_over` for the superiority claim.
`market_anchored` is a separate, clearly labeled product.

### 8.6 One signed promotion contract
Create `config/promotion_contract_v1.json` capturing (frozen, no post-result edits):
`{ book, proof_date_min, proof_date_max, selection_frac, min_rows, bootstrap, alpha,
min_logloss_delta, min_brier_delta, min_auc_delta, exclusions[], exclusion_rationale,
feature_hash, model_hash, calibration_hash }`. The prove command must read these values.

### 8.7 Phase 0 exit criteria (Definition of Done)
- `rg "prob_over_from_pmf|_p_over\(" src scripts` shows a single implementation used everywhere.
- `python3 -m pytest tests/test_prob_over_push.py tests/test_deliver_calibration_wiring.py -q`
  passes.
- `python3 scripts/build_scored_candidates.py ...` + `evaluate_market_superiority.py --mode
  prove ...` run end-to-end and reproduce an honest re-baseline (numbers may equal or differ
  from Section 3; that is fine - the point is a trustworthy measuring stick).
- No performance claim is made in this phase's PR beyond "measurement is now correct."

---

## 9. PHASE A - Binary recalibration to close the 0.03-0.06 nat gap

Branch: `cursor/phase-a-binary-recal-06e5`. This targets log loss/Brier only; it cannot raise AUC.

### 9.1 Fit calibrators against the scored objective
New: `scripts/fit_binary_prob_calibrators.py`. Per `(stat, role_bucket, line_z_bucket)` (same
five buckets as `apply_per_line_calibration`), fit isotonic and Beta on
`(p_over_raw, over_outcome)` from OOF joined to real historical lines
(`data/oof/oof_pmfs_with_lines.parquet`, referenced by `calibrate.py:683` and
`fit_calibrators.py --oof-with-lines`). Push-exclude integer settlements
(pattern at `calibrate.py:160-164`). Select the per-cell winner by **cross-validated log
loss** using chronological folds (no shuffling). Persist `per_line_calibrators.pkl` +
`beta_cal_{stat}.pkl` and a `binary_calibrator_selection.json` recording each cell's winner
and OOF log loss. Add `tests/test_binary_prob_calibrators.py`.

### 9.2 Market-anchored logistic pooling (the realistic first PASS)
Fit `p_final = sigmoid(a*logit(p_model) + b*logit(p_market_novig) + c)` per stat on OOF,
chronologically (log-loss-optimal logistic pooling / stacking). Save coefficients to
`artifacts/models/calibration/market_blend_{stat}.json`. Ship as `market_anchored_p_over`
(Section 8.5). Even where the model cannot beat the market alone, the blend usually pushes
log loss below the market's - this is the near-term path to a negative `logloss_delta`.

### 9.3 Re-prove
Run Section 11 on `pure_forecast_p_over` and, separately, on `market_anchored_p_over`.
Expected first movers: `pts` and `ast` (AUC already near/above market), where honest
recalibration + blend can flip `logloss_delta` and `brier_delta` negative.

### 9.4 Phase A Definition of Done
- `fit_binary_prob_calibrators.py` runs and writes selection JSON.
- Re-prove shows `logloss_delta_ci_high < 0` and `brier_delta_ci_high < 0` for at least `pts`
  (and ideally `ast`) on `market_anchored`; document exact deltas + CIs in the PR.

---

## 10. PHASE B - Discrimination via new information (raises AUC; required for reb/fg3m/ast)

Branch: `cursor/phase-b-tracking-opportunity-06e5`. Recalibration cannot move AUC.

### 10.1 Land tracking/hustle data
User runs `scripts/pull_wnba_tracking_local.py` on a residential IP (Section 6.3) and uploads
`wnba_tracking_2021_2026.parquet` + `wnba_hustle_2021_2026.parquet`. Ingest into the canonical
layer via `scripts/build_canonical_tables.py` keyed on `(GAME_ID, PLAYER_ID)`. Build strictly
lagged rolling features (prior games only) in `src/wnba_props_model/features/build_features.py`.

### 10.2 Declare new feature families (leakage-gated)
In `src/wnba_props_model/features/feature_contract.py` add families and register them so
`assert_no_forbidden_features()` permits them. Model each stat as opportunity x conversion:
- **reb** (worst AUC, -0.055): `rebound_chances_roll5`, `contested_reb_rate`,
  `box_outs_roll5`, `opp_missed_fg_rate_roll5`. Rebounds are opportunity-dominated - highest
  expected AUC lift.
- **ast** (-0.028): `potential_assists_roll5`, `passes_made_roll5`, `touches_roll5`,
  `teammate_fg_pct_on_pass_roll5`. Potential assists decouple playmaking from teammate makes.
- **fg3m** (-0.054): `open_3pa_rate_roll5` (uncontested 3PA share), `catch_shoot_3pa_roll5`,
  `avg_defender_distance_roll5`. Separates 3PA volume from make rate.
- **pts** (AUC ~ market): `contested_fga_split_roll5`, `fta_opportunity_roll5`,
  `paint_touches_roll5`.

### 10.3 Shrinkage for sparse signal
Route new opportunity rates through `src/wnba_props_model/models/shrinkage.py` and
`archetype_shrinkage.py` so rates shrink to role/position priors. This protects log loss from
overfitting on short WNBA tracking histories.

### 10.4 Confirm signal via ablation
Extend `scripts/run_prop_ablation.py` candidate maps
(`config/feature_ablation_maps_v1.json`) to include the new tracking families beyond
G0/S1-S5. Run:
`python3 scripts/run_prop_ablation.py --features
data/processed/wnba_player_game_features_wide.parquet --holdout-dates 25`.
Promote a family into the champion feature set **only if** it lowers OOF NLL AND raises OOF
AUC for that prop. (The prior ablation found G0 optimal precisely because no new information
existed; 10.1 changes that premise.)

### 10.5 Phase B Definition of Done
- Tracking/hustle ingested, lagged, leakage-gated (contract assertion green).
- Ablation shows an OOF AUC gain for at least `reb` and `fg3m`; champion feature set updated
  only for props with a confirmed gain.

---

## 11. PHASE C - Certification (the gate is the product)

Branch: `cursor/phase-c-certification-06e5`. Run per candidate, in order. Paths shown for the
tracking candidate `B_tracking`; substitute artifact paths as produced by earlier phases.

```bash
# 0) Ensure env
python3 -c "import sys; sys.path.insert(0,'src'); import wnba_props_model; print('ok')"

# 1) Rebuild OOF PMFs under the candidate feature set
python3 scripts/build_oof_pmfs.py \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --features-long data/processed/wnba_player_game_features_long.parquet \
  --out-dir data/oof

# 2) Fit PMF-shape + binary P(over) calibrators
python3 scripts/fit_calibrators.py \
  --oof-pmfs data/oof/oof_pmfs.parquet \
  --oof-with-lines data/oof/oof_pmfs_with_lines.parquet \
  --out-dir artifacts/models/calibration
python3 scripts/fit_binary_prob_calibrators.py \
  --oof-with-lines data/oof/oof_pmfs_with_lines.parquet \
  --out-dir artifacts/models/calibration

# 3) Score real model output vs real per-book closing quotes (no book averaging)
python3 scripts/build_scored_candidates.py \
  --oof data/oof/oof_pmfs_calibrated.parquet \
  --quotes data/processed/closing_quotes_by_book.parquet \
  --candidate B_tracking --selection-frac 0.6 \
  --out artifacts/market_feature_proof/scored_candidates.parquet

# 4) Freeze selection on the selection window, then prove on the untouched forward window
python3 scripts/evaluate_market_superiority.py --mode select \
  --input artifacts/market_feature_proof/scored_candidates.parquet \
  --output-dir artifacts/market_feature_proof
python3 scripts/evaluate_market_superiority.py --mode prove \
  --input artifacts/market_feature_proof/scored_candidates.parquet \
  --selected-candidates artifacts/market_feature_proof/selected_candidates.json \
  --output-dir artifacts/market_feature_proof \
  --min-rows 300 --bootstrap 5000 --alpha 0.05 \
  --min-logloss-delta 0.0 --min-brier-delta 0.0 --min-auc-delta 0.0

# 5) Calibration + market + CLV gates
python3 scripts/verify_gates.py calibration --report <calibration_report.parquet>
python3 scripts/verify_gates.py market --min-rows 300 --hard
python3 scripts/generate_clv_report.py --results data/clv_tracking/results.parquet \
  --lookback-days 30 --out-dir artifacts/audits
python3 scripts/verify_gates.py clv_tracking   # signed price_clv, fail-closed
```

### 11.1 Promotion rule (frozen)
A prop is market-beating **iff** `market_superiority_gate == "PASS"` in
`artifacts/market_feature_proof/market_superiority_proof.json` for that prop (all three
sub-gates pass with Holm p <= 0.05 and n >= 300). Only PASS props are added to
`config/champion_manifest.json` `certified_markets` and reach the Edge Board via
`forecast_publication.apply_multistat_forecast`. INSUFFICIENT/FAIL props stay off the board.

### 11.2 Phase C Definition of Done
- `market_superiority_proof.json` shows PASS for the intended prop(s) with recorded deltas/CIs.
- CLV report shows `positive_clv_established == true` OR is honestly reported as not-yet-proven.
- Champion manifest updated only for PASS props; live markets untouched otherwise.

---

## 12. Per-target edge goals (exact quantities to erase, then beat)

Move all three signed deltas across zero on the forward test (Section 1.2). From Section 3:

| Prop | Cut Delta LogLoss by more than | Cut Delta Brier by more than | Raise AUC (delta) by | Primary lever | Likely first PASS via |
|---|---:|---:|---:|---|---|
| pts  | 0.0574 | 0.0215 | +0.006 -> positive | Phase A.2 blend (AUC ~ market) | Phase 0 + A |
| ast  | 0.0307 | 0.0114 | 0.028 | Phase B potential assists + A.2 | Phase B then A |
| reb  | 0.0626 | 0.0234 | 0.055 | Phase B rebound chances/box-outs (AUC-bound) | Phase B then A |
| fg3m | 0.0476 | 0.0176 | 0.054 | Phase B open-3 rate + shot quality | Phase B then A |

**Honesty rule:** do not attempt to promote `reb`/`fg3m` before Phase B ablation (10.4)
confirms an OOF AUC gain; their AUC deficit cannot be recalibrated away.

---

## 13. Branch and PR plan

| Phase | Branch | PR base | Scope |
|---|---|---|---|
| 0 | `cursor/phase0-correctness-proof-06e5` | `main` | 8.1-8.7 correctness + honest re-baseline |
| A | `cursor/phase-a-binary-recal-06e5` | `main` | 9.1-9.4 binary recal + blend |
| B | `cursor/phase-b-tracking-opportunity-06e5` | `main` | 10.1-10.5 tracking + opportunity features |
| C | `cursor/phase-c-certification-06e5` | `main` | 11 certify + promote PASS props |

Branch naming: lowercase, prefix `cursor/`, suffix `-06e5`. Commit per logical change with a
descriptive message. Push `git push -u origin <branch>` (retry with backoff on network error).
Create/update the PR at the end of each turn via the PR tool; PRs are draft by default. Do not
merge, do not enable auto-merge, do not force-push or amend.

---

## 14. Failure modes and rollback

- **Proof shows FAIL after a change:** keep the change only if it strictly improves at least
  one signed delta without regressing others beyond noise; otherwise revert that commit. Never
  loosen the frozen contract to manufacture a PASS.
- **Calibrator artifact missing at delivery:** fail closed (drop row); never ship raw prob as
  if calibrated (Section 8.2).
- **Integer-line push regression:** covered by `tests/test_prob_over_push.py`; a failure here
  blocks the PR.
- **Leakage suspicion (AUC jumps implausibly):** run `assert_no_forbidden_features()` and audit
  the new feature's construction for same-game/future data; treat any leak as a hard revert.
- **Live markets accidentally altered:** revert immediately; the seven live markets are frozen
  until a candidate PASSes.
- **CLV negative under signed scoring:** `verify_gates.py clv_tracking` fails closed; investigate
  timing/book selection before any promotion.

---

## 15. Command appendix (copy-paste)

```bash
# Import / test / lint
python3 -c "import sys; sys.path.insert(0,'src'); import wnba_props_model; print('ok')"
python3 -m pytest tests/ -q
ruff check src scripts tests

# Rebuild feature matrix from raw (if needed)
python3 scripts/pull_bdl_history.py
python3 scripts/build_canonical_tables.py
python3 scripts/build_features.py

# Ablation (confirm feature value)
python3 scripts/run_prop_ablation.py \
  --features data/processed/wnba_player_game_features_wide.parquet \
  --maps config/feature_ablation_maps_v1.json \
  --out-dir artifacts/feature_ablation --holdout-dates 25

# Full certification chain: see Section 11 block.
```

---

_End of blueprint. This file is the contract. If reality diverges from any path or line
anchor here (files move, lines shift), re-grep to locate the symbol and update this document
in the same PR so it stays authoritative._
