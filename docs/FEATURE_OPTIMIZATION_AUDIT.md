# WNBA Player-Prop Feature Optimization Audit

**Repository:** `Risky-Scout/wnba-player-props-pmf-model`  
**Audit snapshot:** 2026-07-20  
**Scope:** the 12 props currently represented by the repository: `pts`, `reb`, `ast`, `fg3m`, `stl`, `blk`, `turnover`, `stocks`, `pts_ast`, `pts_reb`, `reb_ast`, and `pts_reb_ast`.

## Executive finding

The repository does **not** currently contain evidence that any feature set is globally or absolutely optimal, and it does **not** contain a completed proof that a frozen model beats closing no-vig market probabilities on all three requested metrics: log loss, Brier score, and AUC.

That is not a semantic objection. It follows from the current implementation:

1. The historical production path feeds the same global feature matrix to every direct-stat model.
2. The new stat-specific selector uses count-mean MSE permutation importance from a surrogate regressor.
3. The selector does not optimize line-level log loss, Brier, AUC, PMF log score, or market-relative deltas.
4. Its untouched holdout is reserved but not scored as a final proof set.
5. The repository's market gate checks log loss and Brier, but not AUC.
6. The latest selector CI did not produce a realized market backtest artifact because required model/data artifacts were unavailable.
7. The current champion manifest is explicitly **forecast-only**, not market-superiority certified.

Therefore, the only defensible output is:

- an exact, leak-safe, stat-specific candidate map using columns that exist now;
- a prioritized list of missing features that should be engineered;
- a nested prequential selection and proof protocol;
- executable scoring code that can certify or reject a market-beating claim once real frozen predictions and closing prices are supplied.

## Current status by prop

| Prop | Current forecast status | Market log-loss/Brier/AUC proof |
|---|---|---|
| Points | Forecast-certified | Not present |
| Rebounds | Forecast-certified | Not present |
| Assists | Forecast-certified | Not present |
| 3-pointers made | Not forecast-certified | Not present |
| Steals | Forecast-certified | Not present |
| Blocks | Not forecast-certified | Not present |
| Turnovers | Forecast-certified | Not present |
| Stocks | Forecast-certified | Not present |
| Points + assists | Forecast-certified | Not present |
| Points + rebounds | Forecast-certified | Not present |
| Rebounds + assists | Not forecast-certified | Not present |
| Points + rebounds + assists | Forecast-certified | Not present |

## What is wrong with the current feature-selection evidence

### 1. Objective mismatch

The selector's target is the realized count and its importance score is validation MSE. A sportsbook prop decision is a probability question at a specific line. A feature can improve count-mean MSE without improving:

- the probability of going over a posted line;
- the calibration of that probability;
- the tails of the full count PMF;
- discrimination between overs and unders;
- performance relative to a closing no-vig market probability.

Selection must use the same production distribution and the same metrics used for proof.

### 2. Surrogate-model mismatch

A Poisson histogram-gradient-boosting surrogate is used for selection, while the production system has stat-specific distribution behavior, minutes marginalization, calibration, sparse-count handling, and combo construction. Feature interactions can reverse when the model family changes.

### 3. Row-population and missing-value mismatch

The selector filters only on a nonmissing target and explicitly fills missing feature values with zero. Production fitting uses different played-row and missing-value behavior. This changes what the selector learns.

### 4. No final untouched score

Reserving a holdout is not proof unless the final frozen candidate is evaluated on it exactly once. The current selector writes a map and importance table, but not a market-relative final score.

### 5. No genuine subset search

Positive permutation importance is a conditional diagnostic, not a proof of an optimal subset. Correlated variables can split importance; interacting variables can have low marginal importance; and the keep rule guarantees a broad minimum fraction rather than searching a Pareto-efficient subset.

### 6. Combo props are omitted

The selector covers only the seven direct stats. Combo props should not be modeled as independent sums. Their joint PMFs need dependence conditioned on minutes, role, usage, pace, lineup, and game script.

### 7. AUC is absent from the current market gate

The existing market gate can compare model and market log loss and Brier score. It cannot establish the requested AUC improvement.

### 8. The current CLV field is not signed closing-line value

The audited implementation constructs its CLV field from the maximum absolute side edge. That quantity is nonnegative by construction. It cannot prove that a bet beat the closing price. Replace it with actual timestamped open and close no-vig probabilities for the same book, side, line, and limit context.

## Recommended current-column feature sets

The exact runnable lists are in:

- `config/prop_feature_map_candidate_v1.json`
- `config/prop_feature_blueprint_v1.json`

These lists deliberately exclude raw numeric player, team, and opponent IDs from the base candidate. Treating integer IDs as ordered continuous features invites arbitrary threshold effects and weak out-of-sample transfer. Identity effects should be added only through an explicitly regularized categorical, target-encoded, embedding, or hierarchical player component rebuilt identically in training and inference.

### Points

Core signal:

- a probabilistic minutes and availability block;
- scoring rate and shot-volume opportunity;
- usage and role transfer caused by absences;
- efficiency and shot-quality state;
- opponent defense, pace, and implied scoring environment;
- fatigue and game-script effects.

Highest-priority new data: 2PA/3PA/FTA volume, touches, drives, time of possession, shrunk true shooting, and on/off shot share.

### Rebounds

Core signal:

- probabilistic minutes and role;
- rebound rate;
- team and opponent rebound environment;
- opponent rim pressure and expected misses;
- vacated rebounds and lineup availability;
- pace, fatigue, and game script.

Highest-priority new data: separate offensive/defensive rebound rates, rebound chances, opponent expected misses, shot-location mix, and lineup size.

### Assists

Core signal:

- probabilistic minutes and role;
- assist rate and creation/usage load;
- team and opponent assist environment;
- vacated assists, teammate availability, and usage transfer;
- pace, fatigue, and implied team scoring.

Highest-priority new data: potential assists, passes, touches, time of possession, and expected teammate conversion.

### 3-pointers made

Core signal:

- probabilistic minutes;
- 3PM rate and general shot volume;
- usage, corner-three share, efficiency, and shot quality;
- role transfer and absences;
- opponent defense, pace, and game environment.

Highest-priority new data: 3PA per minute, shrunk 3P%, corner/above-break attempt shares, opponent 3PA allowed, and expected contest quality.

### Steals

Core signal:

- probabilistic minutes and defender role;
- shrinkage-stabilized steal rate;
- opponent turnover and pass-risk environment;
- pace, workload, and close-game exposure.

Highest-priority new data: deflections, matchup ballhandler touches, opponent passing volume, live-ball turnover rate, and expected primary assignment.

### Blocks

Core signal:

- probabilistic minutes and positional role;
- shrinkage-stabilized block rate;
- opponent rim pressure and rim-attempt proxy;
- defender role, lineup availability, pace, fatigue, and game script.

Highest-priority new data: personal-foul rate, likely-matchup rim attempts, expected lineup size, contest rate, and rim-protection role.

### Turnovers

Core signal:

- probabilistic minutes;
- turnover rate;
- usage, handling, shooting, and creation load;
- usage transfer caused by absences;
- opponent pressure proxy, pace, fatigue, and game script.

Highest-priority new data: touches, passes, drives, ballhandling share, opponent pressure/trap rate, and on/off turnover share.

### Combo props

Construct the combo from a **joint component PMF**, not a product of independent component PMFs.

- `stocks`: joint `stl + blk`
- `pts_ast`: joint `pts + ast`
- `pts_reb`: joint `pts + reb`
- `reb_ast`: joint `reb + ast`
- `pts_reb_ast`: joint `pts + reb + ast`

Condition dependence on predicted minutes, role, usage, pace, teammate availability, lineup confirmation, spread, blowout probability, and close-game probability. `reb_ast` should remain blocked until its own joint-distribution and market gates pass.

## Required experiment lattice

For every direct prop, compare at least these candidates using the **same production estimator and distribution**:

| Candidate | Purpose |
|---|---|
| G0 | Current global feature set |
| S1 | Stat-specific current-column set in this pack |
| S2 | S1 plus lagged market priors |
| S3 | S1 minus game-script family |
| S4 | S1 minus injury/role-transfer family |
| S5 | S1 minus fatigue family |
| S6 | S1 plus newly engineered minutes-distribution features |
| S7 | S6 plus prop-specific tracking/opportunity features |
| S8 | S7 plus a regularized player-effect component |

For combo props, compare the current combo construction against joint-PMF challengers that vary only the dependence model. Do not select combo features by training a separate count-mean surrogate.

## Proof protocol

### Step 1: Define market truth

For each observation retain:

- game and player;
- prop and exact line;
- actual result;
- model over probability produced before the market timestamp;
- sportsbook over and under prices;
- no-vig over probability;
- book and timestamp;
- selection/test split;
- candidate ID.

Use a consistent closing snapshot. Do not mix opening prices, stale prices, alternate lines, and closing prices without explicit stratification.

### Step 2: Use nested, time-ordered evaluation

1. **Inner prequential selection period:** compare candidates and freeze exactly one feature set per prop.
2. **Outer forward proof period:** evaluate only the frozen candidate. Never use this period to alter features, distributions, calibration, or thresholds.
3. Split by game date, not random rows, so all players from a game/date stay together.
4. Cluster uncertainty by game date at minimum; game-level clustering is also appropriate when identifiers are reliable.

### Step 3: Score the full distribution and the settled line

Full PMF diagnostics:

- PMF negative log likelihood;
- CRPS;
- PIT/calibration diagnostics;
- interval coverage and width;
- zero-mass calibration for sparse stats.

Settled binary prop diagnostics:

- log loss;
- Brier score;
- ROC AUC;
- ECE and calibration slope as diagnostics.

Pushes are excluded from binary metrics but retained for full-PMF scoring.

### Step 4: Compare with closing no-vig probabilities

Define:

- `ΔLogLoss = model_logloss - market_logloss` — negative is better.
- `ΔBrier = model_brier - market_brier` — negative is better.
- `ΔAUC = model_auc - market_auc` — positive is better.

Use paired date-cluster bootstrapping because model and market score the same events.

### Step 5: Control false discoveries

There are 12 prop families and three requested metrics. Apply Holm correction across props within each metric. A feature set passes only when all of the following hold on the untouched proof period:

- upper 95% bootstrap CI for `ΔLogLoss` is below zero;
- upper 95% bootstrap CI for `ΔBrier` is below zero;
- lower 95% bootstrap CI for `ΔAUC` is above zero;
- all three one-sided Holm-adjusted p-values are at most 0.05;
- at least 300 settled observations exist for the prop;
- coverage, timestamp integrity, and calibration checks pass.

AUC is a ranking metric, not a calibration metric. Requiring it in addition to two proper scoring rules is intentionally strict.

### Step 6: Repeat forward

After the proof block, lock the model and run a genuinely live forward shadow period. Report results by prop, sportsbook, line bucket, player role, expected minutes, and season phase. A one-period pass is evidence for that population, not a permanent guarantee.

## Running the supplied evaluator

Expected input columns:

```text
game_date
prop
candidate
split
actual
line
model_prob_over
market_prob_over_no_vig
```

Select one candidate per prop on the selection period:

```bash
python scripts/evaluate_market_superiority.py \
  --input scored_candidates.parquet \
  --mode select \
  --selection-split selection \
  --output-dir artifacts/feature_selection
```

Freeze `artifacts/feature_selection/selected_candidates.json`, then prove it on a separate forward period:

```bash
python scripts/evaluate_market_superiority.py \
  --input scored_candidates.parquet \
  --mode prove \
  --selected-candidates artifacts/feature_selection/selected_candidates.json \
  --test-split test \
  --bootstrap 5000 \
  --min-rows 300 \
  --output-dir artifacts/market_feature_proof
```

The output includes a CSV, JSON, and Markdown proof report. A PASS requires all three requested metrics to beat the market with corrected uncertainty.

## Self-test interpretation

The included synthetic self-test deliberately generates a known superior candidate. It verifies that the evaluator:

- selects the known better candidate on the selection split;
- freezes that choice;
- detects negative log-loss and Brier deltas;
- detects positive AUC delta;
- applies clustered intervals and Holm correction.

It is **not** WNBA market evidence.

## Final decision standard

Do not label a feature map “optimal” because it has positive feature importance, lower in-sample error, or better performance than a historical-average baseline.

Use the phrase **market-superiority certified for the stated proof period** only after the frozen, stat-specific map passes the full protocol. If one prop fails one requested metric, that prop is not certified. If a newer candidate is selected using the proof period, a new untouched proof period is required.
