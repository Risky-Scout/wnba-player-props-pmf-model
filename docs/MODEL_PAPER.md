# WNBA Player Props PMF Model
## A Complete Technical Reference

**Repository:** `Risky-Scout/wnba-player-props-pmf-model`  
**Scope:** Pre-game and live in-play probabilistic mass function (PMF) predictions for WNBA player props and game totals

---

## 1. Overview

The model answers a single question at industrial precision:

> For a given WNBA player in a given game, what is the exact probability distribution over every possible integer outcome (0, 1, 2, … pts/reb/ast/…)?

From that full probability mass function every downstream product follows directly: P(over), P(under), fair American odds, edge vs. market, conformal confidence intervals, and live in-play posterior updates.

The system has two prediction engines that share the same foundation:

- **Pre-game PMF engine** — trained daily, predicts the full distribution before tip-off
- **Live in-play Bayesian engine** — updates the pre-game prior in real time as PBP events arrive

Both engines are evaluated using the same calibration standard: **PIT uniformity** (Kolmogorov–Smirnov), **ECCE-MAD** (binning-free calibration error), and **Expected Calibration Error**.

---

## 2. Data Pipeline

### 2.1 Source

All data comes from the **Ball Don't Lie (BDL) API** (`https://api.balldontlie.io`), the only publicly available WNBA statistics API with real-time data. The model uses 13 distinct endpoints:

| Endpoint | Purpose |
|---|---|
| `/wnba/v1/games` | Schedule, scores, game status |
| `/wnba/v1/player_game_stats` | Per-game box scores |
| `/wnba/v1/player_game_advanced_stats` | USG%, pace, PIE, offensive/defensive rating |
| `/wnba/v1/player_season_advanced_stats` | Season USG% (all measure types) |
| `/wnba/v1/team_game_advanced_stats` | Team DEF rating, pace per game |
| `/wnba/v1/player_shot_locations` | FGA % by zone (restricted, corner 3, mid-range) |
| `/wnba/v1/team_shot_locations` | Team opponent shot distribution |
| `/wnba/v1/player_injuries` | Current injury status, return dates |
| `/wnba/v1/odds` | Game spreads, moneylines, totals (5 vendors) |
| `/wnba/v1/odds/player_props` | Live prop lines over/under (real-time only) |
| `/wnba/v1/plays` | Play-by-play text (live engine only) |
| `/wnba/v1/standings` | Win%, conference rank, games behind |
| `/wnba/v1/players` | Player metadata, teams |

Data is ingested with cursor-based pagination, a 1-second rate-limit buffer, and incremental pulls (only new data since last run). All raw data is stored as Parquet files.

### 2.2 Canonical Layer

Raw Parquet files are normalized into a unified canonical schema via `build_canonical_tables.py`. Key enrichments:

- Games are flagged with `has_player_stats` and `has_odds`
- Player stats receive `opponent_team_id`, `is_home`, `did_play`, and `started_proxy`
- Stat renames: `tov → turnover`, `pa → pts_ast`, `pr → pts_reb`, etc.
- Odds are enriched with `game_date` and `season` from the games table
- A JSON schema manifest is written for downstream validation

---

## 3. Feature Engineering

The feature table (`wnba_player_game_features_wide.parquet`) has one row per player-game and approximately **80+ engineered columns**. All features are **shift-1 safe**: a game at date T uses only information from dates T−1 and earlier. This is enforced by `feature_contract.py`, which maintains a FORBIDDEN set of market and post-game leakage columns.

### 3.1 Feature Families

**Identity/Schedule**
- `game_date`, `season`, `team_id`, `opponent_team_id`, `is_home`, `days_rest`

**Minutes & Role**
- `projected_minutes` — prior game average, adjusted for role changes
- `projected_minutes_bucket` — star / starter / rotation / bench
- `role_status` — active / limited / questionable
- `role_uncertainty_bucket` — stable / volatile
- `minutes_volatility_bucket` — low / medium / high (EWMA of minutes std dev)

**Player Rates (rolling)**
- `player_{stat}_l5_mean`, `player_{stat}_l10_mean` — 5 and 10-game rolling means
- `player_{stat}_ewma_halflife3`, `player_{stat}_ewma_halflife5` — exponentially weighted (PenaltyBlog EWMA)
- `player_{stat}_per_min_l5` — per-minute rate over last 5 games
- `player_{stat}_season_mean` — season average with expanding window

**Team Context**
- `team_pace_l5_ewma` — team pace proxy
- `team_ortg_l5`, `team_drtg_l5` — rolling offensive/defensive rating
- `is_b2b`, `is_second_of_b2b` — back-to-back flags

**Advanced Features (24 new columns from extended BDL endpoints)**
| Feature | Source | Purpose |
|---|---|---|
| `player_usage_pct` | `/player_season_advanced_stats?measure_type=usage` | Season USG% fraction of team possessions |
| `player_usage_pct_ewma10` | Computed | 10-game EWMA of game-level USG% |
| `player_usage_pct_vs_avg` | Computed | Detects role changes (current vs. season average) |
| `player_efg_pct` | `four_factors` measure type | Effective FG% adjusting for 3-point value |
| `player_ft_rate` | `four_factors` measure type | Free throw rate (FTA/FGA) |
| `player_tov_pct` | `four_factors` measure type | Turnover percentage |
| `player_pct_fg_restricted` | `/player_shot_locations` | % of FGA at the rim |
| `player_pct_fg_corner3` | `/player_shot_locations` | % of FGA from corner 3 (most efficient 3) |
| `player_pct_fg_midrange` | `/player_shot_locations` | % of FGA from mid-range (least efficient) |
| `player_fg_pct_restricted` | `/player_shot_locations` | Finishing rate at the rim |
| `shot_quality_score` | Computed | Weighted shot selection quality index |
| `teammate_out_count` | `/player_injuries` | # rotation teammates ruled out |
| `teammate_questionable_count` | `/player_injuries` | # potentially limited teammates |
| `team_total_usage_of_out_players` | Injuries + usage | Redistributable USG% when teammates out |
| `opp_def_rating_ewma10` | `/team_game_advanced_stats` | Opponent EWMA defensive efficiency |
| `opp_pace_ewma10` | `/team_game_advanced_stats` | Opponent EWMA pace |
| `game_pace_predicted` | Computed (both teams) | Expected total possessions |
| `team_playoff_seed` | `/standings` | Seeding affects motivation |
| `team_games_behind` | `/standings` | Playoff race urgency |
| `season_phase` | Computed from game count | Early (0–0.2) / mid (0.2–0.7) / late (0.7–1.0) |

**Injury/Usage Features**
- `dnp_probability` — IPW-corrected DNP probability (causal injury model, E17)
- `usage_transfer_delta` — USG% redistribution from injured teammates
- `projected_usage_given_absences` — UTM-adjusted projected usage

**Pi Ratings** (PenaltyBlog methodology)
- `player_pi_offense` — exponential moving average of actual vs. expected (l5 baseline)
- `team_def_pi` — opponent defensive Pi rating
- Updated at speed β = 0.10 per game for players, β = 0.20 for team defense

**Matchup History**
- `player_vs_opp_career_pts_mean`, `player_vs_opp_career_reb_mean`, etc. — career averages against this specific opponent (min 3 games)

**Game Script / Conditional Minutes**
- `win_probability` — pre-game implied win probability from moneyline
- `blowout_risk` — market-implied probability of >15-point margin at close

**Positional Defense**
- `opp_{pos}_def_rating` — opponent's defensive rating specifically against guard/wing/big

---

## 4. Pre-Game Model Architecture

The pre-game engine predicts **seven stats** simultaneously: `pts`, `reb`, `ast`, `fg3m`, `stl`, `blk`, `turnover`. It also derives five **combo stats** by convolution: `stocks` (stl+blk), `pts_ast`, `pts_reb`, `reb_ast`, `pts_reb_ast`.

The pipeline runs in six stages.

### Stage 1: Minutes Model

Before predicting any counting stat, the model first predicts minutes played. Without an accurate minutes projection, every downstream counting stat prediction is unreliable.

- Uses `HistGradientBoostingRegressor` on role features and schedule features
- **Bimodal distribution** (RotationModel, E19): starters play different minute distributions in close games vs. blowouts
  - Close game (|spread| < 4): starter mean = 32 min, σ = 3.5
  - Normal game: starter mean = 28 min, σ = 4.5  
  - Blowout risk > 0.4: bench players get additional minutes
- Outputs `projected_minutes` as a key feature for all counting stat models
- Conformal prediction intervals also produced for minutes

### Stage 2: DNP Probability (Causal Injury Model, E17)

Before generating any PMF, the model computes `p_dnp` — the probability a player does not play. This uses an **IPW-corrected (Inverse Probability Weighting)** approach that corrects the "healthy-worker survivor effect" bias: players who play heavy minutes appear artificially healthy relative to their true injury risk.

- Propensity model (HistGradientBoostingClassifier) estimates P(observed high minutes | features)
- Stabilized IPW weights correct for the selection bias
- Logistic regression with IPW sample weights produces `p_dnp`
- Final PMF is a mixture: `P(stat=k) = p_dnp × 𝟙[k=0] + (1-p_dnp) × PMF_active(k)`

### Stage 3: Stat Rate Models (Two Types)

**For non-sparse stats (pts, reb, ast, fg3m, turnover) — StatRateModel:**
- `HistGradientBoostingRegressor` trained on `did_play=True` rows
- Predicts E[Y] (expected count)
- Trained separately per stat; ~80 feature columns

**For sparse stats (stl, blk) — HurdleModel:**
- Stage A: `HistGradientBoostingClassifier` → P(Y > 0) on all played rows
- Stage B: `HistGradientBoostingRegressor` → E[Y | Y > 0] on positive-only rows
- Final expectation: E[Y] = P(Y > 0) × E[Y | Y > 0]

**Multi-task shared representation (E13):**  
A `MultiTaskStatRateModel` runs a shared HGB layer first (capturing cross-stat covariance) then private residual models per stat. The residuals from this shared model are used to compute a **learned correlation matrix** that replaces any hardcoded copula correlations.

### Stage 4: PMF Generation

The model converts the rate prediction E[Y] into a full probability mass function over {0, 1, 2, …, K_max}.

**Distribution selection by stat:**

| Stat | Distribution | Why |
|---|---|---|
| `pts` | **Negative Binomial** | Overdispersed count; high variance relative to mean |
| `reb` | Negative Binomial | Similar overdispersion profile |
| `ast` | Negative Binomial | Bursty, game-script dependent |
| `fg3m` | Negative Binomial | Geometric shooting runs |
| `stl` | **Hurdle + NegBinom** | Structural zeros (bench players never get steals) |
| `blk` | Hurdle + NegBinom | Structural zeros (perimeter players rarely block) |
| `turnover` | Negative Binomial | Exposure-dependent; scales with minutes |

The NegBinom dispersion parameter `r` is estimated from OOF residuals:
`r = μ² / (σ² − μ)` where σ² is the empirical variance of residuals.

Role-aware batching: star players get a smaller `r` (fatter tails) than bench players, correctly reflecting that elite players occasionally go supernova.

**Combo stats** are produced via discrete convolution of the individual PMFs:
`PMF_pts_reb(k) = Σ_{j=0}^{k} PMF_pts(j) × PMF_reb(k−j)`

### Stage 5: Joint Distribution (Copula / Monte Carlo)

For markets requiring **correlated multi-stat outcomes** (e.g., "player scores 20+ AND grabs 7+ rebounds"), the model uses a **data-learned correlation matrix** from the MultiTask residuals (E13) fed into a Gaussian copula simulation.

The **Possession-Level Monte Carlo Simulator** (E14) provides the highest-fidelity joint distribution: it simulates individual possessions, draws outcome types (FGM, FT, TO, REB, etc.) probabilistically, and accumulates player stat totals across N=10,000 simulated games. This naturally captures all nonlinear dependencies and game-script effects.

### Stage 6: Calibration (Three Tiers)

Calibration converts raw model probabilities into true probabilities. The system uses three sequential layers.

**Layer 1 — Beta Calibration (Manokhin & Grønhaug, 2026)**  
The optimal method for binary P(over) calibration. Three-parameter family:
```
c(s) = 1 / (1 + exp(-c) · s^a / (1-s)^b)
```
When a=b=1, c=0, this is the identity (no-op) — the model recognizes it is already calibrated and leaves probabilities untouched. Isotonic regression distorts already-calibrated predictions. Applied when n_OOF ≥ 50 per bucket.

**Layer 2 — Isotonic Distributional Regression (Allen et al., 2025)**  
Calibrates the **full CDF** at every integer threshold, guaranteeing monotone, coherent distributions. For each threshold k and role bucket, an `IsotonicRegression` maps model CDF(k) → empirical CDF(k). The calibrated PMF is recovered by differencing the calibrated CDF.

**Layer 3 — Conformal Prediction Intervals (Datta et al., 2025)**  
Distribution-free coverage guarantee: the actual stat value falls inside the predicted interval at least (1−α)×100% of the time. Used to flag props where the line falls inside the interval (no edge — model is too uncertain in that direction).

**Selection logic:**
- n_OOF ≥ 500: Compare Beta, Isotonic, Platt on ECCE-MAD; pick best
- 50 ≤ n_OOF < 500: Beta calibration (identity-safe)  
- n_OOF < 50: Platt scaling (2-parameter, minimal overfitting)

### Stage 7: Edge Calculation

Edge = model P(over) − market implied P(over)

Market implied probability is extracted using **Shin's no-vig method**:
```
p_no_vig_over = p_over_raw / (p_over_raw + p_under_raw)
```
where p_over_raw and p_under_raw are the raw implied probabilities from American odds.

Output columns per prop:
- `model_p_over` — calibrated PMF-derived probability
- `market_p_over_no_vig` — Shin's method implied probability
- `edge` — difference (positive = model favors over vs. market)
- `fair_american_over` — American odds at model's probability  
- `confidence_tier` — "standard" (Shin-z ≤ threshold) vs "high_adversity" (sharp market)

### 4.1 Walk-Forward OOF Validation

The model's calibration is validated using **strict expanding-window chronological cross-validation**:

1. Data is sorted by `game_date`
2. Folds are generated with `validation_window_days = 14`
3. Each fold trains on all games up to `fold_start_date − 1` and validates on the next 14-day window
4. No data leakage is possible: the validation set is always strictly in the future relative to the training set

OOF PMFs are used to:
- Fit and validate calibrators
- Compute PIT values for drift monitoring
- Track ECCE-MAD and ECE per stat-role bucket
- Confirm the model beats market implied probabilities (CLV tracking)

---

## 5. Game Total Coherence Anchoring

The WNBA game total market is the most efficiently priced market — sharps move it quickly and it reflects true expected scoring. If the model's implied total (sum of all player point projections) diverges by more than 3 points from the market total, the model is wrong, not the market.

**Algorithm:**
1. Read `total_value` from `wnba_game_odds.parquet`
2. Compute `model_implied_total = Σ pts_mean (all players)`
3. If `|model_implied - market_total| < 3 pts`: no adjustment
4. Otherwise: compute each team's share of model total → scale home/away proportionally
5. Points scale by factor `s`; assists scale by `1 + 0.3×(s−1)`; rebounds do not scale (rebounding is pace-neutral)
6. Maximum scale factor: 1.15× (prevents wild adjustments)

---

## 6. Usage Transfer Matrix (Injury Response)

When a player is ruled out, their usage (measured as USG% from BDL advanced stats) and projected minutes must redistribute to available teammates.

**Redistribution rule:**
```
extra_usg_i = out_player_usg × (usg_i / Σ_j usg_j)   for j in available teammates
extra_min_i = out_player_minutes × (usg_i / Σ_j usg_j)
```

The `UsageTransferMatrix` is automatically called by `apply_injury_news.py` when any player is marked "out". It reads from `wnba_player_season_advanced.parquet` to get current season USG% for all players.

---

## 7. Live In-Play Engine

The live engine consists of four components that run in a continuous polling loop during games.

### 7.1 PBP Parser

Parses the BDL `/wnba/v1/plays?game_id=X` response using regex pattern matching on play text:
- "A. Wilson made 2-point layup" → `fgm += 1`, `fga += 1`, `pts += 2`
- "A. Wilson made 3-point jumper" → `fg3m += 1`, `fgm += 1`, `fga += 1`, `pts += 3`
- "A. Wilson defensive rebound" → `reb += 1`
- "A. Wilson assist to K. Collier" → `ast += 1`
- "A. Wilson steal/block/turnover" → respective counters
- "A. Wilson made free throw 1 of 2" → `ftm += 1`, `fta += 1`, `pts += 1`

Maintains running totals per player (`LivePlayerState`) and game state (score, period, clock).

### 7.2 Gamma-Poisson Bayesian Updater

This is the mathematical engine of the live model. It uses **Gamma-Poisson conjugacy** for closed-form posterior updates.

**Model:**
```
λ_i ~ Gamma(α, β)                  [pre-game prior]
X_i | λ_i ~ Poisson(λ_i)           [observed events per minute]
```

**Pre-game → prior parameters:**
```
rate_per_min = mean_per_game / projected_minutes
α = rate_per_min × β_scale    (default β_scale = 10)
β = β_scale
```

The β_scale = 10 means the prior has weight equivalent to 10 minutes of observation. This is intentionally loose — early in games, the prior dominates; late in games, observed data dominates.

**After observing k events in t minutes:**
```
α_posterior = α + k
β_posterior = β + t
```

**Posterior predictive for remaining minutes T_rem:**
```
X_remaining | data ~ NegBin(r = α_post, p = β_post / (β_post + T_rem))
```

This gives a **full PMF for the total stat** (observed + remaining), not just a point estimate.

**Live P(over):**
```
P(total > line) = Σ_{k: observed + k_remaining > line} P(X_remaining = k)
```

### 7.3 Live Edge Calculator

Compares the posterior P(over) against the live prop line using the same Shin no-vig method as the pre-game model.

- Fetches live props from `/wnba/v1/odds/player_props?game_id=X` (real-time, not stored historically)
- Computes edge = model_p_over − market_vig_free_p_over
- Flags as bettable when |edge| ≥ 4pp (0.04)

### 7.4 Polling Orchestrator

Adaptive polling frequency:
- Pre-game / halftime: every 60 seconds
- Q1–Q3 normal play: every 15 seconds
- Q4 with < 5 minutes remaining: every 10 seconds

BDL rate limit is ~600 requests/minute. Tracking 2 simultaneous games at 8 polls/minute = 16 total requests/minute — well within budget.

---

## 8. Hidden Markov Model Regime Detection (E22)

A 4-state **Hidden Markov Model** (`GameRegimeHMM`) detects within-game momentum states:
- **Normal** — baseline prediction rates
- **High-scoring** — elevated pts/possessions; scale up scoring rates
- **Defensive** — suppressed scoring; scale down scoring rates
- **Garbage time** — bench players enter; radical minute redistribution

The HMM modulates live player rates by `adjust_live_rate(stat, base_rate, state)`. It is inferred from the sequence of score differentials and game clock positions in the PBP stream.

---

## 9. Quality Gates

Before any predictions are published, they must pass 6 mandatory gates:

| Gate | Metric | Threshold | Action if Failed |
|---|---|---|---|
| 1 | ECCE-MAD (binning-free calibration error) | < 0.05 | Hard fail → retrain |
| 2 | PIT KS test (uniformity) | p > 0.05 | Hard fail → recalibrate |
| 3 | Game total coherence | divergence < 5 pts | Warning |
| 4 | Edge distribution sanity | 10–35% of props show ≥ 4pp edge | Warning |
| 5 | Backtest ROI (last 200 games) | Positive | Warning |
| 6 | Live engine rate corrections | Within [0.7, 1.3] | Hard fail |

Gates are verified by `scripts/verify_gates.py` as the penultimate step in the daily pipeline.

---

## 10. Calibration Drift Monitoring

`check_calibration_drift.py` runs after every post-game scoring cycle. It checks:

1. **PIT uniformity**: Probability Integral Transform values should be Uniform(0,1) if calibrated. KS test; p < 0.05 flags drift.
2. **ECCE-MAD**: max|cumulative(actual − predicted)| / n < 0.05 required.
3. **Brier score trend**: Is the early-window Brier score < late-window? (Monotone improvement required.)
4. **Direction**: Is the model systematically over- or under-projecting?

Hard drift triggers automatic recalibration in the GitHub Actions pipeline.

---

## 11. CLV Tracking

Beating the closing line is the gold standard for predictive model quality. The model tracks CLV on every prediction:

```
CLV = model_p_over_at_bet_time − implied_p_over_at_close
```

**Targets:**
- CLV% (fraction of bets with positive CLV): > 60%
- Mean CLV: > 2 percentage points

`generate_clv_report()` produces rolling CLV breakdown by stat and role bucket.

---

## 12. Output Artifacts

Every day the pipeline produces the following files in `deliveries/today/`:

| File | Contents |
|---|---|
| `full_pmfs_wide.parquet` | Full PMF table: one row per (player, stat, game) with pmf_json, mean, std, P(over) at each line |
| `player_projections_{date}.parquet` | Clean projection table: mean/std/P(over) for each player-stat |
| `player_projections_{date}.json` | Same as above in JSON format |
| `edge_report.parquet` | All edges ≥ 4pp with Shin no-vig market probabilities |
| `publishable_edges.parquet` | Filtered to highest-confidence edges |

Live edges are written to `artifacts/live/live_edges_latest.json` on every poll cycle.

---

## 13. GitHub Actions Automation

| Workflow | Schedule | What it does |
|---|---|---|
| `daily_pipeline.yml` | 9 AM ET daily | Full data pull → features → train → calibrate → predict → edge report |
| `live_tracker.yml` | Every 5 min, 6–11 PM ET | PBP poll → Bayesian update → live edge computation |
| `weekly_calibration.yml` | Weekly | OOF walk-forward → fit Beta/IDR calibrators → promote if ECCE-MAD improves |
| `post_game_scoring.yml` | 2 AM ET daily | Score yesterday's predictions → update CLV tracking → drift check |

---

## 14. Software Stack

| Component | Library |
|---|---|
| Gradient boosting (core model) | `scikit-learn` HistGradientBoostingRegressor/Classifier |
| Beta calibration | `scipy.optimize.minimize` (L-BFGS-B) |
| IDR calibration | `sklearn.isotonic.IsotonicRegression` |
| Gamma-Poisson posterior | `scipy.stats.nbinom` |
| Player embeddings (E12) | `PyTorch` (WNBA2Vec) |
| HMM regime detection (E22) | `hmmlearn.hmm.GaussianHMM` |
| Data storage | `pandas` + `pyarrow` Parquet |
| Feature pipeline | `numpy`, `pandas` |
| API / CLI | `typer` |
| Testing | `pytest` (627 tests, 0 failures) |

---

## 15. Academic References

- Allen, S. et al. (2025). In-sample calibration yields conformal calibration guarantees. *arXiv:2503.03841*
- Datta, J. et al. (2025). Conformal Prediction = Bayes? *arXiv:2512.23308*
- Farran, T. (2026). When Your Model Stops Working: Anytime-Valid Calibration Monitoring. *arXiv:2603.13156*
- Hullman, J. et al. (2025). Conformal Prediction and Human Decision Making. *arXiv:2503.11709*
- Lipiecki, A. et al. (2024). Isotonic distributional regression for day-ahead electricity prices. *Energy Economics*
- Manokhin, V. & Grønhaug, D. (2026). Classifier Calibration at Scale. *arXiv:2601.19944*
- Marx, C.G. et al. (2022). Modular Conformal Calibration. *arXiv:2206.11468*
- Yeh, C.-K. et al. (2020). Evaluating Real-Time Probabilistic Forecasts with Application to NBA Outcome Prediction. *The American Statistician*
