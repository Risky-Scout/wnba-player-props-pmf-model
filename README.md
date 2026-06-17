# WNBA Player Props PMF Model

Production-grade WNBA player prop prediction system. Generates full
probability mass functions (PMFs) for every player stat, calibrated daily
using PenaltyBlog-style isotonic regression, with market edge calculation
via Shin's no-vig method.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Daily Pipeline  (9 AM ET)                        │
│                                                                       │
│  BDL API ──► Canonical Tables ──► Feature Engineering                │
│                                         │                             │
│                              ┌──────────▼──────────────┐             │
│                              │   Stage 4 HGB Engine     │             │
│                              │  Minutes Model           │             │
│                              │  Stat Rate Models        │             │
│                              │  Hurdle Models (blk)     │             │
│                              └──────────┬──────────────┘             │
│                                         │ PMFs                        │
│                              ┌──────────▼──────────────┐             │
│                              │  Stage 6 Calibration     │             │
│                              │  Role-aware isotonic     │             │
│                              │  Bayesian shrinkage      │             │
│                              └──────────┬──────────────┘             │
│                                         │ Calibrated PMFs             │
│                              ┌──────────▼──────────────┐             │
│                              │  D+1 Delivery            │             │
│                              │  full_pmfs_wide.parquet  │             │
│                              │  market_comparison       │             │
│                              │  publishable_edges       │             │
│                              └─────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│               Weekly Calibration  (Monday 6 AM ET)                   │
│                                                                       │
│  Full history ──► OOF Walk-Forward ──► Fit Calibrators               │
│                         │                     │                       │
│                  Fold audit            Fatal gate (ECE/PIT KS)        │
│                                               │                       │
│                                    Only promotes if gate passes       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│               Post-Game Scoring  (2 AM ET daily)                     │
│                                                                       │
│  Yesterday's PMFs + BDL actuals ──► CLV tracking ──► Gate report    │
│                                          │                            │
│                                   results.parquet                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stats Covered

**Direct stats:** `pts` (points), `reb` (rebounds), `ast` (assists),
`fg3m` (3-pointers made), `stl` (steals), `blk` (blocks), `turnover`

**Combo stats:** `stocks` (stl+blk), `pts_ast`, `pts_reb`, `reb_ast`, `pts_reb_ast`

---

## Quick Start: Local D+1 Prediction

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Pull BDL data (requires BDL_API_KEY env var)
BDL_API_KEY=your_key python scripts/pull_bdl_history.py \
  --start-season 2024 --end-season 2026 --out-dir data/raw/bdl

# 3. Build features
python scripts/build_canonical_tables.py --raw-dir data/raw/bdl --out-dir data/processed
python scripts/build_features.py --data-dir data/processed

# 4. Train Stage 4 baseline model
python scripts/train_baseline_pmfs.py

# 5. Predict for tomorrow
python scripts/predict_today.py \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --game-date $(date -v+1d +%Y-%m-%d) \
  --out-dir deliveries/next_game

# 6. Build edge report
python scripts/build_edge_report.py \
  --pmfs deliveries/next_game/full_pmfs_wide.parquet \
  --raw-props data/processed/wnba_player_props.parquet \
  --out-dir deliveries/next_game
```

---

## Manual Override: Injury News & Lineup Changes

### Single player injury update (in place)
```bash
# Mark player as out — redistributes minutes to teammates automatically
python scripts/apply_injury_news.py \
  --player-id 1234 \
  --status out \
  --game-date 2026-06-18

# Mark as limited with explicit minutes cap
python scripts/apply_injury_news.py \
  --player-id 1234 \
  --status limited \
  --minutes-cap 22 \
  --game-date 2026-06-18

# Mark as questionable (scales minutes by 0.75)
python scripts/apply_injury_news.py \
  --player-id 1234 \
  --status questionable \
  --game-date 2026-06-18
```

### Multi-player override (full team, in seconds)
```bash
# Override multiple players at once
python scripts/override_projections.py \
  --slate deliveries/next_game/full_pmfs_wide.parquet \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --overrides '{"1234": {"status": "out"}, "5678": {"projected_minutes": 28}, "9012": {"minutes_multiplier": 0.80}}' \
  --out-dir deliveries/overrides \
  --game-date 2026-06-18
```

Override types supported:
| Key | Effect |
|-----|--------|
| `"status": "out"` | Set to 0 min, redistribute to teammates |
| `"status": "limited"` | Apply 65% multiplier (or use minutes_cap) |
| `"status": "questionable"` | Apply 75% multiplier |
| `"projected_minutes": N` | Hard-set projected minutes |
| `"minutes_multiplier": 0.80` | Scale current projection by factor |
| `"minutes_cap": N` | Ceiling on projected minutes |

---

## Explainability

### What's driving a player's projected minutes?
```bash
python scripts/explain_projection.py explain-minutes \
  --player-id 1234 \
  --game-date 2026-06-18
```
Shows top feature importances from the HGB minutes model with each
feature's direction (above/below league average) for the given player.

### What changed between yesterday and today?
```bash
python scripts/explain_projection.py explain-change \
  --player-id 1234 \
  --game-date 2026-06-18 \
  --reference-date 2026-06-17
```
Compares per-stat mean projections and key feature values between
two dates to explain why a projection moved.

### Full narrative explanation
```bash
python scripts/explain_projection.py \
  --player-id 1234 \
  --game-date 2026-06-18
```

---

## Output Format

See [`docs/output_format.md`](docs/output_format.md) for complete column
documentation for all four output files:

- `deliveries/next_game/full_pmfs_wide.parquet` — All PMFs
- `deliveries/next_game/market_comparison.parquet` — Model vs. market
- `deliveries/next_game/publishable_edges.parquet` — Edges ≥ 4pp
- `deliveries/next_game/betting_sheet_*.csv` — Human-readable sheet

---

## GitHub Actions Workflows

| Workflow | Schedule | Trigger | Purpose |
|---------|----------|---------|---------|
| `daily_pipeline.yml` | 9 AM ET | Scheduled / manual | Pull data, build features, predict D+1, build edges |
| `weekly_calibration.yml` | Mon 6 AM ET | Manual | Full OOF walk-forward, fit calibrators, gate check |
| `post_game_scoring.yml` | 2 AM ET | Scheduled | Score predictions vs. actuals, update CLV tracking |

### Trigger manually
```bash
# Trigger weekly calibration
gh workflow run weekly_calibration.yml --repo Risky-Scout/wnba-player-props-pmf-model

# Trigger daily pipeline for a specific date
gh workflow run daily_pipeline.yml \
  --repo Risky-Scout/wnba-player-props-pmf-model \
  -f game_date=2026-06-18

# Check recent runs
gh run list --repo Risky-Scout/wnba-player-props-pmf-model --limit 5
```

### Key artifacts produced
- `daily-delivery-{date}` — PMFs, edges, betting sheets
- `calibrators-latest` — Fitted isotonic calibrators (updated weekly on gate pass)
- `clv-report-{date}` — CLV tracking and market superiority report

---

## Forward Testing: How to Evaluate After a Week

After 7+ days of live predictions:

```bash
# View CLV report (last 7 days)
python scripts/generate_clv_report.py \
  --results data/clv_tracking/results.parquet \
  --lookback-days 7 \
  --out-dir artifacts/audits

# Check calibration drift
python scripts/check_calibration_drift.py \
  --scored-predictions data/clv_tracking/drift_window.parquet

# Market superiority gate (informational until 300+ samples per stat)
python scripts/verify_gates.py market \
  data/clv_tracking/results.parquet --min-rows 50

# Build historical review package
python scripts/build_historical_review.py --lookback-days 7
```

**Key metrics to track:**
| Metric | Target | Where |
|--------|--------|-------|
| `mean_true_clv` | > 0 per stat | CLV report |
| `logloss_delta` | < 0 vs market | CLV report |
| ECE per stat | < 0.06 (drift) | Drift check |
| `certified_pass` | True per stat | Market gate |

---

## Codebase Structure

```
src/wnba_props_model/
  data/           BDL REST client, ingestion, normalization
  features/       Feature contract, rolling features, role buckets, pi ratings
  models/         HGB PMF engine, hurdle models, shrinkage, bivariate PMF,
                  market utilities, calibration
  evaluation/     NLL, RPS/CRPS, PIT, ECE, bootstrap UCB95, CLV diagnostics
  pipeline/       Training, calibration, prediction, delivery

scripts/
  pull_bdl_history.py         Pull historical BDL player/game data
  build_canonical_tables.py   Normalize raw data to canonical parquets
  build_features.py           Build wide + long feature tables
  train_baseline_pmfs.py      Train Stage 4 HGB models
  build_oof_pmfs.py           Run walk-forward OOF PMF generation
  fit_calibrators.py          Fit isotonic calibrators from OOF PMFs
  predict_today.py            Generate D+1 PMFs for next game slate
  build_next_game_slate.py    Build player slate for target game date
  build_edge_report.py        Compare PMFs vs. market (Shin no-vig)
  override_projections.py     Fast manual override (seconds, no retraining)
  apply_injury_news.py        Single-player injury/status update
  explain_projection.py       Explain projection drivers
  score_daily_predictions.py  Post-game CLV and accuracy scoring
  generate_clv_report.py      Generate rolling CLV report
  check_calibration_drift.py  Detect calibration drift vs. thresholds
  build_historical_review.py  Generate historical review Markdown report
  verify_gates.py             Calibration and market superiority gates

config/model/
  stage4_baseline.yaml        HGB training configuration
  stage5_oof.yaml             OOF walk-forward configuration
  stage6_calibration.yaml     Calibration gate thresholds

docs/
  output_format.md            Complete column documentation for all outputs
```

---

## Install

```bash
git clone https://github.com/Risky-Scout/wnba-player-props-pmf-model
cd wnba-player-props-pmf-model
pip install -e ".[dev]"
```

Requires: Python 3.11+, `BDL_API_KEY` environment variable.
