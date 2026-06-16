# NoVig Output Contract — WNBA Player Prop PMF Model

Version: `v1.0` | Updated: 2026-06-16

## Overview

This document describes the stable, versioned output format produced by the WNBA player prop
model for ingestion into NoVig systems. Outputs are generated daily, **one day before gameday**,
covering all players on the active roster for the next slate of games.

---

## Delivery Schedule

| Output | Timing | Format |
|--------|--------|--------|
| `player_projections_{YYYY-MM-DD}.parquet` | Day prior by 9 AM ET | Apache Parquet |
| `player_projections_{YYYY-MM-DD}.json` | Day prior by 9 AM ET | JSON (array of records) |
| `pmf_distributions_{YYYY-MM-DD}.parquet` | Day prior by 9 AM ET | Apache Parquet |
| `edge_report_{YYYY-MM-DD}.parquet` | Day prior by 9 AM ET | Apache Parquet |

---

## Stats Covered

| Stat | Column Key | Description |
|------|-----------|-------------|
| Points | `pts` | Total points scored |
| Rebounds | `reb` | Total rebounds (offensive + defensive) |
| Assists | `ast` | Assists |
| 3-Pointers Made | `fg3m` | Made three-point field goals |
| Steals | `stl` | Steals |
| Blocks | `blk` | Blocks |
| Turnovers | `turnover` | Turnovers |

---

## Primary Output: `player_projections_{date}.parquet`

One row per player × stat combination for all players with a scheduled game.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `player_id` | int64 | Stable BDL player identifier |
| `player_name` | string | Full player name |
| `team_id` | int64 | Stable BDL team identifier |
| `team_abbreviation` | string | e.g. `IND`, `LAS` |
| `opponent_team_id` | int64 | Opponent BDL team identifier |
| `opponent_abbreviation` | string | Opponent abbreviation |
| `home_away` | string | `home` or `away` |
| `game_id` | int64 | Stable BDL game identifier |
| `game_date` | date | Game date (ET) in `YYYY-MM-DD` |
| `stat` | string | One of the 7 stats above |
| `projected_minutes_mean` | float64 | Model projected minutes (0–45) |
| `projected_minutes_sigma` | float64 | Uncertainty in minutes projection |
| `mean` | float64 | Projected stat value (PMF mean) |
| `median` | int64 | PMF median |
| `mode` | int64 | PMF mode |
| `p0` | float64 | P(stat = 0) |
| `p_ge_1` | float64 | P(stat ≥ 1) |
| `p_ge_3` | float64 | P(stat ≥ 3) |
| `p_ge_5` | float64 | P(stat ≥ 5) |
| `p_ge_10` | float64 | P(stat ≥ 10) |
| `p_ge_15` | float64 | P(stat ≥ 15) |
| `p_ge_20` | float64 | P(stat ≥ 20) |
| `pmf_json` | string | Full discrete PMF as JSON `{"0": p0, "1": p1, ...}` |
| `is_calibrated` | bool | True if role-aware isotonic calibration applied |
| `cal_source` | string | Calibration method (`role_aware_isotonic`, `no_calibrator`, `uncalibrated`) |
| `role_bucket` | string | Player role (`starter_high`, `starter_mid`, `bench_high`, etc.) |
| `projected_minutes_bucket` | string | Minutes tier (`workhorse`, `starter`, `bench_high`, etc.) |
| `dnp_risk` | string | `low`, `moderate`, `high` based on injury status + recent DNP rate |
| `injury_flag` | bool | True if player appears on active injury report |
| `override_applied` | bool | True if manual minutes/DNP override was applied |
| `override_source` | string | `manual`, `injury_report`, or null |
| `model_version` | string | e.g. `wnba_pmf_v1.0_hgb_calibrated` |
| `generated_at` | string | ISO 8601 UTC timestamp of generation |
| `game_date_et` | string | Game date in ET `YYYY-MM-DD` |

### Sample Record

```json
{
  "player_id": 341,
  "player_name": "Brittney Griner",
  "team_id": 2,
  "team_abbreviation": "PHX",
  "opponent_team_id": 5,
  "opponent_abbreviation": "LAS",
  "home_away": "home",
  "game_id": 24857,
  "game_date": "2026-06-16",
  "stat": "pts",
  "projected_minutes_mean": 28.4,
  "projected_minutes_sigma": 4.1,
  "mean": 14.7,
  "median": 14,
  "mode": 13,
  "p0": 0.002,
  "p_ge_1": 0.998,
  "p_ge_3": 0.991,
  "p_ge_5": 0.974,
  "p_ge_10": 0.819,
  "p_ge_15": 0.441,
  "p_ge_20": 0.142,
  "pmf_json": "{\"0\": 0.002, \"1\": 0.003, ...}",
  "is_calibrated": true,
  "cal_source": "role_aware_isotonic",
  "role_bucket": "starter_high",
  "projected_minutes_bucket": "starter",
  "dnp_risk": "low",
  "injury_flag": false,
  "override_applied": false,
  "override_source": null,
  "model_version": "wnba_pmf_v1.0_hgb_calibrated",
  "generated_at": "2026-06-15T13:00:00Z",
  "game_date_et": "2026-06-16"
}
```

---

## PMF Distribution Format

The `pmf_json` field contains a JSON object mapping integer outcome strings to probabilities:

```json
{"0": 0.002, "1": 0.003, "2": 0.009, "3": 0.018, "4": 0.031, ...}
```

**Invariants guaranteed by the model:**
- All probabilities ≥ 0
- Sum of all probabilities = 1.0 (within 1e-6)
- Support starts at 0
- Integer outcomes only
- Maximum support cap per stat: pts=60, reb=30, ast=25, fg3m=15, stl=10, blk=10, turnover=12

---

## Override API

The model supports real-time adjustments before delivery. See `scripts/predict_with_overrides.py`.

### DNP Override

When a player is marked DNP, their projected minutes are zeroed and their expected minutes
are redistributed to active teammates proportionally to baseline minutes shares.

```bash
python scripts/predict_with_overrides.py \
  --game-date 2026-06-16 \
  --dnp 341,419 \
  --out-dir deliveries/overrides
```

### Minutes Override

Manually set projected minutes for specific players (e.g. after lineup confirmation):

```bash
python scripts/predict_with_overrides.py \
  --game-date 2026-06-16 \
  --override-minutes "341:32,419:24" \
  --out-dir deliveries/overrides
```

---

## Explainability

Every projection includes a driver explanation accessible via `scripts/explain_projection.py`:

```bash
python scripts/explain_projection.py \
  --player-id 341 \
  --game-date 2026-06-16 \
  --stat pts
```

**Output includes:**
- Top 5 features driving the minutes projection (with direction vs. player average)
- Top 5 features driving the stat projection
- `minutes_change_flag`: triggered if projected minutes differs from L5 average by > 5 min
- `reason_narrative`: plain-English summary of top drivers

---

## Calibration Quality

The model enforces the following gates (per PenaltyBlog methodology) before certifying outputs:

| Gate | Threshold | Scope |
|------|-----------|-------|
| ECE (Expected Calibration Error) | < 0.03 | Per stat, all roles |
| PIT Kolmogorov–Smirnov | < 0.075 | Per stat/role |
| Mean absolute projection error | < 0.15 | Per stat/role |
| Ignorance Score vs. market | Negative delta (model < market) | Per stat, eligible rows |

Outputs include `is_calibrated: true` only after passing all gates via the weekly calibration workflow.

---

## Historical Performance

A historical review package is generated at `artifacts/historical_review/` covering:
- Per-stat MAE, RMSE, calibration curves
- Hit rates at common market lines (0.5, 1.5, 2.5, ...) vs. model probability
- Over/under accuracy vs. BDL market odds
- Log Loss (Ignorance Score) vs. market baseline per stat

See `scripts/build_historical_review.py` and `docs/HISTORICAL_REVIEW.md`.

---

## Versioning

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-06-16 | Initial NoVig contract |
