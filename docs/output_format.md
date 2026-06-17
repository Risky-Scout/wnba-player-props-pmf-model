# Output Format Documentation

This document describes every column in the four primary output files produced
by the daily pipeline. All files are written to `deliveries/next_game/` after
each daily run.

---

## 1. `full_pmfs_wide.parquet` — Full PMF Predictions

One row per `(player_id × game_id × stat)`. Contains complete probability
mass function data for every player, game, and statistic.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `player_id` | int64 | BDL player identifier | `1234` |
| `player_name` | string | Player display name | `"A'ja Wilson"` |
| `team_id` | int64 | BDL team identifier | `7` |
| `team_abbreviation` | string | Team short code | `"LAS"` |
| `opponent_team_id` | int64 | Opposing team BDL ID | `12` |
| `opponent_team_abbreviation` | string | Opponent short code | `"SEA"` |
| `game_id` | int64 | BDL game identifier | `987654` |
| `game_date` | string/date | Game date (ISO 8601) | `"2026-06-18"` |
| `is_home` | bool/int | 1 if player's team is home | `1` |
| `home_away` | string | `"home"` or `"away"` | `"home"` |
| `stat` | string | Stat category (see below) | `"pts"` |
| `pmf_json` | string | Sparse JSON PMF: `{"0": p0, "1": p1, ...}` | `{"0": 0.02, "12": 0.08, ...}` |
| `pmf_mean` | float64 | PMF expected value (mean projection) | `18.3` |
| `pmf_variance` | float64 | PMF variance | `42.1` |
| `pmf_support_min` | int64 | Minimum support value (always 0) | `0` |
| `pmf_support_max` | int64 | Maximum support value (cap) | `60` |
| `p0` | float64 | P(stat = 0) | `0.018` |
| `p_ge_1` | float64 | P(stat ≥ 1) | `0.982` |
| `p_ge_2` | float64 | P(stat ≥ 2) | `0.941` |
| `p_ge_3` | float64 | P(stat ≥ 3) | `0.887` |
| `p_ge_5` | float64 | P(stat ≥ 5) | `0.791` |
| `p_ge_10` | float64 | P(stat ≥ 10) | `0.563` |
| `minutes_mean` | float64 | Projected minutes (model output) | `32.4` |
| `minutes_sigma` | float64 | Uncertainty in projected minutes | `4.2` |
| `role_bucket` | string | Player role classification | `"starter"` |
| `is_calibrated` | bool | True if isotonic calibration was applied | `true` |
| `cal_source` | string | Calibration method used | `"role_aware_isotonic"` |
| `model_version` | string | Model version tag | `"wnba_pmf_v1.0_hgb_calibrated"` |
| `pmf_source` | string | Stage identifier | `"stage4_baseline"` |

**Stat categories:**

| `stat` value | Description |
|---|---|
| `pts` | Points scored |
| `reb` | Total rebounds |
| `ast` | Assists |
| `fg3m` | 3-pointers made |
| `stl` | Steals |
| `blk` | Blocks |
| `turnover` | Turnovers |
| `stocks` | Steals + Blocks (combo) |
| `pts_ast` | Points + Assists (combo) |
| `pts_reb` | Points + Rebounds (combo) |
| `reb_ast` | Rebounds + Assists (combo) |
| `pts_reb_ast` | Points + Rebounds + Assists (combo) |

**Role bucket values:**

| `role_bucket` | Minutes range | Description |
|---|---|---|
| `starter` | ≥ 30 min | Heavy starter |
| `core` | 24–30 min | Core rotation |
| `rotation` | 18–24 min | Regular rotation |
| `bench` | 12–18 min | Bench contributor |
| `fringe` | < 12 min | Minimal minutes |
| `inactive_risk` | ≤ 3 min or p_inactive ≥ 12% | DNP risk |

---

## 2. `market_comparison.parquet` — Model vs. Market

One row per `(player_id × game_id × stat × market_line)`. Contains market
odds alongside model probabilities for direct comparison and CLV calculation.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `player_id` | int64 | BDL player ID | `1234` |
| `player_name` | string | Player name | `"A'ja Wilson"` |
| `game_id` | int64 | BDL game ID | `987654` |
| `game_date` | string | Game date | `"2026-06-18"` |
| `stat` | string | Stat category | `"pts"` |
| `line` | float64 | Market over/under line | `22.5` |
| `over_odds` | int64 | American odds for OVER | `-115` |
| `under_odds` | int64 | American odds for UNDER | `-105` |
| `vendor` | string | Book/market source | `"BDL"` |
| `market_prob_over_no_vig` | float64 | Shin no-vig probability of OVER | `0.483` |
| `market_prob_under_no_vig` | float64 | Shin no-vig probability of UNDER | `0.517` |
| `shin_z` | float64 | Shin market sharpness (0 = soft, 0.1 = sharp) | `0.042` |
| `model_prob_over` | float64 | Model's P(stat > line) from PMF tail-sum | `0.531` |
| `edge_over` | float64 | Model edge betting OVER (model_prob - market_prob) | `+0.048` |
| `edge_under` | float64 | Model edge betting UNDER (1-model_prob - market_prob_under) | `-0.048` |
| `market_implied_mean` | float64 | Poisson mean implied by market mid-point | `22.1` |
| `pmf_mean` | float64 | Model's mean projection | `23.4` |
| `mean_disagreement` | bool | True if model/market means diverge > 2.0 | `false` |
| `confidence_tier` | string | `standard` or `high_adversity` based on shin_z | `"standard"` |
| `is_calibrated` | bool | Whether model PMF was calibrated | `true` |
| `model_version` | string | Model version tag | `"wnba_pmf_v1.0_hgb_calibrated"` |

---

## 3. `publishable_edges.parquet` — Edges ≥ 4pp

Filtered subset of `market_comparison.parquet` where `|edge_over| ≥ 0.04`.
Sorted by absolute edge descending. Contains all columns from market_comparison
plus:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `confidence_tier` | string | `standard` (shin_z ≤ 0.06) or `high_adversity` | `"standard"` |

**Confidence tier guidance:**

- `standard`: Market shin_z ≤ 0.06 — softer market, lower adverse-selection risk.
  Treat as primary betting candidates.
- `high_adversity`: Market shin_z > 0.06 — sharper market with more informed money.
  Use smaller position sizes or require larger edge threshold.
- `unknown`: shin_z not available for this line.

---

## 4. `betting_sheet_*.csv` — Human-Readable Betting Sheet

Flat CSV version of publishable edges, formatted for manual review or
spreadsheet ingestion.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `game_date` | string | Game date | `"2026-06-18"` |
| `player_name` | string | Player name | `"A'ja Wilson"` |
| `team` | string | Team abbreviation | `"LAS"` |
| `opponent` | string | Opponent abbreviation | `"SEA"` |
| `stat` | string | Stat category | `"pts"` |
| `line` | float | Market line | `22.5` |
| `direction` | string | Bet direction (`OVER` or `UNDER`) | `"OVER"` |
| `model_prob` | float | Model probability for bet direction | `0.531` |
| `market_prob_no_vig` | float | Market no-vig probability | `0.483` |
| `edge` | float | Edge in percentage points | `0.048` |
| `kelly_quarter` | float | Quarter-Kelly suggested stake (fraction of bankroll) | `0.021` |
| `shin_z` | float | Market sharpness score | `0.042` |
| `confidence_tier` | string | `standard` or `high_adversity` | `"standard"` |
| `over_odds` | int | American odds for over | `-115` |
| `under_odds` | int | American odds for under | `-105` |

---

## Quick Start: Ingesting into External Systems

```python
import pandas as pd

# Load publishable edges (all edges >= 4pp from latest slate)
edges = pd.read_parquet("deliveries/next_game/publishable_edges.parquet")

# Filter to standard-market edges only (lower adverse-selection risk)
standard_edges = edges[edges["confidence_tier"] == "standard"]

# Filter to a minimum edge threshold (e.g. >= 5pp)
strong_edges = standard_edges[standard_edges["edge_over"].abs() >= 0.05]

# Sort by edge descending
strong_edges = strong_edges.sort_values("edge_over", key=abs, ascending=False)

# Key columns for ingestion
output_cols = [
    "player_name", "stat", "line", "direction",
    "model_prob_over", "market_prob_over_no_vig",
    "edge_over", "shin_z", "confidence_tier",
    "game_date", "game_id", "player_id",
]
print(strong_edges[output_cols].head(20))


# Load full PMFs for a specific player and stat
pmfs = pd.read_parquet("deliveries/next_game/full_pmfs_wide.parquet")
player_pts = pmfs[(pmfs["player_name"] == "A'ja Wilson") & (pmfs["stat"] == "pts")]

# Parse PMF JSON to numpy array
import json
import numpy as np

def parse_pmf(pmf_json_str: str, cap: int = 60) -> np.ndarray:
    """Parse sparse PMF JSON to dense numpy array."""
    d = json.loads(pmf_json_str)
    arr = np.zeros(cap + 1)
    for k, v in d.items():
        idx = int(k)
        if idx <= cap:
            arr[idx] = float(v)
    total = arr.sum()
    return arr / total if total > 0 else arr

pmf_array = parse_pmf(player_pts["pmf_json"].iloc[0])
print(f"Mean: {np.dot(np.arange(len(pmf_array)), pmf_array):.2f}")
print(f"P(>22.5): {pmf_array[23:].sum():.4f}")
```

---

## File Freshness

All files in `deliveries/next_game/` are regenerated daily by the morning
pipeline (9 AM ET). The `game_date` in each file refers to the **next day's**
games (D+1 predictions, available one full day before tip-off).

For historical CLV tracking data, see `data/clv_tracking/results.parquet`.
