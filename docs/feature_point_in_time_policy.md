# Feature Point-in-Time Policy

## Core Rule

Every feature used to predict a game must be computable from data available **before** the scheduled game start.

```
source_effective_at < scheduled_start_utc
source_updated_at <= prediction_time_utc
feature_cutoff_utc <= prediction_time_utc
```

## Rolling Features

The only permitted pattern for rolling/EWMA features:

```python
df[feature_name] = (
    df.groupby("player_id")[raw_stat]
      .transform(
          lambda s: s.shift(1).rolling(window, min_periods=1).mean()
      )
)
```

The `shift(1)` is mandatory. It excludes the current game from its own feature computation.

This applies to:
- Rolling means, medians, standard deviations
- EWMA features
- Expanding means
- Usage rates
- Per-minute rates
- Season-to-date statistics
- Opponent profiles
- Team pace

## Feature Categories and Temporal Safety

| Category | Safe? | Notes |
|----------|-------|-------|
| Rolling box-score stats (shifted) | ✅ YES | shift(1) required |
| EWMA features (shifted) | ✅ YES | shift(1) required |
| Schedule features (rest, B2B) | ✅ YES | Always pregame |
| Prior-game market features | ⚠️ STRUCTURAL_MODEL_UNSAFE | Allowed as market-model input only |
| Current injury status | ⚠️ LIVE_ONLY | Use for live predictions only |
| Historical injury snapshots | ✅ YES (if archived) | Only if snapshot precedes game |
| Final season records | ❌ NO | Join to all games of that season |
| Current standings | ⚠️ USE_PREGAME_ONLY | Compute from games before target |
| Shot-location full-season data | ❌ NO | Apply shift(1) |
| SVD/PCA embeddings | ✅ IF FOLD-SAFE | Must fit inside training fold |

## Market-Prior Features

The following features are derived from prior-game market data and are safe from same-game leakage (lagged one game):

- `player_market_p_over_prev` — prior closing P(over)
- `player_market_line_prev` — prior closing line
- `player_line_movement_prev` — prior (close - open) movement

**However**, these features must NOT enter the structural outcome model (which must use only basketball information). In safe mode, they are stripped before the HGB feature matrix is built.

## Injury Data Policy

- **Live predictions:** Use current injury status (current-state data is acceptable for same-day predictions)
- **Historical training:** Do NOT join today's injury state to historical games
- **Historical snapshots:** Only use a snapshot if it was archived before the modeled game
- **Imputation:** Do not impute historical injury state from the final box score (outcome leakage)

## Standings Policy

Compute standings from games that occurred before the target game:

```python
# Correct: expanding pre-game sum
standings_before = df.groupby("team_id")["wins"].transform(
    lambda x: x.shift(1).expanding(min_periods=1).sum()
)

# WRONG: season-end standings joined to all games
df["season_wins"] = df.groupby("team_id")["wins"].transform("sum")  # leaky!
```

## Validation

The validator `validate_feature_point_in_time()` in `pipeline/safety.py` checks:
- `feature_cutoff_utc` <= `prediction_time_utc`
- `source_effective_at` < `scheduled_start_utc`

In safe mode with `fail_on_feature_leakage=True`, violations raise `ValueError`.
