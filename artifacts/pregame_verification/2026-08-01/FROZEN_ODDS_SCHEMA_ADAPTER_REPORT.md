# Frozen Odds Schema Adapter Report

## Frozen parquet schema

- File: `wnba_player_props_oddsapi_2026-08-01.parquet`
- Rows: 146
- Columns (27): event_id, game_date, bookmaker, market_key, player_name, line, stat, over_odds, outcome_link_over, event_link, market_link, home_team, away_team, commence_time, last_update, under_odds, outcome_link_under, market_prob_over_no_vig, market_prob_under_no_vig, shin_z, deep_link, pulled_at_utc, source, number_of_books_offering, best_over_odds, best_under_odds, vendor

### Dtypes

```
event_id                        str
game_date                       str
bookmaker                       str
market_key                      str
player_name                     str
line                        float64
stat                            str
over_odds                     int64
outcome_link_over               str
event_link                      str
market_link                  object
home_team                       str
away_team                       str
commence_time                   str
last_update                     str
under_odds                  float64
outcome_link_under              str
market_prob_over_no_vig     float64
market_prob_under_no_vig    float64
shin_z                      float64
deep_link                       str
pulled_at_utc                   str
source                          str
number_of_books_offering      int64
best_over_odds                int64
best_under_odds             float64
vendor                          str
```

## Canonical field correspondence

| Canonical field | Frozen source |
|---|---|
| provider | `source` (= odds_api_v4) |
| Odds API event ID | `event_id` |
| canonical game ID | exact join `event_id` → `GAME_AUDIT.bdl_game_id` |
| bookmaker | `bookmaker` (alias `vendor`) |
| market key | `market_key` |
| player description | `player_name` |
| canonical player ID | exact join `player_name` → frozen slate `player_id` (no fuzzy) |
| line | `line` |
| side | expand from wide `over_odds` / `under_odds` |
| price | `over_odds` / `under_odds` (American) |
| provider quote timestamp | `last_update` |
| market_last_update | `last_update` |
| ingestion timestamp | `pulled_at_utc` |
| scheduled tip | `commence_time` |
| prediction cutoff | frozen `run_metadata.prediction_timestamp_utc` |
| period | `q1` if `market_key` endswith `_q1`, else `game` |
| stat | `stat` |

## Adapter behavior

The adapter performs **column/type transforms only** plus exact frozen-identity joins.
It does **not** call the Odds API, replace timestamps, change prices, invent player IDs,
fuzzy-match names, alter forecasts, or use post-prediction quotes.

Wide rows already contain both American sides for every frozen row
(missing over=0, missing under=0), so exact same-book pairs are reconstructible once
`player_id` / `game_id` are attached via exact joins.

## Original audit failure mode

`quote_pairs_valid=0` because the verification script did not recognize this wide schema
without a `player_id` column — not because opposite sides were absent.
