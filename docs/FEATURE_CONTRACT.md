# Feature Contract

## Feature families

- Identity
- Schedule/rest
- Minutes and role state
- Player per-minute rolling rates
- Team context and opponent context
- Injury/availability
- Lineup fallback
- Sparse-event opportunity

## Leakage rules

Forbidden in training:

- same-game box score values for the target game
- market line or odds
- actual starter/minutes/final score
- post-tip information not available before quote time

## WNBA lineup strategy

Because WNBA BDL docs do not list a lineup endpoint, the default features are:

- `expected_starter`
- `expected_bench`
- `recent_starter_rate5`
- `minutes_lag1`
- `minutes_roll5`
- `team_expected_starters_count`

`lineup_confirmed` and `confirmed_starter` are reserved for a later external lineup feed.
