# Pregame Verification Report — 2026-08-01

**Verdict:** PASS_WITH_WARNINGS

- code_sha: `2caef26fa235a9f2cdccc900bd44f42974915e4d`
- prediction_timestamp: `20260801T162447Z`
- stage_dir: `deliveries/pregame_verification/2026-08-01/20260801T162447Z`
- smoke_status: `PASS`
- publish: false (enforced)

## Games
- LV@CHI tip=2026-08-01T17:00:00Z bdl=24969 odds=d113b66ed1649d47506a6434e06bd1b6 identity=MATCH
- NY@PHX tip=2026-08-01T19:00:00Z bdl=24970 odds=e9d3ed7c6df9b35e3b167038806ecd53 identity=MATCH

## Players
- discovered: 64
- accepted: 57
- rejected: 7
- unresolved: 7
- duplicates: 0

## Status counts
{
  "UNKNOWN": 137
}

## Blocking defects
- none

## Warnings
- BDL dates[]=2026-08-01 includes game 24968 tip ET 2026-07-31 (excluded from ET slate)
- player 631 slate_team=PHX players_table_team=LA
- odds parquet schema not recognized for pairing

## Files for independent review
- `artifacts/pregame_verification/2026-08-01/GAME_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/PLAYER_IDENTITY_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/AVAILABILITY_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/FEATURE_SANITY_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/QUOTE_PAIR_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/PMF_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/PMF_SAMPLES.md`
- `artifacts/pregame_verification/2026-08-01/MONOTONICITY_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/PRICING_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/PRICING_TRACE.csv`
- `artifacts/pregame_verification/2026-08-01/STATUS_AUDIT.csv`
- `artifacts/pregame_verification/2026-08-01/FROZEN_PREDICTION_MANIFEST.json`
- `artifacts/pregame_verification/2026-08-01/FROZEN_PREDICTION_REGISTRY.jsonl`
- `artifacts/pregame_verification/2026-08-01/PREGAME_VERIFICATION_SUMMARY.json`
- staged delivery: `deliveries/pregame_verification/2026-08-01/20260801T162447Z`