# Opportunity V2 — Data Contract

Opportunity V2 is a **parallel, point-in-time** player-prop candidate (`OPP_V2_RAW`). It never routes
through `structural_pmf.py` and never modifies the frozen baseline (`P0`).

## Canonical identity / cutoff (every row)

```
prediction_cutoff_utc   scheduled_tip_utc   game_id   player_id   team_id   opponent_team_id
cutoff_source           proof_eligible      starter_label_quality
```

- `cutoff_source = exact_quote_timestamp` → `proof_eligible = true`
- `cutoff_source = scheduled_tip_minus_90m` → `proof_eligible = false`
- Every historical join enforces `available_at_utc <= prediction_cutoff_utc` (`opportunity/asof.py`).

## Snapshot tables (append-only, immutable)

| Table | Location | Status on this repo |
|---|---|---|
| Availability snapshots | `data/snapshots/availability/` | **FORWARD_ONLY** — no historical archive exists |
| Lineup snapshots | `data/snapshots/lineups/` | **NO_CONTEXT** — no source yet |
| Roster intervals | `data/processed/wnba_roster_intervals.parquet` | **NO_CONTEXT** — only current team known |
| Player tracking | `data/processed/wnba_player_game_opportunity_tracking.parquet` | **NO_CONTEXT** — no tracking source |
| Team tracking | `data/processed/wnba_team_game_opportunity_tracking.parquet` | **NO_CONTEXT** — box possessions only |

Schemas are defined in `opportunity/contracts.py`. Snapshots are written by
`opportunity/snapshot_store.append_snapshot_partition` (payload-SHA256 identity, atomic
temp→fsync→rename, never overwrites a differing record with the same identity).

## Data tiers

- `0` box score only (available historically 2025–2026)
- `1` play-by-play derived (not available)
- `2` tracking derived (not available)

`config/model/opportunity_v2.yaml: data_tiers.require_tier2_for = [ast, reb]` — those props are **not
historically certifiable** on this repository. Only Tier-0 props (`fg3m`, `pts`) are buildable now.

## Forbidden inputs

No market line/price/probability/spread/total/consensus/CLV/book identity may enter model features.
`opportunity/contracts.forbidden_market_columns` + `opportunity/audit.audit_temporal_purity` enforce
this; the OOF fails closed on any violation. Market data enters ONLY the final evaluation join.

## Honesty rules

- Unknown availability stays `unknown` (never coerced to `available`).
- Historical pregame status is never reconstructed from eventual DNP.
- Conditional minutes trained on appearances only; the active PMF is built once and DNP mass is added
  at most once (optional mixture only). Sportsbook settlement reads `active_pmf_json` (void-on-DNP).

See `artifacts/opportunity_v2/DATA_AVAILABILITY_AUDIT.json` for the authoritative per-family verdict.
