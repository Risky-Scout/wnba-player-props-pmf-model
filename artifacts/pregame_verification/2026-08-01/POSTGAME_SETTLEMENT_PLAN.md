# Postgame Settlement Plan — 2026-08-01 Frozen Prospective Evidence

## Scope

Settle **only after** BDL game status is final for:

- 24969 LV @ CHI (tip 2026-08-01T17:00:00Z)
- 24970 NY @ PHX (tip 2026-08-01T19:00:00Z)

Do **not** regenerate predictions. Attach outcomes to frozen prediction IDs in
`FROZEN_PREDICTION_REGISTRY.jsonl`.

Frozen stage: `deliveries/pregame_verification/2026-08-01/20260801T162447Z`  
Workflow run: `30708012898`  
Code SHA: `2caef26fa235a9f2cdccc900bd44f42974915e4d`  
Prediction timestamp: `20260801T162447Z` / `2026-08-01T16:27:11.435636+00:00`

## Identity filters before scoring

1. Primary certified set = identity `ACCEPTED` rows **except** player 631
   (`VOID_IDENTITY_TEAM_MISMATCH` — see `PLAYER_631_ROOT_CAUSE.md`).
2. Rejected seven players (`REJECTED_PLAYER_AUDIT.csv`) → exclude from primary metrics;
   optional secondary `unresolved_identity` bucket if box score resolves the same BDL IDs.
3. Never repair frozen PMF atoms, hashes, or timestamps.

## Retrieval

1. Pull final BDL player statistics for game_ids 24969 and 24970.
2. Resolve to frozen `(game_id, player_id)` keys.
3. Preserve DNP / void / push:
   - DNP + VOID_DNP books → void quoted-line bets; full-distribution metrics use
     actual minutes=0 / did_play=false handling per schema.
   - Integer line hits → push; half-points → no push.
4. Join outcomes onto registry rows by `(game_id, player_id, target)`.

## FULL-DISTRIBUTION METRICS (all settled non-excluded PMF rows)

- atom NLL
- CRPS
- predictive mean error (`actual - predictive_mean`)
- MAE
- squared error
- PIT value

## QUOTED-LINE METRICS (only valid frozen exact pairs)

Valid pairs reconstructed: **146**  
Trace rows prepared: **137**

For each valid pair with a non-void settlement:

- binary log loss
- Brier score
- model probability (`p_over_settled` from frozen PMF)
- no-vig market probability
- model-minus-market log-loss difference

## Calibration

Do **not** calculate ECE meaningfully from one slate.  
If probability buckets are recorded, label:

`INSUFFICIENT_ONE_SLATE_SAMPLE`

## Execution gate

Workflow: `.github/workflows/postgame_settlement_frozen.yml`  
Inputs require `games_final=true` confirmation. Hard-fail if any target game is not final.
Never writes to gh-pages. Never mutates frozen evidence files.

## Outputs (when executed)

- `artifacts/pregame_verification/2026-08-01/settlement/POSTGAME_SETTLEMENT_RESULTS.parquet`
- `artifacts/pregame_verification/2026-08-01/settlement/FULL_DISTRIBUTION_METRICS.csv`
- `artifacts/pregame_verification/2026-08-01/settlement/QUOTED_LINE_METRICS.csv`
- `artifacts/pregame_verification/2026-08-01/settlement/SETTLEMENT_SUMMARY.json`
