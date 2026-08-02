# Audit Verification Gate — Phase 2

**Verdict: `PASS_CONTINUE_TO_REPAIR`**

Starting main: `188f9b99bfe35d187fd8fd5f22bd228fec8c8fc8` (= `origin/main`, includes PR #107).  
Production pointer: `artifacts/releases/wnba-pmf-production-v1.1` (unchanged).

## Feature counts

| Class | Unique production features |
|---|---|
| OBSERVED_BDL | 0 |
| OBSERVED_INJURY_WORKBOOK | 0 |
| DERIVED_POINT_IN_TIME | **284** |
| OPTIONAL_MANUAL_OVERRIDE | 0 |
| MARKET_ONLY | 0 |
| UNAVAILABLE_EXCLUDED | 0 |
| LEAKAGE_PROHIBITED | 0 |
| UNKNOWN_SOURCE | 0 |

All 284 contracted production features are **derived point-in-time** transforms of prior-game BDL box observations (lags/EWMA/z-scores/interactions). “Derived” here means calculated from legitimate observed history — not market-derived, not leakage-prohibited, and not silently invented mandatory inputs.

Component slots = 895; matrix adds 3 `__component_note__*` rows (287 names) — not a source contradiction.

## Pure-model integrity

- Required MARKET_ONLY / UNKNOWN_SOURCE / LEAKAGE_PROHIBITED: **0**
- Unavailable mandatory production features: **0**
- No same-game outcome or postgame starter/minutes fields in contracts

## Silent defaults

- Production contract rows: native NaN missing support — **legitimate**
- `player_is_confirmed_starter`: silent zero in wide matrix, **not** in contracts — dead/unsafe if required
- Raw `flatten_player_stat_row` null→0 coercion — unsafe for auditability; root of fgm/ftm drop when unmapped

## FGM/FTM

API `/wnba/v1/player_stats` returns `fgm`/`ftm` (canary game 3858).  
`STAT_TO_BDL_COL` + `flatten_player_stat_row` drop them before parquet write.  
Normalized caches lack the fields; recovered shooting labels restore **5430** rows offline. Full core-data repull unnecessary.

## Participation (reproduced)

| Season | Active | Inferred DNP | Confirmed inactive |
|---|---|---|---|
| 2023 | 4918 | 876 | 0 |
| 2024 | 4928 | 1001 | 0 |
| 2025 | 5869 | 1250 | 21 |
| 2026 | 4324 | 981 | 7 |

Classes: `minutes>0` → active; `minutes_flag==non_playing` → confirmed inactive; other zero-minute box rows → inferred DNP. Unknown roster eligibility not enumerable without historical roster snapshots.

## Injury workbook

Event columns only; summary excluded; 2023–2026 = 175/203/252/225; identities 811 exact / 42 unresolved / 2 ambiguous; no unrestricted fuzzy match; `date_returned` / `total_games_missed` not onset features; 2026 returns remain open. Leakage rules valid.

## Valid minutes contract

Conditional active minutes buildable from lagged performance, roster, availability, role, teammate absences, rest/schedule, game environment, optional overrides — without starters, lineups, tracking, or BDL projected minutes.

## Inconsistencies (non-blocking)

1. Matrix unique-name count 287 vs contract 284 (component notes).
2. Full offline fgm/ftm restore from normalized cache alone is impossible; partial via shooting labels; defect still confirmed.

## Gate conditions

All proceed conditions hold → continue to Stage B repair (no model fit, no production pointer change).
