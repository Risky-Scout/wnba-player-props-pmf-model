# Vacated-Opportunity Feature — why it can only be evaluated forward

## What it does
`src/wnba_props_model/data/vacated_opportunity.py` takes the set of players who are **OUT**
for a team on a date and redistributes their **strictly-prior** minutes / usage / 3PA /
possessions to the available teammates, in proportion to each teammate's own prior share
(minutes capped, default 40). This produces per-player "vacated opportunity" projections —
the raw signal behind next-man-up prop edges.

## Leakage discipline
- The `prior` frame passed in must be aggregated over games **strictly before** the target
  date (as-of before tip). The function never reads the game being predicted.
- The absence set must come from a **pre-tip** availability pull.

## Why it cannot be backfilled
The absence set requires knowing who was OUT **before tip**. BDL's `player_injuries`
endpoint is **current-state only** — it reports today's injuries, not a historical
pregame-availability archive. Postgame box scores tell you who *did not play*, but using
that to reconstruct pregame status is **leakage** (a late scratch vs. a planned rest vs. an
in-game injury are indistinguishable after the fact, and the market didn't know at tip).

Therefore this feature can only be **evaluated** on availability that is captured going
forward, one slate at a time. `scripts/collect_availability.py` +
`.github/workflows/collect_availability.yml` do exactly that: they append a timestamped,
hashed, pre-tip availability snapshot every slate. After enough forward accrual, the OUT
sets can be joined to strictly-prior aggregates and the vacated-opportunity projections can
be scored against realized outcomes.

## Status
- Feature builder: implemented + unit-tested (`tests/test_vacated_opportunity.py`).
- Forward availability collector: implemented and confirmed pulling live BDL data
  (37 injury rows / 32 OUT players on 2026-07-28; full 5-game ET slate resolved).
- Accrual: begins from the first scheduled `collect_availability.yml` run onward.
