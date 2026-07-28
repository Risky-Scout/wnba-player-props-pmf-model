# Opportunity V2 — Model Card

## Purpose

Test honestly whether **point-in-time, game-specific opportunity** information improves player-prop
discrimination and proper scoring relative to the market and to the frozen baseline `P0`.

## Prediction graph

```
point-in-time features (strictly lagged)
  -> P(active)                         AvailabilityModelV2      (mixture / renorm only)
  -> P(minutes | active)               ConditionalMinutesDistributionV2 (appearances only)
  -> opportunity count per minutes     OpportunityRateModel / TeamOpportunityShareModel
  -> conversion posterior              HierarchicalBetaConversionModel (Beta, exposure-weighted)
  -> ACTIVE prop PMF                   averaged over equal-probability minutes samples
  -> push-safe settled Over prob       settled_over_probability(active_pmf, line)
```

The canonical output is the **active (conditional-on-play) PMF**. It is built once and never
multiplied by `1 - p_dnp`. An optional availability mixture is stored separately.

## Tier-0 construction (current data)

- **FG3M**: 3PA count PMF (per-minute rate × minutes) × Beta(3P%) conversion, marginalized and
  averaged over minutes samples. Fully honest opportunity decomposition.
- **PTS (proxy)**: `(3 × FG3M)` convolved with a **Tier-0 direct non-3PT points count**.
- **PTS (full decomposition, `OPP_V2_PTS_DECOMP`)**: later built by recovering FGM from tracking
  `fieldGoalPercentage × box FGA` and FTM via the points identity (`2PA·2P% ⊛ 3PA·3P% ⊛ FTA·FT%`, shrunk
  Beta conversions from validated INFERRED labels; ungrounded rows fall back to the labeled proxy). It
  was fully OOF-measured and **still fails vs market** (see status below).

## Measured results (exact-quote OOF, 2026 season, weekly walk-forward)

| Prop | n | dates | LogLoss P0 → OPP_V2 (market) | ΔLL vs P0 | ΔLL vs market | 95% CI ΔLL vs market | AUC OPP_V2 / market |
|---|---|---|---|---|---|---|---|
| fg3m | 498 | 55 | 0.768 → 0.676 (0.665) | **−0.092** | +0.010 | [−0.014, +0.036] | 0.593 / 0.618 |
| pts  | 845 | 55 | 0.804 → 0.782 (0.693) | **−0.023** | +0.089 | [+0.057, +0.124] | 0.503 / 0.518 |

## Current status (2026-07-28) — additive infrastructure/measurement only

**No candidate is market-superior and none is production-enabled. `P0` remains the production champion.**

On ONE canonical universe (`CANDIDATE_COMPARISON_ALL.json`, AUC / LogLoss):

| prop | P0 | R1 (RATE) | G1 (TEAM_SHARE) | PTS_DECOMP (full-only) | market |
|---|---|---|---|---|---|
| fg3m | 0.568 / 0.762 | **0.593 / 0.676** | 0.550 / 0.803 | — | 0.617 / 0.665 |
| pts | 0.486 / 0.800 | 0.503 / 0.782 | — | 0.505 / 0.794 | 0.520 / 0.693 |

- **G1 REGRESSES vs R1** on fg3m (AUC 0.550 vs 0.593); it is a lossy reparameterization of R1's direct rate.
- **PTS_DECOMP FAILS vs market** (ΔLL +0.101, AUC 0.505 vs 0.518; Holm p=1.0; not selection-eligible).
- **G2 CANNOT be built from `tracking-data-v1`** — every intended Tier-2 signal is 0.0% non-zero
  (`SCHEMA_PRESENT_SIGNAL_EMPTY`; `G2_TRACKING_FEASIBILITY.json`). Blocked by empty signal, not identity.
- **No candidate beats the market on any prop.** No Opportunity V2 candidate is deployed, calibrated for
  delivery, or eligible for the Edge Board.

## Honest limitations

The distinctive V2 signals (point-in-time availability/lineups, roster-at-cutoff, vacated role,
player/team tracking) **do not exist as usable historical signal** in this repository (the tracking
asset's Tier-2 columns are schema-present but signal-empty). The measured candidates are Tier-0
box-opportunity models. Reaching market parity/superiority requires:
1. forward append-only availability + lineup snapshot collection (starts now), and
2. a WNBA tracking / play-by-play source for Tier-2 opportunity (touches, potential assists, rebound
   chances, catch-and-shoot 3PA).

Until then, `ast`/`reb`/`turnover`/`stl`/`blk` are **not certifiable** and are excluded from the
candidate. See `DATA_AVAILABILITY_AUDIT.json`.
