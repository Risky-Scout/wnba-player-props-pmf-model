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
- **PTS**: `(3 × FG3M)` convolved with a **Tier-0 direct non-3PT points count** (the box lacks FGM/FTM
  makes, so 2P/FT cannot be decomposed). PTS is therefore a documented proxy, not a full decomposition.

## Measured results (exact-quote OOF, 2026 season, weekly walk-forward)

| Prop | n | dates | LogLoss P0 → OPP_V2 (market) | ΔLL vs P0 | ΔLL vs market | 95% CI ΔLL vs market | AUC OPP_V2 / market |
|---|---|---|---|---|---|---|---|
| fg3m | 498 | 55 | 0.768 → 0.676 (0.665) | **−0.092** | +0.010 | [−0.014, +0.036] | 0.593 / 0.618 |
| pts  | 845 | 55 | 0.804 → 0.782 (0.693) | **−0.023** | +0.089 | [+0.057, +0.124] | 0.503 / 0.518 |

- **Information gate: PASS** — OPP_V2 beats P0 on both props (log loss and, for fg3m, AUC 0.564→0.593).
- **Market-superiority gate: NOT PASSED** — fg3m is statistically indistinguishable from the market
  (CI straddles 0); pts is clearly worse (proxy construction).

## Honest limitations

The distinctive V2 signals (point-in-time availability/lineups, roster-at-cutoff, vacated role,
player/team tracking) **do not exist historically** in this repository. The measured candidate is a
Tier-0 box-opportunity model. Reaching market parity/superiority requires:
1. forward append-only availability + lineup snapshot collection (starts now), and
2. a WNBA tracking / play-by-play source for Tier-2 opportunity (touches, potential assists, rebound
   chances, catch-and-shoot 3PA).

Until then, `ast`/`reb`/`turnover`/`stl`/`blk` are **not certifiable** and are excluded from the
candidate. See `DATA_AVAILABILITY_AUDIT.json`.
