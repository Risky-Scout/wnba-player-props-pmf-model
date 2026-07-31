# WNBA Sharp PMF V6 — Final Implementation Report

## Delivered (real, verified corrections)
- **BDL endpoint CORRECTED** — `/wnba/v1/player_stats` returns 200 with FGM/FTM (V5 probed the
  undocumented `/stats`, `/box_scores`, `/season_stats`, all 404; the V5 "tier-blocked" claim was
  **wrong**). `BDL_PLAYER_STATS_ENDPOINT_AUDIT.json`.
- **FGM/FTM recovered** for ~5.4k player-games with the **`pts` identity holding exactly**
  (`2·2PM+3·3PM+FTM = PTS`, 0 violations); reb differs on some rows (team-rebound quirk).
  `SHOOTING_LABEL_RECONCILIATION.json`. (Pull is partial; a resumable full pull is the remaining
  work before structural-points production fitting.)
- **Market-projection mass bug FIXED** — `TiltedDistribution` applies the exponential tilt to the
  **complete** base (incl. its analytic tail) with `Z` computed under a certified remainder bound;
  **stored atoms + overflow = 1 exactly** (audited error `0.0`, test enforces ≤1e-10). The V5 bug
  (normalize stored atoms, then reattach unchanged overflow → sum>1) is gone. Multi-line projection
  returns one proper push-aware distribution; contradictory constraints fail closed.
- **Hurdle analytic variance** (complete second moment, not stored atoms with zero tail); bounded
  tilt basis keeps the transformed infinite tail summable.
- **Live real slate 2026-07-31** (not fixture): 224 players, **35,504 atoms, 14,520 fair lines,
  130 tail-aware market-consistent projections / 14 infeasible**; append-only prospective registry.
- **11 V6 acceptance tests** (+16 V5 +13 V4 +10 V3 +20 pricing) pass; ruff clean.

## Honest scope (not fitted this pass — abstain / market-fallback, not fabricated)
Full FGM/FTM coverage + fitted structural points, shared game-environment + team-constrained
minutes reconciliation, copula joint dependence, Q1 labels/models, first-basket, and a persisted
cross-fit calibration layer remain **not fitted** (labels/compute beyond this pass). The V5
minutes-mixture stat fit is retained unchanged; V6's changes are pricing/distribution correctness +
the endpoint/label correction.

## Merge status (honest)
Not on `origin/main` (`merged_to_main=false`). Agent `gh` is read-only with no merge/ready tool;
maintainer must run `gh pr ready 99 && gh pr merge 99 --squash`. No secrets/private data committed.

## Verdict
The concrete V6 correctness items the mission centered on — the correct BDL endpoint (+ FGM/FTM
recovery), the market-projection tail-mass bug, and the hurdle variance — are genuinely fixed,
tested, and audited, with a live real slate priced through the corrected projection. **No market
superiority is claimed.** Several large components remain honestly not-fitted. **Not on `main`.**
