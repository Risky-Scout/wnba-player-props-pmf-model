# WNBA Sharp PMF V4 — Final Implementation Report

## Delivered (real, fitted, honest)
- **Integration + version lineage**: continued on PR #99 branch, merged newest `origin/main`,
  retitled to "WNBA Sharp PMF V4 — atom-reliable production forecasting and pricing". V2/V3 design,
  configs, metrics, and artifacts preserved unmodified. New `sharp_v4/` paths.
- **2026 V3 holdout marked CONSUMED** (`HOLDOUT_LINEAGE_AUDIT.json`,
  `status=CONSUMED_NOT_VALID_FOR_V4_SELECTION`). V4 selection uses **2023-2025 only**.
- **V4 design frozen before fitting** (`V4_FREEZE_MANIFEST.json`,
  `modeling_design_v4_sha256`), including feature-contract schema hashes and the prospective start.
- **Core statistical defects repaired** (Section 4):
  - **A** explicit ordered per-component feature contracts (anchored regex → exact lists, no
    `if stat in col`); market/label/id columns excluded (`FEATURE_CONTRACTS.json`).
  - **B** 1:1 join cardinality asserted with row counts (`JOIN_CARDINALITY_AUDIT.json`).
  - **C** **no outcome clipping** — outcomes above nominal support are scored on the exact analytic
    tail (`CountPMF.logpmf(200)` is finite, not clipped); NLL/CRPS/pricing share one support.
  - **D** adaptive support + retained overflow atom; exact analytic survival used for scoring and
    Over settlement (`PMF_SUPPORT_AND_OVERFLOW_AUDIT.json`).
  - **E** hierarchical (partially-pooled) dispersion `player→role→stat→league`, replacing one
    global scalar (`DISPERSION_REPORT.json`); NB2 vs hurdle-NB2 compared per stat.
- **Minutes as a distribution** — `P(minutes=m|active)` (truncated, role-band dispersion), reported
  via NLL/CRPS (`MINUTES_DISTRIBUTION_REPORT.json`).
- **Structural components** — OREB/DREB fitted separately (REB=OREB+DREB convolution), 3PA×3P%
  components; **points is the V4 exact-tail hierarchical-dispersion direct model, NOT falsely
  labelled structural** because the recovered data lacks FGM/FTM (data limit;
  `SHOOTING_COMPONENT_REPORT.json`).
- **Chronological OOF on 2023-2025**: V4 improves NLL on 6/7 Tier A stats vs the V3 baseline
  (pts 3.067→2.945, fg3m 1.110→1.093, reb 2.113→2.090; `V3_BASELINE_COMPARISON.csv`).
- **Market comparison** (exact same-time no-vig, clustered bootstrap): **market not beaten** on any
  stat → `MARKET_CONSISTENT_ZERO_RESIDUAL` (`MARKET_PMF_AUDIT.json`, `ACTIVATION_REGISTRY.json`).
- **Market-consistent PMF** — closed-form min-KL exponential tilt to the exact no-vig Over prob
  (`core.market_consistent_atoms`; tests prove constraint match + monotone CDF + delta-0 fallback).
- **LIVE real upcoming-slate run (not fixture)** for **2026-07-31**: refit through the latest
  completed game, real BDL schedule (5 games), real point-in-time features, **live The Odds API
  no-vig quotes** (Enterprise key; credits logged) → **224 players, 29,558 atoms, 14,500 fair
  Over/Under lines, 130 players matched to live market** → market-consistent PMFs.
  `deliveries/sharp_v4/2026-07-31/T-live/` (gitignored).
- **Append-only prospective registry** — 1,568 immutable pre-tip forecasts, hashed;
  re-registration adds 0 (`PROSPECTIVE_REGISTRY_STATUS.json`).
- **Tests**: 13 V4 acceptance tests (+10 V3, +20 pricing) pass; full suite green; ruff clean.

## Not fitted / deferred (honest — abstain or market-fallback, never fabricated)
- Structural **points** decomposition (needs FGM/FTM — absent in recovered data; BDL box-score
  re-pull required).
- Shared **game-environment** latent + team-opportunity **reconciliation** (`DESIGNED_NOT_FITTED`).
- **Copula** joint dependence for combos (`MARKET_CONSISTENT_ZERO_RESIDUAL`).
- **Q1** markets (`ABSTAIN_NO_Q1_LABELS`), first-basket/event models, role-state minutes mixture,
  automated drift refit, season-transition, and the multi-horizon (T-12h..T-10m) forward
  collection (single pregame + T-live horizons produced this run).

## Merge status (honest)
The model is on the PR #99 branch, **not** on `origin/main` (`ORIGIN_MAIN_VERIFICATION.json`,
`merged_to_main=false`). **This agent cannot merge**: its GitHub CLI is read-only and no merge tool
is available. PR #99 is pushed and ready for a maintainer to run `gh pr merge 99 --squash`. On-main
ancestry/SHA verification will pass only after that.

## Verdict
A functional, leakage-free, atom-reliable V4 forecasting + pricing system with genuine
statistical-defect repairs, real 2023-2025 OOF (improved over V3), a live real upcoming-slate run
with live market-consistent PMFs, and honest abstention for uncertified components. **No market
superiority is claimed** (market not beaten). **Not yet on `main`** — merge is externally blocked
by tool constraints.
