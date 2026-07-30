# WNBA Sharp PMF v3 — Final Implementation Report

## What was built (real, fitted, chronologically evaluated)
- **Integration**: `cursor/wnba-sharp-pmf-v3` off `origin/main`; merged PR #98 (pricing engine,
  registry, generator, first-basket, calibration) and PR #97 (recovered data infra, feature/target
  separation, estimator guards, leakage audits, Design V2). One `bdl_client.py` conflict resolved
  in favor of the newer probe-referenced fix. See `PR97_PR98_INTEGRATION_AUDIT.json`.
- **Private data preserved + hash-verified** (never committed): 23,931 pregame player-games × 336
  features, Tier A targets, 256,160 atomic quotes, 50,753 exact same-time settled no-vig pairs.
  Training **fails closed** on hash drift. See `PRIVATE_INPUT_MANIFEST.json`.
- **Modeling Design V3 frozen** before any prediction (`modeling_design_v3_sha256`), V2 archived
  unmodified. See `V3_FREEZE_MANIFEST.json`.
- **Fitted, active-conditional, chronological OOF models** (`src/wnba_props_model/sharp_v3/`):
  participation (HGB + isotonic cross-fit calibration), conditional minutes (HGB), and Tier A
  direct-stat NB2 PMFs (`pts reb ast fg3m stl blk turnover`) with conditional-residual dispersion,
  integrated over the expanding-window dev folds (2024/2025) and a **single-use 2026 holdout**.
- **Leakage caught and fixed**: bare label columns (`pts`,`reb`,…) initially matched the
  stat-name feature contract after the features+targets merge, producing impossible OOF `mean_mae
  ≈ 0.03` and a fake `delta_logloss ≈ −0.48`. Fixed via `LABEL_COLS` exclusion + a fail-closed
  guard; post-fix `pts` MAE ≈ 4.2 and **0 feature==label violations**. See `LEAKAGE_AUDIT.json`.
- **Exact same-time no-vig market comparison** on identical settled rows (push/void removed),
  game-date-clustered bootstrap CI. **The pure model does not beat the market**: dev
  `delta_logloss` is ≈ +0.008 (pts) with clustered 95% upper bound > 0 in every fold →
  `market_beaten = False` everywhere. Consistent on the 2026 holdout.
- **Real-slate pricing (not a fixture)**: newest real slate in the recovered data (2026-07-29),
  93 players, 12,665 atoms, 6,248 fair Over/Under lines, from refit fitted artifacts + real
  point-in-time features, with full lineage. See `deliveries/sharp_v3/2026-07-29/`.

## Honest status by tier
- **Tier A direct** (`pts reb ast fg3m stl blk turnover`): `TRAINED_AND_EVALUATED` (pure);
  `MARKET_FALLBACK` (no certified residual). Production = exact market fallback.
- **Tier A derived combos** (`stocks pts_ast pts_reb reb_ast pts_reb_ast`): `MARKET_FALLBACK` —
  copula/shared-latent dependence is designed + scaffolded but **not fitted/certified** this run
  (`JOINT_DEPENDENCE_AUDIT.json`); combos abstain to market rather than summing independent marginals.
- **Tier B/C** (Q1, double/triple-double, fantasy, first-basket): **ABSTAIN / not fitted** —
  no fabricated fixture/default probabilities (`Q1_COHERENCE_AUDIT.json`, `SHOT_COMPONENT_MODEL_REPORT.json`).

## Not certified / deferred (honest)
- No market superiority is claimed (correctly — the market is not beaten).
- Structural 2PA/3PA/FTA×conversion decomposition, role-state minutes mixture, game-level
  shared-latent reconciliation, copula joint dependence, market-anchored residual boosting, and
  first-basket competing-risk fitting are **designed (V3) and scaffolded** but not fitted/certified.
- The model is **on the branch/PR, not on `origin/main`** (`ORIGIN_MAIN_VERIFICATION.json`,
  `merged_to_main=false`). Not auto-merged.

## Production-readiness verdict
**Functional and safe, not certified-superior.** A leakage-free, chronologically-evaluated Tier A
pure PMF exists with honest OOF + single-use holdout metrics and a real-slate pricing run;
production behavior is exact market fallback where the pure model is not proven better, and honest
abstention where no model/market exists. It is **not** a certified market-beating product and is
**not** on `main`.
