# Modeling Design V2 — Feature-Only WNBA Player-Prop PMF Model (FROZEN)

This is the frozen design that governs feature-driven OOF construction and selection. It exists
because no committed **Modeling Design V1** was ever created (the earlier premature modeling had
no frozen design and is quarantined `INVALID_FOR_MODEL_SELECTION`, stash `13c7195`). Per the
authorization's Section B, execution against V1 is **not executable**; this V2 is frozen before
any outer-fold prediction and requires review before OOF runs.

**Sportsbook data is used nowhere** in fitting, selection, calibration, support, uncertainty, or
tie-breaking. If no feature candidate passes the frozen gates, the target is `WITHHELD` — the
market baseline is never substituted for the model.

## Pipeline
`pregame features → P(participate) → P(minutes | participate) → P(stat=k | minutes, participate) → stat PMF → combination PMFs`

The direct-stat PMF integrates the conditional stat model over the **OOF-predicted minutes PMF**
(never the held-out actual minutes): `P(S=k|X) = Σ_m P(S=k|M=m,X)·P(M=m|X)`.

## Inputs (Stage 9 contract)
- Features: `data/recovered_v2/modeling/wnba_pregame_features_t12.parquet` (325 approved,
  `feature_schema_hash=112e332e30b1f0e3`). Every estimator entry point calls the shared
  `estimator_guard.guard_estimator_frame`.
- Targets: `data/recovered_v2/modeling/wnba_player_targets.parquet` (participation, actual_minutes,
  7 direct + 5 deterministic combos).

## Cutoff classes
`EXACT_T12` (tip−12h) → only these enter outer validation metrics and later market comparison.
`CONSERVATIVE_PRETIP` (tip unknown → game-day 00:00 UTC) → training only. `PIT_INELIGIBLE`
(2nd+ same-day game) → excluded. No tip is ever inferred to increase coverage.

## Chronological OOF (`chronological_oof_v2.yaml`)
Expanding-window by game-date: 6 outer blocks (block 0 = warm-up/priors, blocks 1–5 predicted by
earlier data only). Same date never split across folds; all players of a game share a fold.
3 inner expanding folds for hyperparameter selection inside training history only. Calibration
from prior OOF blocks only. No in-sample fallback; a failed prediction is recorded as an
exclusion, never substituted by a training-fit/same-block/future/market value.

## Candidate registry (`modeling_design_v2.yaml`)
- **Participation:** hierarchical baseline + HGB classifier (log loss primary).
- **Minutes:** hierarchical empirical + HGB quantile grid (q01…q99) → monotone inverse-CDF
  distribution on a 0–50 grid (overtime headroom); CRPS primary.
- **Direct stats:** hierarchical rate baseline, Poisson-HGB, NB2-HGB, minutes-mixture NB2, and a
  hurdle-NB2 for the rare/zero-heavy stl/blk/turnover (rare stats are **not** forced into the pts
  family). Dispersion by conditional-residual moments (bounds [0.5, 500]), training-fold only,
  never changed after outer metrics. NLL primary.
- **Combinations:** independent exact convolution baseline + Gaussian-copula-on-training-residual-
  ranks dependence candidate (PSD-corrected, MC error < 0.01). Selected by combination proper
  scores only.

## PMF support (`pmf_support_v2.yaml`)
Support starts at 0; grows until omitted tail < `1e-4` or the frozen hard cap; overflow recorded.
Support/dispersion/caps never depend on the observed outcome.

## Calibration
Distributional PIT recalibration operating on the monotone CDF (never per-line), trained on prior
OOF blocks only; fallback = no calibration. All PMF validity gates re-run after calibration.

## Uncertainty
Chronological block-bootstrap ensemble (K=20, deterministic seeds) → PMF ensemble + threshold
intervals. Frozen abstention rules (insufficient history, unstable participation/minutes, missing
features, schema mismatch, out-of-population, excessive epistemic width). No threshold derives
from betting ROI.

## Selection gates (`model_selection_gates_v2.yaml`)
PMF validity (0 invalid rows), OOF coverage minimums, strict NLL improvement over the feature
baseline (CRPS not materially worse), PIT KS ≤ 0.05, calibration slope ∈ [0.80,1.25], 80% coverage
∈ [0.72,0.88], no season subgroup NLL > 1.25× overall, deterministic reproduction. Tie-break:
NLL → CRPS → PIT → simpler family; market tie-break forbidden. All gates pass → RESEARCH (or
SHADOW_CANDIDATE_PENDING_FREEZE at freeze); any fail → WITHHELD. `VALIDATED` is unattainable here.

## Seeds
master 20260730; fold = master + outer_fold; inner = master + 100 + inner_fold; bootstrap =
master + 10000 + index.

## Freeze rule
Once the first outer-fold prediction is generated, this design is closed. Any design-semantic
change (candidates, dispersion, support, feature groups, preprocessing, grids, calibration,
gates, uncertainty) requires quarantining V2 OOF artifacts, freezing V3, and restarting OOF.
