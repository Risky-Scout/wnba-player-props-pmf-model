# Premature Modeling — Abort Report

STATUS = INVALID_FOR_MODEL_SELECTION
REASON = STAGES_4_TO_9_NOT_COMPLETED_AND_OOF_DESIGN_NOT_FROZEN

Generated (UTC): see git commit time. Branch: `cursor/per-stat-compact-pmf-v1`.

## What was run (the aborted modeling)

Commands:
- `python3 scripts/run_oof_pipeline.py` (feature-only chronological OOF pipeline; Stages 10–16 attempt).

Files created (now stashed + quarantined, NOT committed to the active branch):
- `src/wnba_props_model/oof/__init__.py`
- `src/wnba_props_model/oof/pmf_pipeline.py`
- `scripts/run_oof_pipeline.py`

Model candidates fitted (per chronological expanding fold, per direct stat):
- `baseline_rate` (shrunk rolling-mean → NB2)
- `poisson_hgb` (HGB Poisson mean → Poisson PMF)
- `nb2_hgb` (HGB Poisson mean → NB2 PMF)
- `minutes_mixture` (per-minute rate × expected minutes → NB2)
- plus a participation classifier and minutes quantile models.

Date ranges used: expanding-window chronological folds over the recovered 2023–2026 seasons
(first block warm-up; later blocks predicted by earlier data).

## Why this work is INVALID for model selection

1. **OOF design was not frozen, and dispersion was changed AFTER observing OOF behavior.**
   The first run raised `poisson_or_nbinom_pmf: omitted tail mass ... exceeds tolerance`. In
   response I changed the dispersion estimator from unconditional method-of-moments to a
   conditional-residual estimator and added adaptive support-cap growth. Adjusting the model
   after inspecting OOF output contaminates any subsequent model-selection claim.
   → validation outcomes WERE inspected before the design was frozen.

2. **No physical feature/target separation.** The estimator consumed the recovered wide matrix,
   which still contains `actual_*`, `actual_minutes`, and `did_play`. Column-list exclusion is
   not a sufficient leakage guard; a physically separate feature-only matrix + target-only
   matrix + runtime estimator-input guard (strengthened Stage 9) were not built.

3. **The run did not complete** and produced no persisted OOF ledger, PMFs, metrics, selection,
   or freeze artifacts. `data/oof` was empty and no modeling audits were written.

No model, metric, PMF, selection, or freeze from this work may be called OOF, selected,
validated, frozen, production-ready, or bettor-ready. No metric will be promoted or reused.

## Ad hoc changes made after inspecting results
- Replaced `_dispersion_r` (unconditional) with `_residual_dispersion_r` (conditional residuals).
- Added `_count_pmf` adaptive support-cap growth.
- Added `_usable_cols` / `_fit_predict` all-NaN-column dropping.
All of the above were made in reaction to observed OOF errors → design not frozen.

## Preservation / quarantine

- Stash id (contains the full aborted implementation): `13c7195b6c7a2be3860f523cf1440e1d1f437180`
  (message `ABORTED_PREMATURE_MODELING_BEFORE_STAGES_4_TO_9`).
- Quarantined copy (gitignored): `artifacts/quarantine/premature_modeling_20260730T014245Z/`
  containing `oof_module/`, `run_oof_pipeline.py`, `data_oof/` (empty).

## Factual note on stage state (accuracy)

To avoid misrepresentation: Stages 4–8 were previously implemented, gated, committed and pushed
on this branch (commits `26c8417` Stage 4, `036af11` Stage 5, `0fdd22e` Stage 6, `60edc56`
Stage 7, `6e5cfd2` Stage 8), and Stage 9 (feature registry + leakage audit) at `d8b46b4`. The
data foundation therefore currently holds **1,159 raw responses / 256,160 atomic rows / 50,753
settled decision pairs**, not the Stage-3 state (382 files / 45,209 rows). The required REASON
string above is reproduced verbatim as instructed; the substantive, accurate reasons the
modeling is invalid are (1) post-hoc dispersion change with an unfrozen OOF design and (2) the
strengthened Stage-9 physical feature/target separation + estimator guard were not built. A
hard git revert to the Stage-3 commit was **not** performed because it would discard valid,
pushed Stages 4–9 work and conflict with the preserved raw cache the directive forbids deleting.
The active branch was NOT modified except to stash the uncommitted modeling code and gitignore
the quarantine directory.
