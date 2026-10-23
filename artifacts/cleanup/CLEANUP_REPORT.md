# Repository Cleanup Report

**Date:** 2026-07-28
**Branch:** `cursor/clv-backtest-actionability` (based on `main`)
**Scope:** Delete provably-unused files (dead scripts, dead `src` modules, stale
artifacts, defunct workflows, junk). Evidence-driven, conservative — bias toward
correctness over aggressiveness.

## Summary

| Category           | Files | Dirs | Bytes reclaimed |
|--------------------|------:|-----:|----------------:|
| Dead scripts       |    37 |    – |         264,473 |
| Dead `src` modules |     1 |    – |           5,256 |
| Stale artifacts    |    32 |    7 |         162,112 |
| Defunct workflows  |     0 |    – |               0 |
| Junk (tracked)     |     0 |    – |               0 |
| **Total**          | **70**|      |     **431,841** |

~422 KB reclaimed across 70 files. Full per-file inventory with non-reference
evidence is in `DELETION_MANIFEST.json` (same directory).

## Verification

- **pytest baseline (before):** `2129 passed, 2 skipped` via
  `python -m pytest tests/ -q --ignore=tests/test_elite_projection_gate.py`
  (`PYTHONPATH=$(pwd)/src`).
- **pytest after deletions:** `2129 passed, 2 skipped` — identical to baseline,
  zero regressions.
- **ruff:** `python -m ruff check src scripts` run for sanity (CI does not gate
  on ruff; used only to surface broken imports).
- **workflow safety:** every deleted script filename was grepped across
  `.github/workflows/` and returned zero matches, so no live cron references a
  deleted entrypoint.

## Method

A ripgrep reference map was built for every candidate, searching **both** the
`*.py` filename (path invocation) and the module stem (`\bstem\b`, catches
`import`/`from ... import`) across: `.github/workflows`, `tests`, `src`,
`scripts` (excluding self), `tools`, `config`, `docs`, `README.md`, `Makefile`,
`pyproject.toml`. A file was a deletion candidate **only** if it had zero
references there and was not a `pyproject` console script. Artifact directories
were additionally cross-checked for live consumers (workflows/tests/src/config)
via explicit `artifacts/<dir>` path greps.

## Categories

### Dead scripts (37)
One-off directive/phase/audit scripts with no inbound reference: Path 2/3
forecast gates, PBP/opportunity/tracking investigations, data-recovery
preservation steps, and diagnostic sprints. Their only outputs were artifacts
nothing consumes.

### Dead `src` module (1)
`evaluation/pmf_calibration.py` — zero `\bpmf_calibration\b` references anywhere
(distinct from the live `evaluation/pmf_recalibration.py` and
`models/binary_probability_calibration.py`, both retained).

### Stale artifacts (7 dirs / 32 files)
`hotfix_baseline_734e6d79`, `mission_state`, `p2`, `publish_ready`,
`pure_model_completion`, `tracking`, `candidate_freeze` — past-investigation
report sets read by no code/test and published by no workflow. Config substring
matches (`p2`, `tracking`) were confirmed to be false positives (no
`artifacts/<dir>` paths). All other artifact dirs (`models`, `audits`, `live`,
`hyperparams`, `p1`, `p3`, `pure_supremacy`, `explainability`, `edge_board`,
`feature_ablation`, `foundation_lock`, `market_feature_proof`, `opportunity_v2`,
`probability_contract`, `availability`, `data_bootstrap`, `hotfix_diagnostics`,
`path_a`, `path_b`) are referenced by live consumers and were retained.

### Defunct workflows (0)
No workflow references only deleted/missing entrypoints. All 29 workflows drive
at least some existing scripts, so none were removed. See risk note below.

### Junk (0)
No tracked `.DS_Store`, `*.pyc`, `__pycache__`, `*.bak`, `*.tmp`, `*.swp`, or
backup files. (Untracked, gitignored `__pycache__` dirs were left as-is.)

## Risk notes / deliberately KEPT despite suspicion

- **`scripts/path_b_live_scan.py`** — no workflow/test/script reference, but it
  is a documented current **Path B deliverable** (listed in
  `artifacts/PATH_A_B_IMPLEMENTATION_REPORT.md`) added in the same recent commit
  series as the Path B acceptance gate. KEPT.
- **`.github/workflows/challenger_train.yml`** — references the missing script
  `scripts/compare_champion_challenger.py` (a **pre-existing** dangling step, not
  created by this cleanup). Because the workflow also drives many live scripts,
  it is a live cron with one broken reference rather than a wholly-defunct
  workflow. KEPT and flagged — recommend fixing/removing that step separately.
- **`artifacts/hotfix_diagnostics/`, `artifacts/calibration_monitor/`** — looked
  stale but are referenced by production code (`pipeline/predict.py`) /
  orchestrator (`run_daily_pipeline.py`). KEPT.
- Several other one-off scripts (`build_critical_path_status.py`,
  `build_historical_review.py`, `run_daily_pipeline.py`, etc.) are not in any
  workflow/test but ARE referenced by docs or other scripts, so they did not
  meet the strict zero-reference bar. KEPT.
