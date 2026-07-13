# Hotfix Baseline Documentation

## Commit 734e6d79
Starting point for reb/ast underprediction hotfix.

## GitHub Actions Run 29210161447 (fcb29d97)
- Overall workflow status: success
- Prediction generation: SUCCESS
- Edge board generation: SUCCESS
- FTP/gh-pages deployment: SUCCESS
- **SHAP feature importance step: FAILED (exit code 1)**
  - This step generates model explainability data and is non-critical
  - Prediction and edge board outputs are unaffected by SHAP failure
  - Root cause unknown — likely missing model artifact or sklearn/shap version mismatch
  - Action required: investigate SHAP failure independently; do not block prediction pipeline on it

## Hotfix Status
STRUCTURAL_HOTFIX_DEPLOYED_VALIDATION_PENDING

Deployed commit: cf08a567
Validation pending:
- Full round-trip PMF integrity (combo mean error after JSON serialization)
- Historical rebound/reb_ast OOF calibration
- PRA trivariate dependence validation

## Hotfix Commits (734e6d79 → cf08a567)

| Commit | Description |
|--------|-------------|
| fcb29d97 | hotfix: replace C.6 absolute-count ratio with per-minute rate for reb/ast |
| c5b86de9 | hotfix v2: fix dead C.6 reb/ast code, enforce combo mean integrity, hard-error on dup PMFs |
| 12e3ac3a | hotfix v3: IPF marginal-preserving combo + fix C.6 dead code + hard duplicate errors |
| cf08a567 | hotfix v3b: fix pts_reb_ast mean drift — use sequential bivariate IPF instead of trivariate MC |

## Validation Items (Pending)

1. **Truncation diagnostics** — per-combo tail mass, mean error, round-trip error
2. **Adaptive cap** — replace fixed DOMAIN_MAX cap with tail_mass_tol=1e-10 adaptive cap
3. **Integrity gates** — in-code gates marking rows combo_suppressed=True on failure
4. **IPF fallback labeling** — independence_fallback explicitly labeled (not silently passing as correlated)
5. **IPF tolerance** — tightened to 1e-12 to drive pre-truncation combo_mean_error < 1e-8
6. **Round-trip validation** — verify_combo_roundtrip.py script
7. **OOF calibration** — historical reb/reb_ast Brier score, ECE, bias
8. **PRA limitation** — sequential bivariate limitation documented in code
9. **SHAP failure** — documented here; non-blocking
