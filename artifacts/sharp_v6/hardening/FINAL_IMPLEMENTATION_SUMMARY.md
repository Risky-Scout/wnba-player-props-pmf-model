# V6 One-Production-Model Hardening — Final Implementation Summary

## Baseline verified before edits

| Field | Value |
|---|---|
| Starting SHA (`origin/main` / HEAD) | `987c07fab0315ac026c446ed0ca7d9c9aedca17c` |
| PR #99 merge SHA | `ee35dc7d25aeca153ae4d05335820f3aee8bf1c4` |
| Last successful daily smoke | `30726462645` (success on `987c07fa…`) |
| V3/V4/V5 `PRODUCTION` | `False` |
| V6 `PRODUCTION` | `True` |
| Baseline bundle (immutable) | `artifacts/releases/wnba-pmf-production-v1` |
| Baseline claimed `model_sha256` | `932d6078c50fa296c2e0970163a986cd0ddcd9a712a56d5c571021134feeba15` |
| Baseline defect | MANIFEST hash computed before final pickle rewrite; `load_bundle` previously `pass`-ed mismatches |

## Authoritative production chain (unchanged contract)

```text
predict_slate → frozen bundle → joint PMFs → publish
```

- Entry point: `wnba_props_model.sharp_v6.inference.predict_slate`
- Production bundle: `artifacts/releases/wnba-pmf-production-v1.1`
- Command: `python scripts/run_wnba_pmf.py --bundle-dir artifacts/releases/wnba-pmf-production-v1.1`
- Daily retrain: disabled

## Selected components (frozen)

| Stat | Family | Calibrator |
|---|---|---|
| pts | structural_shooting | identity |
| reb | structural_oreb_dreb | identity |
| ast | minutes_mixture_nb2 | identity |
| fg3m | minutes_mixture_nb2 | identity |
| stl | hurdle_nb2 | identity |
| blk | minutes_mixture_nb2 | identity |
| turnover | hurdle_nb2 | identity |

- Dependence: Gaussian copula (PSD-projected)
- Q1: `FITTED_NESTED_MINUTES`
- First basket: `FITTED_COMPETING_RISK`
- Market superiority: **NOT_PROVEN**

## Hardening delivered

1. Fail-closed bundle integrity (`verify_bundle_integrity`; hash mismatch no longer ignored)
2. Governed feature missingness classifications (no unclassified NaN-fill policy)
3. Date-effective identity audit with typed statuses / quarantine
4. Unified `_core_pmf_delivery` for live + historical paths
5. Production CLI modes (`production|research|validation|offline_fixture`)
6. Unified release matrix + vacuous-pass prevention (`NOT_EVALUABLE`)
7. Proof generated from repository facts
8. Deployment receipt artifact
9. Legacy `daily_pipeline.yml` marked `LEGACY_CONTROL` / `AUTHORITATIVE_PUBLISH=false` with publish gated
10. Reports under `artifacts/sharp_v6/hardening/`

## Withheld markets

- `fantasy_points` — requires operator scoring configuration
- `double_double` / `triple_double` — joint-sim gate

## Remaining risk (honest)

- Full rolling OOF feature ablation not re-run on this branch (frozen set retained)
- Market validation remains `NOT_PROVEN`
- Raw duplicate classification report is partial until next acquisition pull
- Remote deployment receipt requires a successful post-merge smoke with `SMOKE_RESULT=success`
- Sklearn estimator re-pickle is not bit-stable across loads; canonical bundle bytes are integrity-pinned via MANIFEST/SHA256SUMS

## Deliverable index

- `ROOT_CAUSE_CLOSURE_MATRIX.json`
- `FEATURE_REGISTRY.json` / `FEATURE_ABLATION_REPORT.json`
- `SINGLE_MODEL_ARCHITECTURE.md`
- `REPRODUCIBILITY_MANIFEST.json`
- `STATISTICAL_EVALUATION_REPORT.json`
- `RELEASE_MATRIX.json`
- `DEPLOYMENT_RECEIPT.json`
- `DEPENDENCY_REBUILD_GRAPH.json`
- `artifacts/sharp_v6/ONE_PRODUCTION_MODEL_PROOF.json`
