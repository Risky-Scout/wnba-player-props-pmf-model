# Foundation Lock v1

The Foundation Lock makes every currently completed foundational component **reproducible,
hash-pinned, regression-tested, fail-closed, explicitly classified by what it does and does
not prove, and required by CI**. It is a lockfile-style contract: a locked artifact cannot
change without an explicit manifest update, and the `foundation-lock` CI job fails on any
drift.

> Scope discipline: Foundation Lock v1 changes **no** champion prediction, delivered
> probability, recommendation policy, promotion threshold, or live behavior. It does not
> begin Phase 0. **No locked component is promotion-eligible; none proves a market edge.**

## Files

- `config/foundation_lock_v1.json` - the manifest (source commit, per-component status,
  paths + SHA-256, invariants, required tests, required CI job, evidence, limitations,
  promotion-eligibility).
- `scripts/verify_foundation_lock.py` - verifier / updater / report generator.
- `docs/FOUNDATION_LOCK.md` - this document.
- `artifacts/foundation_lock/FOUNDATION_LOCK_REPORT.md` - generated classification report.

## Commands

```bash
# Verify the working tree against the manifest (this is what CI runs). Exit 1 on any drift.
python scripts/verify_foundation_lock.py

# Regenerate the human-readable report.
python scripts/verify_foundation_lock.py --write-report

# Maintainer-only: recompute in-repo hashes after an INTENTIONAL, reviewed change.
# (Never run in CI. This is the explicit "update the lockfile" action.)
python scripts/verify_foundation_lock.py --update
```

The verifier FAILS on: a missing in-repo path; a hash mismatch (a changed artifact without an
explicit manifest update); a missing required test; a manifest/schema mismatch; or a
component labeled promotion-eligible while its limitations prohibit that status.

Data artifacts declared `availability: data_artifact_untracked` (the git-ignored feature
parquet and the not-yet-landed tracking parquets) are verified when present and reported as
**DEFERRED** (explicitly, never silently) when absent from the checkout. Every in-repo locked
artifact and required test is always checked and can never be silently skipped.

## Locked components (what each does and does NOT prove)

| # | Component | Status | Proves | Does NOT prove |
|---|---|---|---|---|
| 1 | Market-superiority evaluator | `locked` | The gate mechanics (isolation, clustered bootstrap, signs, Holm, fail-closed) are correct and deterministic. | Any real market edge. The self-test is synthetic. |
| 2 | G0 baseline proof | `locked` | The current model FAILS vs the market on every quoted prop (honest baseline). | Nothing positive. Pre-Phase-0; quote identity / push-safety / parity not fixed. |
| 3 | Signed-CLV gate | `locked` | CLV is signed, directional, fail-closed; no nonnegative fallback. | Positive CLV (requires post-game closing quotes to evaluate). |
| 4 | Feature-ablation maps | `locked` | Resolved maps regenerate deterministically; G0 == full contract; no forbidden features. | Any performance claim. |
| 5 | Tracking/hustle collector | `not_landed` | The collector code + output schema contract are correct (mocked). | Anything about tracking data or features - data is NOT landed, features NOT built. |
| 6 | Feature-matrix snapshot | `locked` | The current matrix identity/shape is pinned; drift is detectable. | Model performance. |
| 7 | Prop-ablation runner + verdict | `exploratory_locked` | No tested feature subset improved a **surrogate** model under a **development-only** protocol. | That tracking is the only improvement source; not proof/certification/promotion evidence. |

## Required branch-protection checks

The following checks should be **required** on `main` via repository settings:

```text
lint-workflows
unit-tests
foundation-lock
```

`lint-workflows` and `unit-tests` are provided by `.github/workflows/ci.yml`; `foundation-lock`
is provided by `.github/workflows/foundation_lock.yml`.

**Manual action required (agent cannot change branch protection):** an admin must open
GitHub -> Settings -> Branches -> Branch protection rules for `main` ->
"Require status checks to pass before merging", and add `lint-workflows`, `unit-tests`, and
`foundation-lock` to the required set. Cloud agents do not have permission to modify branch
protection, so this step must be performed manually.
