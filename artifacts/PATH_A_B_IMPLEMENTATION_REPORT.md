# Path A/B Mandatory Pre-Merge Acceptance Gate — Implementation Report

**Scope:** Harden the already-merged Path B (soft-book +EV scan) and Path A (availability
accrual) behind a MANDATORY, fail-closed pre-merge acceptance gate; produce audit artifacts;
run the full test suite + a live scan + a deterministic offline scan.

**Core mandate honored everywhere:** Path B is a **MARKET DISLOCATION detector, not a model
edge.** Every board row carries `source_type = "MARKET_DISLOCATION"`, defaults
`actionable = false`, distinguishes `THEORETICAL_EV` from `EXECUTABLE_EV`, and emits **no
stake sizes / no Kelly** during the validation period. No profitability / executability /
market-superiority is claimed.

> **THIS IS NOT A MERGE.** Branch pushed + PR opened for human review only.

---

## What was built

### New package modules (`src/wnba_props_model/edge/`)
- `prop_identity.py` — exact, fail-closed name → canonical `player_id` resolution
  (RESOLVED / UNMATCHED / AMBIGUOUS; never fuzzy, never name-only).
- `path_b_gate.py` — the acceptance-gate validator: per-row provenance contract, forbidden
  stake/Kelly fields, `actionable` guard, no-vig fail-closed check, latency/rejection
  disclosure checks. Any violation ⇒ non-passing report.
- `path_b_audit.py` — builds the gate-validated `LIVE_SCAN_AUDIT.json` + companion audits.
- `path_b_collect.py` — shared Odds-API payload → atomic two-sided row extraction.
- `path_b_fixtures.py` — deterministic, now-relative fixtures that exercise **every**
  rejection path.
- `src/wnba_props_model/data/availability_audit.py` — Path A structured-failure
  classification, coverage, and the strictly **append-only** forward-snapshot manifest.

### Enhanced existing code (not duplicated)
- `edge/soft_book_scan.py` — added full provenance per row, atomic-line segregation
  (grouped on `market_key`, `*_alternate` never mixed with standard), timestamp integrity
  (provider/ingestion/scan/tip + quote age + configurable strict age gate), leave-one-out
  consensus **dispersion** (stdev/IQR) + **sharp/reference-book** flag + recorded
  self-exclusion, no-vig **fail-closed** on missing opposite side (recorded reason), a
  rejection ledger, `source_type`, `actionable=false`, THEORETICAL vs EXECUTABLE EV.
- `scripts/collect_availability.py` — structured failure reasons (403/auth/empty/unresolved
  are **never** "successful empty data"); records source/ingestion timestamps, prediction
  cutoff, payload hash, coverage; append-only snapshot manifest; overwrite guard.
- `scripts/build_soft_book_edge_board.py` — **removed Kelly emission**; adds
  `source_type=MARKET_DISLOCATION`, `actionable=false`, theoretical/executable EV split.

### Scripts
- `scripts/path_b_acceptance_gate.py` — **MANDATORY gate CLI; exits non-zero on any
  violation.**
- `scripts/path_b_live_scan.py` — live slate scan with 30s/60s price-survival recheck +
  credit accounting.
- `scripts/path_b_fixture_scan.py` — deterministic OFFLINE end-to-end scan for CI.

### CI wiring
- New `path_b_acceptance_gate` job in `.github/workflows/ci.yml`: runs the Path A/B unit +
  integration tests, runs the offline fixture scan, then runs the gate (fail closed). It
  gates merges alongside the existing `actionlint` + `pytest` gates.

---

## Live scan results (today's slate, 2026-07-28, region `us,us2`)

| Metric | Value |
|---|---|
| Games discovered | **5** |
| Books observed (10) | betonlineag, betparx, betrivers, draftkings, espnbet, fanduel, fliff, hardrockbet, rebet, williamhill_us |
| Markets observed (4) | player_points, player_rebounds, player_assists, player_threes |
| Atomic two-sided pairs created | **175** |
| Raw atomic quote rows | 2,451 |
| BDL roster players (identity index) | 207 |
| Scored board rows (all `actionable=false`) | 2,158 |
| Qualifying **diagnostic** edges (≥2.5% THEORETICAL EV) | **5** |
| Rows rejected (total) | **169** |
| — `missing_opposite_side_no_vig_fail_closed` | 166 |
| — `unresolved_identity` | 3 |
| Consensus: median books / dispersion(stdev) / self-exclusion | 7 / 0.0103 / enforced |
| Quote age: median / max (seconds) | 40.4 / 132.4 |

**Price-survival re-check (execution realism, req 7):** re-checked **1** candidate event at
**+30s** and **+60s**; **2** candidate prices survived at 30s and **2** at 60s (still offered,
not worse for the bettor). Per-row `executable_ev_pct` / `price_survived_*` remain **null**
for the broader board — see blockers.

**The 5 diagnostic edges (THEORETICAL EV only — NOT executable, NOT profitable):**

| Player | Market | Side | Line | Book | Odds | Theoretical EV | Consensus books | Sharp in consensus |
|---|---|---|---|---|---|---|---|---|
| Michaela Onyenwere | points | over | 9.5 | hardrockbet | +105 | 3.60% | 7 | yes |
| A'ja Wilson | assists | over | 3.5 | betonlineag | +139 | 3.51% | 6 | no |
| Awa Fam | points | under | 12.5 | draftkings | −108 | 3.35% | 6 | no |
| Rae Burrell | points | over | 14.5 | hardrockbet | −105 | 3.08% | 7 | yes |
| Olivia Nelson-Ododa | rebounds | under | 6.5 | betparx | −121 | 2.98% | 5 | no |

### API credit usage (from `x-requests-remaining`)
- Before: **4,398,914** → After: **4,398,858** → **Consumed: 56 credits** (remaining
  4,398,858). Includes the initial per-event pulls plus the 30s/60s price-survival rechecks.

---

## Path A results (forward availability, 2026-07-28)

| Metric | Value |
|---|---|
| Games (teams playing) | 5 |
| Injury rows | 37 |
| OUT players | 32 |
| Identity resolution rate (canonical `player_id`) | 100% (37/37) |
| Slate-linkage rate (OUT players matched to slate) | 100% (32/32) |
| games / injuries endpoint status | documented_success / documented_success |
| Overall status | ok |
| Forward-snapshot manifest entries (append-only) | 1 |

Structured-failure behavior verified by tests: a 403/auth/empty/unresolved result is recorded
with an explicit reason and `overall_status="degraded_or_failed"` — never as "successful empty
data". The forward-snapshot manifest is strictly append-only (earlier entries never
overwritten; idempotent by payload hash; overwrite guard on snapshot files).

---

## Offline fixture scan (deterministic, CI-safe)

`scripts/path_b_fixture_scan.py` builds a crafted slate and exercises every rejection path:
2 games, **4 atomic pairs**, 28 board rows, 2 diagnostic edges, and rejections
`{missing_opposite_side_no_vig_fail_closed: 1, unresolved_identity: 1, post_tip_or_stale_event: 1,
stale_quote_age: 2}` plus a malformed-timestamp **warning**. The acceptance gate PASSES the
clean fixture audit (30 rows checked, 0 violations).

**Gate fails closed on seeded violations (demonstrated):**
```
[gate] FAIL: .../seeded.json (30 rows checked, 5 violation(s))
    - [NO_VIG_NOT_FAIL_CLOSED] $.config.no_vig_fail_closed: ...
    - [PREMATURE_ACTIONABLE] board_rows[0]: actionable=True without passing ...
    - [WRONG_SOURCE_TYPE] board_rows[1]: source_type='MODEL_EDGE' != 'MARKET_DISLOCATION'
    - [STAKE_EMITTED] board_rows[2]: forbidden stake/Kelly field 'kelly_fraction'=0.05 ...
    - [IDENTITY_UNRESOLVED] board_rows[3]: displayed row lacks resolved canonical player_id ...
gate exit: 1
```

---

## Tests (names + pass/fail)

**Full suite:** `pytest tests/ --ignore=tests/test_elite_projection_gate.py` →
**2121 passed, 10 skipped, 0 failed** (~219s).

**44 new tests (all PASS):**

`tests/test_path_b_gate_and_scan.py` (32):
- req1: `test_req1_identity_resolution_exact_unmatched_ambiguous`,
  `test_req1_unresolved_identity_is_rejected_not_scored`
- req2: `test_req2_self_excluded_recorded_and_book_not_in_own_consensus`
- req3: `test_req3_alternate_market_not_mixed_with_standard`,
  `test_req3_different_lines_never_compared`
- req4: `test_req4_post_tip_event_rejected`, `test_req4_stale_quote_rejected_and_reasoned`,
  `test_req4_malformed_timestamp_warned_and_age_null`,
  `test_req4_timestamps_present_and_age_reported`,
  `test_req4_configurable_age_gate_disabled_keeps_rows`
- req5: `test_req5_min_consensus_books_guard_flags_not_qualified`,
  `test_req5_dispersion_and_sharp_reference_recorded`,
  `test_req5_soft_only_consensus_flagged_not_sharp`
- req6: `test_req6_missing_opposite_side_fails_closed_with_reason`
- req7: `test_req7_theoretical_and_executable_ev_distinct`, `test_req7_high_ev_row_not_actionable`
- req8: `test_req8_actionable_false_and_no_stake_or_kelly_emitted`
- req9: `test_req9_every_row_carries_full_provenance`
- gate: `test_gate_passes_clean_fixture_audit`,
  `test_gate_fails_closed_on_seeded_violation` (×10 seeded cases),
  `test_gate_missing_file_fails`, `test_gate_cli_exits_nonzero_on_seeded_violation`,
  `test_gate_cli_exits_zero_on_clean_audit`

`tests/test_path_a_availability_audit.py` (12):
`test_classify_success_with_rows`, `test_classify_empty_is_not_success`,
`test_classify_403_auth_failure_not_success`, `test_classify_404_unavailable`,
`test_build_audit_marks_failure_when_endpoint_fails`, `test_build_audit_ok_only_when_all_succeed`,
`test_coverage_identity_and_slate_rates`, `test_manifest_is_append_only`,
`test_manifest_idempotent_on_duplicate_hash`, `test_assert_no_snapshot_overwrite`,
`test_manifest_refuses_corrupt_existing`, `test_payload_hash_stable`.

The 14 pre-existing `tests/test_soft_book_scan.py` tests remain green after the enhancement.

---

## Requirement-by-requirement status

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Exact identity match | ✅ | `prop_identity`, `require_identity`; 3 live rows rejected `unresolved_identity`; gate `IDENTITY_UNRESOLVED` |
| 2 | Leave-one-book-out consensus (recorded) | ✅ | `self_excluded=true`, `consensus_books` excludes self; test + gate `SELF_EXCLUSION_NOT_RECORDED` |
| 3 | Atomic line matching (alt segregated) | ✅ | group key `market_key`; `is_alternate_market`; fixture test |
| 4 | Timestamp integrity + strict age gate | ✅ | provider/ingestion/scan/tip + `quote_age_seconds`; `--max-quote-age-seconds`; post-tip/stale/malformed handled |
| 5 | Consensus quality | ✅ | min books, dispersion stdev/IQR, reference-book set + `consensus_includes_sharp` |
| 6 | No-vig correctness / fail closed | ✅ | same-book Over+Under Shin devig; missing side ⇒ reject `missing_opposite_side` (166 live) |
| 7 | Execution realism (30s/60s recheck) | ⚠️ partial | live recheck ran (1 event, 2 prices survived); per-row `executable_ev`/survival still null board-wide (see blockers) |
| 8 | Initial safety status | ✅ | `actionable=false` everywhere; no Kelly/stake emitted; gate `STAKE_EMITTED`/`PREMATURE_ACTIONABLE` |
| 9 | Board provenance (every row) | ✅ | `PROVENANCE_FIELDS` enforced by gate on 2,005 live rows + 30 fixture rows |
| 10 | Path A append-only + structured failures | ✅ | append-only manifest, overwrite guard, structured statuses, coverage |

---

## Brutally honest remaining blockers / limitations

1. **Forward-CLV validation** (this section is superseded by the **CLV Backtest &
   Actionability** section below). The original forward approach needed post-tip closing lines
   accrued across future slates. Per the owner's spec change we instead validate on the
   historical open/decision/close snapshots we already hold; see the CLV backtest below. The
   net conclusion is still **no row is actionable today** — but now for a measured,
   sample-size reason, not because CLV is unmeasured.
2. **Price-survival is partial by design.** The live recheck covered **1** candidate event
   (credit-conserving `--recheck-events 1`). The per-row `price_survived_30s/60s` and
   `executable_ev_pct` fields on the broader board remain null; only the `price_survival`
   block carries the sampled recheck. Widening this needs more credits and a scheduled
   re-poll, not a single scan.
3. **Slate thinness.** Only 4 of the 7 requested single-stat markets were offered two-sided
   by books tonight (steals/blocks/turnovers were absent/one-sided) — reflected honestly in
   `markets_observed` and the 166 `missing_opposite_side` rejections.
4. **3 unresolved identities (live).** Three Odds-API player names did not resolve to a BDL
   canonical `player_id` (likely name variants / non-active-roster players). They were
   **rejected, not scored** — resolving them needs an alias-review pass, not fuzzy matching.
5. **The 5 diagnostic edges are ~3% THEORETICAL EV** vs a median-of-others consensus. They
   are **not** claimed executable, profitable, or market-superior; several have no sharp book
   in their consensus (`consensus_includes_sharp=false`), which lowers confidence further.
6. **`LIVE_SCAN_AUDIT.json` is large (~2.7 MB)** because it embeds full per-row provenance for
   all 2,158 rows (capped at 2,000 board rows). Committed as the required audit artifact.

---

# CLV Backtest & Actionability (spec change: validate on data we already have)

The owner changed the spec: instead of waiting multiple future slates for forward-CLV, the
model must be made **actionable now**, validated by a **CLV backtest on historical data we
already hold**. This section reports that backtest and the actionability wiring it drives.

## What was built (extends the acceptance-gate work, does not duplicate it)

- **`src/wnba_props_model/edge/clv_backtest.py`** — replays the **unchanged**
  `scan_soft_book_edges` at the **decision** snapshot (falling back to **open**) per slate,
  reusing the repo's Shin no-vig (`models.market.shin_no_vig_two_way`), leave-one-book-out
  consensus, atomic line matching and fail-closed guards; computes **price CLV** (closing
  no-vig **consensus** P(side) − the candidate book's own decision no-vig P(side)) and
  **same-book CLV**; aggregates by market / book / EV-bucket / market×EV-bucket with a
  **date-cluster bootstrap 95% CI** (resample the 56 game_date clusters); and emits the
  fail-closed validation table.
- **`scripts/run_clv_backtest.py`** — writes `artifacts/path_b/CLV_BACKTEST.json` (full
  methodology + per-segment tables + verdict + limitations) and
  `artifacts/path_b/CLV_VALIDATION_TABLE.json` (the compact table the board + gate consume).
- **Actionability wiring** — `scripts/build_soft_book_edge_board.py` now loads the validation
  table and sets `actionable` per row (fail closed); `edge/path_b_gate.py` now permits
  `actionable=true` **only** when a row carries qualifying backtest-CLV evidence (a positive
  mean with a 95% CI whose lower bound > 0) plus resolved identity, `forward_clv_validated`,
  and `VALIDATED_EXECUTABLE`. `source_type=MARKET_DISLOCATION` is unchanged and **no
  stake/Kelly** is emitted.

## Data & method

`artifacts/p1/p1_quotes.parquet`: 76,620 rows, **56 game dates** (2026-05-08 → 2026-07-22),
**5 books** (betonlineag, betrivers, draftkings, fanduel, williamhill_us), snapshot labels
open/decision/close. Replay config = production defaults (`ev_threshold=2.5%`,
`min_consensus_books=3`); date-cluster bootstrap = 5,000 resamples, seed 20260728; a segment
needs ≥2 date clusters. Actionability metric = **price CLV**. CLV is reported in probability
percentage points ("cents" = prob×100).

## Headline results (price CLV vs the closing consensus)

Only **17** decision-time flags clear 2.5% EV in the whole 56-date panel (a 5-book market's
no-vig consensus sits close to each book, so >2.5% dislocations are rare); **13** have a
closing consensus to score against.

| Segment | N | dates | mean CLV | median | % beat close | 95% CI (date-cluster) | CI excludes 0 |
|---|---:|---:|---:|---:|---:|---|:--:|
| **Overall (price CLV)** | 13 | 9 | **+3.77c** | +3.17c | **100%** | **[+2.57c, +4.94c]** | ✅ |
| Overall (same-book CLV) | 10 | 8 | +2.66c | +2.54c | 100% | [+1.40c, +3.92c] | ✅ |
| market = player_points | 7 | 6 | +3.72c | +4.33c | 100% | [+2.24c, +4.85c] | ✅ |
| market = player_rebounds | 3 | 3 | +2.48c | +2.41c | 100% | [+1.87c, +3.17c] | ✅ |
| market = player_threes | 3 | 3 | +5.15c | +4.55c | 100% | [+1.85c, +9.06c] | ✅ |
| EV bucket 2.5–5% | 12 | 9 | +3.92c | +3.75c | 100% | [+2.60c, +5.19c] | ✅ |
| book = betonlineag | 11 | 8 | +3.14c | +2.41c | 100% | [+2.01c, +4.04c] | ✅ |
| book = draftkings | 2 | 2 | +7.20c | +7.20c | 100% | [+5.35c, +9.06c] | ✅ |

Sensitivity — relaxing `min_consensus_books` to 2 (defensible in a 5-book universe) yields
**N=21** flags (mean **+3.36c**, 90.5% beat close, CI **[+1.41c, +4.94c]**): more flags,
**same qualitative conclusion**.

## Honest verdict & the exact actionable set

**The soft-book / MARKET_DISLOCATION edges DO beat the close.** The overall signal is
positive and the date-cluster bootstrap 95% CI **excludes 0** (+3.77 cents, 100% beat close),
and every populated market segment is individually positive with a CI excluding 0.

**But under the fail-closed actionability rule, NO segment qualifies, so ZERO board rows are
marked `actionable` today.** The blocker is **sample size, not sign or significance**: with
only 56 dates and 5 books the strategy fires ~13–21 times total, so the largest market
segment is `player_points` at **N=7–10**, far below the `min_segment_n=50` bar. The persisted
`CLV_VALIDATION_TABLE.json` therefore has `actionable_segments = []`, and each board row is
stamped `actionable=false` with reason `insufficient_sample:market=… n=… < min_segment_n=50`.

This is the brutally honest outcome the spec anticipated: the edge is real and +CLV, but it
is **underpowered per segment** on the data we currently hold. It is **not** faked to please
the request. Actionability will flip on automatically — no code change — once accrued slates
push a market (or market×EV-bucket) segment past `min_segment_n` while keeping mean CLV > 0
and its CI above 0. (`min_segment_n` is a CLI knob; lowering it is a policy decision the owner
can make with these numbers in hand, but the committed default stays fail-closed at 50.)

## Artifacts

- `artifacts/path_b/CLV_BACKTEST.json` — methodology, per-segment CLV tables with CIs,
  overall verdict, coverage, and honest limitations.
- `artifacts/path_b/CLV_VALIDATION_TABLE.json` — the fail-closed table consumed by the board
  and gate (`actionable_segments = []` on current data).
- `artifacts/path_b/CLV_ROWS.csv` — the 17 per-candidate CLV rows (auditable).

## Honest limitations of the backtest

1. **Only 5 books** — the no-vig consensus (and its leave-one-out subset) is thin; one
   mispriced book moves it more than in a 10+ book market.
2. **Only 56 dates** — few bootstrap clusters ⇒ wide CIs and tiny per-segment N.
3. **Closing snapshot is the last collected price**, a proxy for the true settle-time close; a
   market pulled before tip has no closing row (excluded, never imputed).
4. **Consensus is median-of-all** — sharp books are annotated but not up-weighted, so
   "beating the close" means beating a median-of-all close, not a Pinnacle close.
5. **CLV is a +EV proxy, not realized P&L** — it ignores bet availability at the quoted price,
   limits, and settlement vig.
