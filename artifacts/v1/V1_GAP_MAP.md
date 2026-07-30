# V1 Gap Map

| Area | Status | Notes |
|------|--------|-------|
| Pricing engine (push-safe, fair odds, margin, yes/no, categorical) | DONE | `pricing/engine.py` + tests |
| Canonical market registry (all Section-2 keys, alternates from base dist, fantasy config) | DONE | `pricing/market_registry.py` |
| Coherent joint generator (shared-latent minutes; identities every sample) | DONE (baseline) | `pricing/joint_generator.py` |
| Direct + combination + alternate + Q1 + event markets priced from one distribution | DONE | fixture run |
| First-basket / first-team / method competing-risk | DONE (needs hazards) | `pricing/first_basket.py` |
| Monotone distributional calibration (no per-line isotonic) | DONE (hooks) | `pricing/calibration.py` |
| BDL client: prop_type, plays non-paginated | DONE | applied on this branch |
| Odds client x-requests-last / atomic-schema rework / full backfill | NOT DONE (Phase-1) | large; lives on cursor/per-stat-compact-pmf-v1; not re-done here |
| Live "today's pricing run" | BLOCKED | needs Phase-1 recovered feature/quote slate (not on this branch) |
| Chronological market evaluation + superiority certification | NOT DONE | separate gate; requires quotes + OOF |
| Market-anchored production track (KL projection + residual) | SCOPED | not run in this pass |
| Immutable RC bundle | DONE | `artifacts/releases/wnba-pricing-pmf-v1.0.0-rc1/` |
