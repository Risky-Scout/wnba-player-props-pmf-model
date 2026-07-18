# P2 — Production Edge invocation & stale-artifact audit

## Successful publisher: pregame_initial (run 29613189392)
- command: `build_edge_report.py --pmfs deliveries/tonight/full_pmfs_wide.parquet --raw-props … --out-dir deliveries/tonight --slate-manifest deliveries/tonight/slate_manifest.json --game-date 2026-07-17 --odds-api-props … --edge-threshold 0.0 --source-policy odds_api_then_bdl` (no `--require-venn-abers`: manifest declares role-aware isotonic)
- exit code: 0 (step "Build edge report (BLOCKING)", continue-on-error: false)
- edge-report status: `market_status=SUCCESS_WITH_MARKETS`
- publishable-edge rows: **274 at |edge| >= 0.00%** (274 standard, 0 high-adversity) of 1476 PMF props
- artifact/release: `release_id=29613189392`, self-contained `latest.json`, committed + FTP-deployed
- source run ID: 29613189392 (triggered by daily 29611816014); source commit: 1e79d0a9
- target game date: 2026-07-17 (resolved via workflow_run_artifact — correct lineage)
- consumed current-run artifact? YES (fresh; not stale). Custom-domain post-deploy verification PASSED.

## Release-blocking incidents
1. **edge-threshold 0.0**: every market prop is published as a "recommendation"; there is no
   validated edge threshold, no side/stat gating, and calibration is role-aware isotonic (not a
   validated betting-probability layer). Per P1, Overs lose (−9.4% ROI) and the model is worse than
   the market on log-loss. The live Edge page therefore shows unvalidated picks.
2. **pregame_final fails every run** at "Apply confirmed lineups / late injury updates (BLOCKING)":
   `apply_injury_updates.py: [FATAL] No feature parquet found — cannot rebuild PMFs`. The final,
   Venn-Abers-locked publish never completes; only the T-1 initial (isotonic, threshold 0) publishes.
3. **daily_pipeline** edge build uses `continue-on-error: true` and omits slate-manifest/threshold/VA
   (preview only), so a failed daily edge build is silently ignored.

## Verdict
No STALE artifact was published (lineage is current), but the live Edge board is an UNVALIDATED,
threshold-0 board and the final-lock publisher is broken. Treated as a release-blocking product-trust
incident; P2 hardens the contract and moves the Edge page to a validated/abstaining state.
