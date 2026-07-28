# Per-Prop Feature-Selection / Ablation Study

Leakage-safe, nested rolling-origin CV feature-selection and group-ablation study
for all seven WNBA player-prop targets. Harness:
`src/wnba_props_model/ablation/feature_ablation.py` (driver
`scripts/run_feature_ablation.py`). Per-prop artifacts:
`artifacts/feature_ablation/FEATURE_ABLATION_<PROP>.json`; combined roll-up:
`artifacts/feature_ablation/FEATURE_ABLATION_SUMMARY.json`.

## Method (identical rigor across all seven props)

- **Candidate features** — a broad union of strictly-lagged (as-of strictly
  before tip) signals organised into seven groups: `player_pbp_rate`,
  `player_box_form`, `opponent_defense`, `pace_env`, `schedule`, `role`,
  `dispersion`. Forward-only (tonight's availability / lineup / game script) and
  market-derived columns are excluded so the study measures **non-market** signal.
- **Nested CV** — feature selection (greedy-forward on inner folds + L1 consensus)
  and any tuning happen inside the inner folds of each outer fold and never see
  the outer evaluation block (`assert_nested_cv_integrity`).
- **Importance** — permutation (gradient-boosted) + L1 path.
- **Ablation** — only-one-group and leave-one-group-out.
- **Evaluation** — outer-OOF. Market-evaluable props (`pts`, `reb`, `ast`,
  `fg3m`) are scored vs the frozen no-vig market with LogLoss / Brier / AUC / ECE
  + paired date-cluster bootstrap and **Holm** correction across the four market
  props. Outcome-only props (`stl`, `blk`, `tov`) — no sportsbook offers these,
  so there are no quotes — use proper count scores (Poisson deviance / PMF
  log-score / CRPS) against a naive P0 baseline; market fields are `null`.

## Data provenance (cloud run)

`BDL_API_KEY` was available, so every BDL-derived input was regenerated on the VM
(none committed — all gitignored / license-restricted): PBP history → parsed
per-player counts (stl/blk/tov exact-match ≥ 0.999 vs the official box) →
`wnba_pbp_opportunity_features`; box `wnba_player_game_stats`; and
`wnba_stlblktov_labels`.

The license-restricted wide feature file
(`...recovered_v2_20260725.parquet`) is not present on the cloud VM. Rather than
drop `player_box_form`, it was **cleanly rebuilt from the BDL box via the standard
feature pipeline** (`scripts/build_features.py`). All seven groups are therefore
**included**. The only columns absent versus the original recovered_v2 build are
the license-restricted learned embeddings (`player_embed_*`, `player_svd_dim_*`,
`team_embed_*`); this is recorded verbatim in every regenerated artifact as:

```
"player_box_form_group": "INCLUDED_rebuilt_from_BDL_standard_feature_pipeline_no_recovered_v2_embeddings"
```

`ast` and `fg3m` were produced earlier from the original recovered_v2 wide file
(with embeddings) and are unchanged here.

## Market-evaluable props (vs. no-vig market)

Positive ΔLogLoss / negative ΔAUC = **worse than the market**. Holm p is adjusted
across the four market props. Lower LogLoss and higher AUC are better.

| Prop | OOF n / dates | Best only-one group (AUC) | Selected LL | Market LL | Selected AUC | Market AUC | ΔLogLoss | ΔAUC | Holm p (LL) | Closes gap | Beats market? | Verdict |
|------|---------------|---------------------------|-------------|-----------|--------------|------------|----------|------|-------------|------------|---------------|---------|
| pts  | 658 / 43 | player_box_form (0.531) | 0.7303 | 0.6920 | 0.5409 | 0.5269 | +0.0382 | +0.0140 | 1.00 | 49.2% | No | CLOSES_GAP |
| reb  | 577 / 43 | pace_env (0.523) | 0.7618 | 0.6815 | 0.4948 | 0.5932 | +0.0803 | −0.0984 | 1.00 | 20.6% | No | INFORMATION_GAP |
| ast  | 414 / 43 | pace_env (0.555) | 0.7968 | 0.6903 | 0.4972 | 0.5403 | +0.1065 | −0.0431 | 1.00 | 5.2% | No | INFORMATION_GAP |
| fg3m | 391 / 43 | pace_env (0.548) | 0.8203 | 0.6600 | 0.4896 | 0.6368 | +0.1604 | −0.1472 | 1.00 | −45.5% | No | INFORMATION_GAP |

**No market prop beats the market.** Every Holm-adjusted p-value is 1.00; the
selected non-market model is worse than the market on LogLoss for all four. `pts`
is the only prop whose selected AUC (0.541) edges above the market's (0.527) and
whose selected model recovers ~49% of the P0→market LogLoss gap, but the
difference is not remotely significant. `fg3m` is the widest information gap — the
market's AUC (0.637) is far beyond anything the non-market features reach (≤0.55
only-one-group; 0.490 selected).

## Outcome-only props (no market — proper count scores)

No book offers stl/blk/tov, so there is nothing to beat; the question is only
whether the selected feature set improves calibrated count prediction over a naive
P0 baseline. Lower Poisson deviance / PMF log-score / CRPS is better; negative
Selected−P0 deviance delta = features help.

| Prop | OOF n / dates | Best only-one group (deviance) | P0 deviance | Selected deviance | Sel−P0 Δ | Selected PMF log-score | Selected CRPS | Verdict |
|------|---------------|--------------------------------|-------------|-------------------|----------|------------------------|---------------|---------|
| stl  | 3262 / 57 | role (1.194) | 1.2190 | 1.1961 | −0.0229 | 1.1513 | 0.4612 | FEATURES_HELP |
| blk  | 3262 / 57 | player_pbp_rate (0.985) | 0.9672 | 0.9897 | +0.0224 | 0.7977 | 0.2888 | NO_GAIN |
| tov  | 3262 / 57 | player_box_form (1.183) | 1.2302 | 1.1958 | −0.0344 | 1.4606 | 0.6246 | FEATURES_HELP |

The count props have far more rows (every player who played, not just those with
quotes) and clear 300-row / 30-date sufficiency. Non-market features give a small
but real deviance improvement for `stl` and `tov`; for `blk` the naive P0 baseline
is already as good as the selected set (features do not help). These verdicts are
about model quality only — there is no market comparison to make.

## Bottom line

- **All four market props show a genuine information gap: nothing beats the
  market** (Holm p = 1.00 everywhere, selected LogLoss worse than market for all
  four). This matches the previously-measured `ast`/`fg3m` result.
- The most consistently useful *non-market* group is `pace_env` (best only-one-
  group for reb/ast/fg3m), with `player_box_form` best for pts and tov.
- For the outcome-only counts, features modestly help stl and tov and do not help
  blk. All three are data-sufficient.
