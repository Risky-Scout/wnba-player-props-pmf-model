# Per-Prop Feature Optimization

## What this adds

Historically every per-stat model trained on the **same global** `MODEL_FEATURES` set
(`feature_contract.py`). This adds the ability to train each prop (`pts, reb, ast, fg3m,
stl, blk, turnover`) on its **own optimal feature subset**, selected leakage-safely.

- `scripts/select_prop_features.py` — ranks features per stat by permutation importance
  aggregated over **expanding chronological folds** (train earlier → score later), keeps
  the stably-useful features, and writes `config/prop_feature_map.json` +
  `artifacts/feature_selection/prop_feature_importance.csv`. Selection touches **only**
  pre-holdout dev folds; the final holdout is never used.
- `training.stat_feature_subset(...)` — when `cfg["prop_feature_map"]` is present, each
  stat model trains on its subset. Inference needs **no change**: each model stores its
  trained columns (`_usable_cols`) and reindexes to them at predict time.
- `.github/workflows/prop_feature_optimization.yml` — dispatchable: rebuild features → run
  selection → publish the map + importance report as artifacts.

## Safety / backward compatibility

- **OFF by default.** With no `prop_feature_map` in config, `stat_feature_subset` returns
  the training matrix unchanged — byte-for-byte the current global-feature behavior. The
  live daily pipeline ships no map, so it is unaffected. (Tests:
  `tests/test_prop_feature_selection.py`.)
- A per-stat subset below `prop_feature_min_cols` (default 8) falls back to the full set,
  so a map can only help, never starve a model.

## How a change reaches the live pages (gated)

The feature map alone changes nothing live. To promote it:

1. Run `prop_feature_optimization.yml` → get `config/prop_feature_map.json`.
2. Pass the map as `cfg["prop_feature_map"]` to a **challenger** retrain (reuse the
   challenger training + OOF path).
3. Run the existing prequential forecast gate (`scripts/p3_*`) on the challenger OOF.
4. **Promote only if** proper scores (CRPS/log score vs baseline) and calibration improve
   and the gates pass — then rebuild the champion package and republish.

## Honest expectation

Gradient-boosted trees already down-weight irrelevant features, so per-prop selection
typically yields **modest** calibration / proper-score gains rather than a large signal
unlock, and it **cannot** create information the model (or the market) does not already
have. It is worth doing for forecast quality; it is unlikely to establish a market-beating
betting edge on public data. Promotion is governed by the gate, not by this document.
