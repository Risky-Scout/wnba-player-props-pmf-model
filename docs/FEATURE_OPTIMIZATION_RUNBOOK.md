# Execution Runbook

## 1. Snapshot the data

Freeze a data version containing only information available before each prediction timestamp. Retain timestamped injuries, expected lineups, model output, exact line, book prices, and the market snapshot used for comparison.

## 2. Train controlled candidates

Use the same estimator, distribution family, minutes marginalization, calibration, random seed policy, and training dates for every candidate.

Start with:

- G0: current global feature contract
- S1: `config/prop_feature_map_candidate_v1.json`
- S2-S5: transforms in `config/feature_ablation_plan_v1.json`

Do not change hyperparameters while attributing a result to features. Run a separate nested search if joint feature/hyperparameter optimization is required.

## 3. Export scored rows

One row per player, prop, exact line, book snapshot, and candidate:

```text
game_date,game_id,player_id,prop,candidate,split,actual,line,
model_prob_over,market_prob_over_no_vig,book,market_timestamp,model_timestamp
```

`selection` dates must precede `test` dates.

## 4. Select

```bash
python scripts/evaluate_market_superiority.py \
  --input scored_candidates.parquet \
  --mode select \
  --selection-split selection \
  --output-dir artifacts/feature_selection
```

Review `selection_metrics.csv`, then freeze `selected_candidates.json`.

## 5. Prove

```bash
python scripts/evaluate_market_superiority.py \
  --input scored_candidates.parquet \
  --mode prove \
  --selected-candidates artifacts/feature_selection/selected_candidates.json \
  --test-split test \
  --bootstrap 5000 \
  --min-rows 300 \
  --min-logloss-delta 0.0025 \
  --min-brier-delta 0.0010 \
  --min-auc-delta 0.0000 \
  --output-dir artifacts/market_feature_proof
```

The nonzero log-loss and Brier margins above mirror a practical, nontrivial superiority standard. Relaxing them to zero tests only literal superiority.

## 6. Promotion

Promote a prop only when `market_superiority_gate == PASS`, full-PMF calibration remains acceptable, and timestamp/coverage audits pass. Props are promoted independently.

## 7. Live shadow confirmation

Freeze the promoted model and repeat on future games without changing features or calibration. Report all failures, coverage exclusions, and sportsbook segmentation.
