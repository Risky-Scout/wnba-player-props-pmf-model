# Per-Prop Feature Ablation (Development-Only, Surrogate)

- Classification: **surrogate feature-subset ablation**
- Protocol: development-only expanding chronological folds; no holdout scored
- Conditional-on-played (DNP excluded): **True**
- Reserved (unscored) tail dates: **25**
- Surrogate: HGB Poisson mean -> NB PMF (MoM dispersion); random_state=0
- Metric: full-count NLL (primary), RPS (tie-break). AUC/accuracy not used.

> This is development-only exploration. It is NOT proof, certification, or promotion
> evidence and is NOT promotion-eligible.

| Prop | Winner | G0 NLL | Winner NLL | Delta vs G0 | Improves G0? |
|---|---|--:|--:|--:|---|
| pts | G0 | 3.0076 | 3.0076 | +0.0000 | False |
| reb | G0 | 2.1652 | 2.1652 | +0.0000 | False |
| ast | G0 | 1.7342 | 1.7342 | +0.0000 | False |
| fg3m | G0 | 1.1610 | 1.1610 | +0.0000 | False |
| stl | G0 | 1.1419 | 1.1419 | +0.0000 | False |
| blk | G0 | 0.7992 | 0.7992 | +0.0000 | False |
| turnover | G0 | 1.4484 | 1.4484 | +0.0000 | False |

## Conclusion

None of the tested feature subsets improved the tested surrogate model under the specified development-only protocol. New tracking data is a high-priority next source of information, but this experiment does not prove it is the only possible source of improvement.

