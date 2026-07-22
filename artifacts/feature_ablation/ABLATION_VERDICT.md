# Per-Prop Feature Ablation Verdict

Leakage-safe expanding chronological folds; HGB Poisson mean -> NB PMF; identical training per
candidate so differences isolate features. Metric: full-count NLL (lower better).

| Prop | Winner | G0 NLL | Best subset NLL (Δ) | Verdict |
|---|---|--:|--:|---|
| pts | G0 | 3.0076 | S1 3.0199 (+0.0123) | no subset improves |
| reb | G0 | 2.1652 | S1 2.2607 (+0.0955) | no subset improves |
| ast | G0 | 1.7342 | S1 1.7450 (+0.0108) | no subset improves |
| fg3m | G0 | 1.1610 | S3 1.1628 (+0.0018) | no subset improves |
| stl | G0 | 1.1419 | S1 1.1465 (+0.0046) | no subset improves |
| blk | G0 | 0.7992 | S3 0.8445 (+0.0453) | no subset improves |
| turnover | G0 | 1.4484 | S3 1.4503 (+0.0019) | no subset improves |

## Conclusion

For every prop the **full global feature set (G0) is optimal** among the tested candidates; every
stat-specific subset (S1-S5) increased NLL. Feature *reduction/selection does not improve* these
gradient-boosted PMFs (they already down-weight irrelevant inputs). No per-prop subset is promoted;
the champion feature set is unchanged. Improving discrimination requires **new information**
(player tracking), not subsetting existing features.
