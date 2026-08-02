# Single-Model Architecture (V6)

One end-to-end forecasting contract — not alternative production models.

```
Point-in-time source snapshots
  → Canonical identities (date-effective)
  → Frozen feature contract + governed missingness
  → Participation → Minutes → Shared environment
  → Direct-stat heads (frozen families per stat)
  → Gaussian-copula dependence + joint sims
  → Full-game / combo / Q1 / first-basket markets
  → Explicit calibration → release matrix → publish
```

- Inference: `wnba_props_model.sharp_v6.inference.predict_slate`
- Baseline bundle: `artifacts/releases/wnba-pmf-production-v1` (immutable)
- Candidate bundle: `artifacts/releases/wnba-pmf-production-v1.1`
- Internal components (participation, minutes, env, stats, dependence, Q1, FB) are parts of one system, not competing production models.
- V3/V4/V5 remain RESEARCH_ONLY / PRODUCTION=False.
- Market odds are external evaluation inputs only.
- Market superiority: NOT_PROVEN.
