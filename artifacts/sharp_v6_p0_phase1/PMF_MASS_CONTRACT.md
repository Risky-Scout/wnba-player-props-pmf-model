# V6 PMF Mass Contract (P0 Phase 1)

Authoritative module: `wnba_props_model.sharp_v6.distribution`.

## Identity

For any materialized support `0..K`:

```
sum(atoms[0..K]) + overflow = 1    within 1e-10
```

where `overflow = P(Y > K)`.

Stored atoms are **never** renormalized independently of overflow.

## Minutes mixture

```
P(Y=y|X) = sum_m P(M=m|X) * P(Y=y|M=m,X)
```

- Mixture weights are validated (finite, nonnegative, positive total) and normalized once.
- No per-state drop threshold (e.g. `1e-4`).
- Exact `probability` / `survival` / moments remain available via analytic components.

## Hurdle

```
P(0) = 1 - p_pos
P(y) = p_pos * P_base(y) / P_base(Y>=1)   for y >= 1
overflow = p_pos * P_base(Y>K) / P_base(Y>=1)
```

`P_base(Y>=1)` includes the analytic tail. Moments use complete analytic formulas.

## Zero-inflated

```
P(0) = pi + (1-pi) P_base(0)
P(y) = (1-pi) P_base(y)   for y > 0
```

## Convolution

Independent discrete convolution without truncating then renormalizing.
Joint overflow = `1 - sum(convolved stored atoms)` after adaptive component expansion.

## Tilted

Normalizer sums transformed atoms until the remaining transformed-tail **upper bound**
is `<= 1e-10`. A bound is never treated as exact mass unless certified at that tolerance.

## Production callers

- `mix_atoms` / `predict_stat_atoms` / `predict_stat_distribution` in `sharp_v6.models`
- `inference._normalize_pmf` validates mass identity and **does not** rewrite atoms
- Pricing / settlement use the same atoms + overflow

Production pointer remains on `wnba-pmf-production-v1.1` for this phase (math-only repair).
