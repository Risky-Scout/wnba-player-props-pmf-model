"""WNBA Pricing PMF v1 — coherent feature-driven player-prop pricing engine.

Public surface:
- market_registry: canonical provider-market -> internal-outcome registry.
- engine: fair-odds pricing from a count PMF (push-safe) + margin/quoted layer.
- joint_generator: coherent joint active-player outcome simulation + atom PMFs.
- calibration: monotone-CDF distributional calibration.
"""
RELEASE_VERSION = "wnba-pricing-pmf-v1.0.0-rc1"
MODEL_VERSION = "wnba-pricing-pmf-v1"
