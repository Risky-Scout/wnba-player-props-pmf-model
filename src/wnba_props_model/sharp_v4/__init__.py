"""WNBA Sharp PMF v4 — atom-reliable production forecasting + pricing.

V4 repairs the V3 core statistical defects (explicit feature contracts, 1:1 join cardinality,
no outcome clipping / exact-tail scoring, adaptive support + overflow atoms, hierarchical
dispersion), adds a structural shooting model + minutes-as-a-distribution, and a market-consistent
atom PMF. Sportsbook data never enters the PURE track.
"""
DESIGN_VERSION = "wnba-sharp-pmf-v4"
SEED = 20260730
