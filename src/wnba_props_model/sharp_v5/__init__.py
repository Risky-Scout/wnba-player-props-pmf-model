"""WNBA Sharp PMF v5 — coherent atom distributions and production pricing.

V5 replaces the PMF abstraction with a single DiscreteDistribution interface whose mass accounting
is correct (exact analytic per-atom tail probability, correct hurdle/zero-inflated/convolution
mass, aggregate overflow used only as a bucket), push-aware A/(A+B) settlement + multi-line
market projection, and minutes-PMF propagation into every stat PMF.
"""
DESIGN_VERSION = "wnba-sharp-pmf-v5"
SEED = 20260730
