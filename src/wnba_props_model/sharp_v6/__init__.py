"""WNBA Sharp PMF v6 — production-complete coherent player distributions.

V6 corrects the BDL shooting-label endpoint (/wnba/v1/player_stats), repairs the market-projection
mass bug with a tail-aware TiltedDistribution (tilt applied to the COMPLETE base incl. tail;
Z computed with a certified remainder bound; stored+overflow=1), and fixes remaining distribution
moment math.
"""
DESIGN_VERSION = "wnba-sharp-pmf-v6"
SEED = 20260730
