"""Game-specific Opportunity Model V2 (parallel candidate ``OPP_V2_RAW``).

A causally valid, point-in-time opportunity-modeling package. Every submodel consumes ONLY
information available at ``prediction_cutoff_utc``; market signals are forbidden as inputs and may
enter only at the final evaluation join. The canonical output is the ACTIVE (conditional-on-play)
PMF; sportsbook settlement reads ``active_pmf_json`` (void-on-DNP), never the availability mixture.

This package is additive: it does not modify or route through the frozen baseline candidate.
"""
from __future__ import annotations

VERSION = "opportunity_v2_v1"

__all__ = ["VERSION"]
