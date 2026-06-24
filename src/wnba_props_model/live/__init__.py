"""Live in-play engine for WNBA player props.

Modules:
  pbp_parser    — Parse BDL play-by-play text into live player stat totals
  bayesian_updater — Gamma-Poisson Bayesian posterior engine
  live_edge     — Compare live model P(over) vs live prop lines from BDL
  orchestrator  — Orchestrate live game tracking loop
"""
from wnba_props_model.live.pbp_parser import PBPParser, LivePlayerState
from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine
from wnba_props_model.live.live_edge import LiveEdgeCalculator
from wnba_props_model.live.orchestrator import LiveGameOrchestrator

__all__ = [
    "PBPParser",
    "LivePlayerState",
    "GammaPoissonLiveEngine",
    "LiveEdgeCalculator",
    "LiveGameOrchestrator",
]
