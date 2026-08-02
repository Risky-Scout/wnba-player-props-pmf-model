"""WNBA Sharp PMF v6 — authoritative production package.

Public entrypoint: ``predict_slate`` (see ``inference.py``).
"""
from wnba_props_model.sharp_v6.inference import predict_slate
from wnba_props_model.sharp_v6.types import PlayerTargetPMF, SlatePMFDelivery

DESIGN_VERSION = "wnba-sharp-pmf-v6"
SEED = 20260730
PRODUCTION = True

__all__ = [
    "DESIGN_VERSION",
    "SEED",
    "PRODUCTION",
    "predict_slate",
    "PlayerTargetPMF",
    "SlatePMFDelivery",
]
