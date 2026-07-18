"""Loader for the canonical recommendation/publication policy (P2 Phase 1).

Both the live edge builder and the historical replay load the SAME file so the two
never drift. The loader validates required keys and returns a typed view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_POLICY_PATH = "config/recommendation_policy.yaml"

_REQUIRED_EDGE_KEYS = {
    "edge_threshold", "min_market_prob", "max_shin_z", "supported_stats",
    "required_calibration", "source_policy", "publication_mode",
}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class RecommendationPolicy:
    version: int
    edge_threshold: float
    min_market_prob: float
    max_shin_z: float
    supported_stats: list
    required_calibration: list
    source_policy: str
    publication_mode: str            # "abstain" | "publish"
    suppress_sides: list = field(default_factory=list)
    suppress_stats: list = field(default_factory=list)
    forecast_publish_stats: list = field(default_factory=list)
    forecast_suppress_stats: list = field(default_factory=list)
    forecast_status: str = ""
    forecast_certified_stats: list = field(default_factory=list)
    forecast_pending_banner: str = ""
    status: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def abstain(self) -> bool:
        return str(self.publication_mode).lower() == "abstain"

    def edge_eligible_stats(self) -> set:
        """Supported stats minus suppressed stats."""
        return set(self.supported_stats) - set(self.suppress_stats)


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> RecommendationPolicy:
    p = Path(path)
    if not p.exists():
        raise PolicyError(f"policy file not found: {p}")
    doc = yaml.safe_load(p.read_text()) or {}
    edge = doc.get("edge", {})
    missing = _REQUIRED_EDGE_KEYS - set(edge)
    if missing:
        raise PolicyError(f"policy 'edge' missing required keys: {sorted(missing)}")
    if str(edge["publication_mode"]).lower() not in ("abstain", "publish"):
        raise PolicyError(f"invalid publication_mode: {edge['publication_mode']}")
    if float(edge["edge_threshold"]) <= 0.0:
        raise PolicyError("edge_threshold must be > 0 (never publish every prop at 0.0)")
    fc = doc.get("forecast", {})
    return RecommendationPolicy(
        version=int(doc.get("version", 0)),
        edge_threshold=float(edge["edge_threshold"]),
        min_market_prob=float(edge["min_market_prob"]),
        max_shin_z=float(edge["max_shin_z"]),
        supported_stats=list(edge["supported_stats"]),
        required_calibration=list(edge["required_calibration"]),
        source_policy=str(edge["source_policy"]),
        publication_mode=str(edge["publication_mode"]).lower(),
        suppress_sides=list(edge.get("suppress_sides", [])),
        suppress_stats=list(edge.get("suppress_stats", [])),
        forecast_publish_stats=list(fc.get("publish_stats", [])),
        forecast_suppress_stats=list(fc.get("suppress_stats", [])),
        forecast_status=str(fc.get("status", "")),
        forecast_certified_stats=list(fc.get("certified_stats", [])),
        forecast_pending_banner=str(fc.get("pending_banner", "")),
        status=str(doc.get("status", "")),
        raw=doc,
    )
