"""P3 #1 — release-status consistency. The policy must never claim launch-ready while
no stat is certified, never certify+suppress the same stat, never publish a stat with no
registry entry, and never mark Edge publish without a validated betting policy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from wnba_props_model.pipeline.policy import load_policy

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "config" / "recommendation_policy.yaml"
REGISTRY = REPO / "config" / "stat_registry.json"
LAUNCH_STATES = {"LAUNCH_READY_FORECAST_ONLY", "LIVE_VALIDATED_FORECAST_ONLY", "LIVE_VALIDATED_EDGE"}


def _registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}


def check_policy_consistency(pol, registry: dict) -> list[str]:
    """Return a list of consistency violations (empty == consistent)."""
    errs = []
    certified = set(pol.forecast_certified_stats)
    published = set(pol.forecast_publish_stats)
    suppressed = set(pol.forecast_suppress_stats)
    status = pol.status

    if status in LAUNCH_STATES and not certified:
        errs.append(f"status={status} but certified_stats is empty")
    if certified & suppressed:
        errs.append(f"stats both certified and suppressed: {sorted(certified & suppressed)}")
    if published and status not in LAUNCH_STATES:
        errs.append(f"published stats {sorted(published)} while status={status} (not launch)")
    # every published/certified stat needs a registry entry with forecast_allowed=true
    reg_ok = {s for s, e in registry.items() if e.get("forecast_allowed")}
    missing = (published | certified) - reg_ok
    if missing:
        errs.append(f"published/certified stats without validated registry entry: {sorted(missing)}")
    # Edge publish requires a validated betting policy
    if not pol.abstain:
        bet_ok = any(e.get("betting_recommendation_allowed") for e in registry.values())
        if not bet_ok:
            errs.append("Edge publication_mode != abstain but no betting_recommendation_allowed stat")
    return errs


def test_policy_currently_consistent():
    pol = load_policy(POLICY)
    errs = check_policy_consistency(pol, _registry())
    assert errs == [], f"policy inconsistencies: {errs}"


def test_current_state_forecast_only_seven_markets_certified():
    # seven markets passed the corrected gates -> LIVE_VALIDATED_FORECAST_ONLY; Edge abstains.
    # reb was promoted via Candidate D (empirical residual + frozen dispersion scale 0.9).
    pol = load_policy(POLICY)
    certified = {"turnover", "pts", "ast", "stl", "stocks", "pts_ast", "reb"}
    assert pol.status == "LIVE_VALIDATED_FORECAST_ONLY"
    assert set(pol.forecast_certified_stats) == certified
    assert set(pol.forecast_publish_stats) == certified
    assert not (certified & set(pol.forecast_suppress_stats))
    assert pol.abstain is True                      # Edge remains abstaining
    reg = _registry()
    for m in certified:
        assert reg.get(m, {}).get("forecast_allowed") is True
        assert reg.get(m, {}).get("betting_recommendation_allowed") is False


def test_launch_without_certified_is_flagged():
    pol = load_policy(POLICY)
    # simulate the contradiction: launch status but nothing certified
    object.__setattr__(pol, "status", "LAUNCH_READY_FORECAST_ONLY")
    object.__setattr__(pol, "forecast_certified_stats", [])
    errs = check_policy_consistency(pol, {})
    assert any("certified_stats is empty" in e for e in errs)


def test_certified_also_suppressed_is_flagged():
    pol = load_policy(POLICY)
    object.__setattr__(pol, "forecast_certified_stats", ["reb"])
    object.__setattr__(pol, "forecast_suppress_stats", ["reb"])
    object.__setattr__(pol, "status", "LIVE_VALIDATED_FORECAST_ONLY")
    errs = check_policy_consistency(pol, {"reb": {"forecast_allowed": True}})
    assert any("both certified and suppressed" in e for e in errs)


def test_published_without_registry_is_flagged():
    pol = load_policy(POLICY)
    object.__setattr__(pol, "forecast_publish_stats", ["reb"])
    object.__setattr__(pol, "forecast_certified_stats", ["reb"])
    object.__setattr__(pol, "status", "LIVE_VALIDATED_FORECAST_ONLY")
    errs = check_policy_consistency(pol, {})  # empty registry
    assert any("without validated registry entry" in e for e in errs)
