"""Contract/enum/prior tests for Opportunity V2."""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.opportunity import contracts as C


def test_status_never_maps_missing_to_available():
    assert C.normalize_availability_status(None) == "unknown"
    assert C.normalize_availability_status("") == "unknown"
    assert C.normalize_availability_status("   ") == "unknown"
    assert C.normalize_availability_status("weird provider string") == "unknown"


def test_status_prior_bounds():
    assert C.status_prior("out") <= 0.01
    assert C.status_prior("available") >= 0.99
    assert C.status_prior("unknown") == C.STATUS_PRIOR["unknown"]
    # every normalized status has a prior
    for s in C.AVAILABILITY_STATUS_NORMALIZED:
        assert 0.0 < C.status_prior(s) < 1.0


def test_normalize_common_provider_strings():
    assert C.normalize_availability_status("Out") == "out"
    assert C.normalize_availability_status("Out For Season") == "out"
    assert C.normalize_availability_status("Doubtful") == "doubtful"
    assert C.normalize_availability_status("GTD") == "questionable"
    assert C.normalize_availability_status("Questionable") == "questionable"
    assert C.normalize_availability_status("Probable") == "probable"
    assert C.normalize_availability_status("Available") == "available"
    assert C.normalize_availability_status("Suspended") == "suspended"


def test_validate_frame_schema_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(C.ContractError):
        C.validate_frame_schema(df, ["a", "b"], "unit")
    C.validate_frame_schema(df, ["a"], "unit")  # ok


def test_forbidden_market_columns_detects_signals():
    cols = ["player_minutes_ewma", "market_prob_over_no_vig", "closing_line", "spread_pts", "touches"]
    found = C.forbidden_market_columns(cols)
    assert "market_prob_over_no_vig" in found
    assert "closing_line" in found
    assert "spread_pts" in found
    assert "player_minutes_ewma" not in found
    assert "touches" not in found


def test_cutoff_required_columns_are_canonical():
    for c in ("prediction_cutoff_utc", "scheduled_tip_utc", "game_id", "player_id",
              "team_id", "opponent_team_id"):
        assert c in C.CUTOFF_REQUIRED_COLUMNS
