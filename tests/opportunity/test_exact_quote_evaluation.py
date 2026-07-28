"""Exact-quote settlement + scoring tests for Opportunity V2 (section 30)."""
from __future__ import annotations

import numpy as np

from wnba_props_model.opportunity.pmf_builders import settled_over_probability


def _log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_settlement_matches_pmf_over_mass_half_line():
    # symmetric-ish PMF, half line 1.5 -> p_over = P(Y>=2)
    pmf = np.array([0.3, 0.4, 0.2, 0.1])
    over, under, push = settled_over_probability(pmf, 1.5)
    assert abs(over - 0.3) < 1e-9
    assert abs(under - 0.7) < 1e-9
    assert push == 0.0


def test_integer_line_push_removed_from_scoring():
    pmf = np.array([0.25, 0.5, 0.25])
    over, under, push = settled_over_probability(pmf, 1.0)
    assert abs(push - 0.5) < 1e-9
    assert abs(over - 0.5) < 1e-9 and abs(under - 0.5) < 1e-9


def test_a_sharper_correct_model_scores_better():
    # a model that concentrates mass on the correct side gets a lower log loss
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, 0.8, 0.2)
    bad = np.full(400, 0.5)
    assert _log_loss(y, good) < _log_loss(y, bad)


def test_settlement_probabilities_are_bounded():
    rng = np.random.default_rng(1)
    for _ in range(50):
        raw = rng.random(8)
        pmf = raw / raw.sum()
        for line in (0.5, 1.0, 2.5, 3.0):
            over, under, push = settled_over_probability(pmf, line)
            assert 0.0 <= over <= 1.0 and 0.0 <= under <= 1.0 and 0.0 <= push <= 1.0
