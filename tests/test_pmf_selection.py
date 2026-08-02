"""PMF feature-selection scoring must be distinct from a binary Over classifier and must
produce coherent, monotone line probabilities from a single distribution (directive S5)."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.ablation.pmf_selection import (
    assert_cdf_coherent,
    better_on_pmf,
    pmf_over_probabilities,
    score_pmfs,
)
from wnba_props_model.opportunity.pmf_builders import poisson_or_nbinom_pmf


def _poisson_pmfs(mus):
    return [poisson_or_nbinom_pmf(float(m), None, maximum_cap=60) for m in mus]


def test_score_pmfs_returns_all_pmf_metrics():
    rng = np.random.default_rng(0)
    y = rng.poisson(8.0, size=300)
    pmfs = _poisson_pmfs(np.full(300, 8.0))
    s = score_pmfs(pmfs, y, lines=np.full(300, 7.5))
    assert s.n == 300
    assert np.isfinite(s.count_log_score) and s.count_log_score > 0
    assert np.isfinite(s.crps) and s.crps > 0
    assert abs(s.mean_bias) < 0.6          # well-specified mean
    assert s.line_log_loss is not None and s.line_brier is not None


def test_well_specified_pmf_beats_misspecified_on_count_log_score():
    rng = np.random.default_rng(1)
    y = rng.poisson(8.0, size=500)
    good = _poisson_pmfs(np.full(500, 8.0))
    bad = _poisson_pmfs(np.full(500, 3.0))   # badly biased mean
    sg = score_pmfs(good, y)
    sb = score_pmfs(bad, y)
    assert sg.count_log_score < sb.count_log_score
    assert sg.crps < sb.crps
    assert abs(sg.mean_bias) < abs(sb.mean_bias)


def test_pmf_derived_line_probabilities_are_always_coherent():
    pmf = poisson_or_nbinom_pmf(9.0, 4.0, maximum_cap=60)
    lines = [4.5, 6.5, 8.5, 10.5, 12.5]
    p_over = pmf_over_probabilities(pmf, lines)
    # non-increasing in the line
    assert np.all(np.diff(p_over) <= 1e-12)
    assert_cdf_coherent(lines, p_over)   # a single PMF is coherent by construction


def test_incoherent_binary_head_is_rejected():
    # a line-aware binary head that predicts a HIGHER P(over) at a HIGHER line is incoherent
    lines = [4.5, 8.5, 12.5]
    p_over_bad = [0.40, 0.55, 0.30]   # P(over 8.5) > P(over 4.5): contradiction
    with pytest.raises(ValueError, match="incoherent line model"):
        assert_cdf_coherent(lines, p_over_bad)


def test_k_max_budget_never_forces_twenty():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_per_stat_feature_policies as B
    assert B.k_max(3000) == 20
    assert B.k_max(400) == 8       # floor(400/50)
    assert B.k_max(49) == 0        # fewer survive -> no forcing


def test_collapse_correlated_groups_redundant_rolling_variants():
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_per_stat_feature_policies as B
    base = np.linspace(0, 10, 200)
    df = pd.DataFrame({
        "roll5": base + np.random.default_rng(0).normal(0, 1e-6, 200),   # ~identical
        "roll10": base + np.random.default_rng(1).normal(0, 1e-6, 200),  # ~identical
        "independent": np.random.default_rng(2).normal(0, 1, 200),
    })
    fams = B.collapse_correlated(df, ["roll5", "roll10", "independent"], threshold=0.90)
    # the two highly-correlated rolling variants collapse into one family
    reps = list(fams.keys())
    assert "independent" in reps
    assert any(set(members) == {"roll5", "roll10"} for members in fams.values())


def test_advancement_decision_on_pmf_scores():
    rng = np.random.default_rng(2)
    y = rng.poisson(8.0, size=400)
    ref = score_pmfs(_poisson_pmfs(np.full(400, 6.0)), y)
    cand = score_pmfs(_poisson_pmfs(np.full(400, 8.0)), y)
    decision = better_on_pmf(cand, ref)
    assert decision["advances"] is True
    # the worse candidate must not advance over the better reference
    decision_rev = better_on_pmf(ref, cand)
    assert decision_rev["advances"] is False
