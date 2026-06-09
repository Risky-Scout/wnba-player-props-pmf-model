import numpy as np

from wnba_props_model.models.simulation import convolve_pmfs, normalize_pmf
from wnba_props_model.models.market import prob_over_from_pmf


def test_normalize_pmf():
    p = normalize_pmf(np.array([0.0, 2.0, 2.0]))
    assert abs(p.sum() - 1) < 1e-12
    assert p[1] == 0.5


def test_convolution_sum():
    a = np.array([0.5, 0.5])
    b = np.array([0.25, 0.75])
    c = convolve_pmfs(a, b)
    assert abs(c.sum() - 1) < 1e-12
    assert len(c) == 3


def test_prob_over_half_line():
    p = np.array([0.1, 0.2, 0.7])
    assert abs(prob_over_from_pmf(p, 1.5) - 0.7) < 1e-12
