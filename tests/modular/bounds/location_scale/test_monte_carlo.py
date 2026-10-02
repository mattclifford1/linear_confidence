'''
LocationScaleConfidence: Monte Carlo pivots cover a logistic class and reproduce the exact Gaussian.
'''
import numpy as np
import pytest
from scipy import stats

from deltas.bounds import GaussianConfidence, LocationScaleConfidence
from deltas.core import ClassSample


def test_logistic_family_covers_logistic_data(coverage):
    cov = coverage(lambda: LocationScaleConfidence('logistic', draws=20000),
                    lambda r, n: r.logistic(0, 1, n), stats.logistic.sf,
                    N=15, delta=0.1, side='low', seed=4)
    assert cov >= 0.9 - 0.025


def test_monte_carlo_pivots_reproduce_the_exact_gaussian():
    s = ClassSample(np.random.default_rng(5).normal(0, 1, 20), 0, 'low')
    exact = GaussianConfidence().fit(s).extremes(0.05)
    mc = LocationScaleConfidence('gaussian', draws=200000).fit(s).extremes(0.05)
    assert mc[0] == pytest.approx(exact[0], rel=0.03)
    assert mc[1] == pytest.approx(exact[1], rel=0.03)
