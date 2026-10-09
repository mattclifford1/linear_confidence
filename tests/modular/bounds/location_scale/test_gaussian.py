'''
GaussianConfidence: the rectangle and the Monte Carlo band cover the true tail
at every boundary at once (checked by simulation), the band needs a fixed
delta, and the band has the right shape.
'''
import numpy as np
import pytest
from scipy import stats

from deltas.bounds import GaussianConfidence, StudentTPredictive
from deltas.confidence import OptimisedDelta
from deltas.core import ClassSample, DeltasEstimator


@pytest.mark.parametrize('side', ['low', 'high'])
def test_gaussian_rectangle_covers(coverage, side):
    tail = (lambda b: stats.norm.sf(b)) if side == 'low' else \
        (lambda b: stats.norm.cdf(b))
    cov = coverage(GaussianConfidence, lambda r, n: r.normal(0, 1, n), tail,
                    N=12, delta=0.1, side=side, seed=1)
    assert cov >= 0.9 - 0.02


def test_gaussian_mc_band_covers_and_is_tighter(coverage):
    cov = coverage(lambda: GaussianConfidence(band='mc', draws=20000),
                    lambda r, n: r.normal(0, 1, n), stats.norm.sf,
                    N=12, delta=0.1, side='low', seed=2)
    assert cov >= 0.9 - 0.025
    s = ClassSample(np.random.default_rng(3).normal(0, 1, 12), 0, 'low')
    b = np.array([2.5])
    rect = GaussianConfidence().fit(s).curve(b, 0.1)
    band = GaussianConfidence(band='mc', draws=20000).fit(s).curve(b, 0.1)
    assert band[0] <= rect[0]


def test_mc_band_needs_a_fixed_delta():
    X = np.r_[np.random.default_rng(0).normal(0, 1, 50), np.random.default_rng(1).normal(3, 1, 10)]
    y = np.r_[np.zeros(50), np.ones(10)]
    with pytest.raises(NotImplementedError, match='fixed'):
        DeltasEstimator(bound=GaussianConfidence(band='mc'),
                        confidence=OptimisedDelta(resolution=20)).fit(X[:, None], y)


def test_fewer_points_mean_more_room():
    '''same mean and s.d., smaller N -> larger band beyond the data'''
    base = np.random.default_rng(6).normal(0, 1, 200)
    base = (base - base.mean()) / base.std(ddof=1)
    small = np.r_[-1.2, -0.6, 0.0, 0.6, 1.2]
    small = (small - small.mean()) / small.std(ddof=1)
    b = np.array([3.0])
    for B in (GaussianConfidence(), StudentTPredictive()):
        big_U = type(B)().fit(ClassSample(base, 0, 'low')).curve(b, 0.05)
        small_U = type(B)().fit(ClassSample(small, 0, 'low')).curve(b, 0.05)
        assert small_U[0] > big_U[0]


def test_band_shape():
    s = ClassSample(np.random.default_rng(7).normal(0, 1, 30), 0, 'low')
    g = GaussianConfidence().fit(s)
    b = np.linspace(-3, 6, 91)
    U = g.curve(b, 0.05)
    assert np.all((U >= 0) & (U <= 1))
    assert np.all(np.diff(U) <= 1e-15)                 # falls away from the class
    assert np.all(g.curve(b, 0.01) >= U - 1e-15)       # smaller delta, wider
    assert g.curve(np.array([-10.0]), 0.05)[0] == 1.0  # inside the class: 1
