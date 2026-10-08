'''
The closed-form minimax boundary of the working notes, the shape-free result
(the boundary does not depend on the tail), and more room for a scarce class.
'''
import numpy as np
import pytest

from deltas.bounds import GaussianConfidence
from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator


def _two_class(seed=9, n0=400, n1=12, m1=3.0, s1=1.5):
    rng = np.random.default_rng(seed)
    z = np.r_[rng.normal(0, 1, n0), rng.normal(m1, s1, n1)]
    return z[:, None], np.r_[np.zeros(n0), np.ones(n1)]


def test_minimax_boundary_matches_the_closed_form():
    X, y = _two_class()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        search='grid').fit(X, y)
    b_star, value = m.bounds_['low'].closed_form_minimax(m.bounds_['high'], 0.05)
    assert m.boundary == pytest.approx(b_star, abs=1e-9)
    assert m.loss == pytest.approx(value, rel=1e-9)


class _GaussianExtremesCantelliTail(GaussianConfidence):
    '''same (mu, sigma) bounds as the Gaussian, Cantelli's tail'''

    def tail(self, k):
        return 1.0 / (1.0 + np.asarray(k) ** 2)


def test_minimax_boundary_does_not_depend_on_the_tail_shape():
    '''the "shape-free" result: only the certified value changes'''
    X, y = _two_class()
    kw = dict(confidence=FixedDelta(0.05), search='grid')
    g = DeltasEstimator(bound=GaussianConfidence(), **kw).fit(X, y)
    c = DeltasEstimator(bound=_GaussianExtremesCantelliTail(), **kw).fit(X, y)
    assert g.boundary == pytest.approx(c.boundary, abs=1e-8)
    assert c.loss > g.loss


def test_scarce_or_spread_minority_gets_more_room():
    '''the minimax boundary moves towards the majority as N_2 shrinks'''
    b = []
    for n1 in (6, 30, 300):
        X, y = _two_class(seed=10, n1=n1, s1=1.0)
        b.append(DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                                 search='grid').fit(X, y).boundary)
    assert b[0] < b[1] < b[2]
