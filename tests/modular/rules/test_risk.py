'''
Risk: equal priors recover the sum rule; real priors spend less on the rare class.
'''
import pytest

from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator
from deltas.rules import Risk, Sum


def test_risk_with_equal_priors_is_the_sum_rule(imbalanced):
    X, y = imbalanced()
    kw = dict(bound='gaussian', confidence=FixedDelta(0.05), search='grid')
    s = DeltasEstimator(rule=Sum(), **kw).fit(X, y)
    r = DeltasEstimator(rule=Risk(priors=(0.5, 0.5)), **kw).fit(X, y)
    assert r.boundary == pytest.approx(s.boundary, abs=1e-9)


def test_risk_with_real_priors_spends_less_on_the_rare_class(imbalanced):
    '''a rare positive class costs little risk: the boundary moves towards it'''
    X, y = imbalanced()
    kw = dict(bound='gaussian', confidence=FixedDelta(0.05), search='grid')
    s = DeltasEstimator(rule=Sum(), **kw).fit(X, y)
    r = DeltasEstimator(rule=Risk(priors={0: 0.95, 1: 0.05}), **kw).fit(X, y)
    assert r.boundary > s.boundary
    # None uses the empirical proportions (300:30)
    e = DeltasEstimator(rule=Risk(), **kw).fit(X, y)
    assert e.boundary > s.boundary
