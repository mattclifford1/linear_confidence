'''
Neyman-Pearson: meets alpha, reports an unreachable alpha, protects either class.
'''
import pytest

from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator
from deltas.rules import NeymanPearson


def test_neyman_pearson_meets_its_constraint(imbalanced):
    X, y = imbalanced()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.2), search='grid').fit(X, y)
    assert m.rule_info_ == {'protected_label': 1, 'constraint_met': True}
    assert m.certified_error()['class 2'] <= 0.2 + 1e-12
    # a tighter alpha pushes the boundary further from the protected minority
    t = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.05), search='grid').fit(X, y)
    assert t.boundary < m.boundary


def test_neyman_pearson_reports_an_unreachable_alpha(imbalanced):
    X, y = imbalanced()
    m = DeltasEstimator(bound='clopper_pearson', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.01)).fit(X, y)
    # 30 minority points cannot certify below ~0.1: the "floor" result of the notes
    assert m.rule_info_['constraint_met'] is False
    assert m.is_fit


def test_neyman_pearson_can_protect_either_class(imbalanced):
    X, y = imbalanced()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.1, protect=0),
                        search='grid').fit(X, y)
    assert m.rule_info_['protected_label'] == 0
    assert m.certified_error()['class 1'] <= 0.1 + 1e-12
