'''
The published fences re-expressed as components (for ablations), and the
risk and Neyman-Pearson rules.
'''
import os
import sys

import numpy as np
import pytest

from deltas.bounds import GaussianConfidence, KthPointFence, PublishedFence
from deltas.classifiers.frozen import FrozenProjection
from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator
from deltas.rules import NeymanPearson, Risk, Sum

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'golden'))
import cases  # noqa: E402


def _fixture(name):
    fx = cases.load_fixture(name)
    return fx['z_train'][:, None], fx['y_train'], float(fx['threshold'])


# ----------------------------------------------------------------- fences ---
def test_published_fence_tracks_the_legacy_method():
    '''
    same objective, different search (grid over b vs grid over delta_1), so
    close but not identical - the published numbers stay with the legacy code
    '''
    from deltas.model import base
    X, y, thr = _fixture('gauss_wide')
    legacy = base.base_deltas(FrozenProjection(thr)).fit(X, y)
    modular = DeltasEstimator(bound=PublishedFence(factor=2.0), rule='sum').fit(X, y)
    gap = X[y == 1].min() - X[y == 0].max()
    assert modular.is_fit
    assert abs(modular.boundary - legacy.boundary) < 0.01 * gap


@pytest.mark.parametrize('factor,flag', [(2.0, 'true'), (1.0, 'false')])
@pytest.mark.parametrize('fixture', cases.FIXTURE_NAMES)
def test_published_fence_is_feasible_exactly_where_the_published_method_is(
        factor, flag, fixture):
    '''
    the legacy base method (no slacks) solves only gauss_wide, under either
    USE_TWO setting, and crashes on the rest (FINDINGS B11); the modular fence
    with the matching factor must agree on feasibility, and on the boundary
    where there is one
    '''
    import json
    with open(os.path.join(os.path.dirname(cases.__file__),
                           f'expected_use_two_{flag}.json')) as f:
        legacy = json.load(f)[f'{fixture}/base']
    X, y, _ = _fixture(fixture)
    m = DeltasEstimator(bound=PublishedFence(factor=factor), rule='sum').fit(X, y)
    assert m.is_fit == bool(legacy.get('is_fit', False))
    if not m.is_fit:
        return
    # same objective: the modular search does at least as well, and the
    # legacy answer is near-optimal on it. The objective is nearly flat when
    # the gap is wide (every feasible b scores ~1/(N1+1) + 1/(N2+1)), so the
    # arg-min itself depends on the search: the legacy delta_1 grid stops at
    # 1e-4, the b-sweep does not.
    def objective(b):
        b = np.array([b])
        return float(m.bounds_['low'].resolve(b)['L'][0] +
                     m.bounds_['high'].resolve(b)['L'][0])
    at_legacy = objective(legacy['boundary'])
    assert m.loss <= at_legacy + 1e-12
    assert at_legacy - m.loss < 1e-3
    if factor == 2.0:
        gap = X[y == 1].min() - X[y == 0].max()
        assert abs(m.boundary - legacy['boundary']) < 0.01 * gap


def test_one_sided_radius_ignores_the_far_side():
    '''a majority point far from the boundary moves the two-sided fence only'''
    X, y, _ = _fixture('gauss_wide')
    Xo = np.r_[X, [[X[y == 0].min() - 6.0]]]
    yo = np.r_[y, 0]
    for radius, moves in (('two_sided', True), ('one_sided', False)):
        a = DeltasEstimator(bound=PublishedFence(radius=radius), rule='sum').fit(X, y)
        b = DeltasEstimator(bound=PublishedFence(radius=radius), rule='sum').fit(Xo, yo)
        if moves:
            assert (not b.is_fit) or abs(b.boundary - a.boundary) > 0.05
        else:
            assert b.is_fit and abs(b.boundary - a.boundary) < 0.05


@pytest.mark.parametrize('aggregate,loss_type', [('min', 'min'),
                                                  ('furthest', 'furthest')])
def test_kth_point_fence_tracks_the_draft(aggregate, loss_type):
    from deltas.model import non_sep
    X, y, thr = _fixture('gauss_wide')
    kw = {'only_furtherest_k': True} if loss_type == 'furthest' else \
        {'loss_type': loss_type}
    legacy = non_sep.deltas(FrozenProjection(thr)).fit(X, y, **kw)
    modular = DeltasEstimator(bound=KthPointFence(aggregate=aggregate),
                              rule='sum').fit(X, y)
    assert abs(modular.boundary - legacy.boundary) < 0.05


def test_fence_certificate_is_average_case():
    X, y, _ = _fixture('gauss_wide')
    m = DeltasEstimator(bound='published_fence', rule='sum').fit(X, y)
    assert m.certificates_['decision']['guarantee'] == 'average_case'
    assert 0 < m.delta1 < 1 and 0 < m.delta2 < 1


# ------------------------------------------------------------------ rules ---
def _imbalanced(seed=0):
    rng = np.random.default_rng(seed)
    z = np.r_[rng.normal(0, 1, 300), rng.normal(2.0, 1, 30)]
    return z[:, None], np.r_[np.zeros(300), np.ones(30)]


def test_risk_with_equal_priors_is_the_sum_rule():
    X, y = _imbalanced()
    kw = dict(bound='gaussian', confidence=FixedDelta(0.05), search='grid')
    s = DeltasEstimator(rule=Sum(), **kw).fit(X, y)
    r = DeltasEstimator(rule=Risk(priors=(0.5, 0.5)), **kw).fit(X, y)
    assert r.boundary == pytest.approx(s.boundary, abs=1e-9)


def test_risk_with_real_priors_spends_less_on_the_rare_class():
    '''a rare positive class costs little risk: the boundary moves towards it'''
    X, y = _imbalanced()
    kw = dict(bound='gaussian', confidence=FixedDelta(0.05), search='grid')
    s = DeltasEstimator(rule=Sum(), **kw).fit(X, y)
    r = DeltasEstimator(rule=Risk(priors={0: 0.95, 1: 0.05}), **kw).fit(X, y)
    assert r.boundary > s.boundary
    # None uses the empirical proportions (300:30)
    e = DeltasEstimator(rule=Risk(), **kw).fit(X, y)
    assert e.boundary > s.boundary


def test_neyman_pearson_meets_its_constraint():
    X, y = _imbalanced()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.2), search='grid').fit(X, y)
    assert m.rule_info_ == {'protected_label': 1, 'constraint_met': True}
    assert m.certified_error()['class 2'] <= 0.2 + 1e-12
    # a tighter alpha pushes the boundary further from the protected minority
    t = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.05), search='grid').fit(X, y)
    assert t.boundary < m.boundary


def test_neyman_pearson_reports_an_unreachable_alpha():
    X, y = _imbalanced()
    m = DeltasEstimator(bound='clopper_pearson', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.01)).fit(X, y)
    # 30 minority points cannot certify below ~0.1: the "floor" result of the notes
    assert m.rule_info_['constraint_met'] is False
    assert m.is_fit


def test_neyman_pearson_can_protect_either_class():
    X, y = _imbalanced()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        rule=NeymanPearson(alpha=0.1, protect=0),
                        search='grid').fit(X, y)
    assert m.rule_info_['protected_label'] == 0
    assert m.certified_error()['class 1'] <= 0.1 + 1e-12
