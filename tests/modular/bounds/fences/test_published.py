'''
The published fence as a component: close to the legacy method, feasible exactly
where it is, one-sided radius, average-case certificate. The published numbers
stay with the legacy code.
'''
import json
import os

import numpy as np
import pytest

import cases
from deltas.bounds import PublishedFence
from deltas.classifiers.frozen import FrozenProjection
from deltas.core import DeltasEstimator


def test_published_fence_tracks_the_legacy_method(golden_xy):
    '''
    same objective, different search (grid over b vs grid over delta_1), so
    close but not identical - the published numbers stay with the legacy code
    '''
    from deltas.model import base
    X, y, thr = golden_xy('gauss_wide')
    legacy = base.base_deltas(FrozenProjection(thr)).fit(X, y)
    modular = DeltasEstimator(bound=PublishedFence(factor=2.0), rule='sum').fit(X, y)
    gap = X[y == 1].min() - X[y == 0].max()
    assert modular.is_fit
    assert abs(modular.boundary - legacy.boundary) < 0.01 * gap


@pytest.mark.parametrize('factor,flag', [(2.0, 'true'), (1.0, 'false')])
@pytest.mark.parametrize('fixture', cases.FIXTURE_NAMES)
def test_published_fence_is_feasible_exactly_where_the_published_method_is(
        golden_xy, factor, flag, fixture):
    '''
    the legacy base method (no slacks) solves only gauss_wide, under either
    USE_TWO setting, and crashes on the rest (FINDINGS B11); the modular fence
    with the matching factor must agree on feasibility, and on the boundary
    where there is one
    '''
    with open(os.path.join(os.path.dirname(cases.__file__),
                           f'expected_use_two_{flag}.json')) as f:
        legacy = json.load(f)[f'{fixture}/base']
    X, y, _ = golden_xy(fixture)
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


def test_one_sided_radius_ignores_the_far_side(golden_xy):
    '''a majority point far from the boundary moves the two-sided fence only'''
    X, y, _ = golden_xy('gauss_wide')
    Xo = np.r_[X, [[X[y == 0].min() - 6.0]]]
    yo = np.r_[y, 0]
    for radius, moves in (('two_sided', True), ('one_sided', False)):
        a = DeltasEstimator(bound=PublishedFence(radius=radius), rule='sum').fit(X, y)
        b = DeltasEstimator(bound=PublishedFence(radius=radius), rule='sum').fit(Xo, yo)
        if moves:
            assert (not b.is_fit) or abs(b.boundary - a.boundary) > 0.05
        else:
            assert b.is_fit and abs(b.boundary - a.boundary) < 0.05


def test_fence_certificate_is_average_case(golden_xy):
    X, y, _ = golden_xy('gauss_wide')
    m = DeltasEstimator(bound='published_fence', rule='sum').fit(X, y)
    assert m.certificates_['decision']['guarantee'] == 'average_case'
    assert 0 < m.delta1 < 1 and 0 < m.delta2 < 1
