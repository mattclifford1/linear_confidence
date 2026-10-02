'''
Shared helpers for the modular tests, as fixtures, so each test file can sit
next to the module it covers. Also puts tests/golden (the fixtures and cases)
and experiments/ (the runners) on sys.path once, for the tests that import
them.
'''
import os
import sys

import numpy as np
import pytest

from deltas.bounds import Bound
from deltas.core import ClassSample

_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_TESTS, 'golden'),
           os.path.join(os.path.dirname(_TESTS), 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPS = 1500


def _coverage(make_bound, draw, true_tail, N, delta, side, seed):
    '''fraction of samples whose envelope covers the truth on a whole grid'''
    rng = np.random.default_rng(seed)
    grid = np.linspace(-6, 6, 241)
    ok = 0
    for _ in range(REPS):
        s = ClassSample(draw(rng, N), 0, side)
        U = make_bound().fit(s).curve(grid, delta)
        ok += np.all(U >= true_tail(grid) - 1e-12)
    return ok / REPS


@pytest.fixture
def coverage():
    '''the simulation check every high-probability bound runs'''
    return _coverage


class _Smooth(Bound):
    '''a toy continuous bound: a logistic tail in the facing distance'''
    needs_delta = True

    def curve(self, b, delta):
        d = self.sample.facing_distance(b)
        return 1.0 / (1.0 + np.exp(d)) + 0.0 * np.asarray(delta, dtype=float)


class _NoDelta(_Smooth):
    needs_delta = False
    guarantee = 'average_case'

    def curve(self, b, delta):
        assert delta is None
        return 1.0 / (1.0 + np.exp(self.sample.facing_distance(b)))


@pytest.fixture
def smooth_bound():
    '''a toy continuous bound class, for policy and search tests'''
    return _Smooth


@pytest.fixture
def no_delta_bound():
    '''a toy average-case bound class with no confidence level'''
    return _NoDelta


def _imbalanced(seed=0):
    rng = np.random.default_rng(seed)
    z = np.r_[rng.normal(0, 1, 300), rng.normal(2.0, 1, 30)]
    return z[:, None], np.r_[np.zeros(300), np.ones(30)]


@pytest.fixture
def imbalanced():
    '''(X, y) with 300 majority and 30 minority points, two Gaussians'''
    return _imbalanced


def _golden_xy(name):
    import cases
    fx = cases.load_fixture(name)
    return fx['z_train'][:, None], fx['y_train'], float(fx['threshold'])


@pytest.fixture
def golden_xy():
    '''(X, y, threshold) of one golden projection fixture, by name'''
    return _golden_xy
