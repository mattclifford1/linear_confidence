'''
Cantelli: needs a score range, and covers bounded data at its nominal level.
'''
import numpy as np
import pytest
from scipy import stats

from deltas.bounds import Cantelli
from deltas.core import ClassSample


def test_bounded_moment_bounds_need_a_range():
    s = ClassSample(np.random.default_rng(12).uniform(0, 1, 50), 0, 'low')
    with pytest.raises(ValueError, match='score_range'):
        Cantelli().fit(s)
    with pytest.raises(ValueError, match='outside'):
        Cantelli(score_range=(0.2, 1.0)).fit(s)


def test_cantelli_covers_bounded_data(coverage):
    cov = coverage(lambda: Cantelli(score_range=(-6, 6)),
                    lambda r, n: np.clip(r.normal(0, 1, n), -6, 6),
                    stats.norm.sf, N=40, delta=0.1, side='low', seed=13)
    assert cov >= 0.9
