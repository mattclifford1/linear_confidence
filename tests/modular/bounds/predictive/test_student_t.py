'''
StudentTPredictive is exact for a fixed rule, over sample and new point together.
'''
import numpy as np
import pytest
from scipy import stats

from deltas.bounds import StudentTPredictive
from deltas.core import ClassSample


def test_student_t_predictive_is_exact_for_a_fixed_rule():
    N, c = 8, 1.7
    rng = np.random.default_rng(8)
    x = rng.normal(0, 1, (200000, N + 1))
    xb, s = x[:, :N].mean(1), x[:, :N].std(1, ddof=1)
    sim = np.mean(x[:, N] <= xb - c * s * np.sqrt(1 + 1 / N))
    exact = stats.t.cdf(-c, N - 1)
    assert abs(sim - exact) < 4 * np.sqrt(exact * (1 - exact) / 200000)
    sample = ClassSample(x[0, :N], 1, 'high')
    b = sample.mean - c * sample.std * np.sqrt(1 + 1 / N)
    assert StudentTPredictive().fit(sample).curve(np.array([b]))[0] == \
        pytest.approx(exact, rel=1e-12)
