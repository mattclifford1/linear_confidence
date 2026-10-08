'''
Sum: the arg-min of L_1 + L_2.
'''
import numpy as np
import pytest

from deltas.rules import Sum


def test_sum_takes_the_argmin():
    c = np.array([0.0, 1.0, 2.0])
    b, losses = Sum().choose(c, np.array([0.5, 0.2, 0.1]),
                             np.array([0.1, 0.2, 0.6]))
    assert b == 1.0 and Sum().combine(0.1, 0.2) == pytest.approx(0.3)
