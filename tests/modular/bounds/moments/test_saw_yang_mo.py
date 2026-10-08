'''
Saw-Yang-Mo: the formula, and the 1/(N+1) floor however far the boundary is.
'''
import numpy as np
import pytest

from deltas.bounds import SawYangMo
from deltas.core import ClassSample


def test_saw_yang_mo_formula_and_floor():
    s = ClassSample(np.random.default_rng(11).normal(0, 1, 10), 0, 'low')
    sym = SawYangMo().fit(s)
    lam = np.array([1.5, 3.0, 50.0])
    b = s.mean + lam * s.std
    N = 10
    expected = np.minimum(1, np.floor((N + 1) * (N ** 2 - 1 + N * lam ** 2) /
                                      (N ** 2 * lam ** 2)) / (N + 1))
    assert np.allclose(sym.curve(b), expected)
    assert sym.curve(np.array([1e6]))[0] == pytest.approx(1 / (N + 1))
