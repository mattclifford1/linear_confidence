'''
OptimisedDelta: the 'expected' and 'union' loss forms.
'''
import numpy as np
import pytest

from deltas.bounds import ClopperPearson
from deltas.confidence import OptimisedDelta
from deltas.core import ClassSample


def test_optimised_delta_forms():
    s = ClassSample(np.arange(30.0), 0, 'low')
    b = np.array([10.5])
    cp = ClopperPearson().fit(s)
    e = OptimisedDelta(resolution=200, form='expected').prepare(cp)(b)
    u = OptimisedDelta(resolution=200, form='union').prepare(cp)(b)
    assert e['L'][0] == pytest.approx((1 - e['delta'][0]) * e['U'][0] + e['delta'][0])
    assert u['L'][0] == pytest.approx(u['U'][0] + u['delta'][0])
    assert e['L'][0] <= u['L'][0]
    with pytest.raises(ValueError):
        OptimisedDelta(form='nope')
