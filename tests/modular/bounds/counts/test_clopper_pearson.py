'''
Clopper-Pearson: the union correction over the N + 1 thresholds, applied at a fixed delta.
'''
import numpy as np

from deltas.bounds import ClopperPearson, clopper_pearson_upper
from deltas.confidence import FixedDelta
from deltas.core import ClassSample


def test_fixed_delta_applies_the_union_correction():
    s = ClassSample(np.arange(20.0), 0, 'low')
    b = np.array([4.5, 15.5])
    cp = ClopperPearson().fit(s)
    got = FixedDelta(0.05).prepare(cp)(b)
    m = s.wrong_side_count(b)
    assert np.array_equal(got['U'], clopper_pearson_upper(m, 20, 0.05 / 21))
    assert np.all(got['delta'] == 0.05)
    no_union = ClopperPearson(union_bound=False).fit(s)
    assert np.all(FixedDelta(0.05).prepare(no_union)(b)['U'] < got['U'])
