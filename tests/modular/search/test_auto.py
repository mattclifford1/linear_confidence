'''
Auto: midpoints alone for step curves, midpoints plus a grid otherwise.
'''
import numpy as np

from deltas.bounds import DKW, ClopperPearson
from deltas.core import ProjectedData
from deltas.search import Auto, DataMidpoints


def test_auto_adds_a_grid_only_for_continuous_curves(smooth_bound):
    d = ProjectedData(np.r_[0.0, 1.0, 1.0, 4.0], np.array([0, 0, 1, 1]))
    mids = DataMidpoints().generate(d)
    counts = {'low': ClopperPearson(), 'high': DKW()}
    assert np.array_equal(Auto().generate(d, counts), mids)
    smooth = {'low': ClopperPearson(), 'high': smooth_bound()}
    union = Auto(n=11).generate(d, smooth)
    assert set(mids) <= set(union) and len(union) > len(mids)
