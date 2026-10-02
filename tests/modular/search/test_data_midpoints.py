'''
DataMidpoints: midpoints padded at both ends, and a single distinct value.
'''
import numpy as np

from deltas.core import ProjectedData
from deltas.search import DataMidpoints


def test_midpoints_are_padded_and_handle_one_value():
    d = ProjectedData(np.r_[0.0, 1.0, 1.0, 4.0], np.array([0, 0, 1, 1]))
    mids = DataMidpoints().generate(d)
    assert np.allclose(mids[1:-1], [0.5, 2.5])
    assert mids[0] < 0.0 and mids[-1] > 4.0
    single = ProjectedData(np.r_[2.0, 2.0], np.array([0, 1]))
    assert list(DataMidpoints().generate(single)) == [1.0, 3.0]
