'''
Grid: n points over the range widened by the margin.
'''
import numpy as np

from deltas.core import ProjectedData
from deltas.search import Grid


def test_grid_spans_the_widened_range():
    d = ProjectedData(np.r_[0.0, 1.0, 1.0, 4.0], np.array([0, 0, 1, 1]))
    g = Grid(n=11, margin=0.5).generate(d)
    assert g[0] == -2.0 and g[-1] == 6.0 and len(g) == 11
