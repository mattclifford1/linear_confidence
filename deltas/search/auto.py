'''
Auto: data midpoints when every curve is a step function at data points,
otherwise the midpoints and a grid together. The default search.
'''
import numpy as np

from deltas.search.base import CandidateSet
from deltas.search.data_midpoints import DataMidpoints
from deltas.search.grid import Grid
from deltas.core.registry import register


@register('search', 'auto')
class Auto(CandidateSet):
    '''
    data midpoints when every curve is a step function; otherwise the union
    of the midpoints and a grid
    '''

    def __init__(self, n=2001, margin=0.25):
        self.n = n
        self.margin = margin

    def generate(self, data, bounds=None):
        mids = DataMidpoints().generate(data)
        if bounds is not None and not any(
                getattr(b, 'continuous', False) for b in bounds.values()):
            return mids
        grid = Grid(self.n, self.margin).generate(data)
        return np.unique(np.concatenate([mids, grid]))
