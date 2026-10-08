'''
Grid: evenly spaced boundaries over the (widened) data range, for smooth
curves.
'''
import numpy as np

from deltas.search.base import CandidateSet, _all_values
from deltas.core.registry import register


@register('search', 'grid')
class Grid(CandidateSet):
    '''
    n evenly spaced boundaries over the data range, widened by `margin` times
    the range on each side (smooth curves keep changing beyond the data)
    '''

    def __init__(self, n=2001, margin=0.25):
        self.n = n
        self.margin = margin

    def generate(self, data, bounds=None):
        allz = _all_values(data)
        lo, hi = allz[0], allz[-1]
        span = (hi - lo) if hi > lo else 1.0
        return np.linspace(lo - self.margin * span, hi + self.margin * span,
                           self.n)
