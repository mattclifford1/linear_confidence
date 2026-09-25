'''
Slot: which boundaries are tried.

For count-based curves, which only change at data points, the midpoints
between consecutive distinct projected values (plus one point beyond each end)
contain every distinct pair of counts, so nothing is missed. Smooth curves need
a grid, and benefit from refining the grid choice afterwards.
'''
import numpy as np

from deltas.core.components import CandidateSet
from deltas.core.registry import register


def _all_values(data):
    return np.unique(np.concatenate([data.low.z, data.high.z]))


@register('search', 'data_midpoints')
class DataMidpoints(CandidateSet):
    '''midpoints between consecutive distinct values, padded at both ends'''

    def generate(self, data, bounds=None):
        allz = _all_values(data)
        if len(allz) == 1:
            span = 1.0
            return np.array([allz[0] - span, allz[0] + span])
        mids = (allz[:-1] + allz[1:]) / 2.0
        pad = (allz[-1] - allz[0]) * 0.01 + 1e-12
        return np.concatenate([[allz[0] - pad], mids, [allz[-1] + pad]])


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
