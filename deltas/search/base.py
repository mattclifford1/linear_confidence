'''
Base class of the search slot: which boundaries are tried.

For count-based curves, which only change at data points, the midpoints
between consecutive distinct projected values (plus one point beyond each end)
contain every distinct pair of counts, so nothing is missed. Smooth curves need
a grid, and benefit from refining the grid choice afterwards.

_all_values is the sorted distinct scores both classes together, shared by
the candidate sets.
'''
import numpy as np

from deltas.core.component import Component


class CandidateSet(Component):
    def generate(self, data, bounds):
        '''array of boundaries to try'''
        raise NotImplementedError


def _all_values(data):
    return np.unique(np.concatenate([data.low.z, data.high.z]))
