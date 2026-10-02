'''
Data midpoints: the candidate set that is exact for count-based curves. See
deltas/search/base.py for why.
'''
import numpy as np

from deltas.search.base import CandidateSet, _all_values
from deltas.core.registry import register


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
