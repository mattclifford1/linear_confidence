'''
Mean and s.d. bounded without a shape assumption, via a known score range:
empirical Bernstein for the mean, Maurer & Pontil (2009, Thms 4 and 10) for
the s.d. The base of Cantelli and VysochanskijPetunin, which supply the tail.
'''
import numpy as np

from deltas.bounds.location_scale.base import _LocationScale


class _BoundedMoments(_LocationScale):
    '''mean and s.d. bounded without a shape assumption, via a known range'''

    def __init__(self, score_range=None):
        self.score_range = score_range

    def fit(self, sample):
        if self.score_range is None:
            raise ValueError(
                f'{type(self).__name__} needs score_range=(lo, hi): bounding '
                'the mean and s.d. without a shape assumption needs a known '
                'range. Squash the score monotonically if it has none.')
        lo, hi = self.score_range
        if sample.N and (sample.z[0] < lo or sample.z[-1] > hi):
            raise ValueError(f'scores outside score_range={self.score_range}')
        super().fit(sample)
        self.r = float(hi - lo)
        # a zero spread is fine here: sigma_up is still r * sqrt(...) > 0
        self.degenerate = self.N < 2
        if not self.degenerate and not (self.s > 0):
            self.s = 0.0
        return self

    def extremes(self, delta):
        '''delta split equally between the mean and the s.d.'''
        delta = np.asarray(delta, dtype=float)
        n, r, s = self.N, self.r, self.s
        log_mean = np.log(4.0 / delta)           # ln(2 / (delta / 2))
        shift = np.sqrt(2.0 * s ** 2 * log_mean / n) + \
            7.0 * r * log_mean / (3.0 * (n - 1))
        sigma_up = s + r * np.sqrt(2.0 * np.log(2.0 / delta) / (n - 1))
        return shift, sigma_up
