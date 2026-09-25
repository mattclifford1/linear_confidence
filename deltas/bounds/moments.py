'''
Level 1 of the assumption ladder: bounds from the mean and spread, with no
shape assumed.

SawYangMo      Chebyshev's inequality with the mean and s.d. *estimated*
               (Saw, Yang & Mo 1984): valid for any exchangeable sample, even
               one with no finite variance. Average-case, like the published
               1/(N+1) - and it keeps that floor however far b is. Looser than
               the ranks inside the data (working notes, Fig. 4b).
Cantelli       one-sided Chebyshev, P(Z - mu >= k sigma) <= 1/(1 + k^2), with
               mu and sigma bounded from the sample.
VysochanskijPetunin
               its unimodal refinement (Mercadier & Strobel 2021),
               4 / (9 (1 + k^2)) for k^2 >= 5/3.

Cantelli and VP need confidence bounds on mu and sigma that hold without a
shape assumption. For that the score must live in a known range [lo, hi]
(squash it monotonically if it does not): empirical Bernstein for the mean,
Maurer & Pontil (2009, Thms 4 and 10) for the s.d. At N = 10 the s.d. term
alone is 0.91 of the range, so these are comparators for large classes, not
the answer for scarce ones.
'''
import numpy as np

from deltas.bounds.location_scale import _LocationScale
from deltas.core.components import Bound
from deltas.core.registry import register


@register('bound', 'saw_yang_mo')
class SawYangMo(Bound):
    guarantee = 'average_case'
    needs_delta = False
    continuous = True     # a step curve in b, but not stepping at data points
    smooth = False

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        self.xbar = sample.mean
        self.s = sample.std
        self.degenerate = self.N < 2 or not (self.s > 0)
        return self

    @staticmethod
    def bound(lam, N):
        '''P(|Z_{N+1} - xbar| >= lam s) <= this (two-sided; s with ddof = 1)'''
        lam = np.asarray(lam, dtype=float)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            v = np.floor((N + 1) * (N ** 2 - 1 + N * lam ** 2) /
                         (N ** 2 * lam ** 2)) / (N + 1)
        return np.where(lam > 0, np.minimum(1.0, v), 1.0)

    def curve(self, b, delta=None):
        b = np.asarray(b, dtype=float)
        if self.degenerate:
            return np.ones(b.shape)
        lam = self.sample.facing_distance(b) / self.s
        return np.where(lam > 0, self.bound(lam, self.N), 1.0)


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


@register('bound', 'cantelli')
class Cantelli(_BoundedMoments):
    def tail(self, k):
        k = np.asarray(k, dtype=float)
        return 1.0 / (1.0 + k ** 2)


@register('bound', 'vysochanskij_petunin')
class VysochanskijPetunin(_BoundedMoments):
    '''one-sided VP: assumes the class distribution is unimodal'''

    def tail(self, k):
        k2 = np.asarray(k, dtype=float) ** 2
        return np.where(k2 >= 5.0 / 3.0, 4.0 / (9.0 * (1.0 + k2)),
                        4.0 / (3.0 * (1.0 + k2)) - 1.0 / 3.0)
