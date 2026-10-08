'''
Saw, Yang & Mo (1984): Chebyshev's inequality with the mean and s.d.
estimated. Average-case, floored at 1/(N+1). See
deltas/bounds/moments/__init__.py.
'''
import numpy as np

from deltas.bounds.base import Bound
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
