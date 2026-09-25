'''
Average-case curves: the expected error of a boundary placed at b.

StudentTPredictive
    the chance that the *next* class point falls on the wrong side of b,
    averaged over the uncertainty in (mu, sigma) of a Gaussian class:

        P_pred(b) = T_{N-1}( d(b) / (s sqrt(1 + 1/N)) ) tail

    For a fixed rule b = xbar + c s sqrt(1 + 1/N) this is exact over sample and
    new point together, and it is the Bayesian posterior predictive under the
    reference prior p(mu, sigma) ~ 1/sigma. Small N gives heavier tails and a
    wider scale; a larger s gives a wider scale. It is the spread-aware heir
    of the published 1/(N+1), and the recommended *decision* curve (working
    notes, Sec. 6.1). It has no delta, so it is not a certificate - pair it
    with certify=[...] at a fixed delta.
'''
import numpy as np
from scipy import stats

from deltas.core.components import Bound
from deltas.core.registry import register


@register('bound', 'predictive_t')
class StudentTPredictive(Bound):
    guarantee = 'average_case'
    needs_delta = False
    continuous = True

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        self.xbar = sample.mean
        self.s = sample.std
        self.degenerate = self.N < 2 or not (self.s > 0)
        return self

    def curve(self, b, delta=None):
        b = np.asarray(b, dtype=float)
        if self.degenerate:
            return np.ones(b.shape)
        scale = self.s * np.sqrt(1.0 + 1.0 / self.N)
        # facing distance > 0 means b is beyond the mean, towards the boundary
        return stats.t.sf(self.sample.facing_distance(b) / scale, self.N - 1)
