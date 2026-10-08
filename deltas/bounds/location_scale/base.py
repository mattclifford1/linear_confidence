'''
The machinery every location-scale envelope shares: fit the sample moments,
turn (shift, sigma_up) at a confidence delta into the curve
    U(b) = psi((d(b) - shift) / sigma_up),
and the closed-form minimax boundary of the working notes. Subclasses supply
`extremes(delta)` and `tail(k)`.
'''
import numpy as np

from deltas.bounds.base import Bound


class _LocationScale(Bound):
    guarantee = 'high_probability'
    continuous = True

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        self.xbar = sample.mean
        self.s = sample.std
        self.degenerate = self.N < 2 or not (self.s > 0)
        return self

    def extremes(self, delta):
        '''(shift, sigma_up) at confidence delta; arrays broadcast with delta'''
        raise NotImplementedError

    def tail(self, k):
        '''psi(k): the standardised tail probability beyond k >= 0'''
        raise NotImplementedError

    def curve(self, b, delta):
        b = np.asarray(b, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if self.degenerate:
            return np.ones(np.broadcast(b, delta).shape)
        shift, sigma_up = self.extremes(delta)
        k = (self.sample.facing_distance(b) - shift) / sigma_up
        return np.where(k >= 0, self.tail(np.maximum(k, 0.0)), 1.0)

    def closed_form_minimax(self, other, delta):
        '''
        The "closed-form minimax boundary" of the working notes: with the same tail on both sides the
        minimax boundary splits the gap between the two cautious means in
        proportion to the two inflated spreads. `other` is the other class's
        fitted bound. Returns (b*, value), or None when the cautious means
        cross (no boundary certifies below psi(0)).
        '''
        by_side = {self.sample.side: self, other.sample.side: other}
        lo, hi = by_side['low'], by_side['high']
        s_lo, g_lo = lo.extremes(delta)
        s_hi, g_hi = hi.extremes(delta)
        a1, a2 = lo.xbar + s_lo, hi.xbar - s_hi
        if not a2 > a1:
            return None
        b = (g_hi * a1 + g_lo * a2) / (g_lo + g_hi)
        return float(b), float(self.tail((a2 - a1) / (g_lo + g_hi)))
