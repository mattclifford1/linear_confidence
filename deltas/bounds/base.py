'''
Base classes of the bound slot.

A Bound is one class's error curve at any boundary b: an upper bound on the
class-conditional error (high-probability) or an estimate of it
(average-case). This is where a concentration inequality lives. The
implementations are one per file under deltas/bounds/.
'''
import numpy as np

from deltas.core.component import Component


class Bound(Component):
    '''
    one class's error curve

    fit(sample) stores the class sample (and precomputes whatever the bound
    needs); curve(b, delta) evaluates it at an array of boundaries.
    '''
    #: 'high_probability' - U(b) >= e(b) for all b at once, w.p. >= 1 - delta
    #: 'average_case'     - the expected error of a rule placing b there
    guarantee = 'high_probability'
    #: False for bounds with no confidence level (e.g. predictive curves)
    needs_delta = True
    #: False for step curves that only change at data points (counts): the
    #: data midpoints then contain every distinct value, no grid is needed
    continuous = True
    #: False for curves with flat steps anywhere (counts, the Monte Carlo
    #: band, Saw-Yang-Mo): refining the grid choice by root finding or 1-D
    #: minimisation needs a curve without flat stretches
    smooth = True
    #: True for bounds that fix delta themselves from b (the published
    #: fence); they implement resolve(b) -> {'L', 'delta', 'U'}
    self_resolving = False

    def fit(self, sample):
        self.sample = sample
        return self

    def curve(self, b, delta):
        '''values in [0, 1] at every boundary in b (and delta, broadcast)'''
        raise NotImplementedError

    def resolve(self, b):
        '''for self-resolving bounds only'''
        raise NotImplementedError


class CountBound(Bound):
    '''
    a bound that depends on b only through the count m(b) of training points
    on the wrong side (Clopper-Pearson, DKW, ...). Rank-based: flat between
    data points and floored beyond them (see the working notes' "flat" and "floor" results).

    Subclasses supply upper(m, N, delta); the rest is shared.
    '''
    continuous = False
    smooth = False
    #: above (N+1) x resolution cells, the optimised-delta table is built by
    #: ternary search instead of a dense grid; only worth it when upper() is
    #: expensive, so the default is never
    dense_table_limit = np.inf

    def __init__(self, dense_table_limit=None):
        if dense_table_limit is not None:
            self.dense_table_limit = dense_table_limit

    def upper(self, m, N, delta):
        '''the statistical core: an upper bound on the error given m of N'''
        raise NotImplementedError

    def delta_correction(self, N):
        '''divisor applied to delta to pay for choosing the threshold'''
        return 1.0

    def state(self, b):
        return self.sample.wrong_side_count(b)

    def curve(self, b, delta):
        N = self.sample.N
        return self.upper(self.state(b), N,
                          np.asarray(delta, dtype=float) / self.delta_correction(N))
