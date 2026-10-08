'''
Base classes of the confidence slot.

A DeltaPolicy decides how the confidence level delta is handled, fixed in
advance or traded off against the bound as in the published loss, and turns
one fitted bound into a PreparedCurve: b -> (L, delta, U). NoDeltaCurve is
the curve both policies fall back to for bounds that carry no delta.
'''
import numpy as np

from deltas.core.component import Component


class PreparedCurve:
    '''a per-class curve with its delta handling resolved: b -> (L, delta, U)'''

    def __call__(self, b):
        '''dict of arrays over b: L (the decision curve), delta, U'''
        raise NotImplementedError


class DeltaPolicy(Component):
    def prepare(self, bound):
        '''a PreparedCurve for one fitted bound'''
        raise NotImplementedError


class NoDeltaCurve(PreparedCurve):
    '''for curves with no confidence level (average-case, predictive)'''

    def __init__(self, bound):
        self.bound = bound

    def __call__(self, b):
        if self.bound.self_resolving:
            return self.bound.resolve(b)
        U = np.asarray(self.bound.curve(b, None), dtype=float)
        return {'L': U, 'delta': np.full(U.shape, np.nan), 'U': U}
