'''
Minimax: minimise the larger of the two per-class curves.

The natural heir of the published constraint that the two certified regions
meet: its optimum equalises the two curves wherever that is attainable, and
since G-mean >= 1 - max(e1, e2) it guards the G-mean.
'''
import numpy as np
from scipy.optimize import brentq

from deltas.core.components import DecisionRule
from deltas.core.registry import register


def plateau_midpoint(candidates, losses, tolerance=1e-12):
    '''the midpoint, in value, of the run of candidates tied for the minimum'''
    plateau = np.flatnonzero(losses <= losses.min() + tolerance)
    return 0.5 * float(candidates[plateau[0]] + candidates[plateau[-1]])


@register('rule', 'minimax')
class Minimax(DecisionRule):
    #: losses within this of the minimum count as tied
    tie_tolerance = 1e-12

    def choose(self, candidates, L_low, L_high):
        '''
        A count-based curve is flat over long runs of candidates, so the
        arg-min is often a plateau, not a point. Every boundary on the plateau
        carries the same value, so the remaining freedom is spent on the one
        furthest from the training points at either end: the midpoint of the
        plateau *in value*. (The middle index would depend on how the points
        happen to be spaced, and breaks mirror symmetry.)
        '''
        losses = np.maximum(L_low, L_high)
        return plateau_midpoint(candidates, losses, self.tie_tolerance), losses

    def combine(self, l_low, l_high):
        return max(l_low, l_high)

    def refine(self, curve_low, curve_high, weights, bracket):
        '''
        for continuous curves, the minimax point is where they cross: find the
        root of the difference inside the bracket, if the sign changes there
        '''
        w_low, w_high = weights

        def diff(b):
            return (w_low * curve_low(np.array([b]))['L'][0] -
                    w_high * curve_high(np.array([b]))['L'][0])
        lo, hi = bracket
        f_lo, f_hi = diff(lo), diff(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return None
        if f_lo == 0:
            return float(lo)
        if f_hi == 0:
            return float(hi)
        return float(brentq(diff, lo, hi, xtol=1e-12, rtol=1e-12))
