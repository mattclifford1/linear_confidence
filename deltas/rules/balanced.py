'''
Sum: minimise L_1(b) + L_2(b), a bound on (twice) the balanced error.

With count-based curves this rule degenerates when the minority is scarce:
its curve is flat, so the majority term takes over and pushes the boundary
towards the minority (FINDINGS.md §10.4). With smooth curves it is well posed.
'''
import numpy as np
from scipy.optimize import minimize_scalar

from deltas.core.components import DecisionRule
from deltas.core.registry import register
from deltas.rules.minimax import plateau_midpoint


@register('rule', 'sum')
class Sum(DecisionRule):
    '''
    tie_break='midpoint' takes the middle (in value) of the candidates tied
    for the minimum, like minimax: symmetric under mirroring the data.
    tie_break='first' takes the first of them - what the overlap methods have
    always done, kept so 'CP Sum' / 'DKW Sum' reproduce exactly.
    '''

    def __init__(self, tie_break='midpoint'):
        if tie_break not in ('midpoint', 'first'):
            raise ValueError("tie_break must be 'midpoint' or 'first'")
        self.tie_break = tie_break

    def choose(self, candidates, L_low, L_high):
        losses = L_low + L_high
        if self.tie_break == 'first':
            return float(candidates[int(np.argmin(losses))]), losses
        return plateau_midpoint(candidates, losses), losses

    def combine(self, l_low, l_high):
        return l_low + l_high

    def refine(self, curve_low, curve_high, weights, bracket):
        w_low, w_high = weights

        def total(b):
            b = np.array([b])
            return w_low * curve_low(b)['L'][0] + w_high * curve_high(b)['L'][0]
        lo, hi = bracket
        if not hi > lo or not (np.isfinite(total(lo)) and np.isfinite(total(hi))):
            return None       # e.g. an infeasible stretch of a fence curve
        res = minimize_scalar(total, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-12})
        return float(res.x) if res.success else None
