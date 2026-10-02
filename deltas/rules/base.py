'''
Base class of the rule slot: how the two per-class curves become one
boundary. A rule may see the data first (bind) and set the per-class weights
(class_weights); it gets a private copy per fit. plateau_midpoint is the
tie-break the count-based rules share.
'''
import numpy as np

from deltas.core.component import Component


class DecisionRule(Component):
    def bind(self, data):
        '''
        see the data before choosing (e.g. which class is protected); called on
        a private copy, so it may store state. Returns the rule.
        '''
        return self

    def class_weights(self, data, costs):
        '''(w_low, w_high) multiplying the two curves; costs are by label'''
        c_low = costs[0] if data.low.label == 0 else costs[1]
        c_high = costs[0] if data.high.label == 0 else costs[1]
        return c_low, c_high

    def choose(self, candidates, L_low, L_high):
        '''(boundary, losses over the candidates)'''
        raise NotImplementedError

    def combine(self, l_low, l_high):
        '''the objective value of one boundary'''
        raise NotImplementedError

    def refine(self, curve_low, curve_high, weights, bracket):
        '''
        optional: polish the chosen boundary within `bracket` when both curves
        are continuous. Returns a boundary, or None to keep the grid choice.
        '''
        return None


def plateau_midpoint(candidates, losses, tolerance=1e-12):
    '''the midpoint, in value, of the run of candidates tied for the minimum'''
    plateau = np.flatnonzero(losses <= losses.min() + tolerance)
    return 0.5 * float(candidates[plateau[0]] + candidates[plateau[-1]])
