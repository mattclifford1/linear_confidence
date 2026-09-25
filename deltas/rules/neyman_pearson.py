'''
Neyman-Pearson: minimise the error of one class subject to the other's
curve staying at or below alpha.

The rule for "miss at most alpha of the minority, with confidence 1 - delta"
- the guaranteed-protection use case of the working notes (Sec. 6.1). With a
count-based bound on held-out data this is the umbrella algorithm of Tong,
Feng & Li (2018); with a location-scale bound it is the parametric version
of Tong et al. (2020).

protect: 'minority' (the smaller class; label 1 on a tie) or a class label.
Costs are ignored: alpha is compared with the protected class's own curve.
If no candidate meets the constraint, the boundary that comes closest is
returned and `rule_info_['constraint_met']` is False on the estimator.
'''
import numpy as np

from deltas.core.components import DecisionRule
from deltas.core.registry import register
from deltas.rules.minimax import plateau_midpoint


@register('rule', 'neyman_pearson')
class NeymanPearson(DecisionRule):
    def __init__(self, alpha=0.1, protect='minority'):
        if not 0.0 < alpha < 1.0:
            raise ValueError('alpha must be in (0, 1)')
        self.alpha = alpha
        self.protect = protect

    def bind(self, data):
        if self.protect == 'minority':
            label = 0 if data.N[0] < data.N[1] else 1
        else:
            label = self.protect
        self._protected_low = data.low.label == label
        self.info_ = {'protected_label': label}
        return self

    def class_weights(self, data, costs):
        return 1.0, 1.0

    def _split(self, L_low, L_high):
        return (L_low, L_high) if self._protected_low else (L_high, L_low)

    def choose(self, candidates, L_low, L_high):
        L_prot, L_other = self._split(L_low, L_high)
        feasible = L_prot <= self.alpha
        self.info_['constraint_met'] = bool(feasible.any())
        if feasible.any():
            losses = np.where(feasible, L_other, np.inf)
        else:
            losses = L_prot
        return plateau_midpoint(candidates, losses), losses

    def combine(self, l_low, l_high):
        l_prot, l_other = (l_low, l_high) if self._protected_low \
            else (l_high, l_low)
        return l_other if self.info_.get('constraint_met', True) else l_prot
