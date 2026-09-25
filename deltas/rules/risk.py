'''
Risk: minimise pi_1 c_1 L_1(b) + pi_2 c_2 L_2(b).

Decision cost is a *risk*, in which the class priors multiply the error rates
alongside the costs. Weighting the rates by the costs alone over-weights the
minority by the imbalance factor - the negative result of FINDINGS.md §13.
This rule puts the priors back.

priors: {label: pi} or (pi_0, pi_1) in label order; None uses the class
proportions of the data the deltas fit sees. Those are often *not* the
deployment priors (training sets are frequently thinned to a target ratio),
so pass the real ones when known.
'''
from deltas.core.registry import register
from deltas.rules.balanced import Sum


@register('rule', 'risk')
class Risk(Sum):
    def __init__(self, priors=None, tie_break='midpoint'):
        super().__init__(tie_break)
        self.priors = priors

    def _prior(self, data, label):
        if self.priors is None:
            return data.N[label] / float(data.N[0] + data.N[1])
        if isinstance(self.priors, dict):
            return float(self.priors[label])
        return float(self.priors[label])

    def class_weights(self, data, costs):
        c_low, c_high = super().class_weights(data, costs)
        return (c_low * self._prior(data, data.low.label),
                c_high * self._prior(data, data.high.label))
