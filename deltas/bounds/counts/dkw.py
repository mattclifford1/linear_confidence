'''
DKW: the one-sided Dvoretzky-Kiefer-Wolfowitz bound on the empirical CDF,
uniform over every threshold by construction. Level 0 of the assumption
ladder, see deltas/bounds/counts/__init__.py.
'''
import numpy as np

from deltas.bounds.base import CountBound
from deltas.core.registry import register


def dkw_upper(m, N, delta, two_sided=False):
    '''
    one-sided DKW upper bound, uniform over all thresholds
        e <= m/N + sqrt(ln(1/delta) / (2N))
    two_sided=True uses ln(2/delta) (only needed if both tails are used)
    '''
    m = np.asarray(m, dtype=float)
    delta = np.asarray(delta, dtype=float)
    inside = np.log((2.0 if two_sided else 1.0) / delta) / (2.0 * N)
    return np.clip(m / N + np.sqrt(inside), 0.0, 1.0)


@register('bound', 'dkw')
class DKW(CountBound):
    '''
    one-sided Dvoretzky-Kiefer-Wolfowitz bound on the empirical CDF: uniform
    over every threshold by construction, so no union bound is paid. Looser
    than Clopper-Pearson pointwise. Valid for delta <= 1/2 (Massart 1990).
    '''

    def upper(self, m, N, delta):
        return dkw_upper(m, N, delta)
