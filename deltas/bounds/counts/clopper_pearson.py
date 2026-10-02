'''
Clopper-Pearson: the exact binomial upper limit on the class error, from the
count of training points on the wrong side of b. Level 0 of the assumption
ladder, see deltas/bounds/counts/__init__.py.
'''
import numpy as np
from scipy.stats import beta

from deltas.bounds.base import CountBound
from deltas.core.registry import register


def clopper_pearson_upper(m, N, delta):
    '''
    exact one-sided upper confidence limit on a binomial proportion
        P(e <= BetaInv(1 - delta; m + 1, N - m)) >= 1 - delta
    m may be an array; delta a scalar or matching array
    '''
    m = np.asarray(m, dtype=float)
    N = float(N)
    delta = np.asarray(delta, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ub = beta.ppf(1.0 - delta, m + 1.0, N - m)
    # m == N -> no information, error could be 1; also guard nan from ppf
    ub = np.where(m >= N, 1.0, ub)
    ub = np.where(np.isnan(ub), 1.0, ub)
    return np.clip(ub, 0.0, 1.0)


@register('bound', 'clopper_pearson')
class ClopperPearson(CountBound):
    '''
    exact binomial (Clopper-Pearson) upper limit on the class error

    Pointwise in b, so with union_bound=True delta is divided by the N + 1
    distinct values m can take as b sweeps the line - the guarantee then holds
    at whichever boundary is selected.
    '''
    #: each value is a Beta quantile, so a dense (N+1) x resolution table is
    #: 10^8 evaluations at N = 50k; above this size, search instead
    dense_table_limit = 5_000_000

    def __init__(self, union_bound=True, dense_table_limit=None):
        super().__init__(dense_table_limit)
        self.union_bound = union_bound

    def upper(self, m, N, delta):
        return clopper_pearson_upper(m, N, delta)

    def delta_correction(self, N):
        return (N + 1.0) if self.union_bound else 1.0
