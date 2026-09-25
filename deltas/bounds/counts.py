'''
Level 0 of the assumption ladder: bounds from counts alone.

For a boundary b, the number m(b) of class training points on the wrong side
is Binomial(N, e(b)) when the points are i.i.d. draws from the class. Both
bounds below turn m into an upper bound on the true class error e(b), assuming
nothing about the shape of the class distribution. The price (working notes,
Props 2-3): they are flat between data points and cannot certify below about
ln(1/delta)/N however far b is from the data.

Moved here from deltas/model/overlap.py unchanged; the original lives on in
deltas/legacy/overlap/ as the reference the equivalence tests compare against.
'''
import numpy as np
from scipy.stats import beta

from deltas.core.components import CountBound
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


@register('bound', 'dkw')
class DKW(CountBound):
    '''
    one-sided Dvoretzky-Kiefer-Wolfowitz bound on the empirical CDF: uniform
    over every threshold by construction, so no union bound is paid. Looser
    than Clopper-Pearson pointwise. Valid for delta <= 1/2 (Massart 1990).
    '''

    def upper(self, m, N, delta):
        return dkw_upper(m, N, delta)
