'''
Optimised delta: trade the confidence level off against the bound.

For each boundary and class, choose the delta minimising

    form='expected'  (1 - delta) U(b; delta) + delta     (the published loss)
    form='union'     U(b; delta) + delta                  (unconditional form)

Both read "with probability 1 - delta the error is at most U, otherwise at
most 1" as a bound on the *expected* error (working notes, Sec. 3.3); the
first is exact when the error on the good event is at most U. This is a good
*decision* device. The delta it picks is chosen from the data, so it should
not be quoted as a confidence; certify at a fixed delta instead.

For count bounds the curve depends on b only through m, so the minimisation is
done once per possible m and read off a table. That is what makes the overlap
methods fast; the code is the overlap methods', moved here unchanged.
'''
import numpy as np

from deltas.core.components import CountBound, DeltaPolicy, PreparedCurve
from deltas.confidence.fixed import NoDeltaCurve
from deltas.core.registry import register


@register('confidence', 'optimised')
class OptimisedDelta(DeltaPolicy):
    def __init__(self, resolution=2000, form='expected'):
        if form not in ('expected', 'union'):
            raise ValueError("form must be 'expected' or 'union'")
        self.resolution = resolution
        self.form = form

    def deltas(self):
        '''the delta grid, excluding the degenerate end points'''
        r = self.resolution
        return np.linspace(1.0 / r, 1.0 - 1.0 / r, r)

    def _loss(self, bound_values, deltas):
        if self.form == 'expected':
            return (1.0 - deltas) * bound_values + deltas
        return bound_values + deltas

    # ------------------------------------------------------ count bounds ---
    def table(self, bound, N, dense_table_limit=None):
        '''
        for every possible m in 0..N, minimise the loss over delta. Returns the
        minimised loss, the argmin delta and the bound value at that delta.
        '''
        if dense_table_limit is None:
            dense_table_limit = bound.dense_table_limit
        deltas = self.deltas()
        eff = deltas / bound.delta_correction(N)      # union bound correction
        ms = np.arange(N + 1)

        if (N + 1) * self.resolution > dense_table_limit:
            return self._sparse_table(bound, ms, deltas, eff, N)

        # (N+1, resolution) grids
        bounds = bound.upper(ms[:, None], N, eff[None, :])
        # the delta paid in the loss is the *nominal* per-class confidence
        L = self._loss(bounds, deltas[None, :])
        idx = np.argmin(L, axis=1)
        return {'loss': L[ms, idx],
                'delta': deltas[idx],
                'bound': bounds[ms, idx]}

    def _sparse_table(self, bound, ms, deltas, eff, N):
        '''
        same table, found by ternary search instead of evaluating every cell

        The dense path costs (N+1) x resolution evaluations of the bound, which
        is 10^8 Beta quantiles at N = 50k and takes minutes. The loss is
        unimodal in delta at fixed m - it tends to 1 at both ends, since a
        vanishing delta buys a vacuous bound and a delta near 1 pays for itself
        - so a ternary search over grid *indices* finds the same grid minimum
        in ~40 evaluations per m rather than 2000.

        Returns the same losses as the dense path; only the delta reported at
        an exact tie may differ, and tied deltas give the same loss.

        Only safe where the loss is *strictly* unimodal: not for DKW, whose
        bound clips at 1 for small delta, leaving a flat plateau a ternary
        search can step across (DKW keeps an infinite dense_table_limit).
        '''
        def loss_at(idx):
            idx = np.clip(idx, 0, len(deltas) - 1)
            b = bound.upper(ms, N, eff[idx])
            return self._loss(b, deltas[idx]), b

        lo = np.zeros(len(ms), dtype=int)
        hi = np.full(len(ms), len(deltas) - 1, dtype=int)
        while np.any(hi - lo > 2):
            third = np.maximum((hi - lo) // 3, 1)
            m1, m2 = lo + third, hi - third
            L1, _ = loss_at(m1)
            L2, _ = loss_at(m2)
            take_left = L1 <= L2
            hi = np.where(take_left, np.maximum(m2 - 1, lo), hi)
            lo = np.where(take_left, lo, np.minimum(m1 + 1, hi))

        # settle the last few indices exactly, keeping the lowest index on ties
        # so the tie-breaking matches np.argmin on the dense grid
        best_L = np.full(len(ms), np.inf)
        best_d = np.zeros(len(ms))
        best_b = np.zeros(len(ms))
        for offset in range(0, 3):
            idx = np.clip(lo + offset, 0, hi)
            L, b = loss_at(idx)
            better = L < best_L
            best_L = np.where(better, L, best_L)
            best_d = np.where(better, deltas[idx], best_d)
            best_b = np.where(better, b, best_b)
        return {'loss': best_L, 'delta': best_d, 'bound': best_b}

    # ------------------------------------------------------------ prepare ---
    def prepare(self, bound):
        if not bound.needs_delta:
            return NoDeltaCurve(bound)
        if isinstance(bound, CountBound):
            return _TableCurve(bound, self.table(bound, bound.sample.N))
        return _GridCurve(bound, self)


class _TableCurve(PreparedCurve):
    '''a count bound with delta optimised per m: a lookup'''

    def __init__(self, bound, table):
        self.bound = bound
        self.table = table

    def __call__(self, b):
        m = self.bound.state(b)
        return {'L': self.table['loss'][m],
                'delta': self.table['delta'][m],
                'U': self.table['bound'][m]}


class _GridCurve(PreparedCurve):
    '''a continuous bound with delta optimised on the grid, per boundary'''

    #: boundaries per chunk, to bound memory at (chunk x resolution)
    chunk = 512

    def __init__(self, bound, policy):
        self.bound = bound
        self.policy = policy

    def __call__(self, b):
        b = np.atleast_1d(np.asarray(b, dtype=float))
        deltas = self.policy.deltas()
        out = {k: np.empty(len(b)) for k in ('L', 'delta', 'U')}
        for s in range(0, len(b), self.chunk):
            bb = b[s:s + self.chunk]
            U = self.bound.curve(bb[:, None], deltas[None, :])
            L = self.policy._loss(U, deltas[None, :])
            idx = np.argmin(L, axis=1)
            rows = np.arange(len(bb))
            out['L'][s:s + len(bb)] = L[rows, idx]
            out['delta'][s:s + len(bb)] = deltas[idx]
            out['U'][s:s + len(bb)] = U[rows, idx]
        return out
