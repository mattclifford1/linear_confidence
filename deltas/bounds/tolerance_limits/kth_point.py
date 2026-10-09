'''
The k-th furthest point tolerance limit of the non-separable draft (Dec 2024) as a
per-class curve. See deltas/bounds/tolerance_limits/__init__.py.
'''
import numpy as np

from deltas.bounds.tolerance_limits.base import _ToleranceLimit
from deltas.core.registry import register


@register('bound', 'kth_point_tolerance_limit')
class KthPointToleranceLimit(_ToleranceLimit):
    #: the loss jumps as b passes each training point's tolerance limit
    smooth = False

    def __init__(self, factor=2.0, aggregate='min', loss_form='product'):
        if aggregate not in ('min', 'max', 'mean', 'furthest'):
            raise ValueError("aggregate must be 'min', 'max', 'mean' or "
                             "'furthest'")
        if loss_form not in ('product', 'expected'):
            raise ValueError("loss_form must be 'product' or 'expected'")
        self.factor = factor
        self.aggregate = aggregate
        self.loss_form = loss_form

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        self.R = sample.radius
        # two-sided distances from the mean, ascending (as the draft)
        self.d_train = np.sort(np.abs(sample.z - sample.mean))
        self.min_conc = self.factor * self.R / np.sqrt(self.N)
        return self

    def _one(self, b):
        N, R = self.N, self.R
        d_bias = float(self.sample.facing_distance(b))
        if not R > 0 or d_bias < self.min_conc:
            return np.inf, np.nan
        comp = d_bias > self.d_train + self.min_conc
        k0 = 1 if comp.all() else N + 1 - int(np.argmin(comp))
        ks = [k0] if self.aggregate == 'furthest' else list(range(k0, N + 1)) or [k0]
        losses, deltas = [], []
        for k in ks:
            d_point = 0.0 if N - k < 0 else self.d_train[N - k]
            B = d_bias - d_point
            delta = np.exp(-np.square(B / (self.factor * R / np.sqrt(N)) - 2.0) / 2.0)
            err = k / (N + 1.0)
            losses.append(err * delta if self.loss_form == 'product'
                          else (1.0 - delta) * err + delta)
            deltas.append(delta)
        losses = np.asarray(losses)
        if self.aggregate == 'mean':
            return float(losses.mean()), float(np.mean(deltas))
        i = int(np.argmax(losses) if self.aggregate == 'max' else np.argmin(losses))
        return float(losses[i]), float(deltas[i])

    def resolve(self, b):
        b = np.atleast_1d(np.asarray(b, dtype=float))
        out = np.array([self._one(v) for v in b]).reshape(len(b), 2)
        L, delta = out[:, 0], out[:, 1]
        return {'L': L, 'delta': delta,
                'U': np.where(np.isfinite(L), np.minimum(L, 1.0), 1.0)}
