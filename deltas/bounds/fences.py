'''
The published fences re-expressed as per-class curves, so the rest of the
method can be held fixed while only the bound is swapped (ablations).

These are *re-expressions*, not the frozen implementations: the published
numbers come from deltas/legacy/. The search differs (a grid over b rather
than over delta_1), so boundaries agree closely but not bit for bit.

PublishedFence (ECAI 2024)
    For a boundary at b, the confidence delta(b) is the one that puts the
    class's fence exactly at b:
        d(b) = R + factor * (R / sqrt(N)) * (2 + sqrt(2 ln 1/delta))
    and the curve is the published loss (1 - delta)/(N + 1) + delta.
    Where no delta in (0, 1] reaches b the curve is infinite - so with
    rule='sum' the method is infeasible exactly when the published one is.
    Minimising the sum over b is the published constrained problem, with
    the constraint (fences meet) built into b. When the gap is wide the
    objective is nearly flat (every feasible b scores about
    1/(N1+1) + 1/(N2+1)), so the arg-min depends on the search: the legacy
    delta_1 grid stops at 1e-4, this b-sweep does not. The objective values
    agree; the boundaries can differ.

    radius='two_sided' is the published radius max|z - mean|; 'one_sided'
    uses only the points facing the boundary (the fix for the "two-sided distances" problem of the
    working notes). `factor` is the USE_TWO factor, passed explicitly.

KthPointFence (non-separable draft, Dec 2024)
    generalise from the k-th furthest training point: error k/(N+1) with the
    delta that puts that point's fence at b, aggregated over k by
    'min' / 'max' / 'mean', or only the furthest point ('furthest').
    loss_form='product' is the draft's delta * k/(N+1) (which bounds
    nothing); 'expected' is (1 - delta) k/(N+1) + delta.
'''
import numpy as np

from deltas.core.components import Bound
from deltas.core.registry import register


class _Fence(Bound):
    guarantee = 'average_case'
    needs_delta = False
    self_resolving = True
    continuous = True

    def curve(self, b, delta=None):
        return self.resolve(b)['U']


@register('bound', 'published_fence')
class PublishedFence(_Fence):
    def __init__(self, factor=2.0, radius='two_sided'):
        if radius not in ('two_sided', 'one_sided'):
            raise ValueError("radius must be 'two_sided' or 'one_sided'")
        self.factor = factor
        self.radius = radius

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        if self.radius == 'two_sided':
            self.R = sample.radius
        else:
            far = sample.z[-1] - sample.mean if sample.side == 'low' \
                else sample.mean - sample.z[0]
            self.R = float(max(far, 0.0))
        return self

    def delta_at(self, b):
        '''the delta putting this class's fence at b (nan where none does)'''
        d = self.sample.facing_distance(np.asarray(b, dtype=float))
        if not self.R > 0:
            return np.full(d.shape, np.nan)
        unit = self.factor * self.R / np.sqrt(self.N)
        q = (d - self.R) / unit - 2.0
        return np.where(q >= 0, np.exp(-0.5 * np.maximum(q, 0.0) ** 2), np.nan)

    def resolve(self, b):
        delta = self.delta_at(b)
        ok = np.isfinite(delta)
        e = 1.0 / (self.N + 1.0)
        L = np.where(ok, (1.0 - np.nan_to_num(delta)) * e + np.nan_to_num(delta),
                     np.inf)
        U = np.where(ok, np.minimum(1.0, e + np.nan_to_num(delta)), 1.0)
        return {'L': L, 'delta': delta, 'U': U}


@register('bound', 'kth_point_fence')
class KthPointFence(_Fence):
    #: the loss jumps as b passes each training point's fence
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
