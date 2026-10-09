'''
The published (ECAI 2024) tolerance limit as a per-class curve. See
deltas/bounds/tolerance_limits/__init__.py.
'''
import numpy as np

from deltas.bounds.tolerance_limits.base import _ToleranceLimit
from deltas.core.registry import register


@register('bound', 'published_tolerance_limit')
class PublishedToleranceLimit(_ToleranceLimit):
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
        '''the delta putting this class's tolerance limit at b (nan where none does)'''
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
