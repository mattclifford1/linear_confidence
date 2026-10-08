'''
Fixed delta: the confidence level is set in advance.

This is the setting under which a high-probability bound can be *quoted*: with
probability at least 1 - delta, every class error is below its curve at every
boundary at once, including the one the method chose (working notes, the "uniform for free" lemma).
'''
import numpy as np

from deltas.confidence.base import DeltaPolicy, NoDeltaCurve, PreparedCurve
from deltas.core.registry import register


@register('confidence', 'fixed')
class FixedDelta(DeltaPolicy):
    def __init__(self, delta=0.05):
        if not 0.0 < delta < 1.0:
            raise ValueError('delta must be in (0, 1)')
        self.delta = delta

    def prepare(self, bound):
        if not bound.needs_delta:
            return NoDeltaCurve(bound)
        return _FixedCurve(bound, self.delta)


class _FixedCurve(PreparedCurve):
    def __init__(self, bound, delta):
        self.bound = bound
        self.delta = delta

    def __call__(self, b):
        U = np.asarray(self.bound.curve(b, self.delta), dtype=float)
        return {'L': U, 'delta': np.full(U.shape, self.delta), 'U': U}
