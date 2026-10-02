'''
Cantelli: one-sided Chebyshev, P(Z - mu >= k sigma) <= 1/(1 + k^2), with mu
and sigma bounded from the sample. See deltas/bounds/moments/__init__.py.
'''
import numpy as np

from deltas.bounds.moments.base import _BoundedMoments
from deltas.core.registry import register


@register('bound', 'cantelli')
class Cantelli(_BoundedMoments):
    def tail(self, k):
        k = np.asarray(k, dtype=float)
        return 1.0 / (1.0 + k ** 2)
