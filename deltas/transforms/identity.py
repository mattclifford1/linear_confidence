'''
Identity: the score as it is.
'''
import numpy as np

from deltas.transforms.base import Transform
from deltas.core.registry import register


@register('transform', 'identity')
class Identity(Transform):
    def __call__(self, z):
        return np.asarray(z, dtype=float)

    def inverse(self, t):
        return np.asarray(t, dtype=float)
