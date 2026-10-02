'''
Vysochanskij-Petunin: the unimodal refinement of Cantelli (Mercadier &
Strobel 2021). See deltas/bounds/moments/__init__.py.
'''
import numpy as np

from deltas.bounds.moments.base import _BoundedMoments
from deltas.core.registry import register


@register('bound', 'vysochanskij_petunin')
class VysochanskijPetunin(_BoundedMoments):
    '''one-sided VP: assumes the class distribution is unimodal'''

    def tail(self, k):
        k2 = np.asarray(k, dtype=float) ** 2
        return np.where(k2 >= 5.0 / 3.0, 4.0 / (9.0 * (1.0 + k2)),
                        4.0 / (3.0 * (1.0 + k2)) - 1.0 / 3.0)
