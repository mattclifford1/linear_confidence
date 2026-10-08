'''
Logit: log(p / (1 - p)) for probability scores.
'''
import numpy as np
from scipy.special import expit, logit

from deltas.transforms.base import Transform
from deltas.core.registry import register


@register('transform', 'logit')
class Logit(Transform):
    '''
    log(p / (1 - p)) for probability scores, clipped to [eps, 1 - eps] so
    scores of exactly 0 or 1 stay finite
    '''

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, z):
        return logit(np.clip(np.asarray(z, dtype=float), self.eps, 1 - self.eps))

    def inverse(self, t):
        return expit(np.asarray(t, dtype=float))
