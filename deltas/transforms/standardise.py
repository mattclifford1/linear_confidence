'''
Standardise: (z - centre) / scale, fitted. Fit it on the classifier's fit
split, never on the calibration data.
'''
import numpy as np

from deltas.transforms.base import Transform
from deltas.core.registry import register


@register('transform', 'standardise')
class Standardise(Transform):
    '''(z - centre) / scale; fitted, so fit it on the fit split'''
    requires_fit = True

    def __init__(self):
        self.centre_ = None
        self.scale_ = None

    def params(self):
        return {}

    def fit(self, z, y=None):
        z = np.asarray(z, dtype=float).ravel()
        self.centre_ = float(np.mean(z))
        self.scale_ = float(np.std(z)) or 1.0
        return self

    @property
    def is_fitted(self):
        return self.centre_ is not None

    def __call__(self, z):
        return (np.asarray(z, dtype=float) - self.centre_) / self.scale_

    def inverse(self, t):
        return np.asarray(t, dtype=float) * self.scale_ + self.centre_
