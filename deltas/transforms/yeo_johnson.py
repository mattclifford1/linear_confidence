'''
Yeo-Johnson: a single power transform (maximum likelihood lambda) of the
pooled scores, then standardised. Fitted, so fit it on the classifier's fit
split, never on the calibration data.
'''
import numpy as np
from scipy import stats

from deltas.transforms.base import Transform
from deltas.core.registry import register


@register('transform', 'yeo_johnson')
class YeoJohnson(Transform):
    '''
    a single Yeo-Johnson power transform (maximum likelihood lambda) of the
    pooled scores, followed by standardisation. Strictly increasing for every
    lambda, so it is a legitimate reparametrisation of the threshold.
    '''
    requires_fit = True

    def __init__(self):
        self.lambda_ = None
        self.centre_ = None
        self.scale_ = None

    def params(self):
        return {}

    def fit(self, z, y=None):
        z = np.asarray(z, dtype=float).ravel()
        t, self.lambda_ = stats.yeojohnson(z)
        self.lambda_ = float(self.lambda_)
        self.centre_ = float(np.mean(t))
        self.scale_ = float(np.std(t)) or 1.0
        return self

    @property
    def is_fitted(self):
        return self.lambda_ is not None

    def __call__(self, z):
        t = stats.yeojohnson(np.asarray(z, dtype=float), lmbda=self.lambda_)
        return (t - self.centre_) / self.scale_

    def inverse(self, t):
        y = np.asarray(t, dtype=float) * self.scale_ + self.centre_
        lam = self.lambda_
        out = np.empty_like(y)
        pos = y >= 0
        if abs(lam) > 1e-12:
            out[pos] = np.power(lam * y[pos] + 1.0, 1.0 / lam) - 1.0
        else:
            out[pos] = np.expm1(y[pos])
        if abs(lam - 2.0) > 1e-12:
            out[~pos] = 1.0 - np.power(1.0 - (2.0 - lam) * y[~pos],
                                       1.0 / (2.0 - lam))
        else:
            out[~pos] = -np.expm1(-y[~pos])
        return out
