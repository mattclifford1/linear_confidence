'''
Slot: a strictly increasing map of the score.

A threshold rule does not see these: {z > b} = {g(z) > g(b)}, so every count
and every error is unchanged. What changes is the *shape* the class
distributions have, which matters to any bound that assumes one (the
location-scale envelopes). So the choice is free for the classifier and
consequential for the bound.

A transform with parameters learnt from data (Yeo-Johnson) must be fitted on
data the certificate never sees - the classifier's fit split - for the same
reason the calibration split exists. The estimator refuses an unfitted one.
'''
import numpy as np
from scipy import stats
from scipy.special import expit, logit

from deltas.core.components import Transform
from deltas.core.registry import register


@register('transform', 'identity')
class Identity(Transform):
    def __call__(self, z):
        return np.asarray(z, dtype=float)

    def inverse(self, t):
        return np.asarray(t, dtype=float)


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
