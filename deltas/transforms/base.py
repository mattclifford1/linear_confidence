'''
Base class of the transform slot: a strictly increasing map of the score,
with its inverse. Threshold classifiers are invariant to these, so the choice
is free - but a data-dependent one must be fitted on data the certificate
never sees.
'''
from deltas.core.component import Component


class Transform(Component):
    '''a strictly increasing map of the score, with its inverse'''
    #: True when the map has parameters learnt from data. Such a transform
    #: must be fitted *before* the estimator sees the calibration data.
    requires_fit = False

    def fit(self, z, y=None):
        return self

    @property
    def is_fitted(self):
        return True

    def __call__(self, z):
        raise NotImplementedError

    def inverse(self, t):
        raise NotImplementedError
