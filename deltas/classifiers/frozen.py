'''
A classifier stand-in for projections that were computed elsewhere.

Every deltas estimator needs exactly one thing from a classifier -
`get_projection(X) -> (n, 1)` - and passes 1-D input straight through. So a
projection computed in another process (see experiments/export_projections.py,
which fits models under the sibling projection_models environment) can be fed
to deltas directly, provided something with the right interface is there to
satisfy the constructor checks.

That is all this is: the identity on an already-projected column, plus the
decision threshold that the original model used, so the untouched model's own
predictions stay available as the `Baseline`.

    z  = exported projections, shape (n,)
    clf = FrozenProjection(threshold)
    model = overlap.binomial_deltas(clf).fit(z[:, None], y)
'''
import numpy as np


class FrozenProjection:
    '''identity projector over pre-computed 1-D projections'''

    def __init__(self, threshold=0.0, name=None):
        self.threshold = float(threshold)
        self.name = name

    def fit(self, X, y=None):
        '''nothing to fit - the projection is already fixed'''
        return self

    def get_projection(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:, None]
        if X.shape[1] != 1:
            raise ValueError(
                f'FrozenProjection expects an already-projected (n, 1) array, '
                f'got shape {X.shape}. The projection has to be computed by '
                f'the model that produced it.')
        return X

    def get_bias(self):
        '''deltas stores the boundary as a bias, i.e. negated'''
        return -self.threshold

    def predict(self, X):
        z = self.get_projection(X).squeeze(axis=-1)
        return (z > self.threshold).astype(int)

    def __repr__(self):
        return (f'FrozenProjection(threshold={self.threshold:.4g}'
                f'{"" if self.name is None else f", name={self.name!r}"})')
