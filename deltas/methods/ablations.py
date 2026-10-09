'''
The published tolerance limits as modular compositions, so one slot can be swapped at a
time. Close to, not identical with, the legacy code (deltas/bounds/tolerance_limits/__init__.py
explains why); the published numbers stay with 'Slacks Deltas' etc.
'''
from deltas.bounds import KthPointToleranceLimit, PublishedToleranceLimit
from deltas.core import DeltasEstimator
from deltas.methods.base import Method
from deltas.rules import Sum

REF = 'deltas-other-concentration working notes, "What goes wrong"'


def _tolerance_limit(bound):
    def build(clf):
        return DeltasEstimator(clf, bound=bound, rule=Sum())
    return build


METHODS = [
    Method('ECAI Tolerance Limit (modular)', _tolerance_limit(PublishedToleranceLimit(factor=2.0)),
           family='ablation', reference=REF,
           description='the published objective, no slacks, factor 2'),
    Method('ECAI Tolerance Limit (one-sided radius)',
           _tolerance_limit(PublishedToleranceLimit(factor=2.0, radius='one_sided')),
           family='ablation', reference=REF,
           description='published objective, radius facing the boundary only'),
    Method('k-th Point Tolerance Limit (min, expected loss)',
           _tolerance_limit(KthPointToleranceLimit(aggregate='min', loss_form='expected')),
           family='ablation', reference=REF,
           description='non-separable draft with (1-delta)k/(N+1) + delta'),
]
