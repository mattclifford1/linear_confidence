'''
The published fences as modular compositions, so one slot can be swapped at a
time. Close to, not identical with, the legacy code (deltas/bounds/fences.py
explains why); the published numbers stay with 'Slacks Deltas' etc.
'''
from deltas.bounds import KthPointFence, PublishedFence
from deltas.core import DeltasEstimator
from deltas.methods.base import Method
from deltas.rules import Sum

REF = 'deltas-other-concentration working notes, "What goes wrong"'


def _fence(bound):
    def build(clf):
        return DeltasEstimator(clf, bound=bound, rule=Sum())
    return build


METHODS = [
    Method('ECAI Fence (modular)', _fence(PublishedFence(factor=2.0)),
           family='ablation', reference=REF,
           description='the published objective, no slacks, factor 2'),
    Method('ECAI Fence (one-sided radius)',
           _fence(PublishedFence(factor=2.0, radius='one_sided')),
           family='ablation', reference=REF,
           description='published objective, radius facing the boundary only'),
    Method('k-th Point Fence (min, expected loss)',
           _fence(KthPointFence(aggregate='min', loss_form='expected')),
           family='ablation', reference=REF,
           description='non-separable draft with (1-delta)k/(N+1) + delta'),
]
