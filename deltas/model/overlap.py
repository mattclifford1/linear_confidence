'''
Overlap-native deltas (Clopper-Pearson / DKW) - now a thin shim over the
modular DeltasEstimator.

The classes keep their names, signatures, attributes and exact numbers; they
are fixed compositions of components:

    binomial_deltas(objective=...)  = ClopperPearson + OptimisedDelta + rule
    dkw_deltas(objective=...)       = DKW + OptimisedDelta + rule
    (search: data midpoints; certificate: the deciding curves)

Bit-for-bit equality with the original implementation is checked against the
frozen copy in deltas/legacy/overlap/overlap.py by
tests/modular/test_overlap_equivalence.py, and pinned by tests/golden/.

For new work, build the composition directly:

    from deltas.core import DeltasEstimator
    DeltasEstimator(clf, bound='clopper_pearson', rule='minimax')

Validity caveat (unchanged): the counts need the projected points to be i.i.d.
draws from the class; they are not when the classifier was fitted to them, so
compute the certificate on a calibration split (CALIBRATION.md).
'''
import numpy as np

from deltas.bounds.counts import (DKW, ClopperPearson, clopper_pearson_upper,
                                  dkw_upper)
from deltas.confidence.optimised import OptimisedDelta
from deltas.core.estimator import DeltasEstimator
from deltas.rules import Minimax, Sum
from deltas.search.candidates import DataMidpoints

__all__ = ['base_overlap_deltas', 'binomial_deltas', 'dkw_deltas',
           'clopper_pearson_upper', 'dkw_upper']


class base_overlap_deltas(DeltasEstimator):
    '''
    sweep the boundary over every distinct split of the projected training data
    and pick the minimiser of the certified loss
    '''
    #: set by subclasses
    bound_name = None
    #: see OptimisedDelta.table - the size above which the table is searched
    dense_table_limit = np.inf

    def __init__(self, clf=None, dim_reducer=None, objective='sum',
                 union_bound=True, delta_resolution=2000, dev=False):
        '''
            objective:        'sum'     minimise c_1 L_1 + c_2 L_2
                              'minimax' minimise max_i c_i L_i (ties broken
                                        towards the middle of the plateau)
            union_bound:      correct delta_i for the number of candidate
                              thresholds (Clopper-Pearson only - DKW is
                              already uniform)
            delta_resolution: grid size for the per-class delta_i search
        '''
        if clf is not None and not hasattr(clf, 'get_projection'):
            raise AttributeError(
                f"Classifier {clf} needs 'get_projection' method")
        if objective not in ('sum', 'minimax'):
            raise ValueError("objective must be 'sum' or 'minimax'")
        self.clf = clf
        self.dim_reducer = dim_reducer
        self.objective = objective
        self.union_bound = union_bound
        self.delta_resolution = delta_resolution
        self.dev = dev
        self.is_fit = False
        self.delta_report = 0.05

    def _make_bound(self):
        raise NotImplementedError

    def _components(self):
        # the original sum rule takes the first of tied minima
        rule = Sum(tie_break='first') if self.objective == 'sum' else Minimax()
        return {'bound': self._make_bound(),
                'confidence': OptimisedDelta(resolution=self.delta_resolution),
                'rule': rule,
                'search': DataMidpoints(),
                'transform': None,
                'certify': 'decision',
                'refine': False}

    # -- the original private API, kept for callers and tests --------------
    def _upper_bound(self, m, N, delta):
        return self._make_bound().upper(m, N, delta)

    def _delta_correction(self, N):
        return self._make_bound().delta_correction(N)

    def _per_class_loss_table(self, N):
        return OptimisedDelta(resolution=self.delta_resolution).table(
            self._make_bound(), N, dense_table_limit=self.dense_table_limit)


class binomial_deltas(base_overlap_deltas):
    '''Clopper-Pearson (exact binomial) upper bound on the class error'''
    bound_name = 'Clopper-Pearson'
    dense_table_limit = ClopperPearson.dense_table_limit

    def _make_bound(self):
        return ClopperPearson(union_bound=self.union_bound,
                              dense_table_limit=self.dense_table_limit)


class dkw_deltas(base_overlap_deltas):
    '''DKW empirical-CDF bound - already uniform over every threshold'''
    bound_name = 'DKW'

    def _make_bound(self):
        return DKW(dense_table_limit=self.dense_table_limit)
