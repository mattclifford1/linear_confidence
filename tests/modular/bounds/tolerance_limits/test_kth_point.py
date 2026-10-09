'''
The k-th point tolerance limit as a component tracks the non-separable draft.
'''
import pytest

from deltas.bounds import KthPointToleranceLimit
from deltas.classifiers.frozen import FrozenProjection
from deltas.core import DeltasEstimator


@pytest.mark.parametrize('aggregate,loss_type', [('min', 'min'),
                                                  ('furthest', 'furthest')])
def test_kth_point_tolerance_limit_tracks_the_draft(golden_xy, aggregate, loss_type):
    from deltas.model import non_sep
    X, y, thr = golden_xy('gauss_wide')
    kw = {'only_furtherest_k': True} if loss_type == 'furthest' else \
        {'loss_type': loss_type}
    legacy = non_sep.deltas(FrozenProjection(thr)).fit(X, y, **kw)
    modular = DeltasEstimator(bound=KthPointToleranceLimit(aggregate=aggregate),
                              rule='sum').fit(X, y)
    assert abs(modular.boundary - legacy.boundary) < 0.05
