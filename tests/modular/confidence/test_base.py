'''
Both policies on a continuous bound and on a bound with no confidence level (NoDeltaCurve).
'''
import numpy as np

from deltas.confidence import FixedDelta, OptimisedDelta
from deltas.core import ClassSample


def test_policies_on_continuous_and_delta_free_bounds(smooth_bound, no_delta_bound):
    s = ClassSample(np.random.default_rng(0).normal(0, 1, 50), 0, 'low')
    b = np.linspace(-1, 3, 7)
    smooth = smooth_bound().fit(s)
    opt = OptimisedDelta(resolution=50).prepare(smooth)(b)
    assert opt['L'].shape == b.shape and np.all(np.isfinite(opt['L']))
    nd = no_delta_bound().fit(s)
    for policy in (FixedDelta(), OptimisedDelta(resolution=50)):
        out = policy.prepare(nd)(b)
        assert np.all(np.isnan(out['delta']))
        assert np.array_equal(out['L'], out['U'])
