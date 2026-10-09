'''
The published tolerance limits re-expressed as per-class curves, so the rest of the
method can be held fixed while only the bound is swapped (ablations).

These are *re-expressions*, not the frozen implementations: the published
numbers come from deltas/legacy/. The search differs (a grid over b rather
than over delta_1), so boundaries agree closely but not bit for bit.

PublishedToleranceLimit (ECAI 2024)
    For a boundary at b, the confidence delta(b) is the one that puts the
    class's tolerance limit exactly at b:
        d(b) = R + factor * (R / sqrt(N)) * (2 + sqrt(2 ln 1/delta))
    and the curve is the published loss (1 - delta)/(N + 1) + delta.
    Where no delta in (0, 1] reaches b the curve is infinite - so with
    rule='sum' the method is infeasible exactly when the published one is.
    Minimising the sum over b is the published constrained problem, with
    the constraint (tolerance limits meet) built into b. When the gap is wide the
    objective is nearly flat (every feasible b scores about
    1/(N1+1) + 1/(N2+1)), so the arg-min depends on the search: the legacy
    delta_1 grid stops at 1e-4, this b-sweep does not. The objective values
    agree; the boundaries can differ.

    radius='two_sided' is the published radius max|z - mean|; 'one_sided'
    uses only the points facing the boundary (the fix for the "two-sided distances" problem of the
    working notes). `factor` is the USE_TWO factor, passed explicitly.

KthPointToleranceLimit (non-separable draft, Dec 2024)
    generalise from the k-th furthest training point: error k/(N+1) with the
    delta that puts that point's tolerance limit at b, aggregated over k by
    'min' / 'max' / 'mean', or only the furthest point ('furthest').
    loss_form='product' is the draft's delta * k/(N+1) (which bounds
    nothing); 'expected' is (1 - delta) k/(N+1) + delta.

    base.py        _ToleranceLimit
    published.py   PublishedToleranceLimit (registry name 'published_tolerance_limit')
    kth_point.py   KthPointToleranceLimit (registry name 'kth_point_tolerance_limit')
'''
from deltas.bounds.tolerance_limits.kth_point import KthPointToleranceLimit
from deltas.bounds.tolerance_limits.published import PublishedToleranceLimit

__all__ = ['KthPointToleranceLimit', 'PublishedToleranceLimit']
