'''
The published fences re-expressed as per-class curves, so the rest of the
method can be held fixed while only the bound is swapped (ablations).

These are *re-expressions*, not the frozen implementations: the published
numbers come from deltas/legacy/. The search differs (a grid over b rather
than over delta_1), so boundaries agree closely but not bit for bit.

PublishedFence (ECAI 2024)
    For a boundary at b, the confidence delta(b) is the one that puts the
    class's fence exactly at b:
        d(b) = R + factor * (R / sqrt(N)) * (2 + sqrt(2 ln 1/delta))
    and the curve is the published loss (1 - delta)/(N + 1) + delta.
    Where no delta in (0, 1] reaches b the curve is infinite - so with
    rule='sum' the method is infeasible exactly when the published one is.
    Minimising the sum over b is the published constrained problem, with
    the constraint (fences meet) built into b. When the gap is wide the
    objective is nearly flat (every feasible b scores about
    1/(N1+1) + 1/(N2+1)), so the arg-min depends on the search: the legacy
    delta_1 grid stops at 1e-4, this b-sweep does not. The objective values
    agree; the boundaries can differ.

    radius='two_sided' is the published radius max|z - mean|; 'one_sided'
    uses only the points facing the boundary (the fix for the "two-sided distances" problem of the
    working notes). `factor` is the USE_TWO factor, passed explicitly.

KthPointFence (non-separable draft, Dec 2024)
    generalise from the k-th furthest training point: error k/(N+1) with the
    delta that puts that point's fence at b, aggregated over k by
    'min' / 'max' / 'mean', or only the furthest point ('furthest').
    loss_form='product' is the draft's delta * k/(N+1) (which bounds
    nothing); 'expected' is (1 - delta) k/(N+1) + delta.

    base.py        _Fence
    published.py   PublishedFence (registry name 'published_fence')
    kth_point.py   KthPointFence (registry name 'kth_point_fence')
'''
from deltas.bounds.fences.kth_point import KthPointFence
from deltas.bounds.fences.published import PublishedFence

__all__ = ['KthPointFence', 'PublishedFence']
