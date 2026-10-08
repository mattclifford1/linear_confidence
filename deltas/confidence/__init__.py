'''
Slot: how the confidence level delta is handled.

    fixed      FixedDelta(delta)        delta set in advance - quotable
    optimised  OptimisedDelta(...)      delta traded off against the bound,
                                        as in the published loss - a decision
                                        device, not a quotable confidence
'''
from deltas.confidence.base import DeltaPolicy, NoDeltaCurve, PreparedCurve
from deltas.confidence.fixed import FixedDelta
from deltas.confidence.optimised import OptimisedDelta

__all__ = ['DeltaPolicy', 'FixedDelta', 'NoDeltaCurve', 'OptimisedDelta',
           'PreparedCurve']
