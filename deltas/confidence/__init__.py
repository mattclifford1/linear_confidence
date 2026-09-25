'''
Slot: how the confidence level delta is handled.

    fixed      FixedDelta(delta)        delta set in advance - quotable
    optimised  OptimisedDelta(...)      delta traded off against the bound,
                                        as in the published loss - a decision
                                        device, not a quotable confidence
'''
from deltas.confidence.fixed import FixedDelta, NoDeltaCurve
from deltas.confidence.optimised import OptimisedDelta

__all__ = ['FixedDelta', 'NoDeltaCurve', 'OptimisedDelta']
