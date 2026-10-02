'''
A fence is a self-resolving, average-case curve: for a boundary b it fixes
the delta that puts the class's fence at b and reports the published loss.
'''
from deltas.bounds.base import Bound


class _Fence(Bound):
    guarantee = 'average_case'
    needs_delta = False
    self_resolving = True
    continuous = True

    def curve(self, b, delta=None):
        return self.resolve(b)['U']
