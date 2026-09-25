'''
Slot: per-class error curves - the concentration inequalities.

Level 0, counts only (rank-based; flat between points, floored beyond them):
    clopper_pearson   ClopperPearson(union_bound=True)
    dkw               DKW()

See bounds/README.md for the assumption ladder.
'''
from deltas.bounds.counts import (DKW, ClopperPearson, clopper_pearson_upper,
                                  dkw_upper)

__all__ = ['ClopperPearson', 'DKW', 'clopper_pearson_upper', 'dkw_upper']
