'''
Slot: how the two per-class curves become one boundary.

    minimax   Minimax()    min max(L_1, L_2): equal errors, guards the G-mean
    sum       Sum()        min L_1 + L_2: balanced error
'''
from deltas.rules.minimax import Minimax
from deltas.rules.balanced import Sum

__all__ = ['Minimax', 'Sum']
