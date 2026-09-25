'''
Slot: how the two per-class curves become one boundary.

    minimax         Minimax()               min max(L_1, L_2): equal errors,
                                            guards the G-mean
    sum             Sum()                   min L_1 + L_2: balanced error
    risk            Risk(priors)            min pi_1 c_1 L_1 + pi_2 c_2 L_2
    neyman_pearson  NeymanPearson(alpha,    min L_other s.t. L_protected <= alpha
                                  protect)
'''
from deltas.rules.balanced import Sum
from deltas.rules.minimax import Minimax, plateau_midpoint
from deltas.rules.neyman_pearson import NeymanPearson
from deltas.rules.risk import Risk

__all__ = ['Minimax', 'NeymanPearson', 'Risk', 'Sum', 'plateau_midpoint']
