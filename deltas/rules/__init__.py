'''
Slot: how the two per-class curves become one boundary.

    minimax         Minimax()               min max(L_1, L_2): equal errors,
                                            guards the G-mean
    sum             Sum()                   min L_1 + L_2: balanced error
    risk            Risk(priors)            min pi_1 c_1 L_1 + pi_2 c_2 L_2
    neyman_pearson  NeymanPearson(alpha,    min L_other s.t. L_protected <= alpha
                                  protect)

One rule per file, named after its registry name; base.py holds DecisionRule
and the shared plateau_midpoint tie-break.
'''
from deltas.rules.base import DecisionRule, plateau_midpoint
from deltas.rules.minimax import Minimax
from deltas.rules.neyman_pearson import NeymanPearson
from deltas.rules.risk import Risk
from deltas.rules.sum import Sum

__all__ = ['DecisionRule', 'Minimax', 'NeymanPearson', 'Risk', 'Sum',
           'plateau_midpoint']
