'''
Slot: which boundaries are tried.

    data_midpoints  DataMidpoints()   exact for count-based curves
    grid            Grid(n, margin)   for smooth curves
    auto            Auto(n, margin)   midpoints, plus a grid if any curve is smooth

One candidate set per file, named after its registry name; base.py holds
CandidateSet and the shared _all_values helper.
'''
from deltas.search.auto import Auto
from deltas.search.base import CandidateSet
from deltas.search.data_midpoints import DataMidpoints
from deltas.search.grid import Grid

__all__ = ['Auto', 'CandidateSet', 'DataMidpoints', 'Grid']
