'''
Slot: which boundaries are tried.

    data_midpoints  DataMidpoints()   exact for count-based curves
    grid            Grid(n, margin)   for smooth curves
    auto            Auto(n, margin)   midpoints, plus a grid if any curve is smooth
'''
from deltas.search.candidates import Auto, DataMidpoints, Grid

__all__ = ['Auto', 'DataMidpoints', 'Grid']
