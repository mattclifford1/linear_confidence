'''
Level 0 of the assumption ladder: bounds from counts alone.

For a boundary b, the number m(b) of class training points on the wrong side
is Binomial(N, e(b)) when the points are i.i.d. draws from the class. Both
bounds below turn m into an upper bound on the true class error e(b), assuming
nothing about the shape of the class distribution. The price (the working
notes' "flat" and "floor" results): they are flat between data points and
cannot certify below about
ln(1/delta)/N however far b is from the data.

The code came from deltas/model/overlap.py unchanged; the original lives on in
deltas/legacy/overlap/ as the reference the equivalence tests compare against.

    clopper_pearson.py   ClopperPearson, clopper_pearson_upper
    dkw.py               DKW, dkw_upper
'''
from deltas.bounds.counts.clopper_pearson import ClopperPearson, clopper_pearson_upper
from deltas.bounds.counts.dkw import DKW, dkw_upper

__all__ = ['ClopperPearson', 'DKW', 'clopper_pearson_upper', 'dkw_upper']
