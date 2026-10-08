'''
Level 1 of the assumption ladder: bounds from the mean and spread, with no
shape assumed.

SawYangMo      Chebyshev's inequality with the mean and s.d. *estimated*
               (Saw, Yang & Mo 1984): valid for any exchangeable sample, even
               one with no finite variance. Average-case, like the published
               1/(N+1) - and it keeps that floor however far b is. Looser than
               the ranks inside the data (working notes, Fig. 4b).
Cantelli       one-sided Chebyshev, P(Z - mu >= k sigma) <= 1/(1 + k^2), with
               mu and sigma bounded from the sample.
VysochanskijPetunin
               its unimodal refinement (Mercadier & Strobel 2021),
               4 / (9 (1 + k^2)) for k^2 >= 5/3.

Cantelli and VP need confidence bounds on mu and sigma that hold without a
shape assumption. For that the score must live in a known range [lo, hi]
(squash it monotonically if it does not): empirical Bernstein for the mean,
Maurer & Pontil (2009, Thms 4 and 10) for the s.d. At N = 10 the s.d. term
alone is 0.91 of the range, so these are comparators for large classes, not
the answer for scarce ones.

    saw_yang_mo.py             SawYangMo
    base.py                    _BoundedMoments: mean and s.d. bounded via a known range
    cantelli.py                Cantelli
    vysochanskij_petunin.py    VysochanskijPetunin
'''
from deltas.bounds.moments.cantelli import Cantelli
from deltas.bounds.moments.saw_yang_mo import SawYangMo
from deltas.bounds.moments.vysochanskij_petunin import VysochanskijPetunin

__all__ = ['Cantelli', 'SawYangMo', 'VysochanskijPetunin']
