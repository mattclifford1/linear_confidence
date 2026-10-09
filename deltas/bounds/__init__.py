'''
Slot: per-class error curves - the concentration inequalities.

Level 0, counts only (rank-based: flat between points, floored beyond them):
    clopper_pearson       ClopperPearson(union_bound=True)
    dkw                   DKW()
Level 1, moments without a shape:
    saw_yang_mo           SawYangMo()                       average-case
    cantelli              Cantelli(score_range)
    vysochanskij_petunin  VysochanskijPetunin(score_range)  unimodal
Level 2, a location-scale shape:
    gaussian              GaussianConfidence(band)          exact pivots
    location_scale_mc     LocationScaleConfidence(family)   Monte Carlo pivots
    predictive_t          StudentTPredictive()              average-case
Published tolerance limits, for ablations:
    published_tolerance_limit       PublishedToleranceLimit(factor, radius)
    kth_point_tolerance_limit       KthPointToleranceLimit(factor, aggregate, loss_form)

One public class per file: counts/, moments/, location_scale/, predictive/
and tolerance_limits/ are packages, base.py holds Bound and CountBound. See
bounds/README.md for the assumption ladder.
'''
from deltas.bounds.base import Bound, CountBound
from deltas.bounds.counts import (DKW, ClopperPearson, clopper_pearson_upper,
                                  dkw_upper)
from deltas.bounds.tolerance_limits import KthPointToleranceLimit, PublishedToleranceLimit
from deltas.bounds.location_scale import (GaussianConfidence,
                                          LocationScaleConfidence)
from deltas.bounds.moments import Cantelli, SawYangMo, VysochanskijPetunin
from deltas.bounds.predictive import StudentTPredictive

__all__ = ['Bound', 'Cantelli', 'ClopperPearson', 'CountBound', 'DKW',
           'GaussianConfidence',
           'KthPointToleranceLimit', 'LocationScaleConfidence', 'PublishedToleranceLimit',
           'SawYangMo', 'StudentTPredictive', 'VysochanskijPetunin',
           'clopper_pearson_upper', 'dkw_upper']
