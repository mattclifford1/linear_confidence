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
    location_scale        LocationScaleConfidence(family)   Monte Carlo pivots
    predictive_t          StudentTPredictive()              average-case
Published fences, for ablations:
    published_fence       PublishedFence(factor, radius)
    kth_point_fence       KthPointFence(factor, aggregate, loss_form)

See bounds/README.md for the assumption ladder.
'''
from deltas.bounds.counts import (DKW, ClopperPearson, clopper_pearson_upper,
                                  dkw_upper)
from deltas.bounds.fences import KthPointFence, PublishedFence
from deltas.bounds.location_scale import (GaussianConfidence,
                                          LocationScaleConfidence)
from deltas.bounds.moments import Cantelli, SawYangMo, VysochanskijPetunin
from deltas.bounds.predictive import StudentTPredictive

__all__ = ['Cantelli', 'ClopperPearson', 'DKW', 'GaussianConfidence',
           'KthPointFence', 'LocationScaleConfidence', 'PublishedFence',
           'SawYangMo', 'StudentTPredictive', 'VysochanskijPetunin',
           'clopper_pearson_upper', 'dkw_upper']
