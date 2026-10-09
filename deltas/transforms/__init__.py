'''
Slot: strictly increasing maps of the score (free for a threshold rule,
consequential for any bound that assumes a shape).

    identity      Identity()
    logit         Logit(eps)        for probability scores
    standardise   Standardise()     fitted - fit it on the fit split
    yeo_johnson   YeoJohnson()      fitted - fit it on the fit split

A threshold rule does not see these: {z > b} = {g(z) > g(b)}, so every count
and every error is unchanged. What changes is the *shape* the class
distributions have, which matters to any bound that assumes one (the
location-scale bands). So the choice is free for the classifier and
consequential for the bound.

A transform with parameters learnt from data (Yeo-Johnson) must be fitted on
data the certificate never sees - the classifier's fit split - for the same
reason the calibration split exists. The estimator refuses an unfitted one.

One transform per file, named after its registry name; base.py holds
Transform.
'''
from deltas.transforms.base import Transform
from deltas.transforms.identity import Identity
from deltas.transforms.logit import Logit
from deltas.transforms.standardise import Standardise
from deltas.transforms.yeo_johnson import YeoJohnson

__all__ = ['Identity', 'Logit', 'Standardise', 'Transform', 'YeoJohnson']
