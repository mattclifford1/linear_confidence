'''
Slot: strictly increasing maps of the score (free for a threshold rule,
consequential for any bound that assumes a shape).

    identity      Identity()
    logit         Logit(eps)        for probability scores
    standardise   Standardise()     fitted - fit it on the fit split
    yeo_johnson   YeoJohnson()      fitted - fit it on the fit split
'''
from deltas.transforms.monotone import Identity, Logit, Standardise, YeoJohnson

__all__ = ['Identity', 'Logit', 'Standardise', 'YeoJohnson']
