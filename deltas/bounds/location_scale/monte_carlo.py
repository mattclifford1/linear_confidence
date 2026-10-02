'''
Confidence envelope for any location-scale family (logistic, t, laplace,
gaussian) from Monte Carlo pivots. See deltas/bounds/location_scale/__init__.py
for the model.
'''
import functools

import numpy as np
from scipy import stats

from deltas.bounds.location_scale.base import _LocationScale
from deltas.core.registry import register


_FAMILIES = {
    'gaussian': lambda df: stats.norm,
    'logistic': lambda df: stats.logistic,
    'laplace': lambda df: stats.laplace,
    't': lambda df: stats.t(df),
}


@functools.lru_cache(maxsize=256)
def _pivot_draws(family, df, N, draws, seed):
    '''Monte Carlo draws of A = mean/sd and B = sd of standard samples'''
    dist = _FAMILIES[family](df)
    rng = np.random.default_rng(seed)
    draws = int(max(2000, min(draws, 20_000_000 // max(N, 1))))
    eps = dist.rvs(size=(draws, N), random_state=rng)
    s = eps.std(axis=1, ddof=1)
    return eps.mean(axis=1) / s, s


@register('bound', 'location_scale')
class LocationScaleConfidence(_LocationScale):
    '''
    confidence envelope for any location-scale family, from Monte Carlo
    pivots. For a sample mu + sigma * eps, A = (xbar - mu)/s and B = s/sigma
    have a law free of (mu, sigma), so their quantiles, simulated once per
    (family, N), give exact (up to Monte Carlo error) confidence bounds:
        mu towards the boundary  <= xbar -+ q_A(delta/2) s
        sigma                    <= s / q_B(delta/2)
    Heavier-tailed families (logistic, t, laplace) hedge against a Gaussian
    tail being too optimistic. `sigma` is the family's own scale parameter.
    '''

    def __init__(self, family='logistic', df=5, draws=20000, seed=0):
        if family not in _FAMILIES:
            raise ValueError(f'family must be one of {sorted(_FAMILIES)}')
        self.family = family
        self.df = df
        self.draws = draws
        self.seed = seed

    def extremes(self, delta):
        delta = np.asarray(delta, dtype=float)
        A, B = _pivot_draws(self.family, self.df, self.N, self.draws, self.seed)
        if self.sample.side == 'low':
            # mu <= xbar - q_A(delta/2) s  (q_A(delta/2) < 0)
            shift = -np.quantile(A, delta / 2.0) * self.s
        else:
            # mu >= xbar - q_A(1 - delta/2) s
            shift = np.quantile(A, 1.0 - delta / 2.0) * self.s
        sigma_up = self.s / np.quantile(B, delta / 2.0)
        return shift, sigma_up

    def tail(self, k):
        return _FAMILIES[self.family](self.df).sf(k)
