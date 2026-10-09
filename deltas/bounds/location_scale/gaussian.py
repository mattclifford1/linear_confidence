'''
Gaussian confidence band: exact t / chi^2 pivots (Bonferroni rectangle),
or a Monte Carlo calibrated simultaneous band. See
deltas/bounds/location_scale/__init__.py for the model.
'''
import functools

import numpy as np
from scipy import stats

from deltas.bounds.location_scale.base import _LocationScale
from deltas.core.registry import register


@functools.lru_cache(maxsize=256)
def _gaussian_band(N, delta, p_min, n_p, draws, seed):
    '''
    a one-sided tolerance band valid for every tail probability p in
    [p_min, 0.5] at once: the pointwise non-central-t factor k(p) at a level
    delta' calibrated by Monte Carlo over the exact pivots
        A = (xbar - mu) / sigma ~ N(0, 1/N),  B = s / sigma ~ sqrt(chi2/(N-1))
    Returns (p grid ascending, k descending).
    '''
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(draws) / np.sqrt(N)
    B = np.sqrt(rng.chisquare(N - 1, draws) / (N - 1))
    ps = np.geomspace(p_min, 0.5, n_p)
    zs = stats.norm.isf(ps)

    def ks(d):
        return stats.nct.ppf(1.0 - d, N - 1, zs * np.sqrt(N)) / np.sqrt(N)

    def coverage(d):
        k = ks(d)
        bad = np.zeros(draws, dtype=bool)
        for j in range(0, n_p, 25):       # chunked: draws x n_p is large
            bad |= np.any(zs[None, j:j + 25] - k[None, j:j + 25] * B[:, None]
                          > A[:, None], axis=1)
        return 1.0 - bad.mean()

    lo, hi = delta / (4.0 * n_p), delta      # coverage falls as d grows
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if coverage(mid) >= 1.0 - delta:
            lo = mid
        else:
            hi = mid
    return ps, ks(lo)


@register('bound', 'gaussian')
class GaussianConfidence(_LocationScale):
    '''
    Gaussian confidence band.

    band='rectangle'  mu and sigma bounded separately at delta/2 each by the
                      exact t and chi^2 pivots; valid for all b and all delta
                      (works with any delta policy)
    band='mc'         a simultaneous band over tail levels [p_min, 0.5],
                      calibrated by Monte Carlo: about half the cost of the
                      rectangle (working notes, "Level 2: location-scale bands"). Fixed delta only;
                      beyond the p_min level it reports p_min.
    '''

    def __init__(self, band='rectangle', p_min=1e-6, n_p=200, draws=40000,
                 seed=0):
        if band not in ('rectangle', 'mc'):
            raise ValueError("band must be 'rectangle' or 'mc'")
        self.band = band
        # the Monte Carlo band is a fine staircase in b, not a smooth curve
        self.smooth = band == 'rectangle'
        self.p_min = p_min
        self.n_p = n_p
        self.draws = draws
        self.seed = seed

    def extremes(self, delta):
        delta = np.asarray(delta, dtype=float)
        N = self.N
        shift = stats.t.ppf(1.0 - delta / 2.0, N - 1) * self.s / np.sqrt(N)
        sigma_up = self.s * np.sqrt((N - 1) / stats.chi2.ppf(delta / 2.0, N - 1))
        return shift, sigma_up

    def tail(self, k):
        return stats.norm.sf(k)

    def curve(self, b, delta):
        if self.band == 'rectangle' or self.degenerate:
            return super().curve(b, delta)
        delta = np.asarray(delta, dtype=float)
        if delta.size != 1:
            raise NotImplementedError(
                "band='mc' is calibrated per delta: use it with a fixed delta "
                "(confidence='fixed'), or use band='rectangle'")
        ps, ks = _gaussian_band(self.N, float(delta), self.p_min, self.n_p,
                                self.draws, self.seed)
        kappa = self.sample.facing_distance(np.asarray(b, dtype=float)) / self.s
        # smallest grid p whose factor the boundary clears; ks is descending
        idx = np.searchsorted(-ks, -kappa, side='left')
        return np.where(idx < len(ps), ps[np.minimum(idx, len(ps) - 1)], 1.0)
