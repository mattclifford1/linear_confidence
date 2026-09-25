'''
Level 2 of the assumption ladder: location-scale envelopes.

Model (after any fixed increasing transform of the score): the class-i scores
are mu_i + sigma_i * eps with eps ~ F_0 known (Gaussian by default). A
confidence region for (mu, sigma) gives, by Lemma 1 of the working notes, an
envelope valid at every boundary at once:

    U(b) = sup over the region of P(error at b)
         = psi((d(b) - shift) / sigma_up)   if that argument is >= 0, else 1

where d(b) is the distance from the class mean towards the boundary, `shift`
moves the mean towards the boundary and sigma_up inflates the spread - both
by more when N is small. This is where "fewer points, or more spread, gives
more room" comes from.

GaussianConfidence       exact t / chi^2 pivots (Bonferroni rectangle), or a
                         Monte Carlo calibrated simultaneous band ('mc',
                         tighter; fixed delta only)
LocationScaleConfidence  any location-scale F_0 (logistic, t_nu, laplace,
                         gaussian) via Monte Carlo pivots

Degenerate samples (N < 2, or no spread) carry no location-scale information:
the envelope is 1 everywhere.
'''
import functools

import numpy as np
from scipy import stats

from deltas.core.components import Bound
from deltas.core.registry import register


class _LocationScale(Bound):
    guarantee = 'high_probability'
    continuous = True

    def fit(self, sample):
        super().fit(sample)
        self.N = sample.N
        self.xbar = sample.mean
        self.s = sample.std
        self.degenerate = self.N < 2 or not (self.s > 0)
        return self

    def extremes(self, delta):
        '''(shift, sigma_up) at confidence delta; arrays broadcast with delta'''
        raise NotImplementedError

    def tail(self, k):
        '''psi(k): the standardised tail probability beyond k >= 0'''
        raise NotImplementedError

    def curve(self, b, delta):
        b = np.asarray(b, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if self.degenerate:
            return np.ones(np.broadcast(b, delta).shape)
        shift, sigma_up = self.extremes(delta)
        k = (self.sample.facing_distance(b) - shift) / sigma_up
        return np.where(k >= 0, self.tail(np.maximum(k, 0.0)), 1.0)

    def closed_form_minimax(self, other, delta):
        '''
        Prop. 4 of the working notes: with the same tail on both sides the
        minimax boundary splits the gap between the two cautious means in
        proportion to the two inflated spreads. `other` is the other class's
        fitted bound. Returns (b*, value), or None when the cautious means
        cross (no boundary certifies below psi(0)).
        '''
        by_side = {self.sample.side: self, other.sample.side: other}
        lo, hi = by_side['low'], by_side['high']
        s_lo, g_lo = lo.extremes(delta)
        s_hi, g_hi = hi.extremes(delta)
        a1, a2 = lo.xbar + s_lo, hi.xbar - s_hi
        if not a2 > a1:
            return None
        b = (g_hi * a1 + g_lo * a2) / (g_lo + g_hi)
        return float(b), float(self.tail((a2 - a1) / (g_lo + g_hi)))


# --------------------------------------------------------------- Gaussian ---
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
    Gaussian confidence envelope.

    band='rectangle'  mu and sigma bounded separately at delta/2 each by the
                      exact t and chi^2 pivots; valid for all b and all delta
                      (works with any delta policy)
    band='mc'         a simultaneous band over tail levels [p_min, 0.5],
                      calibrated by Monte Carlo: about half the cost of the
                      rectangle (working notes, Sec. 5.1). Fixed delta only;
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


# -------------------------------------------------- other location-scale ---
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
