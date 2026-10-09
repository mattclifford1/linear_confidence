'''
Level 2 of the assumption ladder: location-scale bands.

Model (after any fixed increasing transform of the score): the class-i scores
are mu_i + sigma_i * eps with eps ~ F_0 known (Gaussian by default). A
confidence region for (mu, sigma) gives, by the "uniform for free" lemma of the working notes, an
band valid at every boundary at once:

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
the band is 1 everywhere.

    base.py          _LocationScale: the shared band and closed_form_minimax
    gaussian.py      GaussianConfidence (registry name 'gaussian')
    monte_carlo.py   LocationScaleConfidence (registry name 'location_scale_mc')
'''
from deltas.bounds.location_scale.gaussian import GaussianConfidence
from deltas.bounds.location_scale.monte_carlo import LocationScaleConfidence

__all__ = ['GaussianConfidence', 'LocationScaleConfidence']
