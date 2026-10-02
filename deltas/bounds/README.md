# `deltas/bounds` — the concentration inequalities

Each bound turns one class's projected sample into a curve over boundaries b:
an upper bound on (or estimate of) that class's error at b. One public class
per file; `base.py` holds the slot's base classes `Bound` and `CountBound`,
and each family is a package with its shared machinery in its own `base.py`.
Ordered by what they assume, which is also what they can say beyond the data
(see the working notes, *deltas-other-concentration*, sections "A framework:
tail envelopes" and "The menu"):

| level | assumption | file | registry name | guarantee | beyond the data |
|---|---|---|---|---|---|
| 0 | exchangeability only | `counts/clopper_pearson.py` | `clopper_pearson` | high-probability | flat, floored at ≈ ln(1/δ)/N |
| 0 | exchangeability only | `counts/dkw.py` | `dkw` | high-probability | flat, floored at ≈ ln(1/δ)/N |
| 1 | + moments | `moments/saw_yang_mo.py` | `saw_yang_mo` | average-case | floored at 1/(N+1) |
| 1 | + moments, known range | `moments/cantelli.py` | `cantelli` | high-probability | falls polynomially; loose at small N |
| 1 | + moments, known range, unimodal | `moments/vysochanskij_petunin.py` | `vysochanskij_petunin` | high-probability | falls polynomially; loose at small N |
| 2 | + a Gaussian shape | `location_scale/gaussian.py` | `gaussian` (exact; `band='mc'` tighter, fixed δ only) | high-probability | keeps falling |
| 2 | + a location–scale shape | `location_scale/monte_carlo.py` | `location_scale` (logistic, *t*, laplace, gaussian) | high-probability | keeps falling |
| 2 | + Gaussian shape | `predictive/student_t.py` | `predictive_t` | average-case | keeps falling |
| — | the published fence | `fences/published.py` | `published_fence` | average-case | — (for ablations) |
| — | the k-th point fence | `fences/kth_point.py` | `kth_point_fence` | average-case | — (for ablations) |

Shared machinery: `location_scale/base.py` (`_LocationScale`, the envelope
and `closed_form_minimax`), `moments/base.py` (`_BoundedMoments`, which
subclasses `_LocationScale`, so the moments package depends on the
location-scale one), `fences/base.py` (`_Fence`).

Rank-based bounds (level 0) are flat between data points and cannot certify
below about ln(1/δ)/N (the notes' "flat" and "floor" results). To give a scarce class room that
grows with its spread, a bound has to assume a tail shape (level 2), and the
assumption should be stated and checked. The location–scale bounds implement
`closed_form_minimax` (the notes' "closed-form minimax boundary"): with the same tail on both sides, the
minimax boundary does not depend on the tail shape (the "shape-free" result).

Any monotone transform of the score (`deltas/transforms/`) is free for the
classifier and matters to level 1–2 bounds. Fit a data-dependent one on the
classifier's fit split, never on the calibration data.

Adding a bound: a new file under the family it belongs to, one class,
`@register('bound', 'name')`, an import line in that family's `__init__.py`
and the name in this table. Import it in `deltas/bounds/__init__.py` too if it
should be reachable as `from deltas.bounds import ...`.
