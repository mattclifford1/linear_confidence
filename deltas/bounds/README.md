# `deltas/bounds` — the concentration inequalities

Each bound turns one class's projected sample into a curve over boundaries b:
an upper bound on (or estimate of) that class's error at b. Ordered by what
they assume, which is also what they can say beyond the data (see the working
notes, *deltas-other-concentration*, sections "A framework: tail envelopes"
and "The menu"):

| level | assumption | file | registry name | guarantee | beyond the data |
|---|---|---|---|---|---|
| 0 | exchangeability only | `counts.py` | `clopper_pearson`, `dkw` | high-probability | flat, floored at ≈ ln(1/δ)/N |
| 1 | + moments | `moments.py` | `saw_yang_mo` | average-case | floored at 1/(N+1) |
| 1 | + moments, known range | `moments.py` | `cantelli`, `vysochanskij_petunin` (unimodal) | high-probability | falls polynomially; loose at small N |
| 2 | + a location–scale shape | `location_scale.py` | `gaussian` (exact; `band='mc'` tighter, fixed δ only), `location_scale` (logistic, *t*, laplace) | high-probability | keeps falling |
| 2 | + Gaussian shape | `predictive.py` | `predictive_t` | average-case | keeps falling |
| — | the published fences | `fences.py` | `published_fence`, `kth_point_fence` | average-case | — (for ablations) |

Rank-based bounds (level 0) are flat between data points and cannot certify
below about ln(1/δ)/N (the notes' "flat" and "floor" results). To give a scarce class room that
grows with its spread, a bound has to assume a tail shape (level 2), and the
assumption should be stated and checked. The location–scale bounds implement
`closed_form_minimax` (the notes' "closed-form minimax boundary"): with the same tail on both sides, the
minimax boundary does not depend on the tail shape (the "shape-free" result).

Any monotone transform of the score (`deltas/transforms/`) is free for the
classifier and matters to level 1–2 bounds. Fit a data-dependent one on the
classifier's fit split, never on the calibration data.
