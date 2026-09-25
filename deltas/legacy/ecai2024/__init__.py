'''
FROZEN: the published method (Clifford et al., ECAI 2024) - moved verbatim.

    base.py                base_deltas: separable case (fences meet)
    downsample.py          downsample_deltas: + binary/continuous slacks - THE
                           published method ('Slacks Deltas')
    equations.py           the paper's equations (loss, constraint, delta2(delta1))
    radius.py              the concentration term, R bounds
    optimise_deltas.py     the delta_1 grid search / SLSQP
    optimise_contraint.py  finding a feasible starting point

Reads deltas.misc.use_two at import time. Known bugs that feed the published
numbers (FINDINGS.md §7.1 B1, B2, B8, B11) are kept deliberately; the old
import paths (deltas.model.base, deltas.utils.radius, ...) are aliases.
'''
