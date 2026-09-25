# `deltas/legacy` — FROZEN

The code behind every reported number, moved here verbatim (only its own import
lines changed). **Do not edit it to change behaviour.** Fix forward in the
modular packages. `tests/golden/` pins the exact outputs of everything here
under both settings of `deltas.misc.use_two.USE_TWO`.

| folder | what | old import paths (still work, same module objects) |
|---|---|---|
| `ecai2024/` | the published method: `base.base_deltas` (separable), `downsample.downsample_deltas` (+ slacks — *the* published method), `equations`, `radius`, `optimise_deltas`, `optimise_contraint` | `deltas.model.base`, `deltas.model.downsample`, `deltas.utils.equations`, `deltas.utils.radius`, `deltas.optimisation.*` |
| `non_separable/` | the Dec 2024 draft: `non_sep.deltas` (k-th point fences), `data_info` | `deltas.model.non_sep`, `deltas.model.data_info` |
| `exploratory/` | `SSL`, `reprojection`, `SVM_supports` — used by neither paper | `deltas.model.SSL`, `…reprojection`, `…SVM_supports` |
| `overlap/` | the original Clopper–Pearson / DKW code | (none: `deltas.model.overlap` is now a shim over `DeltasEstimator`, proven bit-identical to this) |

Known bugs kept on purpose, because they feed the published numbers
(`FINDINGS.md` §7.1, table rows in brackets): the exact-float grid filter
(B1), `support_max_hit` read on a path where it may be unassigned (B2), the
silent `max_trials` cap (B8), and the infeasible-fit crash, where
`base_deltas.fit` raises `TypeError` on an infeasible problem (B11).

The per-method API reference that used to live in `deltas/model/README.md` is
below.

## `ecai2024/base.py` — `base_deltas`

`fit(X, y, costs=(1,1), clf=None, grid_search=True)` computes `data_info` (R,
D, M, means, counts), optimises δ₁ (δ₂ follows from the constraint) and sets
`boundary`. Attributes: `delta1`, `delta2`, `boundary`, `class_nums`,
`solution_possible`, `solution_found`, `data_info`. Crashes on an infeasible
problem (the infeasible-fit crash, FINDINGS B11).

## `ecai2024/downsample.py` — `downsample_deltas`

The published method. When the constraint is infeasible it removes support
points (class-proportionally) until it is, penalising the loss by
`α·(removed/N)` per class. `continuous_slacks=True` gives Appendix B.
Parallelised with `multiprocessing`; serial and parallel runs agree exactly.

## `non_separable/non_sep.py` — `deltas`

Generalises from the k-th furthest point and sweeps the bias.
`loss_type ∈ {'min','max','mean'}`, or `only_furtherest_k=True`. Requires a
`clf` with `get_projection` at construction. The loss it uses (δ·k/(N+1))
bounds nothing; the modular `KthPointFence(loss_form='expected')` offers the
corrected form.
