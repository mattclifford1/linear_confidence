# deltas/model

Core algorithm classes for computing confidence bounds.

## Files

### `base.py` — `base_deltas`

Primary class for the separable case. Scikit-learn style API.

**Constructor:**
```python
base_deltas(clf=None, dim_reducer=None)
```

**Key methods:**
- `fit(X, y, costs=(1,1), clf=None, grid_search=True)` — fits the model; computes data_info, optimises delta1/delta2, sets `self.boundary`
- `predict(X)` — projects X and applies the threshold boundary
- `get_data_info(X, y, clf)` — static; computes R, D, M, means, class counts
- `print_params()` — prints R, N, margin, D, costs
- `print_deltas()` — prints fitted delta1/delta2 and constraint value
- `plot_data(...)` — visualises projected data and radii

**Attributes after `fit`:**
- `delta1`, `delta2` — optimised confidence parameters ∈ [0,1]
- `boundary` — 1D classification threshold
- `class_nums` — which class label is assigned to each side of boundary
- `solution_possible`, `solution_found` — flags for optimisation status
- `data_info` — dict of empirical statistics (R1, R2, D, M, N1, N2, means)

### `downsample.py` — `downsample_deltas`

⭐ **This is the published ECAI 2024 method**, not a generic utility. It
subclasses `base_deltas` and implements the binary slack variables of §4.2: when
the non-overlap constraint is infeasible it iteratively removes support points
(class-proportionally) until it is, penalising the loss by `α·(removed/N)` per
class. `continuous_slacks=True` gives the Appendix B variant. Parallelised over
trials with `multiprocessing`.

### `non_sep.py` — `deltas`

⭐ The non-separable follow-up (working draft, Dec 2024). Replaces slacks with
the *k*-th furthest order statistic and sweeps the bias directly. Requires `clf`
with `get_projection` at construction time (not optional).
`loss_type ∈ {'min','max','mean'}` or `only_furtherest_k=True`.

### `data_info.py`

Computes and stores the projected data statistics. **Used only by `non_sep.py`**
— it is a class with attributes (`R1_emp`, `emp_xp1`, `min_conc_1`, …), and is
*not* interchangeable with the plain dict returned by
`base_deltas.get_data_info()` (keys like `'empirical R1'`). Adds the sorted
per-point distances `d_1`/`d_2` needed for the order statistics.

### `SVM_supports.py`

`SVM_supports_deltas` — a `downsample_deltas` subclass that fits a *secondary*
SVM to produce the 1-D projection, for classifiers with no `get_projection`.
Exploratory; not used by either paper.

### `SSL.py`

`SSL_deltas` — **Superset** learning (not semi-supervised): points may be
relabelled rather than removed. Exploratory; not used by either paper.

### `reprojection.py`

`reprojectioner` / `reprojection_deltas` — fit a second model to produce the 1-D
projection. Exploratory; not used by either paper.

## Data Flow

```
X, y  ──►  clf.get_projection(X)  ──►  1D projected data
                                              │
                                    projection.make_calcs()
                                              │
                                         data_info dict
                                    (R1, R2, D, M, N1, N2, means)
                                              │
                                    optimise_deltas.optimise()
                            grid search over delta1 (default), OR
                            scipy.minimize with analytic Jacobian
                            — returns None if no solution exists
                                              │
                                       delta1, delta2
                                              │
                                    radius.R_upper_bound()
                                              │
                                         boundary = (upper_min + lower_max) / 2
                                              │
                                       predict(X)
```

## Constraint

delta2 is derived analytically from delta1 via `ds.delta2_given_delta1_matt`. The constraint `ds.contraint_eq7(delta1, delta2, data_info) <= 0` must be satisfied for a solution to exist.

`is_fit == False` after `fit()` means **no solution was found** — a normal
outcome when the projected classes overlap, not an error. The experiment
runners silently skip such seeds; see `FINDINGS.md` §6.1 for why that matters.

⚠️ The grid-search feasibility filter in `optimise_deltas.py:82` uses exact
float equality (`J[constraints != 0]`), which discards ~25% of genuinely valid
grid points at ~1e-16 residuals. See `FINDINGS.md` §7.1 B1.
