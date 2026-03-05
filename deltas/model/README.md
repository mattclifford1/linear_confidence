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

### `non_sep.py` — `deltas`

Variant for non-separable (overlapping) classes. Requires `clf` with `get_projection` at construction time (not optional).

### `data_info.py`

Computes and stores the projected data statistics used throughout the algorithm.

### `SVM_supports.py`

Utilities for extracting SVM support vectors and margin information.

### `SSL.py`

Semi-supervised learning extension.

### `reprojection.py`

Re-projection utilities for adjusting the classification boundary.

### `downsample.py`

Downsampling strategies for class-imbalanced data.

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
                                    grid search + scipy.minimize
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
