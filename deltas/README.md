# deltas package

Python package implementing the DELTAS algorithm for confidence bounds on linear classifiers.

## Directory Structure

```
deltas/
  model/          Core algorithm classes
  data/           Data loading and preprocessing
  optimisation/   Delta parameter optimisation
  utils/          Mathematical utilities
  classifiers/    Neural network classifiers
  plotting/       Visualisation
  pipeline/       End-to-end pipeline helpers
  misc/           Global configuration
  costcla_local/  Cost-sensitive classification utilities (local copy)
```

## Subpackages

### `model/`
Core algorithm. `base_deltas` (in `base.py`) is the primary entry point — a scikit-learn style class with `fit`/`predict`. `non_sep.py` handles the overlapping (non-separable) case.

### `data/`
Dataset utilities. `loaders/` contains individual dataset loaders; each returns `(train_data, test_data)` where each split is a dict with keys `X`, `y`, `feature_names`, and optionally `costs`.

### `optimisation/`
`optimise_deltas.py` runs a 1D grid search over delta1 ∈ [0,1], then refines with `scipy.optimize.minimize`. delta2 is derived analytically via the constraint equation.

### `utils/`
- `projection.py` — project data via classifier, compute class splits
- `radius.py` — compute empirical and upper-bound radii
- `equations.py` — loss functions, constraint equations, Jacobians

### `classifiers/`
Neural network classifiers (PyTorch) that implement `get_projection(X)`. Includes large-margin networks, MIMIC networks, and MNIST classifiers.

### `plotting/`
`plots.py` — visualise projected data, radii, and delta boundaries.

### `misc/`
`use_two.py` — global config flags:
- `USE_TWO` — use factor of 2 in constraint equations (default `True`)
- `USE_GLOBAL_R` — use global R across all data vs per-class (default `False`)
- `RANDOM_STATE` — random seed (default `0`)

## Key Entry Points

```python
from deltas.model.base import base_deltas      # separable case
from deltas.model.non_sep import deltas        # non-separable case
```

## Classifier Interface

Any classifier passed to `base_deltas` must implement:
```python
clf.get_projection(X)  # returns 1D projection, shape (N,) or (N,1)
```
