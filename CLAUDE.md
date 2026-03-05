# CLAUDE.md — linear_confidence

## Project Summary

This repo implements the **DELTAS algorithm**: a method for learning confidence bounds (delta1, delta2) for linear classifiers. Given a linear classifier that projects data to a 1D decision space, DELTAS finds tight upper bounds on the probability of misclassification via a constrained optimisation over delta parameters.

## Environment & Package

- Package name: `deltas`
- Install: `pdm install` (or `pip install -e .` also works via PEP 517)
- Python: 3.10+
- Conda env: `deltas`

## Architecture Overview

### Core class
`deltas/model/base.py` — `base_deltas` (scikit-learn style)
- `fit(X, y, costs=(1,1), clf=None)` — computes data_info, optimises delta1/delta2, builds boundary
- `predict(X)` — projects X via clf, applies boundary threshold
- `get_data_info(X, y, clf)` — static method; computes R, D, M, means, class counts

### Classifier requirement
Any classifier passed to `base_deltas` must implement `get_projection(X)` returning a 1D projection (shape `(N, 1)` or `(N,)`).

### Non-separable case
`deltas/model/non_sep.py` — `deltas` class for overlapping classes (distinct from `base_deltas`).

### Global config flags
`deltas/misc/use_two.py`:
- `USE_TWO` — use factor of 2 in Dario's equations (default True)
- `USE_GLOBAL_R` — use global R across all data vs per-class (default False)
- `RANDOM_STATE` — random seed (default 0)

## Data Loaders

`deltas/data/loaders/` — each loader returns `(train_data, test_data)` tuple of dicts:
```python
{
    'X': np.ndarray,
    'y': np.ndarray,
    'feature_names': list[str],
    'costs': (c1, c2)   # optional
}
```

## Optimisation

`deltas/optimisation/optimise_deltas.py`:
- 1D grid search over delta1 ∈ [0, 1]
- Refinement via `scipy.optimize.minimize` (with Jacobian from `ds.J_derivative`)
- delta2 derived analytically from delta1 via `ds.delta2_given_delta1_matt`

## Notebooks

- `notebooks/` — development notebooks (start here for exploration)
- `notebooks-ECAI/` — publication experiments (cross-validation on benchmark datasets)
- `notebooks-non-sep/` — non-separable / overlapping case experiments
- `dev/` — miscellaneous development work

## No Test Suite

Verification is done through notebooks. No pytest or unittest setup currently.

## Common Workflows

```bash
conda activate deltas
jupyter notebook notebooks/Gaussian.ipynb    # basic separable example
jupyter notebook notebooks-ECAI/CV_1_pima.ipynb  # cross-validation experiment
```
