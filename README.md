# DELTAS: Confidence Bounds for Linear Classifiers

DELTAS is an algorithm for computing tight confidence bounds (delta1, delta2) on the misclassification probability of linear classifiers. Given a classifier that projects data onto a 1D decision axis, DELTAS finds the smallest delta values consistent with the empirical data geometry — enabling statistically grounded prediction boundaries.

The algorithm characterises each class by its projected mean and radius, then solves a constrained optimisation to find delta1 and delta2 (per-class error probabilities) subject to a joint constraint equation.

## Setup

Clone the repo:
```
git clone https://github.com/mattclifford1/linear_confidence
cd linear_confidence
```

Create a Python environment:
```
conda create -n deltas python=3.10 -y
conda activate deltas
```

Install as an editable package:
```
pip install -e .
```

## Quick Start

```python
from deltas.model.base import base_deltas
from sklearn.svm import SVC

# Your classifier must implement get_projection(X)
clf = ...  # e.g. a wrapped SVC with get_projection method

model = base_deltas(clf=clf)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"delta1: {model.delta1:.4f}, delta2: {model.delta2:.4f}")
model.print_params()
```

## Package Structure

```
deltas/
  model/        Core algorithm: base_deltas class, non-separable variant
  data/         Data utilities and loaders for benchmark datasets
  optimisation/ Delta optimisation (grid search + scipy minimize)
  utils/        Projection, radius, and equation utilities
  classifiers/  Neural network classifiers with get_projection support
  plotting/     Visualisation utilities
  pipeline/     End-to-end pipeline helpers
  misc/         Global config flags (use_two.py)
  costcla_local/ Local copy of cost-sensitive classification utilities
```

## Experiments

- `notebooks/` — Development notebooks; start here for exploration
- `notebooks-ECAI/` — Publication experiments (cross-validation on benchmark datasets)
- `notebooks-non-sep/` — Non-separable / overlapping case experiments

## Dependencies

Core: `numpy`, `scipy`, `scikit-learn`, `matplotlib`
Neural classifiers: `torch`
Notebooks: `jupyter`

Install all via `pip install -e .`