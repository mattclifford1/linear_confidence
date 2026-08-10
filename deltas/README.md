# `deltas` package internals

Map of the code, and how the modules line up with the equations in the ECAI
paper (`m598.tex`).

## The pipeline in one line

```
X, y ──clf.get_projection──> 1-D scores ──data_info──> (mean_i, R̄_i, N_i, D̂)
     ──optimise δ_i──> boundary ──> predict
```

Everything the method needs lives in the **projected (1-D) space**. The
classifier is only ever used as a projection function, which is what makes the
method classifier-agnostic.

See also `model/README.md` for a method-by-method API reference of
`base_deltas`, and `data/loaders/readme.md` for the dataset list.

## `model/` — the estimators

All are sklearn-shaped: `.fit(X, y)`, `.predict(X)`, `.predict_proba(X)`
(delegates to the wrapped classifier), `.get_bias()`, and `.is_fit`.

| Module | Class | What it is |
|---|---|---|
| `base.py` | `base_deltas` | The core separable method. `get_data_info()` builds the stats dict; `_optimise()` solves for `δ₁, δ₂`; `_make_boundary()` places the bias. **Fails when the projected classes overlap.** |
| `downsample.py` | `downsample_deltas` | ⭐ **The published method.** Wraps `base_deltas` and, when the constraint is infeasible, iteratively removes support points (the binary slack variables, §4.2 of the paper) until it is, penalising the loss by `α·(removed/N)`. `continuous_slacks=True` gives the Appendix B variant. Parallelised over trials with `multiprocessing`. |
| `non_sep.py` | `deltas` | ⭐ **The non-separable follow-up.** Replaces slacks with the *k*-th furthest order statistic. Sweeps candidate biases and picks the argmin of the loss. `loss_type ∈ {'min','max','mean'}`, or `only_furtherest_k=True` to recover the ECAI behaviour. |
| `data_info.py` | `data_info` | Successor to `base.get_data_info()`'s dict. Used **only** by `non_sep`. Adds sorted per-point distances `d_i` (needed for the order statistics) and `min_conc_i`. |
| `SSL.py` | `SSL_deltas` | Superset learning — lets points be relabelled rather than removed. Exploratory. |
| `reprojection.py` | `reprojectioner`, `reprojection_deltas` | Fit a *second* model (e.g. an SVM) to produce the 1-D projection, for classifiers that have no `get_projection`. Exploratory. |
| `SVM_supports.py` | `SVM_supports_deltas` | `downsample_deltas` + SVM reprojection. Exploratory. |

The three exploratory ones are not used by either paper.

**Note the two incompatible `data_info` representations** — `base.py` uses a
plain dict with keys like `'empirical R1'`, `non_sep.py` uses the class with
attributes like `R1_emp`. They are not interchangeable.

## `utils/equations.py` — the maths

Direct correspondence with the paper:

| Function | Paper |
|---|---|
| `class_cost`, `loss` | Eq. 6, `L = Σ (1−δᵢ)/(Nᵢ+1) + δᵢ` |
| `loss_one_delta` | Eq. 6 with `δ₂` eliminated via the constraint |
| `contraint_eq7` / `eq7_matt` | Eq. 7, `R̂₁ + R̂₂ − D̂ = 0` |
| `delta2_given_delta1_matt` | Eq. 8, `δ₂ = exp[−½(B√N₂/R̄₂ − 2)²]` |
| `dd2_dd1`, `J_derivative` | Eqs. 9, 10 (analytic gradients for SLSQP) |
| `contraint_eq8` | margin-based variant, unused |

`utils/radius.py`:

- `supremum(X, x0)` → `R̄ᵢ`, the empirical class support (max distance from the
  mean in the projected space).
- `error_upper_bound(R, N, δ)` → `(R/√N)(2 + √(2 ln 1/δ))`, doubled when
  `USE_TWO`. This is the concentration term of Eq. 4/5.
- `R_upper_bound` → Eq. 5.

## `optimisation/`

- `optimise_deltas.optimise()` — the solver. Default path is a **1-D grid search
  over `δ₁`** on 10 000 points (`grid_search=True`), keeping only points where
  the constraint holds. Falls back to `scipy.minimize` with the analytic
  gradient when `grid_search=False`, and to a 2-D unconstrained grid when
  `grid_2D=True` and the constraint is infeasible. Returns `None` when
  unsolvable — that `None` is what becomes `is_fit == False`.
- `optimise_contraint.py` — helper to find a feasible starting `(δ₁, δ₂)`.

> The constraint filter on line 82 uses exact float equality; see
> `FINDINGS.md` §7.1 B1.

## `pipeline/` — experiment glue

| Module | Role |
|---|---|
| `data.py` | `get_real_dataset(name, seed, scale)` dispatches to `data/loaders/`; `get_data(...)` makes the synthetic 2-Gaussian set; `get_SMOTE_data`; PCA/UMAP reducers for plotting. |
| `classifier.py` | `get_classifier(data_clf, model=...)` trains the baseline **and all comparison methods in one call**: Baseline, SMOTE, Balanced Weights, BMR, Threshold. Returns a `{name: clf}` dict. `model ∈ {'Linear', 'SVM', 'SVM-linear', 'SVM-rbf', 'SVM-rbf-fixed', 'MLP', 'MLP-small', 'MLP-deep', 'MLP-Gaussian', 'MIMIC', 'MIMIC-cross-val', 'MNIST'}`. `'SVM-rbf'` does a 5-fold grid search over C and gamma (three times — original, weighted, SMOTE). |
| `evaluation.py` | `eval_test(clfs_dict, test_data)` → DataFrame of Accuracy / G-Mean / F1, plus the projected-space boundary plots. |
| `cached.py` | Disk-cached `get_dataset` / `get_classifiers` / `get_data_and_classifiers`. Takes `calibration=<float>`; `get_deltas_fit_data(data_clf)` then returns the right `(X, y)` to fit a deltas model on. |
| `calibration.py` | ⭐ The calibration split. `split_calibration` (stratified, refuses rather than return a useless split) and `fit_calibrated` (fit-part classifier + calibration-part certificate). See `CALIBRATION.md`. |
| `pipeline_old.py` | Superseded. |

**Why calibration lives here and not in the estimator:** by the time
`model.fit(X, y, clf=clf)` runs, `clf` is already trained, so splitting inside
that call would still count points the classifier had fitted to. The split has
to happen upstream of classifier training.

## `classifiers/models.py`

`SVM`, `linear`, `NN` — sklearn subclasses that add `get_projection` and
`get_bias`.

- `SVM.get_projection` uses `decision_function(X) − intercept_` for non-linear
  kernels, and the normalised `X·wᵀ` for linear.
- `NN` carries **~350 lines of vendored sklearn `MLPClassifier` internals**
  (`_fit_weighted`, `_fit_stochastic_weighted`, `_backprop_weighted`) so that
  `class_weight='balanced'` works, which upstream still does not support. This
  is the most version-fragile code in the repo — it imports private symbols
  from `sklearn.neural_network._multilayer_perceptron`.
- `delta_adjusted_clf` — a bare boundary+class-order predictor, used where a
  full deltas object isn't wanted.

`classifiers/frozen.py` — `FrozenProjection`, the identity projector over
projections computed in another process. This is what lets models fitted under
the sibling `projection_models` environment (whose sklearn is too new to import
here) be used by every deltas estimator: the estimators only ever call
`get_projection`, and all of them pass 1-D input straight through. See
`experiments/README.md`.

Also in `classifiers/`: torch nets for MNIST and MIMIC, and a large-margin
loss implementation (Elsayed et al.). None feed the published results.

## `data/`

- `loaders/` — one module per dataset, each returning `(train, test)` dicts with
  `X`, `y`, and relabelling so **class 1 is the minority**. Registered in
  `pipeline/data.py::get_real_dataset`.
- `datasets/` — the small CSVs, committed.
- `utils.py` — `normaliser` (MinMax to [−1,1], **fit on train only**),
  `shuffle_data`, `proportional_split(data, size, ratio)` where `ratio` forces
  a train-set imbalance (e.g. `ratio=10` → 10:1).

## `costcla_local/`

Self-contained port of the parts of `costcla` needed for the BMR
(`Bahnsen et al.`) and Thresholding (`Sheng & Ling`) baselines, because the
upstream package is unmaintained. `models.BMR` and `models.Thresholding` are
the entry points.

## `misc/use_two.py` — global config ⚠️

```python
USE_TWO = True        # factor of 2 on the concentration error term
USE_GLOBAL_R = False  # use sup||proj(x)|| over all data instead of per-class R̄_i
RANDOM_STATE = 0
```

Module-level constants read at import time throughout the package. Changing
them changes the maths of every experiment, and nothing records which setting
produced a given results file. The `results-two/` and `results-two2/`
directories are ablations produced by hand-editing this file.

## `plotting/plots.py`

`plot_classes`, `plot_decision_boundary` (feature space, optional PCA/UMAP
reduction for >2-D), `plot_projection` / `deltas_projected_boundary` (the 1-D
projected space with the `R̂ᵢ` bars — the paper's Figs 2, 4, 5) and
`conc_projected_boundary` (the `non_sep` version).
