# Modularity review of the deltas code and experiments

*Written 1 October 2026 against the working tree of branch `modular-deltas`, HEAD `ace3b41`, which is seven commits ahead of `main` (`86345c0`). `main` has none of `deltas/core`, `bounds`, `methods` or `legacy`, so everything said about those packages applies to the branch only. Nothing in the repo was changed for this review. The non-golden test suite passes, 305 tests in 65 s.*

Matt asked how to make the data setup, the classifiers, the experiment sweeps, the results and figures, and the deltas method itself easier to read and easier to swap pieces in and out of. The method code in `deltas/core/` and the slot packages is already in good shape, and most of the friction is in the machinery around it, where several things exist in three slightly different copies. Most of the suggestions below collapse those copies into one place.

## What already works

- The six-slot `DeltasEstimator` in `deltas/core/` with a registry that accepts a name, a `(name, kwargs)` pair or an object, deep-copies components, and is a real sklearn estimator. `describe()` gets stamped into every `config.json`.
- Flags on `Bound` for the guarantee type, whether a delta is needed, continuity, smoothness and self-resolution let the estimator pick the search and the refinement without the bound knowing about either.
- Per-class bounds as a dict, and `certify=[...]` at a fixed delta, cover the decide-with-one-curve, certify-with-another design from the working notes.
- The test layering. Golden tests pin the frozen code under both `USE_TWO` settings, the equivalence test shows the shim is bit-identical to the original overlap code on 22 data sets, and every high-probability bound has a simulation coverage test.
- The two-step wide grid. Exporting 1-D projections to `.npz` files, so that `run_wide.py` never sees a model, is the clearest expression of the classifier-agnostic claim anywhere in the repo.
- The disk cache keyed on a config dict plus library versions, with atomic writes.

## Constructing the deltas method

### Fit leaves 25 attributes behind

`DeltasEstimator.fit` in `deltas/core/estimator.py` sets about 25 attributes, among them `N1`, `N2`, `costs`, `class_nums`, `_low_sorted`, `_high_sorted`, `bounds_`, `_L_low`, `_L_high`, `boundary_t_`, `boundary`, `candidates`, `losses`, `m_low`, `m_high`, `loss`, `delta_low`, `delta_high`, `bound_low`, `bound_high`, `delta1`, `delta2`, `error_bound_1`, `error_bound_2`, `rule_info_`, `certificates_`, `solution_possible`, `solution_found` and `is_fit`. Many exist so that `deltas.model.overlap` stays bit-identical with the legacy overlap code, which the equivalence test checks attribute by attribute.

Have the fit produce one small result object holding the boundary, the value of each class's curve at that boundary, the candidates and losses, the rule information and the certificates. The estimator exposes that object plus `boundary`, `is_fit` and `certificates_`, and the legacy attribute names move onto `base_overlap_deltas` in the shim as read-only properties. The equivalence test keeps passing and the estimator stops carrying names like `error_bound_1` whose meaning depends on orientation.

### Labels, sides and class numbers

The code uses labels 0 and 1, sides `low` and `high`, and the strings `'class 1'` and `'class 2'`, where `'class 1'` means label 0. Under the repo convention that class 1 is the minority, `certified_error()['class 1']` is therefore the majority certificate, and `run_wide.py` has to translate it with `rec['U_0'] = ce['class 1']`. This is the most confusing name in the package, and it is in the output that other code reads.

Index certificates by label, so `certificates_['clopper_pearson'][1]` is the minority. Keep `'class 1'` and `'class 2'` on the shim.

### Certify is half a slot

`certify` is a list of bound specs evaluated by a hard-wired `FixedDelta(delta_report)`. It cannot express the delta accounting the working notes recommend when two certificates are reported together, nor the fixed-sequence certificate for the Neyman–Pearson rule that needs no union correction, nor a certificate computed on a different sample from the one that chose the boundary. Name collisions are handled by appending an apostrophe.

Make `certifier` a sixth registered component kind with `certify(data, boundary) -> dict`. The default reproduces today's behaviour, a second implementation splits delta across the listed bounds, and each entry is named from its component's `describe()`.

### Costs, priors and alpha scattered across fit and the rules

`costs` is a `fit` keyword, `priors` is a constructor argument of `Risk`, and `alpha` is a constructor argument of `NeymanPearson`. Costs are not data, so they belong with the rule. Move them there, `Sum(costs=...)` and `Risk(costs=..., priors=...)`, and let `fit(X, y)` take data alone. The shim keeps the `costs=` keyword for the published runners.

### Policy and bound know about each other

`OptimisedDelta.prepare` branches on `isinstance(bound, CountBound)` and `NoDeltaCurve` branches on `bound.self_resolving`. Adding a new bound family with its own fast path means editing `deltas/confidence/`. Let the bound build its prepared curve from a policy, `bound.prepare(policy)`, with `CountBound` implementing the table path and the fences ignoring the policy. The confidence package then defines what a policy is and nothing else.

### Moment bookkeeping copied across bounds

`_LocationScale.fit`, `SawYangMo.fit` and `StudentTPredictive.fit` each set `N`, `xbar`, `s` and `degenerate` from the sample. One base class, or a `degenerate` property on `ClassSample`, removes the copies.

### `deltas/core` imports plotting, which imports the legacy code

`deltas/core/estimator.py` imports `deltas.plotting.plots`, which imports matplotlib and `deltas.legacy.ecai2024.radius`, which binds `USE_TWO` at import. Measured in a fresh interpreter, `import deltas.core` loads matplotlib, the legacy radius module and `deltas.misc.use_two`, so the modular package is not independent of the global flag after all, and `plot_loss` calls `plt.show()` from inside an estimator. Remove the plotting import and give the plotting module functions that take a fitted estimator, see the section on figures below.

### Legacy and modular methods under one `Method` class

`Method` wraps either a modular composition, configured through estimator parameters and `with_params()`, or a legacy class, configured through `fit_kwargs` and `configured()`. `describe()` special-cases the legacy ones through `target`. Method names are display strings with spaces that double as command-line arguments and CSV keys, and `combine_tables.PAPER_METHODS` holds a third set of names for LaTeX.

Wrap each legacy estimator once in a thin sklearn class whose constructor takes its fit options, so every method is an estimator class plus parameters. `Method` becomes a name, an estimator factory, a family and a reference, `describe()` is uniform, and `clone` works everywhere. Give each `Method` a slug for files and command lines and a `latex` label for tables.

### Interoperating with projection_models

A fitted deltas model reports `get_bias()`, the negated threshold, while the sibling package renamed its method to `get_threshold()` on purpose. `as_deltas_classifier` bridges the mismatch on the way in, and nothing bridges it on the way out. `ThresholdCalibrator` in `projection_models/calibration.py` has the same shape as a deltas estimator, a wrapper around a fitted model that replaces the threshold and leaves the projection alone.

Give `DeltasEstimator` a `get_threshold()` and make it quack like a `ProjectionModel`. A fitted deltas model then drops into the sibling's `report()`, `plot_threshold()` and `EnsembleTrainer` with no further code, and a deltas estimator can be offered as a `ThresholdCalibrator` strategy later.

### Smaller points

- `dim_reducer` and `dev` on `DeltasEstimator` are unused in the modular path. Keep both on the shim class.
- The `_coverage` helper in `tests/modular/test_bounds_shape.py` is what the planned certificate-validity experiment needs. Move it to `deltas/diagnostics.py` and have the tests import it from there.
- `search` and `refine` are decided in two places, the `Auto` candidate set reading `continuous` and the estimator reading `smooth`. A search object that owns both choices, for example `GridThenRefine`, keeps the estimator out of it.

## Setting up data

### Split policy repeated in `export_projections`, `run_costs` and `sibling`

The deltas split convention is half the data to train, the training minority thinned towards 10:1, a balanced test set, and a floor on minority counts. `experiments/export_projections.load_split` implements it with `TARGET_RATIO`, `MIN_MINORITY_TRAIN`, `MIN_MINORITY_TEST` and `MAJORITY_MAX`. `experiments/run_costs.load` implements it again with `clip(target, 8, n_min // 2)` and no test minimum. `deltas/data/loaders/sibling.get_sibling_dataset` implements it a third time with `clip(wanted, 1, max(1, n_min // 2))`. The three disagree at the edges, which is where the scarce-minority datasets are.

Write one `SplitSpec` dataclass and one `make_split(spec, seed)` returning fit, calibration and test parts plus metadata, use it from all three callers, and write the spec into the results next to the method specs. The sibling's `proportional_split` already does the mechanics and returns new dicts while leaving the input alone, so this is a thin layer over it.

### Untyped data dictionary

`data_clf` carries `'data'`, `'data_test'`, `'data_cal'`, `'data_full'`, `'dim_reducer'`, `'clf'`, `'mean1'` and `'mean2'` depending on which path built it, and `get_deltas_fit_data` reads `data_clf.get('data_cal') or data_clf['data']`. Replace it with a dataclass with `fit`, `calibration` and `test` parts, where `calibration` is `None` when no split was requested, and a `certify_on` property that returns whichever part the certificate should be read from.

### Legacy and sibling datasets behind one resolver

`get_real_dataset('Breast Cancer')` is the legacy loader, `'Breast Cancer Wisconsin'` falls through to `toy_datasets`, and `'Pima Indian Diabetes'` and `'Diabetes Pima Indian'` are the same data with different splits from different code. A results row cannot say which. Add an explicit `source=` argument and move the frozen loaders under `deltas/legacy/data/`, so the default path has one answer. The loaders themselves stay unchanged because the published splits depend on their shuffling.

### Loading a dataset fits a PCA

`make_data_dim_reducer` wraps `get_real_dataset` and fits a PCA on every load of a dataset with more than two features, whether or not anything will be plotted, and `deltas/pipeline/data.py` imports `umap` at module level. Measured in a fresh interpreter, `import deltas.pipeline.cached` takes 5.6 s and loads torch and umap, the torch coming from `pipeline/classifier.py` importing the MNIST and MIMIC trainers at module level. Fit reducers when plotting, through the sibling's `fit_dim_reducer`, and import the torch trainers inside the branch that uses them.

### In-place mutation and boolean seeds

`deltas.data.utils.proportional_split` mutates the input dictionary and `normaliser.__call__` overwrites `data['X']`, where the sibling's versions return new objects. New code should take integer seeds, since the `seed=True` idiom is what produced the seed 0 and seed 1 collision recorded in `FINDINGS.md`.

## Classifiers and baselines

### Model tables in `models.py`, `sibling.py`, `export_projections.py` and `run_costs.py`

A classifier can come from `deltas/classifiers/models.py`, from `deltas.classifiers.sibling.build`, from `MODELS` in `export_projections.py`, or from `MODELS` in `run_costs.py`, and the last three repeat the same hyperparameters, `Linear(max_iter=2000)`, `MLP((64, 32), max_iter=600)` and `RandomForest(n_estimators=200)`. Keep one registry, name to factory plus capabilities such as `supports_class_weight` and `supports_random_state`, imported by the exporter and the cost experiment. The classifier cache key then includes the factory's parameters as data, which removes the manual `CACHE_VERSION` bump that `CLAUDE.md` asks for whenever a hyperparameter inside `get_classifier` changes.

### `get_classifier` trains everything

`deltas/pipeline/classifier.get_classifier` is 292 lines with 15 flags. It trains the baseline, the SMOTE model, the balanced-weights model, BMR and Thresholding, grid-searches the SVM, plots decision boundaries, and saves paper figures. Split it into a function that trains one model from the registry and a family of comparison methods, described next.

### Threshold is two different algorithms

`run_experiments.py` reports `Threshold` from `costcla_local.Thresholding`, which optimises on `predict_proba` with the Sheng and Ling cost matrix. `run_wide.py` reports `Threshold` from `_best_balanced_threshold`, which minimises balanced error on the fit projections. `run_costs.py` has a third, `best_threshold_by_cost`. The rows share one label in `PAPER_METHODS` and `METHOD_ORDER`.

Register the post-hoc baselines as methods with the same `callable(clf, X, y) -> predictor` interface as the deltas methods, under a `baseline` family, and give the balanced-error threshold and the cost-matrix threshold distinct names. SMOTE and balanced weights need the feature space, so they belong on the classifier side as training recipes, plain, SMOTE or balanced, in the model registry. `export_projections.py` already does this implicitly with `pred_smote` and `pred_bw`.

### Code that feeds no result

The torch networks under `deltas/classifiers/`, the vendored `costcla_local` package, `deltas/pipeline/pipeline_old.py` and the `dev/` scripts feed no reported number. Moving them under `deltas/legacy/exploratory/` or an `archive/` folder shortens the map a new reader has to hold.

## Running sweeps

### Separate loops in every runner

`run_experiments.py`, `run_wide.py` with `export_projections.py`, `run_costs.py`, `validate_bounds.py` and the sweep inside `make_figures.py` each have their own loop over datasets, models and seeds, their own `METRICS` dictionary, their own row schema and their own config stamp. `run_wide.py` records seconds, a failure reason, per-class errors, certificates and coverage, and `run_experiments.py` records none of those. `merge_wide.py` exists because partial runs had to be stitched by hand, and `make_figures.py` keeps its own joblib memo for its sweep.

### One artefact, one schema, one runner

1. Make the exported projection cell the artefact every runner consumes. A cell is `z_fit`, `z_cert`, `z_test`, their labels, the model's threshold, the baseline predictions and the metadata, keyed by dataset, model, seed and calibration mode. The six-dataset table of the papers becomes a subset of the grid once the MIMIC local loader is one more source for the exporter. One cache replaces the joblib classifier cache, the `.npz` files and `overlap_sweep.joblib`.
2. Use one long-format row schema, dataset, model, seed, mode, method, the metrics, `U_0`, `U_1`, `delta_0`, `delta_1`, `covered_0`, `covered_1`, seconds and reason, written by one `score()` function. Every analysis reads that schema.
3. Use one runner that takes a list of cells and a list of method names, runs under joblib with the per-fit timeout, and is resumable per cell. That removes `merge_wide.py`.

### Importable shared code

`combine_tables.py` does `from run_experiments import EXPERIMENTS, SHORT_NAMES, METRICS, RESULTS`, which works from inside `experiments/` and nowhere else. Put the cell loader, the scorer, the runner and the analysis in a `deltas/bench/` package, with thin command-line scripts left in `experiments/`, so it can be tested like the rest of the package.

### What `config.json` records

`config.json` already records `USE_TWO`, the seeds, the method specifications and the Python version. Add the git commit, whether the working tree was clean, the numpy and scikit-learn versions, and the split specification.

## Showing results and figures

### Plotting code in the package, the experiments and the notes

`deltas/plotting/plots.py` is the ECAI-era module. It imports the legacy radius code, hard-codes font sizes, works on the `data_info` dictionary, and calls `plt.show()`. `deltas/pipeline/evaluation.eval_test` computes metrics, plots and saves figures in one function. `experiments/make_figures.py` has its own palette and rcParams, writes directly into the Overleaf draft folder, and reads the per-class curves from the private attributes `_L_low` and `_L_high`. The working notes' `figs/make_figures.py` has a third palette and its own `ecai_eps`, `cp_upper`, `gauss_rect`, `env_right`, `env_left` and `sym_bound`, with no import of `deltas.bounds`, so a change to a bound in the package would not change the figure that illustrates it.

### One style, functions that take fitted objects

- `deltas/plotting/style.py` with the palette and rcParams, imported by both figure scripts.
- `plot_projection(ax, data, boundary=None)` for each class's projected points as a strip, `plot_curves(ax, estimator)` for the per-class curves, the objective and the chosen boundary, and `plot_envelope(ax, bound, delta)` for one bound's curve. Each takes an axis and returns it, with no `plt.show()` in library code.
- A public `curves_` attribute, or a `curves()` method, on the estimator holding the candidates, both per-class curves and the losses, so figure scripts stop reading underscored attributes.
- The notes' figure script calls `deltas.bounds` for every curve it draws.

### Tables

Three LaTeX emitters exist, `run_experiments.to_latex`, `combine_tables.build` and `analyse_wide.latex_coverage`, and the number formatting is written out twice. One emitter reading the long-format schema and taking its row labels from the method registry covers all three.

## Layout, added 2 October 2026

*Done on 2 October 2026 on `modular-deltas`, commits `5fde043` to `a8f0be9`: base classes moved, bounds, rules, search and transforms split one class per file, tests mirroring the tree, docs, and the registry rename `location_scale` to `location_scale_mc`. All 464 tests pass, golden included. Not pushed.*

Matt's preference is one public class per file and nested packages, as in `toy_datasets/data_loaders/loaders/local_loaders/<dataset>.py` and `projection_models/sklearn/<model>.py`, because per-component files give clean diffs, clean `git log` on one component, and a tree that can be navigated without grep. The modular packages here are close to that already, with the largest files being `core/estimator.py` at 332 lines, `bounds/location_scale.py` at 226 and `core/components.py` at 203, and the layout below finishes the job.

### Rules

- One public class per file, named after its registry name where it has one, so `bounds/location_scale/gaussian.py` holds the component registered as `gaussian`.
- Private helpers stay in the file of their only consumer. `_gaussian_band` stays with `GaussianConfidence`, `_pivot_draws` and `_FAMILIES` stay with `LocationScaleConfidence`.
- A helper or base class shared inside one package goes in that package's `base.py`. No `utils.py` or `shared.py`, which become the next big file.
- Each `__init__.py` re-exports its public names and imports every module, since components register themselves on import. The existing registry test catches a forgotten import.
- The public import surface stays flat, `from deltas.bounds import GaussianConfidence`, and the deep paths are for navigation and diffs.
- `deltas/legacy/` keeps its current layout. It is frozen and the alias modules depend on its paths.
- Tests mirror the tree, `tests/modular/bounds/test_gaussian.py` and so on, so the test for a file is found by name.

### Tree

```
deltas/bounds/
  __init__.py                 re-exports, the menu docstring
  base.py                     Bound, CountBound (moved here from core/components.py)
  counts/
    __init__.py
    clopper_pearson.py        ClopperPearson, clopper_pearson_upper
    dkw.py                    DKW, dkw_upper
  moments/
    __init__.py
    base.py                   _BoundedMoments
    saw_yang_mo.py
    cantelli.py
    vysochanskij_petunin.py
  location_scale/
    __init__.py
    base.py                   _LocationScale, closed_form_minimax
    gaussian.py               GaussianConfidence, _gaussian_band
    monte_carlo.py            LocationScaleConfidence, _pivot_draws, _FAMILIES
  predictive/
    __init__.py
    student_t.py              StudentTPredictive
  fences/
    __init__.py
    base.py                   _Fence
    published.py              PublishedFence
    kth_point.py              KthPointFence

deltas/confidence/   base.py (DeltaPolicy, PreparedCurve), fixed.py, optimised.py
deltas/rules/        base.py (DecisionRule), minimax.py, sum.py, risk.py, neyman_pearson.py
deltas/search/       base.py (CandidateSet, _all_values), data_midpoints.py, grid.py, auto.py
deltas/transforms/   base.py (Transform), identity.py, logit.py, standardise.py, yeo_johnson.py
deltas/core/         component.py, registry.py, sample.py, estimator.py
```

Each slot's base class moves out of `core/components.py` into the `base.py` of its own package, so the base class lives next to its implementations and `core/` keeps `Component`, the registry, the samples and the estimator. No import cycle follows, because the estimator reaches components through the registry and the base modules import `Component` from `core`.

### Points the move surfaces

- The registry name `location_scale` currently names one implementation, the Monte Carlo pivot version, while the package name covers every location-scale envelope. Renaming the registry entry to `location_scale_mc`, or the file to `monte_carlo.py` with the registry name to match, removes the clash.
- `moments/base.py` subclasses `_LocationScale`, so the moments package imports from the location-scale package. That dependency is real today, hidden in one import line, and the move makes it visible.
- `methods/` stays one file per family. Each entry is five lines of configuration, and one file per method would be finer than useful.
- `core/estimator.py` is not split by file size. Its length falls out of the result-object change in the first section.
- A future `deltas/bench/` package follows the same rule, `cells.py`, `score.py`, `runner.py`, `analysis/coverage.py`, `analysis/ranks.py`, `tables/latex.py`.

### Sequencing

Do the move first, as one mechanical commit per package with `git mv` and import-line edits and no behaviour change, so the golden and equivalence tests prove nothing moved, and every later change shows up as a per-component diff. Add the modular public names to `tests/test_import_surface.py` at the same time, since it guards the legacy paths and nothing else today.

## Suggested order

0. The layout move above, one mechanical commit per package. Done, see the Layout section.
1. Index certificates by label and add `get_threshold()`. Small, touches the runners, and removes the most confusing name in the output.
2. One split specification and one model registry, imported by the exporter, the cost experiment and the sibling loader.
3. One row schema and one runner reading projection cells, with `run_experiments.py` reduced to a preset over the same path. Delete `merge_wide.py`.
4. Plotting rewrite with the shared style, then point the working notes' figure script at `deltas.bounds`.
5. `deltas/core` clean-up. Result object, certifier slot, costs on the rule, `bound.prepare(policy)`, and no plotting import.

Each step would be done behind the golden and equivalence tests, so a reported number that moved would show up as a failing test before it reached a table.

## Not found

This read turned up no correctness problem in the modular path, with the simulation coverage tests, the closed-form minimax check and the bit-identity test all passing at this commit, and the numbers quoted in the working notes were re-derived from the package in the earlier review.
