# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Research code for **deltas** — a post-hoc, classifier-agnostic method for handling
class imbalance by moving the *bias term* of a trained binary classifier using
class-conditional concentration inequalities.

Two papers live off this code:

| Paper | Source | Status |
|---|---|---|
| *Learning Confidence Bounds for Classification with Imbalanced Data* (ECAI 2024, arXiv:2407.11878) | `../../Repos/Overleaf/deltas/ecai-2024-deltas-arxiv-3-sup-mat/m598.tex` | Published |
| *Deltas non-separable* (working draft) | `../../Repos/Overleaf/deltas/deltas-non-separable/main.tex` | Rough draft, Dec 2024 |
| Review notes on the published paper | `../../Repos/Overleaf/deltas/claude-improvement-notes.md` | Notes only |

Read `FINDINGS.md` in this repo for the research state, known bugs and the
open research directions. Read `deltas/README.md` for the package architecture.
Read `CALIBRATION.md` for the calibration split — what it fixes (the reported
certificate is optimistic when computed on the classifier's own training data),
why it has to sit in the pipeline rather than the estimator, and what it costs.

Other docs in the tree: a README in each package folder (`deltas/core/`,
`deltas/bounds/` — the assumption ladder — `confidence/`, `rules/`, `search/`,
`transforms/`, `methods/`, `legacy/` — the frozen code's API reference),
`tests/golden/README.md`, `deltas/data/loaders/readme.md`,
`notebooks/README.md`, `notebooks-ECAI/README.md`, `notebooks-non-sep/README.md`.

The shape-aware ("other concentration") direction is planned in the Overleaf
working notes `../../Repos/Overleaf/deltas/deltas-other-concentration/main.tex`.

## Environment — **uv**

```bash
uv sync --group dev      # creates .venv, installs deltas + both siblings editable
uv run pytest tests -q                  # everything (~2 min)
uv run pytest tests -q -m "not golden"  # skip the golden tests while iterating
uv run python experiments/run_experiments.py
```

Python 3.13, sklearn 1.9, numpy 2.4. `uv.lock` is committed — that is the
known-good set, so prefer `uv run` over activating anything.

There is no conda env any more, and no pdm. The old
`/home/matt/anaconda3/envs/deltas` (python 3.10, sklearn 1.3.2) still exists on
this machine but is **stale — do not use it**.

The two sibling repos are **path dependencies** declared in
`[tool.uv.sources]`, installed editable, so edits there are picked up here
immediately:

```toml
toy-datasets      = {path = "../../Repos/toy_datasets", editable = true}
projection-models = {path = "../../Repos/projection_models", editable = true}
```

### The sklearn pin is gone

`deltas/classifiers/models.py` used to vendor ~350 lines of sklearn 1.3.x
`MLPClassifier` internals (`_fit_weighted`, `_fit_stochastic_weighted`,
`_backprop_weighted`) purely to get sample-weighted MLP training for the
`Balanced Weights` baseline. scikit-learn#25646 landed `sample_weight` in
`MLPClassifier.fit` upstream, so the copy was deleted and `class_weight=
'balanced'` is now an ordinary weighted fit — same algorithm, maintained by
sklearn. The file went 499 → 155 lines. **Do not reintroduce private sklearn
imports.**

Watch for two things the upgrade surfaced:

- `_validate_data` was removed in sklearn 1.6; use
  `sklearn.utils.validation.validate_data(self, X, ...)`.
- `Series.to_numpy()` can return a **read-only** array under numpy 2, so
  in-place relabelling (`y[y == 2] = 0`) raises. Three loaders had rotted this
  way. Copy first.

## Repo map

```
deltas/              the package (see deltas/README.md)
  core/              ⭐ DeltasEstimator + the slot base classes + registry
  bounds/            ⭐ the concentration inequalities, one file per family
  confidence/        delta handling (fixed / optimised)
  rules/             minimax, sum, risk, Neyman-Pearson
  search/            candidate boundaries
  transforms/        monotone score maps (identity, logit, Yeo-Johnson)
  methods/           ⭐ named methods — what the runners select by name
  legacy/            FROZEN: the code behind every reported number
    ecai2024/          the published method (base, downsample, equations, radius, optimisation)
    non_separable/     the Dec 2024 draft (non_sep, data_info)
    exploratory/       SSL, reprojection, SVM_supports
    overlap/           the original CP/DKW code (reference for the shim)
  model/             old import paths: aliases of legacy/, and overlap.py (a shim)
  pipeline/          data -> classifier -> evaluation glue used by experiments
  data/loaders/      one loader per dataset
  misc/use_two.py    GLOBAL config flags read by the legacy code only
experiments/         ⭐ the current runner — start here for new experiments
  export_projections.py  fits models under the SIBLING venv (see below)
  run_wide.py            the 32-dataset x 7-model x 2-calibration-mode grid
notebooks-ECAI/      experiments + figures for the published paper (legacy)
notebooks-non-sep/   experiments for the non-separable follow-up (legacy)
notebooks/           scratch/dev notebooks
dev/                 dead-end / exploratory scripts (MNIST, MIMIC, large-margin)
jonny/               collaborator's independent implementation
data/                large local datasets (MIMIC-III/IV, MNIST, IMDB) — gitignored
cache/               joblib cache of datasets + trained classifiers — gitignored
tests/golden/        exact outputs of every legacy estimator, both USE_TWO settings
tests/modular/       the modular components, estimator contract, equivalence
```

## The modular deltas — how to add a method

Every method is one pipeline with six slots: **bound** (the concentration
inequality: how much of class *i* lies beyond *b*), **confidence** (how δ is
set), **rule** (how two per-class curves give one *b*), **search** (which *b*
are tried), **transform** (a monotone map of the score) and **certify** (what
is reported). `deltas.core.DeltasEstimator` runs it; each slot takes a
registered name, a `(name, kwargs)` pair or a component object.

- A new concentration inequality is a `core.Bound` subclass in `deltas/bounds/`
  with `@register('bound', 'name')`. Test its coverage by simulation in
  `tests/modular/`.
- A new rule, δ policy, search or transform goes in its folder the same way.
- A new named method is a `Method` in `deltas/methods/` (`envelope.py` for the
  shape-aware family). The runners pick methods from `deltas.methods.METHODS`.
  `run_wide.py --methods` accepts any registered name.

**Never edit `deltas/legacy/` to change behaviour.** It produced the published
numbers and the 34-dataset grid. `tests/golden/` pins its exact outputs under
both `USE_TWO` settings. If a number is *meant* to move, re-record with
`uv run python tests/golden/record.py` and say so in the commit. Old import
paths (`deltas.model.base`, `deltas.utils.radius`, ...) are aliases of the
legacy modules: the same objects, so notebooks are unaffected.
`deltas.model.overlap` is a shim over `DeltasEstimator`, bit-identical to
`legacy/overlap/`.

## Running experiments

**Use `experiments/`, not the `notebooks-*/run_all*.py` scripts.** The latter
are kept for provenance but have the reproducibility problems in `FINDINGS.md`
§6 (open-ended seed search that hides failures, truncated numbers, no raw
output, no config recorded).

```bash
cd experiments
python run_experiments.py                 # all datasets, all methods, seeds 0-9
python run_experiments.py --datasets 2 3  # by index (see EXPERIMENTS)
python run_experiments.py --seeds 30
python combine_tables.py                  # multi-row LaTeX for the draft
python make_figures.py                    # figures into the Overleaf draft
```

Outputs: `experiments/results/raw/*.csv` (per-seed, keep these),
`agg-*.csv`, `Results-*.txt` (LaTeX fragments), `config.json`.

Classifier training is cached via `deltas/pipeline/cached.py`, so the ~270 s
per-seed MIMIC MLP training is paid once. The cache key includes library
versions and a `CACHE_VERSION` — bump `deltas/utils/cache.py::CACHE_VERSION` by
hand if you change what a cached artefact *means* without changing its config.
Inspect with `deltas.utils.cache.info()`, wipe with `cache.clear()`.

## The sibling repos (`toy_datasets`, `projection_models`) — prefer them

`/home/matt/Repos/toy_datasets` (54 datasets) and
`/home/matt/Repos/projection_models` (11 model families exposing
`get_projection`) are now ordinary dependencies, and **new work should use them
rather than the local loaders and models**:

```python
from deltas.data.loaders import sibling as data_sibling
from deltas.classifiers import sibling as clf_sibling

train, test = data_sibling.get_sibling_dataset('Thyroid Sick', seed=0, ratio=10)
clf = clf_sibling.build('RandomForest').fit(X, y)   # get_bias() shim included
```

`deltas.pipeline.data.get_real_dataset` falls through to `toy_datasets`
automatically for any name it does not recognise, so
`get_real_dataset('Stroke Prediction')` just works.

Why prefer them: every dataset the papers use exists in `toy_datasets` plus ~30
more, and it is maintained against current numpy/sklearn (three local loaders
had silently rotted under numpy 2). `projection_models` covers 11 model
families against the local 3, and its MLP supports `sample_weight` /
`class_weight` directly.

**What is kept local, and why:** `deltas/data/loaders/*.py` and
`deltas/classifiers/models.py` produced the published results — their exact
shuffling and hyperparameters define those splits and numbers. Keep them
working; do not delete them.

The one API mismatch is naming and sign: `projection_models` reports
`get_threshold()` (`predict = projection > t`), deltas wants `get_bias()`
(`t = -b`). `deltas/classifiers/sibling.py::as_deltas_classifier` bridges it.

Watch out for two things found the hard way:

1. `toy_datasets.proportional_split`'s `minority_reduce_scaler` **is the target
   ratio** — the minority training count is set to `len(majority_train) /
   scaler`, so it must be sized against the majority count. Asking for more
   minority training points than exist leaves that class an empty test index
   list, and `np.concatenate` then returns a float array which fails as an
   index.
2. Its convention (`class 1 is the minority`) matches this repo's, but the raw
   loaders do not guarantee it — relabel first.

## Conventions that matter

- **Class 1 is always the minority / positive class.** Loaders relabel to
  enforce this (see `sklearn_toy.get_breast_cancer`, `MIMIC_III.get_mortality`).
  Metrics treat class 1 as positive.
- Any classifier passed to a deltas model **must** expose `get_projection(X) ->
  (n, 1)` and ideally `get_bias()`. See `deltas/classifiers/models.py`.
- Deltas estimators are sklearn-shaped: `.fit(X, y)`, `.predict(X)`,
  `.get_bias()`, `.is_fit`. `is_fit == False` means *no solution was found*.
  `experiments/` records it as unsolved; the old `notebooks-*/run_all*.py`
  scripts silently skipped the seed. `DeltasEstimator` is a real sklearn
  `BaseEstimator` (`clone`, `get_params` work); certificates are in
  `certified_error()` / `certificates_`.
- `deltas/misc/use_two.py` holds **module-level global flags** that change the
  maths (`USE_TWO` toggles the factor of 2 in the concentration bound). Results
  directories `results-two/`, `results-two2/` are ablations produced by hand-
  editing this file. Treat changes to it as changing every experiment.
  ⚠️ **HEAD has `USE_TWO = True`; the published ECAI results used `False`.**
  `True` is the mathematically correct setting (derivation in `FINDINGS.md`
  §5A: one `ε` lifts the training max into true coordinates, a second drops the
  test point back into empirical ones) — the paper's Eq. 5 drops it while its
  own Eq. 4 keeps it. So don't "fix" HEAD back to `False`; it is the published
  numbers that used the loose bound.
  Only the legacy code reads it — new components take every setting (e.g. the
  fence's `factor`) as an explicit parameter, and `deltas.methods` imports the
  legacy code lazily. To reproduce the paper, set it before any other `deltas`
  import (the legacy modules bind the value at import time):
  ```python
  import deltas.misc.use_two as ut; ut.USE_TWO = False
  from deltas.pipeline import data, classifier, evaluation   # after, not before
  ```
  Details and the verified numbers are in `FINDINGS.md` §6.0.

## Style

Match the existing style: plain numpy/sklearn, `_print` / `_plot` keyword flags
for verbosity. Results and certificates are plain dicts. The modular code is the
one agreed exception to "no classes for configuration": components are small
classes (they have to be, to slot in), and per-class data is a
`core.ClassSample` — the successor of both legacy `data_info` forms (the dict in
`legacy/ecai2024/base.py`, the class in `legacy/non_separable/data_info.py`).
Don't mix the three.

Comparisons are written `if x == True:` throughout. Leave existing ones alone.

## Gotchas found in the code

Full list with reproductions in `FINDINGS.md`. The ones most likely to bite:

Items 1–3 are in the frozen `deltas/legacy/` code and are kept on purpose
(they feed the published numbers); `base_deltas.fit` also crashes on an
infeasible problem (the "infeasible-fit crash", FINDINGS B11).

1. `optimise_deltas.optimise()` filters the grid with `J[constraints != 0]` —
   exact float equality. ~25% of genuinely valid grid points get discarded by
   1e-16 residuals. `tol_constraint = 1e-6` is defined and unused.
2. `downsample.fit()` reads `support_max_hit` on a path where it may be
   unassigned; safe today only via `or` short-circuiting. Don't reorder that
   condition. The `method` name validation is commented out at line 60.
3. `non_sep.optimise()` calls `np.delete(line, 0)` without assigning the result
   (no-op; harmless because `get_valid_linspace` already filtered).
4. `heart_disease.get_HD` hits the UCI network API on every call.
5. ⚠️ **`seed == True` also matches the integer `1`** (python: `1 == True`).
   `data/utils.py` used that idiom, so seed 1 was silently replaced by
   `RANDOM_STATE = 0` and **seeds 0 and 1 gave identical data** — every
   `range(10)` run really used nine distinct datasets. Fixed with `is True`;
   the comment there says not to tidy it back. The repo's house style really is
   `== True` everywhere else, so leave those alone, but never for a seed.
   (`toy_datasets` has the same idiom with `RANDOM_STATE = 42`, which is
   harmless for seeds 0–9 but the same trap outside that range.)
