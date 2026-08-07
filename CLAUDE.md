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

Other docs in the tree: `deltas/model/README.md` (per-class API reference),
`deltas/data/loaders/readme.md`, `notebooks/README.md`,
`notebooks-ECAI/README.md`, `notebooks-non-sep/README.md`.

## Environment

There is a working conda env — **use it, don't create a new one**:

```bash
/home/matt/anaconda3/envs/deltas/bin/python   # python 3.10.13, sklearn 1.3.2, numpy 1.26.4
# or: conda activate deltas
```

The package is installed editable, so `import deltas` resolves to this working
tree. Packaging moved to **pdm** in `c1c0a96` — deps are in `pyproject.toml`
(`pdm install`, or `pip install -e .` via PEP 517). `setup.py` and
`requirements.txt` no longer exist. `pyproject.toml` declares no version
bounds; the conda env above is the known-good set.

**Do not upgrade scikit-learn.** `deltas/classifiers/models.py` vendors ~350
lines of sklearn 1.3.x `MLPClassifier` internals (`_fit_weighted`,
`_fit_stochastic_weighted`, `_backprop_weighted`) to get sample-weighted
training. It imports private symbols from
`sklearn.neural_network._multilayer_perceptron` and will break on upgrade.

## Repo map

```
deltas/              the package (see deltas/README.md)
  model/             the deltas estimators — this is the core research code
  optimisation/      scipy / grid-search over delta_1
  utils/equations.py the paper's equations (loss, constraint, delta2(delta1))
  pipeline/          data -> classifier -> evaluation glue used by experiments
  data/loaders/      one loader per dataset
  misc/use_two.py    GLOBAL config flags (USE_TWO, USE_GLOBAL_R, RANDOM_STATE)
notebooks-ECAI/      experiments + figures for the published paper
notebooks-non-sep/   experiments for the non-separable follow-up
notebooks/           scratch/dev notebooks
dev/                 dead-end / exploratory scripts (MNIST, MIMIC, large-margin)
jonny/               collaborator's independent implementation
data/                large local datasets (MIMIC-III/IV, MNIST, IMDB) — gitignored
```

## Running experiments

```bash
cd notebooks-ECAI     && python run_all.py            # ECAI slacks method, 5 datasets
cd notebooks-non-sep  && python run_all_non_sep.py    # non-sep loss variants
cd notebooks-ECAI     && python Guassian_plots.py     # paper's synthetic figures
cd notebooks-ECAI     && python projection_plots.py   # paper's diagram figures
```

`run_all*.py` writes LaTeX table fragments into `results*/` and then
`combine_tables()` stitches them into `combined_table*.txt`, which is pasted
straight into the `.tex`. Note `main()` in `run_all_non_sep.py` currently has
the experiment loop **commented out** — it only re-combines existing tables.

MIMIC-III is the slow one (~10+ min just to train the three MLPs for one seed,
×10 seeds). Everything else is seconds. Anything that trains a classifier
should be cached — see `FINDINGS.md` §"Caching".

## Conventions that matter

- **Class 1 is always the minority / positive class.** Loaders relabel to
  enforce this (see `sklearn_toy.get_breast_cancer`, `MIMIC_III.get_mortality`).
  Metrics treat class 1 as positive.
- Any classifier passed to a deltas model **must** expose `get_projection(X) ->
  (n, 1)` and ideally `get_bias()`. See `deltas/classifiers/models.py`.
- Deltas estimators are sklearn-shaped: `.fit(X, y)`, `.predict(X)`,
  `.get_bias()`, `.is_fit`. `is_fit == False` means *no solution was found* —
  the experiment runners silently skip that seed and try the next one.
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
  To reproduce the paper, set it before any other `deltas` import (modules bind
  the value at import time):
  ```python
  import deltas.misc.use_two as ut; ut.USE_TWO = False
  from deltas.pipeline import data, classifier, evaluation   # after, not before
  ```
  Details and the verified numbers are in `FINDINGS.md` §6.0.

## Style

Match the existing style: plain numpy/sklearn, `_print` / `_plot` keyword flags
for verbosity, dicts (`data_info`) passed around rather than dataclasses.
The newer `deltas/model/data_info.py` class is the successor to the
`get_data_info()` dict in `base.py` — the non-separable code uses the class, the
ECAI code uses the dict. Don't mix them.

Comparisons are written `if x == True:` throughout. Leave existing ones alone.

## Gotchas found in the code

Full list with reproductions in `FINDINGS.md`. The ones most likely to bite:

1. `optimise_deltas.optimise()` filters the grid with `J[constraints != 0]` —
   exact float equality. ~25% of genuinely valid grid points get discarded by
   1e-16 residuals. `tol_constraint = 1e-6` is defined and unused.
2. `downsample.fit()` reads `support_max_hit` on a path where it may be
   unassigned; safe today only via `or` short-circuiting. Don't reorder that
   condition. The `method` name validation is commented out at line 60.
3. `breast_cancer_W.get_Wisconsin_breast_cancer` calls `shuffle_data(data)`
   without the seed → that dataset is not reproducible.
4. `non_sep.optimise()` calls `np.delete(line, 0)` without assigning the result
   (no-op; harmless because `get_valid_linspace` already filtered).
5. `heart_disease.get_HD` hits the UCI network API on every call.
