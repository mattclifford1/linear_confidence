# `experiments` — the current experiment runner

Supersedes `notebooks-ECAI/run_all*.py` and `notebooks-non-sep/run_all_non_sep.py`,
which are kept for provenance but have the reproducibility problems documented
in `FINDINGS.md` §6.

## What is different from the old runners

| | old `run_all*.py` | here |
|---|---|---|
| seeds | searched `0,1,2,…` until 10 *successes* | fixed list, `range(10)` by default |
| a method that finds no solution | seed silently dropped **for every method, including the baselines** | recorded as `NaN`, reported as a `solved n/10` column |
| per-seed numbers | discarded | written to `results/raw/*.csv` |
| number formatting | truncated (`str(x)[1:5]`, so `.6729 → .672`) | rounded |
| config (`USE_TWO`, seeds, versions) | not recorded anywhere | LaTeX comments + `results/config.json` |
| classifier training | retrained every script, every seed | cached (`deltas/pipeline/cached.py`) |

The seed fix matters: under the old scheme the *Baseline* score on Breast Cancer
moved from .917 to .912 purely because `USE_TWO` changed which seeds survived
the filter — a flag that cannot touch the baseline classifier.

## Usage

```bash
# use `uv run <cmd>` from the repo root
cd experiments

uv run python run_experiments.py                  # all datasets, all methods, seeds 0-9
uv run python run_experiments.py --datasets 2 3   # by index, see EXPERIMENTS in the script
uv run python run_experiments.py --seeds 30       # more seeds
uv run python run_experiments.py --no-cache       # bypass the classifier cache

uv run python combine_tables.py                   # multi-row LaTeX table for the draft
uv run python make_figures.py                     # both figures -> the Overleaf draft
uv run python make_figures.py 1                   # just the loss-landscape figure
```

## Outputs (`results/`)

| File | What |
|---|---|
| `raw/<n>-<dataset>.csv` | one row per (seed, method) — **keep these**, they are what makes significance testing possible |
| `agg-<n>-<dataset>.csv` | mean/std/solved per method |
| `Results-<n>-<dataset>.txt` | LaTeX `tabular` fragment with a config header |
| `combined_table-groundup.txt` | the multi-row table pasted into the draft |
| `config.json` | seeds, flags, python version, cache summary |
| `overlap_sweep.joblib` | memoised sweep data for `make_figures.py 2` |

## Methods under test

Literature baselines come from `pipeline.classifier.get_classifier`
(Baseline, SMOTE, Balanced Weights, BMR, Threshold). The deltas methods are in
`DELTAS_METHODS` at the top of `run_experiments.py`:

- `Slacks Deltas` — the published ECAI method (`model/downsample.py`)
- `Min / Max / Avg / F Deltas` — the non-separable loss variants (`model/non_sep.py`)
- `CP Sum / CP Minimax` — Clopper–Pearson overlap-native (`model/overlap.py`)
- `DKW Sum / DKW Minimax` — DKW overlap-native

Add a method by adding one lambda to that dict.

## Timing

Everything except MIMIC-III is seconds per seed. MIMIC is ~330 s per seed on the
**first** pass (training three MLPs) and ~35 s thereafter, because the classifier
cache makes training a one-off cost. Wipe with
`python -c "import deltas.utils.cache as c; c.clear()"`.

## The wide grid (`export_projections.py` + `run_wide.py`)

The scripts above cover the six datasets of the papers. The wide grid covers
**32 datasets × 7 models × 10 seeds × 2 calibration modes**, using the sibling
repos `../../Repos/toy_datasets` and `../../Repos/projection_models`.

### Why it is still two steps

It used to be two *environments*: the siblings need sklearn ≥ 1.6 and this repo
was pinned to 1.3.2 by the vendored `MLPClassifier` internals, so models were
fitted under the sibling venv and their projections shipped across. That pin is
gone (see `CLAUDE.md`) and everything runs under one `uv` environment now.

The export step is kept because it is still worth having:

- **it is a cache.** Fitting 224 (dataset, model) pairs × 10 seeds × 2
  calibration modes takes ~30 min; the deltas methods are then re-runnable in
  minutes without refitting anything.
- **it keeps the classifier out of the deltas code.** A deltas estimator needs
  exactly one thing from a classifier, `get_projection(X) -> (n, 1)`. Exporting
  that and nothing else makes the classifier-agnostic claim structural rather
  than incidental — `run_wide.py` never sees a model.
  `deltas/classifiers/frozen.py::FrozenProjection` is the identity shim that
  satisfies the estimators' constructor checks.

```bash
# step 1 - fit models and export projections (resumable; --force to redo)
uv run python export_projections.py --seeds 10 --jobs 8

# step 2 - run every deltas method over them
uv run python run_wide.py --seeds 10 --jobs 8

# step 3 - coverage, ranks, solve rates, LaTeX
uv run python analyse_wide.py

# or all of it
./reproduce_wide.sh
```

Outputs: `projections/<dataset>__<model>.npz` (one per pair, all seeds and both
modes), then `results/wide.csv`, `results/wide_coverage.csv`,
`results/wide_ranks.csv`, `results/wide_solve.csv`.

### Grid

- **Models** (`MODELS` in `export_projections.py`): Linear, LDA, SVM-rbf, MLP,
  RandomForest, GradientBoosting, NearestClassMean. `NearestClassMean` is the
  interesting control — centroid-plus-radius is exactly the geometry the
  published ECAI derivation assumes.
- **Datasets**: 4 synthetic, 26 tabular (natural imbalance from 1.1:1 to
  19.5:1), 2 MedMNIST image sets flattened to 784 features.
- **Hyperparameters are left at defaults**, unlike `run_experiments.py` which
  grid-searches the SVM. The claim under test is about post-hoc bias
  correction, not tuning, and a 5-fold search over 63 parameter settings per
  cell would dominate the runtime.
- **Splitting** follows the deltas convention: half the data to train, the
  training minority thinned towards 10:1, balanced test set. Datasets that
  cannot leave ≥ 8 minority training and ≥ 10 minority test points are skipped.

### Calibration modes

Each cell is run twice:

- `naive` — classifier fitted on all training data, certificate computed on the
  same points (what the papers do)
- `split` — classifier fitted on 65%, certificate computed on the held-out 35%

See `CALIBRATION.md` for what this fixes and why it cannot live inside the
estimator.

## Caveat on the `Slacks Deltas` rows

With `USE_TWO=True` the published slack method frequently finds no solution on
these fixed seeds (Pima 0/10, Heart Disease 4/10). Where the `solved` count is
low, its mean is taken over only those seeds and is **not** comparable with the
other rows. That is the point of reporting the column.
