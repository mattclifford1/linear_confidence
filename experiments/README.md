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
conda activate deltas
cd experiments

python run_experiments.py                  # all datasets, all methods, seeds 0-9
python run_experiments.py --datasets 2 3   # by index, see EXPERIMENTS in the script
python run_experiments.py --seeds 30       # more seeds
python run_experiments.py --no-cache       # bypass the classifier cache

python combine_tables.py                   # multi-row LaTeX table for the draft
python make_figures.py                     # both figures -> the Overleaf draft
python make_figures.py 1                   # just the loss-landscape figure
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

## Caveat on the `Slacks Deltas` rows

With `USE_TWO=True` the published slack method frequently finds no solution on
these fixed seeds (Pima 0/10, Heart Disease 4/10). Where the `solved` count is
low, its mean is taken over only those seeds and is **not** comparable with the
other rows. That is the point of reporting the column.
