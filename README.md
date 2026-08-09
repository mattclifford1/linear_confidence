# deltas — learning confidence bounds for classification with imbalanced data

Post-hoc, classifier-agnostic class-imbalance correction. Given any binary
classifier of the form `f(x) = sgn(⟨φ(x), w⟩ + b)`, **deltas** moves the bias
term `b` to a position justified by class-conditional concentration
inequalities, rather than by a heuristic such as inverse-frequency
thresholding.

The intuition: the minority class has fewer samples, so we are less certain
that its training samples cover its true distribution. That uncertainty is
quantified per class and the decision boundary is pushed away from the minority
class by exactly that amount.

📄 **Paper:** *Learning Confidence Bounds for Classification with Imbalanced
Data*, Clifford, Erskine, Hepburn, Santos-Rodríguez & Garcia-Garcia, ECAI 2024.
[arXiv:2407.11878](https://arxiv.org/abs/2407.11878)

For the research state, open questions, known bugs and planned work see
[`FINDINGS.md`](FINDINGS.md). For the package internals see
[`deltas/README.md`](deltas/README.md).

---

## Install

```bash
git clone https://github.com/mattclifford1/linear_confidence
cd linear_confidence

conda create -n deltas python=3.10 -y
conda activate deltas

pdm install        # or: pip install -e .   (works via PEP 517)
```

Dependencies are declared in `pyproject.toml` (pdm-backend). `setup.py` and
`requirements.txt` were removed in `c1c0a96`.

> **Pin scikit-learn to 1.3.x.** `deltas/classifiers/models.py` vendors private
> `MLPClassifier` internals to support sample-weighted training and will break
> on newer versions. `pyproject.toml` currently declares no version bounds at
> all — worth fixing. Known-good: python 3.10.13, scikit-learn 1.3.2,
> numpy 1.26.4, pandas 2.2.1, scipy 1.11.4, imbalanced-learn 0.12.0,
> torch 2.1.1.

`costcla`, `torch`/`torchvision` and `umap-learn` are only needed for datasets
and experiments outside the papers (Credit Scoring, MNIST, UMAP plots). The
BMR and Thresholding baselines use the self-contained `deltas/costcla_local/`,
not the `costcla` package — so these three could be optional extras.

MIMIC-III / MIMIC-IV data is not distributed (licensing) — see
`deltas/data/loaders/MIMIC_III.py` for the processing pipeline it expects, and
place the CSVs under `data/MIMIC-III/`.

## Quick start

Any classifier you pass in must expose `get_projection(X) -> (n, 1)`, giving
the 1-D score before the bias. `deltas.classifiers.models` provides
drop-in `SVM`, `linear` (logistic regression) and `NN` (MLP) subclasses that
already do.

```python
import deltas.classifiers.models as models
from deltas.model import downsample, non_sep, overlap

clf = models.SVM(kernel='rbf').fit(X_train, y_train)   # class 1 = minority

# published method (separable + binary slacks)
d = downsample.downsample_deltas(clf).fit(X_train, y_train, max_trials=10000)

# non-separable follow-up (k-th furthest order statistic)
d = non_sep.deltas(clf).fit(X_train, y_train, loss_type='min')

# overlap-native (recommended): never infeasible, no slacks, no R
d = overlap.binomial_deltas(clf, objective='minimax').fit(X_train, y_train)
d = overlap.dkw_deltas(clf, objective='minimax').fit(X_train, y_train)

d.predict(X_test)
d.get_bias()          # the corrected bias term
d.is_fit              # False => no solution was found (never for overlap.*)
d.certified_error()   # overlap.* only: the bounds actually certified
```

**Convention: class `1` is always the minority / positive class.** All the
dataset loaders relabel to enforce this and all metrics assume it.

`is_fit == False` is a normal outcome, not an error — it means the projected
classes overlap too much for a solution to exist. Handling that case properly
is the current research direction (see `FINDINGS.md` §4).

## Running experiments

Use `experiments/` — it supersedes the `notebooks-*/run_all*.py` scripts:

```bash
cd experiments
python run_experiments.py       # all datasets, all methods, fixed seeds 0-9
python combine_tables.py        # multi-row LaTeX table
python make_figures.py          # figures
```

It writes per-seed raw CSV (so significance tests are possible after the fact),
reports how many seeds each method actually solved rather than silently
dropping failures, rounds instead of truncating, and stamps the config into the
output. Classifier training is cached, so MIMIC's ~270 s/seed is paid once.

The original paper scripts are kept for provenance:

```bash
cd notebooks-ECAI    && python run_all.py           # ECAI Table 3
cd notebooks-ECAI    && python "run_all continuous.py"  # ECAI Table 4 (continuous slacks)
cd notebooks-ECAI    && python Guassian_plots.py    # ECAI Figs 3,4,5,6
cd notebooks-ECAI    && python projection_plots.py  # ECAI Figs 1,2
cd notebooks-non-sep && python run_all_non_sep.py   # non-separable draft tables
```

⚠️ **HEAD does not reproduce the published paper as-is.** `deltas/misc/use_two.py`
has `USE_TWO = True` (set for the non-separable follow-up); the ECAI results
used `False`. Verified on Breast Cancer, seeds 0–9: `USE_TWO=False` gives
Baseline .917/.917/.914 and Our Method .945/.943/.947 (published: .917/.917/.913
and .943/.942/.946); `USE_TWO=True` gives .912/.912/.908 and .951/.950/.954.

Also read `FINDINGS.md` §6 before trusting a re-run — the runners use an
open-ended seed search that silently drops seeds where deltas finds no solution
(which shifts even the *baseline* numbers), keep only aggregated `mean ± std`,
and truncate rather than round.

Per-directory notes: [`notebooks-ECAI/README.md`](notebooks-ECAI/README.md),
[`notebooks-non-sep/README.md`](notebooks-non-sep/README.md).

## Repository layout

```
deltas/               the installable package — see deltas/README.md
  model/              the estimators        — see deltas/model/README.md
  data/loaders/       dataset loaders       — see deltas/data/loaders/readme.md
notebooks-ECAI/       experiments + figures for the published paper
notebooks-non-sep/    experiments for the non-separable follow-up
notebooks/            scratch / development notebooks — see notebooks/README.md
dev/                  exploratory dead ends (MNIST, MIMIC, large-margin nets)
jonny/                a collaborator's independent implementation
data/                 large local datasets (gitignored; MIMIC, MNIST, IMDB)
```

## Citation

```bibtex
@inproceedings{clifford2024learning,
  title     = {Learning Confidence Bounds for Classification with Imbalanced Data},
  author    = {Clifford, Matt and Erskine, Jonathan and Hepburn, Alexander
               and Santos-Rodr{\'i}guez, Ra{\'u}l and Garcia-Garcia, Dario},
  booktitle = {ECAI},
  year      = {2024}
}
```
