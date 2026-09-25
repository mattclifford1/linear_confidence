# `deltas` package internals

Map of the code: how a deltas method is put together, where each piece lives,
and how the frozen code behind the published numbers is kept apart.

## The pipeline in one line

Every deltas method, old or new, is the same pipeline with six slots:

```
scores ─[transform]→ per-class samples ─[bound]→ ─[confidence]→ curves L₁(b), L₂(b)
       over [search] candidates ─[rule]→ boundary b ─[certify]→ certificates
```

Everything the method needs lives in the **projected (1-D) space**: the
classifier is only ever used as a projection function (`get_projection(X) ->
(n, 1)`), which is what makes the method classifier-agnostic.

| slot | question it answers | folder | options |
|---|---|---|---|
| bound | how much of class *i* lies beyond *b*? | `bounds/` | Clopper–Pearson, DKW; Gaussian/logistic/*t* envelopes; Student-*t* predictive; Saw–Yang–Mo, Cantelli, VP; the published fences |
| confidence | how is δ set? | `confidence/` | `FixedDelta(δ)`, `OptimisedDelta` (the published loss) |
| rule | how do two curves give one *b*? | `rules/` | minimax, sum, risk, Neyman–Pearson |
| search | which *b* are tried? | `search/` | data midpoints, grid, auto |
| transform | which monotone map of the score? | `transforms/` | identity, logit, standardise, Yeo–Johnson |
| certify | what is reported? | (estimator argument) | the deciding curves, or any bounds at a fixed δ |

## Folder map

```
deltas/
├── core/          DeltasEstimator + the slot base classes + ClassSample/ProjectedData + registry
├── bounds/        the concentration inequalities (one file per family)
├── confidence/    delta handling
├── rules/         decision rules
├── search/        candidate boundaries
├── transforms/    monotone score maps
├── methods/       named methods (what the experiment runners select by name)
├── legacy/        FROZEN: the code behind every reported number
│   ├── ecai2024/        the published method (base, downsample, equations, radius, optimisation)
│   ├── non_separable/   the Dec 2024 draft (non_sep, data_info)
│   ├── exploratory/     SSL, reprojection, SVM_supports
│   └── overlap/         the original Clopper–Pearson/DKW code (reference for the shim)
├── model/         old import paths: aliases of legacy/ modules, and overlap.py (a shim)
├── optimisation/  old import paths: aliases of legacy/ecai2024
├── utils/         cache.py, projection.py, data.py; equations.py/radius.py are aliases
├── pipeline/      data -> classifier -> evaluation glue used by the experiments
├── classifiers/   models exposing get_projection (local, sibling, frozen)
├── data/          dataset loaders
├── plotting/      plots.py
├── costcla_local/ vendored BMR / Thresholding baselines
└── misc/          use_two.py: the global flags the legacy code reads
```

Each new folder has its own short README.

## Using it

```python
from deltas.core import DeltasEstimator
from deltas.bounds import StudentTPredictive, GaussianConfidence

model = DeltasEstimator(clf,
                        bound=StudentTPredictive(),       # decide (average-case)
                        rule='minimax',
                        certify=[GaussianConfidence(), 'clopper_pearson'],
                        delta_report=0.05)
model.fit(X_cal, y_cal)          # the calibration split (CALIBRATION.md)
model.predict(X_test); model.get_bias(); model.certificates_

# or by name, as the experiment runners do
from deltas.methods import METHODS
fitted = METHODS['CP Minimax'](clf, X, y)
```

Slots take a registered name (`'clopper_pearson'`), a `(name, kwargs)` pair
(`('fixed', {'delta': 0.1})`) or a component object. `bound` may also be a dict
by class label, e.g. counts for a large majority and a Gaussian envelope for a
scarce minority. `DeltasEstimator` is an sklearn `BaseEstimator`: `clone`,
`get_params` and `set_params` work, and component objects passed in are never
mutated. `model.describe()` gives a JSON-friendly spec for results files.

The default composition (Clopper–Pearson, optimised δ, minimax, data
midpoints) is bit-for-bit `deltas.model.overlap.binomial_deltas(objective='minimax')`.

## Adding something new

| to add | do |
|---|---|
| a concentration inequality | subclass `core.Bound` (or `core.CountBound` if it only depends on the count of wrong-side points) in a new file under `bounds/`, decorate with `@register('bound', 'name')`, export it from `bounds/__init__.py` |
| a way to handle δ | subclass `core.DeltaPolicy` in `confidence/` |
| a decision rule | subclass `core.DecisionRule` in `rules/` (override `bind` if it needs the data, `class_weights` if it reweights the classes) |
| a search | subclass `core.CandidateSet` in `search/` |
| a transform | subclass `core.Transform` in `transforms/` (set `requires_fit` if it learns from data) |
| a named method | add a `Method` to `methods/envelope.py` (or a new module listed in `methods/__init__.py`) |

Tests for a new component go in `tests/modular/`. Check coverage by simulation
for any high-probability bound.

## Legacy code

`deltas/legacy/` is frozen: it is the code behind the published numbers and
the 34-dataset grid, moved verbatim (only its own import lines changed). Known
bugs that feed the published results are deliberately kept: the exact-float
grid filter, the possibly-unassigned `support_max_hit`, the silent
`max_trials` cap and the infeasible-fit crash (`FINDINGS.md` §7.1, rows B1,
B2, B8, B11). `tests/golden/` pins the exact outputs of every legacy
estimator under both settings of `USE_TWO`. Fix forward in the modular code,
never in `legacy/`.

The old import paths still work and are the *same* module objects
(`deltas.model.base is deltas.legacy.ecai2024.base`), so every notebook and
script is unaffected. `deltas/model/overlap.py` is the one exception: it is a
shim over `DeltasEstimator`, proven bit-identical to the original in
`legacy/overlap/`.

`misc/use_two.py` holds module-level flags (`USE_TWO`, `USE_GLOBAL_R`,
`RANDOM_STATE`) read at import time by the legacy code only. To reproduce the
paper, set `USE_TWO = False` before importing anything else from `deltas`.
`deltas.methods` imports the legacy code lazily, so importing it first is
safe. New components never read these flags: every setting (e.g. the fence's
`factor`) is an explicit parameter.

## `pipeline/`, `classifiers/`, `data/`

| module | role |
|---|---|
| `pipeline/data.py` | `get_real_dataset(name, seed, scale)` dispatches to `data/loaders/` and falls through to the sibling `toy_datasets` |
| `pipeline/classifier.py` | trains the baseline and every comparison method (SMOTE, Balanced Weights, BMR, Threshold) in one call |
| `pipeline/evaluation.py` | Accuracy / G-Mean / F1, and projected-space boundary plots |
| `pipeline/cached.py` | disk-cached datasets and classifiers; `calibration=0.35` holds out a calibration split |
| `pipeline/calibration.py` | the calibration split (`CALIBRATION.md`) |
| `classifiers/models.py` | `SVM`, `linear`, `NN` with `get_projection` / `get_bias` (the published results' models) |
| `classifiers/sibling.py` | the 11 `projection_models` families, with a `get_bias` shim — prefer these for new work |
| `classifiers/frozen.py` | `FrozenProjection`: the identity projector over pre-computed projections |
| `data/loaders/` | one module per dataset; class 1 is always the minority |

See also `data/loaders/readme.md` and `CALIBRATION.md`.
