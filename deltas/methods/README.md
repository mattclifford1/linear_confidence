# `deltas/methods` — named methods

The single place the experiment runners get their methods from. Each is a
`Method`: `callable(clf, X, y) -> fitted model`, with `make(clf)`,
`configured(**fit_options)`, `with_params(**estimator_params)` and `describe()`.

| family | file | methods |
|---|---|---|
| published | `legacy.py` | Slacks Deltas, Slacks Deltas (continuous) |
| non_separable | `legacy.py` | Min / Max / Avg / F Deltas |
| overlap | `overlap.py` | CP Sum, CP Minimax, DKW Sum, DKW Minimax |
| band | `band.py` | Gaussian Predictive, Gaussian Band, Gaussian Band (MC), Logistic Band, Gaussian Neyman-Pearson, Saw-Yang-Mo Minimax |
| ablation | `ablations.py` | ECAI Tolerance Limit (modular), ECAI Tolerance Limit (one-sided radius), k-th Point Tolerance Limit (min, expected loss) |

The legacy entries import their code lazily: the legacy modules read `USE_TWO`
at import time, and importing `deltas.methods` must not fix the flag before a
caller sets it.

```python
from deltas.methods import METHODS, available
METHODS['Slacks Deltas'].configured(max_trials=2000, parallel=False)   # as run_wide does
available('band')
```
