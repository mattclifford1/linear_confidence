# `deltas/search` — which boundaries are tried

One candidate set per file, named after its registry name. `base.py` holds
the slot's base class `CandidateSet` and `_all_values`, the sorted distinct
scores of both classes that the candidate sets share.

| file | registry name | what |
|---|---|---|
| `data_midpoints.py` | `data_midpoints` | midpoints between consecutive distinct scores, plus one beyond each end. Exact for count-based curves, which only change at data points |
| `grid.py` | `grid` | `Grid(n, margin)`: n evenly spaced boundaries over the data range widened by `margin` × range |
| `auto.py` | `auto` | midpoints if every curve is a step function at data points, else midpoints ∪ grid (the default) |

For smooth curves the estimator then refines the grid choice between its
neighbours (root of L₁ − L₂ for minimax, 1-D minimisation for sum).
