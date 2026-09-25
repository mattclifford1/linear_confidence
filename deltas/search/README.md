# `deltas/search` — which boundaries are tried

| registry name | what |
|---|---|
| `data_midpoints` | midpoints between consecutive distinct scores, plus one beyond each end. Exact for count-based curves, which only change at data points |
| `grid` | `Grid(n, margin)`: n evenly spaced boundaries over the data range widened by `margin` × range |
| `auto` | midpoints if every curve is a step function at data points, else midpoints ∪ grid (the default) |

For smooth curves the estimator then refines the grid choice between its
neighbours (root of L₁ − L₂ for minimax, 1-D minimisation for sum).
