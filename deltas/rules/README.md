# `deltas/rules` — how two curves become one boundary

| file | registry name | chooses b to | notes |
|---|---|---|---|
| `minimax.py` | `minimax` | min max(L₁, L₂) | equal errors; guards the G-mean. Ties broken at the plateau midpoint (in value) |
| `balanced.py` | `sum` | min L₁ + L₂ | balanced error. `tie_break='midpoint'` (default) or `'first'` (the original overlap behaviour, kept in the shim) |
| `risk.py` | `risk` | min π₁c₁L₁ + π₂c₂L₂ | priors × costs (FINDINGS.md §13); priors default to the data's proportions |
| `neyman_pearson.py` | `neyman_pearson` | min L_other s.t. L_protected ≤ α | "miss at most α of the minority"; reports `constraint_met` |

A rule can see the data (`bind`) and set the per-class weights
(`class_weights`); it gets a private copy per fit.
