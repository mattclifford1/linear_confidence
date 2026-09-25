# `deltas/core` — the abstract machinery

Nothing here knows about any particular inequality.

| file | what |
|---|---|
| `estimator.py` | `DeltasEstimator`: the one sklearn-shaped class every modular method is an instance of. It projects, orients the classes, fits one bound per class, resolves δ, searches, applies the rule (with optional refinement for smooth curves) and reports certificates |
| `components.py` | the slot base classes: `Bound`, `CountBound`, `DeltaPolicy`, `PreparedCurve`, `DecisionRule`, `CandidateSet`, `Transform` |
| `sample.py` | `ClassSample` (one class's sorted scores and which side of the boundary it sits on) and `ProjectedData` (both, oriented by mean). The successor of `data_info` |
| `registry.py` | name → component; `register`, `get`, `resolve`, `available` |

Flags on a `Bound` that the estimator reads:

| flag | meaning |
|---|---|
| `guarantee` | `'high_probability'` (U ≥ e at every b at once, w.p. ≥ 1 − δ) or `'average_case'` (the expected error of a rule placing b there) |
| `needs_delta` | False for curves with no confidence level (predictive, Saw–Yang–Mo) |
| `continuous` | False when the curve only changes at data points: the data midpoints then suffice, and no grid is needed |
| `smooth` | False when the curve has flat steps anywhere: no root-finding refinement |
| `self_resolving` | True when the bound fixes δ itself from b (the published fence) |

The fit, in order: `ProjectedData` → `bound.fit(sample)` per class (deep copies)
→ `confidence.prepare(bound)` → `search.generate` → curves over the candidates
→ `rule.bind(data)` (a private copy), `rule.class_weights`, `rule.choose` →
refine if every curve is smooth → recount at the boundary → certificates.
