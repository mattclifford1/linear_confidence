# `deltas/confidence` — how δ is handled

| file | registry name | what |
|---|---|---|
| `fixed.py` | `fixed` | `FixedDelta(δ)`: δ set in advance. The only setting under which a high-probability bound can be *quoted* |
| `optimised.py` | `optimised` | `OptimisedDelta(resolution, form)`: per boundary, the δ minimising `(1 − δ)U + δ` (`form='expected'`, the published loss) or `U + δ` (`'union'`). A good decision device, but the δ it picks is chosen from the data, so certify at a fixed δ instead. For count bounds it builds one table over m (with the ternary-search path for large N), exactly as the overlap methods always did |

`NoDeltaCurve` handles bounds with no confidence level (average-case curves)
and bounds that fix δ themselves (the published fence).
