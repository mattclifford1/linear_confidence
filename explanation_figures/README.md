# `explanation_figures` — thesis figures in one style

Thesis explanation figures, regenerated from code in one shared style.

```bash
uv run python explanation_figures/make_all.py              # all figures
uv run python explanation_figures/make_all.py projection   # one module
uv run python explanation_figures/make_all.py --thesis     # all, then copy changed ones into the thesis
uv run python explanation_figures/sync_thesis.py --dry-run # list what a copy would change
```

PDFs go to `out/<chapter>/...`, mirroring the thesis's `chapters/<chapter>/figures/...`.
`sync_thesis.py` copies changed files across (set `THESIS_DIR` to point elsewhere); committing the thesis stays manual.

| Module | Output (under `out/`) | Thesis label |
|---|---|---|
| `imbalance_pipeline.py` | `02-literature-review/imbalance_pipeline` | new, for the class imbalance review |
| `moving_bias.py` | `deltas/diagrams/diagrams_data_original` | `deltas:fig:moving-bias-diagram` |
| `projection.py` | `deltas/diagrams/projection` | `deltas:fig:projection-diagram` |
| `gaussian_classifiers.py` | `deltas/synthetic/Gaussian_data` | `deltas:fig:sythetic-LR-clfs` |
| `gaussian_projection.py` | `deltas/synthetic/Gaussian-training`, `deltas/synthetic/Gaussian-optimised` | `deltas:fig:sythetic-deltas-data`, `deltas:fig:sythetic-deltas-optimised` |
| `gaussian_eval.py` | `deltas/synthetic/Gaussian_eval` | `deltas:fig:sythetic-test` |
| `follow_up.py` | `deltas/follow-up/f1_simulation`, `deltas/follow-up/f4_margins` | `deltas:fig:f1`, `deltas:fig:f4` |

Change the shared look in `style.py` and the shared drawing pieces in `draw.py`.
Figures are drawn at printed size, so include each at the `WIDTH` set in its module
(`\includegraphics[width=<WIDTH>\textwidth]{...}`).

## Adding a figure

Write a module with a `make()` that draws with `style` and `draw` and ends in `style.save(fig, '<chapter>/<name>')`,
then add it to `FIGURES` in `make_all.py` and to the table above.

## Differences from the published ECAI figures

- The separated data of `moving_bias` and `projection` is a new draw: the paper's came from an unseeded RNG.
  The two boundaries in `moving_bias` are schematic, as in the paper.
- `projection` is one figure; the paper's was assembled by hand from two.
- `Gaussian-optimised` marks the new boundary with a dotted line, and `Gaussian-training` puts
  the two overlapping $\hat{\bar R}_i$ on separate rows.
- Decision regions are drawn on a fine grid with the boundary as a line instead of a coarse filled contour.
