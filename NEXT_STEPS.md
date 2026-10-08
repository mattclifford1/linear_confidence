# Next steps

Where the deltas work stands, and what to do next, in order. Last updated
2 October 2026.

The research plan behind this list is in the Overleaf working notes,
`../../Repos/Overleaf/deltas/deltas-other-concentration/main.tex` (16 pages).
`FINDINGS.md` has the full research record, and `deltas/README.md` the code map.

## Where things stand

- **The modular refactor is finished on branch `modular-deltas`, but not merged
  into `main`.** Every deltas method is now one `DeltasEstimator` with six
  swappable slots (bound, confidence, rule, search, transform, certify). The code
  behind the published numbers is frozen in `deltas/legacy/`, and golden tests
  pin its exact outputs. Nothing reported has moved.
- **No experiments have been run** with the new shape-aware methods (Gaussian
  envelopes, the Student-*t* predictive curve and so on). They exist and are
  tested, but have not been tried on real data.
- **The research questions below are still open.** The experiments can start
  without answers, but the answers decide the headline method.

## 1. Merge `modular-deltas` into `main`

- Push the branch, then open the PR:
  <https://github.com/mattclifford1/linear_confidence/compare/main...modular-deltas?expand=1>
  (`gh` is not installed on this machine, so this has to go through the browser,
  or install `gh` and run `gh pr create --base main --head modular-deltas`.)
- What the PR should say, in short:
  - **New structure:** `deltas/core/` (the estimator, registry, per-class samples),
    one package per slot (`bounds/`, `confidence/`, `rules/`, `search/`,
    `transforms/`), named methods in `deltas/methods/` (both runners use them),
    and the frozen code in `deltas/legacy/`.
  - **Nothing reported moved:** the golden tests were written before any code
    changed and pass unchanged; old import paths are aliases of the same module
    objects; Clopper–Pearson/DKW are bit-identical to the originals; both runners
    reproduce their old method tables exactly.
  - **What callers might notice:** the CP/DKW classes now return real
    `get_params()`, reject unknown names in `set_params()`, and have lost the
    private `_sparse_loss_table`; `config.json` records each method's full
    specification.
  - **Found on the way, recorded in `FINDINGS.md` and not fixed in the frozen
    code:** the infeasible-fit crash (`base_deltas.fit` raises instead of
    reporting no solution, B11), the flat published objective (with a wide gap,
    where the published method puts the boundary depends on its search grid),
    and the count-based sum rule's first-tie tie-break, which is not
    mirror-symmetric.
  - **Tests:** `uv run pytest tests -q` (about 2 min; `-m "not golden"` skips the
    slow ones).

## 2. Code clean-up, before the experiments

`MODULARITY-REVIEW.md` (1 October 2026) reviews how easy the code is to read and
swap pieces in. Its suggested order, after the one-class-per-file layout move
(done):

1. **Certificates by label.** Index certificates by class label, so the minority's
   certificate is under `1`, not `'class 2'`; add `get_threshold()`.
2. **One split, one model registry.** A single split specification and model table,
   shared by the projection exporter, the cost experiment and the sibling loader.
3. **One runner.** One results-row schema and one runner reading projection cells;
   `run_experiments.py` becomes a preset of it; delete `merge_wide.py`.
4. **One plotting style.** Rewrite the plotting with a shared style, and point the
   working notes' figure script at `deltas.bounds`.
5. **Tidy `deltas/core`.** A fit-result object instead of 25 attributes, certify
   as a proper slot, costs on the rule, and no plotting import.

Do each behind the golden and equivalence tests. The certificates-by-label step
is the one worth doing before the experiments, because the experiments read
certificates.

## 3. Answer the research questions

Each has a recommendation in the working notes ("Open questions" section).

| question | the choice | recommendation |
|---|---|---|
| **Average or guaranteed?** | place the boundary for best average performance (Student-*t* predictive), for guaranteed minority protection (confidence envelope), or decide with the first and certify with the second | both: decide average-case, certify high-probability |
| **Is a shape assumption OK?** | a stated "Gaussian after a monotone transform" headline method, with the assumption-free Clopper–Pearson certificate always reported next to it | yes |
| **Fix δ or keep optimising it?** | optimising δ on the same data makes the reported confidence invalid; fixed, δ becomes a reported confidence or a caution dial | fix it |
| **Calibration for tiny minorities** | always split off calibration data (costs about a third of the minority), cross-fit, or split only the majority; and what is the smallest minority that matters | always split for the main tables; cross-fitting later |
| **Which score transform?** | a per-model rule (identity for margins, logit for probabilities) or a fitted Yeo–Johnson | decide after the projection-shapes experiment |
| **Which imbalances are in scope?** | sample size and spread for sure; also shape, costs, prior shift, multi-class | size and spread, shape as a robustness check, costs later |
| **Extension or new paper?** | a journal extension of the ECAI paper, or a new paper set against parametric Neyman–Pearson classification (Tong et al. 2020) | open: decides which baselines are mandatory |

## 4. Run the experiments

On a new branch `envelope-deltas`, off `main` once the merge is done. In this
order, because the first two can change the defaults. The first four reuse the
238 cached projections in `experiments/projections/`, so no classifier is refitted.

1. **Projection shapes.** What do real projected class scores look like (skew,
   tail weight, normality), before and after a logit or Yeo–Johnson map? Picks the
   default family and transform. Hours, not days.
2. **Certificate validity.** Does each certificate hold at its nominal level when
   its shape assumption is right, and how badly does it fail when it is not
   (skewed, heavy-tailed, two-subgroup minorities, minority size 5 to 500)?
3. **Boundary placement.** Synthetic sweeps of separation × minority size ×
   spread ratio × shape: where should the boundary go, and how cautious should it
   be? Includes a learned projection, run with and without the calibration split.
   Redoes properly the quick check in the notes' "How much room?" table.
4. **Real-data grid.** All 238 cells × 10 seeds, with and without the calibration
   split: average ranks with significance tests, coverage and tightness of both
   certificates, fit time. `run_wide.py --methods` already accepts the new names
   (`Gaussian Predictive`, `Gaussian Envelope`, …).
5. **One-slot ablations.** Swap one component at a time: rule, family, Monte Carlo
   band or rectangle, transform, fixed or optimised δ, calibration fraction.
6. **Costs and extensions.** The risk rule on the cost datasets (does it fix the
   costs negative result in `FINDINGS.md` §13?), the Neyman–Pearson rule,
   cross-fitting, multi-class.

**Baselines:** the untouched classifier, threshold moving, post-hoc logit
adjustment (new: not implemented yet), the published slack method, and CP/DKW
minimax.

**What would count as success:** the model-based certificate holds on the
calibration-split grid (or its failures are understood); a better G-mean rank than
CP/DKW minimax for small minorities and no worse for large ones; certificates at
least twice as tight as Clopper–Pearson for minorities of 50 or fewer; and one
default amount of caution that is close to ideal on average. If the certificate
often fails, fall back to the hybrid (counts inside the data, the shape model only
beyond it). If only the ranking fails, the contribution is a much tighter valid
certificate.

## Loose ends

- The **deltas-non-separable** Overleaf project has an old uncommitted edit to
  `main.tex` (from 18 September 2026) that was never pushed.
- Known bugs in the frozen code are **kept on purpose** because they feed the
  published numbers: the exact-float grid filter, the possibly-unassigned
  `support_max_hit`, the silent `max_trials` cap and the infeasible-fit crash
  (`FINDINGS.md` §7.1). Fix forward in the modular code, never in `deltas/legacy/`.
