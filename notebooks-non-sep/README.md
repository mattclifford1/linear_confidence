# `notebooks-non-sep` — the non-separable follow-up


> **Superseded.** New experiments should use `experiments/run_experiments.py`,
> which fixes the seed-selection bias, keeps per-seed raw output, rounds instead
> of truncating, records the config, and caches classifier training. The scripts
> here are kept for provenance. See `experiments/README.md`.

Experiments for the working draft
`../../../Repos/Overleaf/deltas/deltas-non-separable/main.tex` (Dec 2024).
Backed by `deltas/model/non_sep.py`.

## The idea being tested

The published method only ever generalises from the **furthest** training point
of a class, giving a test-time error bound of `1/(N+1)`. Extending the same
exchangeability argument to the *k*-th furthest point gives `k/(N+1)`, so a
boundary can cut *inside* a class's training data with a quantified cost. That
removes the need for slack variables entirely — overlap is expressed by
choosing an inner order statistic rather than by deleting points.

Three-step procedure (§2 of the draft):

1. pick a candidate bias between the two empirical means (offset by
   `min_conc_i = 2R̄ᵢ/√Nᵢ` so the problem stays solvable);
2. per class, choose `kᵢ` and solve for the `δᵢ` that reaches that bias;
3. combine `kᵢ/(1+Nᵢ)` and `δᵢ` into a score, and take the argmin over the sweep.

Step 3 is where the four variants come from — **min**, **max**, **mean** over
the available `k`, or **furthest** (`k=1`, which recovers ECAI).

## Files

| File | What |
|---|---|
| `run_all_non_sep.py` | The full table: all 4 variants + slacks + the literature baselines, on 5 datasets. Writes `results-non-sep/` → `combined_table-non-sep.txt`. |
| `Compare_Loss.ipynb` | Side-by-side of the 4 variants on a real dataset (Breast Cancer by default). Produces the draft's `BC_*.png` figures. |
| `Compare_Loss Gaussian.ipynb` | Same on synthetic 2-Gaussian data → the draft's `Gaussian_*.png`. |
| `Gaussian_non_sep.ipynb` | Development notebook for the projection/loss plots. |
| `datasets.ipynb` | Sweeps the dataset menu. |

⚠️ **`run_all_non_sep.main()` has its experiment loop commented out** — as
committed it only re-runs `combine_tables()` over the existing results.
Uncomment the `for exp_num, exp in experiments.items():` block to regenerate.

⚠️ The `Compare_Loss*` notebooks use `N1=10, N2=100`, i.e. class **0** is the
minority there — the reverse of the repo-wide convention. That is why the
draft's Gaussian table (Baseline .994) looks nothing like
`results-non-sep/Results-0-Gaussian.txt` (Baseline .519), which uses the
standard `N1=1000, N2=10`. Worth reconciling before the two go in the same
paper.

## ⚠️ The "Slacks Deltas" baseline is not the published method

These results were run with `USE_TWO = True` (HEAD's setting), whereas the ECAI
paper was run with `USE_TWO = False`. The draft's Breast Cancer "Slacks Deltas"
row (.951/.949/.953) matches `notebooks-ECAI/results-two2/` exactly, not the
published .943/.942/.946. So the draft compares the new variants against a
*differently configured* version of the published method. See `FINDINGS.md`
§6.0 — this needs resolving before the draft goes further.

## Results so far

From `results-non-sep/` and the draft's tables:

- **min** and **furthest** are the good variants, roughly tied with the ECAI
  slacks method across all five datasets.
- **max** is bad (Breast Cancer .78 accuracy vs .95) and misbehaves entirely on
  real data — "Maximum doesn't do anything sensible" in the draft.
- **mean** is in between and never best.
- On MIMIC ICU, **furthest** gives the best G-Mean (.493) and F1 (.219) in the
  whole table.

So: the new formulation **matches** the slacks method but does not yet beat it.
Closing that gap is the open problem.

## Known issue with the loss

`non_sep._single_class_loss` uses `loss = error * delta`, with
`error*(1-delta) + delta` sitting commented out directly above it. Only the
second form is an upper bound on the expected error (it is the published Eq. 6
with `1/(N+1)` replaced by `k/(N+1)`), and it makes this method a strict
generalisation of the published one. See `FINDINGS.md` §3 — this is the
highest-value one-line experiment in the repo.

## Running

```bash
conda activate deltas
cd notebooks-non-sep
python run_all_non_sep.py     # after uncommenting main()'s loop
```

Note the runner requires **all four** variants plus the slacks method to
succeed on a seed before it counts that seed for anybody — see `FINDINGS.md`
§6.1.
