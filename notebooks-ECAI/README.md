# `notebooks-ECAI` — experiments for the published paper


> **Superseded.** New experiments should use `experiments/run_experiments.py`,
> which fixes the seed-selection bias, keeps per-seed raw output, rounds instead
> of truncating, records the config, and caches classifier training. The scripts
> here are kept for provenance. See `experiments/README.md`.

Everything here backs *Learning Confidence Bounds for Classification with
Imbalanced Data* (ECAI 2024, arXiv:2407.11878). Paper source:
`../../../Repos/Overleaf/deltas/ecai-2024-deltas-arxiv-3-sup-mat/m598.tex`.

## Scripts

| Script | Produces | Paper |
|---|---|---|
| `run_all.py` | `results/` → `combined_table.txt` | Table 3 (binary slacks) |
| `run_all continuous.py` | `results-continuous/` → `combined_table-continuous.txt` | Table 4 (continuous slacks, Appendix B) |
| `run_all_two.py` | `results-two/` → `combined_table-two.txt` | not in the paper — `USE_TWO` ablation |
| `run_all_two copy.py` | `results-two2/` → `combined_table-two2.txt` | not in the paper — second `USE_TWO` ablation |
| `projection_plots.py` | `plots/diagrams*.png` | Fig. 1 is `diagrams_data_original.png` (copied verbatim to `figs/diagrams/`). Fig. 2 (`figs/diagrams/projection.png`) has no matching filename here — it looks assembled/edited from `diagrams-training.png`, so it is **not** reproducible by re-running this script alone. |
| `Guassian_plots.py` | `plots/Gaussian*.png` | Figs. 3–6: `Gaussian_data`, `Gaussian-training`, `Gaussian-optimised`, `Gaussian_eval` — filenames match `figs/synthetic/` one-to-one. |
| `presentation/bias_gif.py` | `presentation/bias.gif` | talk material |
| `results-locked/combine_table.py` | re-stitches the locked tables | — |

`run_all_two.py` and `run_all_two copy.py` are byte-identical to `run_all.py`
apart from the output directory names. The ablation was performed by
**hand-editing `deltas/misc/use_two.py` between runs** — the flag value is not
recorded anywhere in the results files. Effect is real (Breast Cancer accuracy
.944 → .951).

## ⚠️ `USE_TWO` before you re-run anything

The published results used **`USE_TWO = False`**; HEAD has `True` (flipped in
`583e663`, 12 Nov 2024, for the non-separable work). Verified reproduction on
Breast Cancer, seeds 0–9: `USE_TWO=False` gives Baseline .917/.917/.914 and Our
Method .945/.943/.947, matching published Table 3; `USE_TWO=True` gives
.912/.912/.908 and .951/.950/.954. See `FINDINGS.md` §6.0.

## `results*/` directories

| Directory | Date | `USE_TWO` | Notes |
|---|---|---|---|
| `results-locked/` | 12–15 Jul 2024 | False | Earlier locked run. Matches published Table 3 on Breast Cancer / Hepatitis / Heart Disease; differs by one last-digit unit on Pima (.673 vs .672) and MIMIC. |
| `results/` | 26 Aug 2024 | False | Matches published Table 3 on Pima, Heart Disease and MIMIC; off by one last-digit unit on Breast Cancer (.944 vs .943) and Hepatitis (.778 vs .775). |
| `results-continuous/` | 21 Aug 2024 | False | ⭐ Reproduces published **Table 4** exactly (continuous slacks, Appendix B). Essentially identical to Table 3 — the continuous-slack elaboration buys nothing, which is a finding in itself. |
| `results-two/` | 6 Nov 2024 | False | Identical to `results/`. |
| `results-two2/` | 6 Nov 2024 | **True** | The setting HEAD is in. This is what the non-separable draft uses for its "Slacks Deltas" row. |

**No directory reproduces published Table 3 exactly.** Table 3's LaTeX comments
are timestamped 26/08/2024 17:55–17:56 and 21/08/2024 16:24:35 (MIMIC); those
files were overwritten by later runs the same day. The published Table 3 is
therefore a mix that no longer exists on disk in per-dataset form —
`combined_table.txt` at the top of this directory is the closest surviving
artefact, and it is the 17:36 run, not the 17:55 one.

Each `Results-<n>-<dataset>.txt` is a LaTeX `tabular` fragment with a three-line
comment header recording the timestamp, run count and train/test class counts.
`combine_tables()` stitches them into the multi-row table pasted into the `.tex`.

## Cross-validation notebooks

`CV_0_Gaussian.ipynb` … `CV_5_MIMIC.ipynb` — one per dataset, the interactive
versions of what `run_all.py` automates. Useful for inspecting a single seed's
projection and loss curve.

`Gaussian_vary_imbal.ipynb` / `Gaussian_vary_sep.ipynb` — sweeps over imbalance
ratio and class separation on the synthetic data. These were made for the
reviewer-2 response (`73ffddb reviwer 2 varying graphs`) and are the closest
thing that exists to the **overlap sweep** recommended in `FINDINGS.md` §4.
Worth reviving.

## Running

```bash
conda activate deltas
cd notebooks-ECAI
python run_all.py
```

Expect ~50 min, ~90% of it retraining the MIMIC MLPs (see `FINDINGS.md` §8).
The runner keeps trying seeds `0, 1, 2, …` until 10 of them yield a solvable
projection, and **drops the failing seeds for every method including the
baselines** — see `FINDINGS.md` §6.1 before drawing conclusions from a re-run.
