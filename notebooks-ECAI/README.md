# notebooks-ECAI/

Publication experiment notebooks for the ECAI paper. These run cross-validated experiments across benchmark datasets and generate the figures/tables in the paper.

## Cross-Validation Experiments

Each `CV_*.ipynb` notebook runs k-fold cross-validation on one dataset, comparing DELTAS against baseline classifiers.

| Notebook | Dataset |
|----------|---------|
| `CV_0_Gaussian.ipynb` | Synthetic Gaussian |
| `CV_1_pima.ipynb` | Pima Indians Diabetes |
| `CV_2_breast.ipynb` | Wisconsin Breast Cancer |
| `CV_3_hep.ipynb` | Hepatitis |
| `CV_4_HD.ipynb` | Heart Disease |
| `CV_5_MIMIC.ipynb` | MIMIC (requires local data) |

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `Gaussian_vary_imbal.ipynb` | Vary class imbalance on Gaussian data |
| `Gaussian_vary_sep.ipynb` | Vary class separation on Gaussian data |

## Presentation Notebooks

`presentation/example_results.ipynb` and `presentation/example_results_G.ipynb` — generate paper-ready figures.

## Results

Cross-validation results are saved to `results/` subdirectories within this folder (CSV files and plots per dataset).

## Running

Run notebooks in order (CV_0 through CV_5) to reproduce the full paper experiments. Each notebook is self-contained and saves its own results.
