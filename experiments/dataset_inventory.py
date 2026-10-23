'''
Size up every toy_datasets dataset for the extra-experiments paper.

    uv run --with medmnist python dataset_inventory.py
    uv run python dataset_inventory.py --tables   # tables from the saved CSV
    uv run python dataset_inventory.py --fresh    # size up every dataset again

Writes results/dataset_inventory.csv and a LaTeX table into the paper
(override the paper folder with EXTRA_PAPER_OUT).
'''
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from data_loaders import AVAILABLE_DATASETS, get_dataset   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_projections import (MIN_MINORITY_TEST, MIN_MINORITY_TRAIN,  # noqa: E402
                                _as_minority_class_1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'results', 'dataset_inventory.csv')
OUT_TEX = os.path.join(os.environ.get('EXTRA_PAPER_OUT', os.path.join(
    HERE, '..', '..', 'papers', 'ecai-extra-experiments')), 'tables')

#: group -> datasets, in table order
GROUPS = {
    'Synthetic': ['Gaussian', 'Moons', 'Circles', 'Blobs', 'XOR',
                  'Sklearn Normal', 'Truncated Normal', 'Madelon'],
    'Medical tabular': [
        'Arrhythmia', 'Breast Cancer', 'Breast Cancer Coimbra',
        'Breast Cancer Prognostic', 'Breast Cancer Wisconsin',
        'Cervical Cancer', 'Chronic Kidney Disease', 'Diabetes Pima Indian',
        'Framingham CHD', 'Habermans Breast Cancer', 'HCC Survival',
        'Heart Disease', 'Heart Failure', 'Hepatitis', 'Indian Liver Patient',
        'Mammographic Mass', 'Parkinsons', 'SPECTF Heart', 'Stroke Prediction',
        'Thoracic Surgery', 'Thyroid Sick', 'Z-Alizadeh Sani CAD'],
    'Clinical ICU': ['MIMIC-III Mortality', 'MIMIC-III Sepsis',
                     'MIMIC-IV Ready for Discharge'],
    'Other tabular': ['Abalone Gender', 'Banknote Authentication',
                      'Ionosphere', 'Sonar Rocks vs Mines', 'Wheat Seeds',
                      'Iris', 'Wine'],
    'Cost-sensitive': ['German Credit', 'Costcla Credit Scoring Kaggle 2011',
                       'Costcla Credit Scoring PAKDD 2009',
                       'Costcla Direct Marketing'],
    'OpenML': ['Spambase', 'Phoneme', 'MAGIC Gamma Telescope', 'EEG Eye State',
               'Default of Credit Card Clients', 'QSAR Biodegradation',
               'Bioresponse', 'Hill-Valley', 'Steel Plates Fault',
               'KC1 Software Defects', 'Blood Transfusion', 'Ozone Level 8hr'],
    'Medical image': ['PneumoniaMNIST', 'BreastMNIST', 'DermaMNIST',
                      'BloodMNIST', 'PathMNIST', 'OCTMNIST'],
    'Natural image': ['MNIST', 'Fashion-MNIST', 'SVHN', 'EuroSAT', 'CIFAR-10',
                      'CIFAR-10N', 'CIFAR-100'],
}
IMAGE_GROUPS = ('Medical image', 'Natural image')
#: the datasets the published ECAI 2024 paper evaluates on
ECAI_DATASETS = ['Gaussian', 'Diabetes Pima Indian', 'Breast Cancer', 'Hepatitis',
                 'Heart Disease', 'MIMIC-III Mortality']


def _label_names(loader):
    try:
        names = loader.get_label_names()
    except Exception:
        return None
    return None if names is None else [str(n) for n in names]


def size_up(name, group):
    t0 = time.time()
    loader = get_dataset(name)
    X, y = loader.get_X(), loader.get_y()
    p = int(np.prod(X.shape[1:]))
    row = {'dataset': name, 'group': group, 'n': int(len(y)), 'features': p}
    try:
        _, yb = _as_minority_class_1(X, y)
        n_maj, n_min = np.bincount(yb, minlength=2)
        names = _label_names(loader)
        flipped = int(np.bincount(np.asarray(y).astype(int))[1]) != n_min
        row.update(binary=True, majority=int(n_maj), minority=int(n_min),
                   ratio=float(n_maj / n_min),
                   minority_label=(names[0 if flipped else 1]
                                   if names and len(names) == 2 else ''))
    except ValueError:
        row.update(binary=False)

    if group in IMAGE_GROUPS:
        native = get_dataset(name, binary=False)
        counts = np.bincount(np.asarray(native.get_y()).astype(int))
        names = _label_names(native) or [str(i) for i in range(len(counts))]
        rare = int(np.argmin(counts))
        row.update(classes=int(len(counts)), rarest_class=names[rare],
                   rarest_count=int(counts[rare]),
                   largest_count=int(counts.max()))

    if row.get('binary'):
        n_min = row['minority']
        row['max_minority_train'] = int(max(
            0, min(n_min // 2, n_min - MIN_MINORITY_TEST)))
        row['eligible'] = row['max_minority_train'] >= MIN_MINORITY_TRAIN
    row['seconds'] = round(time.time() - t0, 1)
    return row


def _fmt_int(v):
    return '' if pd.isna(v) else f'{int(v):,}'.replace(',', '{,}')


def to_latex(df):
    lines = [r'\begin{longtable}{@{}l r r r r r l@{}}',
             r'\caption{Every dataset we have, in its default binary form '
             r'(class 1 is the minority). $N_1^{\max}$ is the largest minority '
             r'training set that still leaves half the minority for testing. '
             r'ECAI: \checkmark\ evaluated in the ECAI 2024 paper. Generated by '
             r'\texttt{experiments/dataset\_inventory.py}.}'
             r'\label{tab:inventory}\\',
             r'\toprule',
             r'Dataset & $n$ & features & minority & ratio & $N_1^{\max}$ & ECAI \\',
             r'\midrule\endfirsthead',
             r'\toprule',
             r'Dataset & $n$ & features & minority & ratio & $N_1^{\max}$ & ECAI \\',
             r'\midrule\endhead',
             r'\bottomrule\endfoot']
    for group in GROUPS:
        part = df[df.group == group]
        if part.empty:
            continue
        lines.append(rf'\multicolumn{{7}}{{@{{}}l}}{{\textit{{{group}}}}}\\')
        for _, r in part.iterrows():
            ratio = '' if pd.isna(r.get('ratio')) else f'{r.ratio:.1f}'
            name = r.dataset.replace('&', r'\&')
            lines.append(
                rf'\quad {name} & {_fmt_int(r.n)} & {_fmt_int(r.features)} & '
                rf'{_fmt_int(r.get("minority"))} & {ratio} & '
                rf'{_fmt_int(r.get("max_minority_train"))} & '
                rf'{r"\checkmark" if r.dataset in ECAI_DATASETS else ""} \\')
        lines.append(r'\addlinespace')
    lines.append(r'\end{longtable}')
    return '\n'.join(lines) + '\n'


def summary_latex(df):
    lines = [r'\begin{tabular}{@{}l r r r r@{}}', r'\toprule',
             r'Group & datasets & in ECAI & minority size & natural ratio \\',
             r'\midrule']
    for group in GROUPS:
        part = df[df.group == group]
        lines.append(
            rf'{group} & {len(part)} & {int(part.dataset.isin(ECAI_DATASETS).sum())} & '
            rf'{_fmt_int(part.minority.min())}--{_fmt_int(part.minority.max())} & '
            rf'{part.ratio.min():.1f}--{part.ratio.max():.1f} \\')
    lines.append(r'\midrule')
    lines.append(rf'Total & {len(df)} & {int(df.dataset.isin(ECAI_DATASETS).sum())} & & \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def write_tables(df):
    os.makedirs(OUT_TEX, exist_ok=True)
    ok = df[df['error'].isna()] if 'error' in df else df
    with open(os.path.join(OUT_TEX, 'inventory.tex'), 'w') as f:
        f.write(to_latex(ok))
    with open(os.path.join(OUT_TEX, 'inventory_summary.tex'), 'w') as f:
        f.write(summary_latex(ok))
    return len(df) - len(ok)


def main():
    if '--tables' in sys.argv:
        write_tables(pd.read_csv(OUT_CSV))
        return
    listed = [d for names in GROUPS.values() for d in names]
    missing = sorted(set(AVAILABLE_DATASETS) - set(listed))
    if missing:
        raise SystemExit(f'datasets with no group: {missing}')

    done = {} if '--fresh' in sys.argv or not os.path.exists(OUT_CSV) else {
        r['dataset']: r for r in pd.read_csv(OUT_CSV).to_dict('records')}
    rows = []
    for group, names in GROUPS.items():
        for name in names:
            if name in done:
                rows.append({**done[name], 'group': group})
                continue
            try:
                rows.append(size_up(name, group))
                print(f'{name:40s} ok  {rows[-1]["seconds"]}s', flush=True)
            except Exception as exc:
                rows.append({'dataset': name, 'group': group,
                             'error': f'{type(exc).__name__}: {exc}'[:200]})
                print(f'{name:40s} FAILED {rows[-1]["error"]}', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    failed = write_tables(df)
    print(json.dumps({'csv': OUT_CSV, 'tex': OUT_TEX, 'failed': failed}))


if __name__ == '__main__':
    main()
