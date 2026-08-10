'''
Merge partial wide-grid runs into results/wide.csv.

The grid was produced in pieces:
  * the main run, launched before the last projection file had finished
    exporting (PneumoniaMNIST / GradientBoosting)
  * a re-run of Stroke Prediction, Cervical Cancer and Thyroid Sick after the
    train/test sizing rule was corrected - the old rule gave those three as few
    as 10 test points per class, which is too coarse to measure coverage
    against at all

Rows from the later files REPLACE any rows for the same
(dataset, model) pair, so re-running a cell and merging is always safe.

    python merge_wide.py wide_fixed3.csv wide_extra.csv
'''
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
KEY = ['dataset', 'model']


def merge(base='wide.csv', extras=(), backup=True):
    base_path = os.path.join(RESULTS, base)
    df = pd.read_csv(base_path)
    if backup and not os.path.exists(base_path + '.orig'):
        shutil.copy(base_path, base_path + '.orig')

    for name in extras:
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            print(f'  skip {name} (not found)')
            continue
        new = pd.read_csv(path)
        pairs = set(map(tuple, new[KEY].drop_duplicates().values))
        before = len(df)
        mask = df[KEY].apply(tuple, axis=1).isin(pairs)
        df = pd.concat([df[~mask], new], ignore_index=True)
        print(f'  {name}: {len(pairs)} (dataset, model) pairs, '
              f'replaced {int(mask.sum())} rows, {before} -> {len(df)}')

    df = df.sort_values(['dataset', 'model', 'seed', 'mode']).reset_index(
        drop=True)
    df.to_csv(base_path, index=False)
    print(f'wrote {base_path}: {len(df)} rows, '
          f'{df["dataset"].nunique()} datasets x {df["model"].nunique()} '
          f'models')
    missing = (df.groupby(['dataset', 'model']).size()
               .unstack().isna().sum().sum())
    if missing:
        print(f'WARNING: {int(missing)} (dataset, model) cells missing')
    return df


if __name__ == '__main__':
    merge(extras=sys.argv[1:] or ['wide_fixed3.csv', 'wide_extra.csv'])
