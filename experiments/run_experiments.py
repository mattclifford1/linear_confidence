'''
Unified, reproducible experiment runner for every deltas method.

Fixes the reproducibility problems of the older notebooks-*/run_all*.py:
  * a FIXED seed list, not an open-ended search for seeds that happen to work
  * failures are recorded as NaN and reported ("solved 9/10"), never silently
    dropped - the old behaviour also dropped that seed for the baselines, which
    let a flag that cannot touch the baseline change the baseline's score
  * per-seed raw results are written to CSV, so significance tests are possible
  * the config (USE_TWO, USE_GLOBAL_R, seeds, library versions) is stamped into
    a JSON sidecar and into the LaTeX comments
  * classifier training is cached (deltas.pipeline.cached), so the ~270s/seed
    MIMIC MLP training is paid once rather than once per script

usage:
    python run_experiments.py                 # everything
    python run_experiments.py --datasets 2 3  # by index (see EXPERIMENTS)
    python run_experiments.py --seeds 30      # more seeds
'''
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score

import deltas.misc.use_two as use_two_cfg
from deltas.pipeline import cached, evaluation  # noqa: F401
from deltas.model import downsample, non_sep, overlap
import deltas.utils.cache as cache


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
RAW = os.path.join(RESULTS, 'raw')

EXPERIMENTS = {
    0: {'dataset': 'Gaussian', 'model': 'Linear'},
    1: {'dataset': 'Pima Indian Diabetes', 'model': 'SVM-rbf'},
    2: {'dataset': 'Breast Cancer', 'model': 'SVM-rbf'},
    3: {'dataset': 'Hepatitis', 'model': 'SVM-rbf'},
    4: {'dataset': 'Heart Disease', 'model': 'SVM-rbf'},
    5: {'dataset': 'MIMIC-III-mortality', 'model': 'MIMIC'},
}

SHORT_NAMES = {'Gaussian': 'Gaussian',
               'Pima Indian Diabetes': 'Pima Diabetes',
               'Breast Cancer': 'Breast Cancer',
               'Hepatitis': 'Hepatitis',
               'Heart Disease': 'Heart Disease',
               'MIMIC-III-mortality': 'MIMIC ICU'}

# the deltas methods under test. name -> callable(clf) -> fitted model
DELTAS_METHODS = {
    'Slacks Deltas': lambda clf, X, y: downsample.downsample_deltas(clf).fit(
        X, y, max_trials=10000, parallel=True),
    'Min Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(
        X, y, loss_type='min'),
    'Max Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(
        X, y, loss_type='max'),
    'Avg Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(
        X, y, loss_type='mean'),
    'F Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(
        X, y, only_furtherest_k=True),
    'CP Sum': lambda clf, X, y: overlap.binomial_deltas(
        clf, objective='sum').fit(X, y),
    'CP Minimax': lambda clf, X, y: overlap.binomial_deltas(
        clf, objective='minimax').fit(X, y),
    'DKW Sum': lambda clf, X, y: overlap.dkw_deltas(
        clf, objective='sum').fit(X, y),
    'DKW Minimax': lambda clf, X, y: overlap.dkw_deltas(
        clf, objective='minimax').fit(X, y),
}

METRICS = {'Accuracy': accuracy_score,
           'G-Mean': geometric_mean_score,
           'F1': f1_score}


def _score(y_true, y_pred):
    return {name: float(fn(y_true, y_pred)) for name, fn in METRICS.items()}


def _nan_scores():
    return {name: np.nan for name in METRICS}


def run_dataset(dataset, model, seeds, use_cache=True, _print=True):
    '''one row per (seed, method)'''
    rows = []
    counts = {}
    for seed in seeds:
        t0 = time.time()
        data_clf, clfs = cached.get_data_and_classifiers(
            dataset, model, seed=seed, scale=True, use_cache=use_cache)
        X, y = data_clf['data']['X'], data_clf['data']['y']
        Xt, yt = data_clf['data_test']['X'], data_clf['data_test']['y']
        clf = clfs['Baseline']
        counts = {'train': np.unique(y, return_counts=True)[1].tolist(),
                  'test': np.unique(yt, return_counts=True)[1].tolist()}

        # literature baselines - these never fail
        for name, model_obj in clfs.items():
            preds = np.asarray(model_obj.predict(Xt)).squeeze()
            rows.append({'seed': seed, 'method': name, 'fit': True,
                         **_score(yt, preds)})

        # deltas methods
        for name, factory in DELTAS_METHODS.items():
            try:
                fitted = factory(clf, X, y)
                ok = bool(getattr(fitted, 'is_fit', False))
                if ok:
                    preds = np.asarray(fitted.predict(Xt)).squeeze()
                    rows.append({'seed': seed, 'method': name, 'fit': True,
                                 **_score(yt, preds)})
                else:
                    rows.append({'seed': seed, 'method': name, 'fit': False,
                                 **_nan_scores()})
            except Exception as exc:      # a failure is data, not a crash
                if _print == True:
                    print(f'    ! {name} seed {seed}: {type(exc).__name__}: {exc}')
                rows.append({'seed': seed, 'method': name, 'fit': False,
                             **_nan_scores()})

        if _print == True:
            n_fail = sum(1 for r in rows
                         if r['seed'] == seed and r['fit'] == False)
            print(f'  seed {seed}: {time.time() - t0:5.1f}s'
                  f'{"" if n_fail == 0 else f"  ({n_fail} method(s) unsolved)"}',
                  flush=True)

    df = pd.DataFrame(rows)
    df.attrs['counts'] = counts
    return df


def aggregate(df):
    '''mean/std over the seeds where the method produced a solution'''
    order = [m for m in df['method'].unique()]
    out = {}
    for method in order:
        sub = df[df['method'] == method]
        solved = int(sub['fit'].sum())
        rec = {'n': solved, 'total': len(sub)}
        for metric in METRICS:
            vals = sub[metric].dropna()
            rec[f'{metric} mean'] = vals.mean() if len(vals) else np.nan
            rec[f'{metric} std'] = vals.std() if len(vals) else np.nan
        out[method] = rec
    return pd.DataFrame.from_dict(out, orient='index')


def to_latex(agg, dataset, model, counts, n_seeds):
    '''LaTeX tabular fragment; values ROUNDED (the old runner truncated)'''
    methods = agg.index.to_list()
    lines = []
    now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    lines.append(f'% {now}')
    lines.append(f'% {dataset} - {n_seeds} seeds {model} model')
    lines.append(f"% train:{counts.get('train')}, test:{counts.get('test')}")
    lines.append(f'% USE_TWO={use_two_cfg.USE_TWO} '
                 f'USE_GLOBAL_R={use_two_cfg.USE_GLOBAL_R} '
                 f'RANDOM_STATE={use_two_cfg.RANDOM_STATE}')
    lines.append('\\begin{tabular}{@{}l' + 'c' * len(METRICS) + 'c@{}}')
    lines.append('\\toprule')
    lines.append('Methods & ' + ' & '.join(METRICS) + ' & solved \\\\')
    lines.append('\\midrule')

    best = {m: np.nanargmax(agg[f'{m} mean'].values) for m in METRICS}
    for i, method in enumerate(methods):
        cells = []
        for metric in METRICS:
            mu = agg.loc[method, f'{metric} mean']
            sd = agg.loc[method, f'{metric} std']
            if np.isnan(mu):
                cells.append('--')
                continue
            mu_s = f'{mu:.3f}'.lstrip('0')
            sd_s = f'{sd:.2f}'.lstrip('0') if not np.isnan(sd) else '--'
            if i == best[metric]:
                mu_s = f'\\textbf{{{mu_s}}}'
            cells.append(f'${mu_s} \\pm {sd_s}$')
        solved = f"{int(agg.loc[method, 'n'])}/{int(agg.loc[method, 'total'])}"
        lines.append(f'{method} & ' + ' & '.join(cells) + f' & {solved} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', type=int,
                        default=list(EXPERIMENTS))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    os.makedirs(RAW, exist_ok=True)

    config = {'seeds': seeds,
              'USE_TWO': use_two_cfg.USE_TWO,
              'USE_GLOBAL_R': use_two_cfg.USE_GLOBAL_R,
              'RANDOM_STATE': use_two_cfg.RANDOM_STATE,
              'methods': list(DELTAS_METHODS),
              'python': sys.version.split()[0],
              'started': datetime.now().isoformat()}
    print(json.dumps({k: config[k] for k in
                      ('seeds', 'USE_TWO', 'USE_GLOBAL_R')}), flush=True)

    for idx in args.datasets:
        exp = EXPERIMENTS[idx]
        dataset, model = exp['dataset'], exp['model']
        print(f'\n=== [{idx}] {dataset} ({model}) ===', flush=True)
        t0 = time.time()
        df = run_dataset(dataset, model, seeds, use_cache=not args.no_cache)
        counts = df.attrs['counts']
        df.to_csv(os.path.join(RAW, f'{idx}-{dataset}.csv'), index=False)

        agg = aggregate(df)
        agg.to_csv(os.path.join(RESULTS, f'agg-{idx}-{dataset}.csv'))
        with open(os.path.join(RESULTS, f'Results-{idx}-{dataset}.txt'),
                  'w') as f:
            f.write(to_latex(agg, dataset, model, counts, len(seeds)))

        print(agg[['n'] + [f'{m} mean' for m in METRICS]].round(3).to_string(),
              flush=True)
        print(f'  ({time.time() - t0:.1f}s)', flush=True)

    config['finished'] = datetime.now().isoformat()
    config['cache'] = cache.info()
    with open(os.path.join(RESULTS, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print('\nwrote', RESULTS)


if __name__ == '__main__':
    main()
