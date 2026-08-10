'''
Run every deltas method over the wide (dataset x model x seed x calibration)
grid, using the projections exported by export_projections.py.

Two-process design - see export_projections.py for why. That script fits the
models under the sibling projection_models environment and writes 1-D
projections; this one runs in the deltas environment and never sees a model.

    python run_wide.py                       # everything that has been exported
    python run_wide.py --datasets Hepatitis
    python run_wide.py --methods "CP Minimax" "DKW Minimax"
    python run_wide.py --jobs 8

Writes experiments/results/wide.csv - one row per
(dataset, model, seed, mode, method) with the test metrics, the certified
bounds and whether those bounds actually held.
'''
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score

import deltas.misc.use_two as use_two_cfg
from deltas.classifiers.frozen import FrozenProjection
from deltas.model import downsample, non_sep, overlap


HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, 'projections')
RESULTS = os.path.join(HERE, 'results')

DELTAS_METHODS = {
    'Slacks Deltas': lambda clf, X, y: downsample.downsample_deltas(clf).fit(
        X, y, max_trials=2000, parallel=False),
    'Min Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(
        X, y, loss_type='min'),
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


def _scores(yt, preds):
    out = {name: float(fn(yt, preds)) for name, fn in METRICS.items()}
    out['err_0'] = float((preds[yt == 0] != 0).mean()) if (yt == 0).any() \
        else np.nan
    out['err_1'] = float((preds[yt == 1] != 1).mean()) if (yt == 1).any() \
        else np.nan
    return out


def _nan_scores():
    return {name: np.nan for name in
            list(METRICS) + ['err_0', 'err_1']}


def _best_balanced_threshold(z, y):
    '''
    Thresholding baseline: the threshold on the fit projections minimising
    balanced error. This is the projection-space analogue of Sheng & Ling's
    threshold moving, and unlike SMOTE/BW it needs no feature space.
    '''
    cands = np.unique(z)
    if len(cands) < 2:
        return float(cands[0]) if len(cands) else 0.0
    mids = (cands[:-1] + cands[1:]) / 2.0
    z0, z1 = z[y == 0], z[y == 1]
    if len(z0) == 0 or len(z1) == 0:
        return float(np.median(z))
    err = 0.5 * (((z0[:, None] > mids[None, :]).mean(axis=0)) +
                 ((z1[:, None] <= mids[None, :]).mean(axis=0)))
    return float(mids[int(np.argmin(err))])


def run_file(path, methods, seeds):
    rows = []
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['meta'][0]))
        keys = set(f.files)
        for seed in seeds:
            for mode in ('naive', 'split'):
                p = f's{seed}_{mode}_'
                if p + 'z_cert' not in keys:
                    continue
                z_cert, y_cert = f[p + 'z_cert'], f[p + 'y_cert'].astype(int)
                z_test, y_test = f[p + 'z_test'], f[p + 'y_test'].astype(int)
                z_fit, y_fit = f[p + 'z_fit'], f[p + 'y_fit'].astype(int)
                thr = float(f[p + 'threshold'][0])
                rows += _one(meta, seed, mode, z_cert, y_cert, z_fit, y_fit,
                             z_test, y_test, thr, f, p, keys, methods)
    return rows


def _one(meta, seed, mode, z_cert, y_cert, z_fit, y_fit, z_test, y_test, thr,
         f, p, keys, methods):
    # how optimistic is this classifier on the points it was fitted to? This
    # is the quantity the calibration split exists to neutralise, so record it
    # per row: the coverage shortfall should track it.
    def _bal_err(z, yy, t):
        if not (yy == 0).any() or not (yy == 1).any():
            return np.nan
        return 0.5 * ((z[yy == 0] > t).mean() + (z[yy == 1] <= t).mean())

    fit_err = _bal_err(z_fit, y_fit, thr)
    test_err = _bal_err(z_test, y_test, thr)

    base = {'dataset': meta['dataset'], 'model': meta['model'],
            'seed': seed, 'mode': mode,
            'n_cert': int(len(y_cert)),
            'n_cert_minority': int((y_cert == 1).sum()),
            'train_ratio': meta.get('train_ratio', np.nan),
            'n_features': meta.get('n_features', np.nan),
            'clf_fit_err': fit_err,
            'clf_test_err': test_err,
            'optimism': test_err - fit_err}
    rows = []

    def add(method, preds=None, fitted=None, ok=True):
        rec = {**base, 'method': method, 'fit': ok}
        rec.update(_scores(y_test, preds) if ok else _nan_scores())
        rec.update({'U_0': np.nan, 'U_1': np.nan, 'delta_0': np.nan,
                    'delta_1': np.nan, 'covered_0': np.nan,
                    'covered_1': np.nan})
        if ok and fitted is not None and hasattr(fitted, 'certified_error'):
            ce = fitted.certified_error()
            rec['U_0'], rec['U_1'] = ce['class 1'], ce['class 2']
            rec['delta_0'], rec['delta_1'] = ce['delta1'], ce['delta2']
            rec['covered_0'] = bool(rec['err_0'] <= ce['class 1'])
            rec['covered_1'] = bool(rec['err_1'] <= ce['class 2'])
        rows.append(rec)

    # ---- baselines ------------------------------------------------------
    add('Baseline', preds=(z_test > thr).astype(int))
    t_star = _best_balanced_threshold(z_fit, y_fit)
    add('Threshold', preds=(z_test > t_star).astype(int))
    for label, name in (('smote', 'SMOTE'), ('bw', 'Balanced Weights')):
        if p + f'pred_{label}' in keys:
            add(name, preds=f[p + f'pred_{label}'].astype(int))
        else:
            add(name, ok=False)

    # ---- deltas methods --------------------------------------------------
    clf = FrozenProjection(thr, name=meta['model'])
    Xc, Xt = z_cert[:, None], z_test[:, None]
    for name in methods:
        try:
            fitted = DELTAS_METHODS[name](clf, Xc, y_cert)
            if bool(getattr(fitted, 'is_fit', False)):
                preds = np.asarray(fitted.predict(Xt)).squeeze().astype(int)
                add(name, preds=preds, fitted=fitted)
            else:
                add(name, ok=False)
        except Exception:
            add(name, ok=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--methods', nargs='*', default=list(DELTAS_METHODS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--out', default='wide.csv')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(PROJ, '*.npz')))
    if args.datasets:
        want = {d.replace(' ', '-') for d in args.datasets}
        files = [f for f in files
                 if os.path.basename(f).split('__')[0] in want]
    if args.models:
        files = [f for f in files
                 if os.path.basename(f).split('__')[1][:-4] in args.models]
    if not files:
        raise SystemExit(f'no exported projections in {PROJ} - run '
                         f'export_projections.py first (with the '
                         f'projection_models venv)')

    seeds = list(range(args.seeds))
    print(f'{len(files)} exported (dataset, model) files, {len(seeds)} seeds, '
          f'{len(args.methods)} deltas methods', flush=True)
    t0 = time.time()

    if args.jobs > 1:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_file)(f, args.methods, seeds) for f in files)
    else:
        chunks = []
        for i, f in enumerate(files):
            chunks.append(run_file(f, args.methods, seeds))
            print(f'  [{i + 1}/{len(files)}] {os.path.basename(f)[:-4]} '
                  f'({time.time() - t0:.0f}s)', flush=True)

    df = pd.DataFrame([r for c in chunks for r in c])
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, args.out)
    df.to_csv(out, index=False)

    config = {'USE_TWO': use_two_cfg.USE_TWO,
              'USE_GLOBAL_R': use_two_cfg.USE_GLOBAL_R,
              'seeds': seeds, 'methods': args.methods,
              'n_files': len(files), 'n_rows': len(df)}
    with open(os.path.join(RESULTS, 'wide_config.json'), 'w') as fh:
        json.dump(config, fh, indent=2)

    print(f'\nwrote {out}  ({len(df)} rows, {time.time() - t0:.0f}s)')
    solved = df.groupby('method')['fit'].mean().sort_values(ascending=False)
    print('\nsolve rate by method:')
    print(solved.round(3).to_string())


if __name__ == '__main__':
    main()
