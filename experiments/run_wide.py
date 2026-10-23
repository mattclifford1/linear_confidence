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
import contextlib
import glob
import json
import os
import signal
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score

import deltas.misc.use_two as use_two_cfg
import fingerprint as fp
from deltas.classifiers.frozen import FrozenProjection
from deltas.methods import METHODS, describe_all


HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, 'projections')
RESULTS = os.path.join(HERE, 'results')
CACHE = os.path.join(RESULTS, '.cache')
#: bump to recompute every cached method result after a change the hashes miss
CACHE_VERSION = 1

# the grid's default methods, from the single registry (deltas/methods/).
# The slack method runs with a smaller budget and serially here: each grid
# cell is already a joblib worker. Any other registered method can be named
# with --methods.
DELTAS_METHODS = {
    'Slacks Deltas': METHODS['Slacks Deltas'].configured(max_trials=2000,
                                                         parallel=False),
    **{name: METHODS[name] for name in (
        'Min Deltas', 'F Deltas', 'CP Sum', 'CP Minimax', 'DKW Sum',
        'DKW Minimax')},
}
LOOKUP = {**METHODS, **DELTAS_METHODS}

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


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def time_budget(seconds):
    '''
    abort a fit that overruns

    Needed once MIMIC-IV joined the grid: at N = 54k the published slack
    method does not finish at all (its downsampling loop is O(N) iterations,
    each O(N)), and the order-statistic variants take minutes per fit. A
    method that cannot produce an answer in the budget is recorded as
    unsolved-with-reason 'timeout', which is a scaling result rather than a
    missing measurement. Zero or None disables the budget.
    '''
    if not seconds:
        yield
        return

    def _fire(signum, frame):
        raise _Timeout()

    try:
        old = signal.signal(signal.SIGALRM, _fire)
    except ValueError:          # not the main thread - run unbounded
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
    above_0 = len(z0) - np.searchsorted(np.sort(z0), mids, side='right')
    below_1 = np.searchsorted(np.sort(z1), mids, side='right')
    err = 0.5 * (above_0 / len(z0) + below_1 / len(z1))
    return float(mids[int(np.argmin(err))])


def _probes():
    '''small score sets, overlapping, separable and with few minority points'''
    rng = np.random.default_rng(0)
    for n0, n1, gap in ((80, 10, 1.0), (80, 10, 8.0), (40, 4, 2.0)):
        z = np.concatenate([rng.normal(0, 1, n0), rng.normal(gap, 1, n1)])
        yield z[:, None], np.r_[np.zeros(n0, int), np.ones(n1, int)]


def _run_probes(name):
    with open(os.devnull, 'w') as quiet:
        for X, y in _probes():
            try:
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    fitted = LOOKUP[name](FrozenProjection(0.0), X, y)
                    if bool(getattr(fitted, 'is_fit', False)):
                        fitted.predict(X)
                        if hasattr(fitted, 'certified_error'):
                            fitted.certified_error()
            except Exception:
                pass


def method_code(methods):
    '''
    per method, every input of its results other than its data and settings;
    of the deltas package, only the files whose functions it runs on `_probes`
    '''
    deltas_dir = os.path.dirname(os.path.dirname(use_two_cfg.__file__))
    shared = [CACHE_VERSION, fp.code_hash(_method_row, _row, _scores, time_budget),
              use_two_cfg.USE_TWO, use_two_cfg.USE_GLOBAL_R]
    code = {}
    for name in methods:
        files, _ = fp.executed_files(lambda: _run_probes(name), deltas_dir)
        code[name] = fp.digest([shared, fp.files_hash(files, deltas_dir)])
    return code


def _read_store(path):
    try:
        return pd.read_pickle(path)
    except Exception:
        return {}


def _add_to_store(path, new):
    '''merge into what is on disk, so two runs sharing a store lose nothing'''
    store = {**_read_store(path), **new}
    pd.to_pickle(store, path + '.tmp')
    os.replace(path + '.tmp', path)


def _store_path(path, cache):
    return os.path.join(cache, os.path.basename(path)[:-4] + '.pkl')


def _cells(f, seeds):
    '''(seed, mode, key prefix, arrays, fingerprint) for each exported (seed, mode)'''
    meta = json.loads(str(f['meta'][0]))
    keys = set(f.files)
    for seed in seeds:
        for mode in ('naive', 'split'):
            p = f's{seed}_{mode}_'
            if p + 'z_cert' not in keys:
                continue
            arrays = {k: f[k] for k in sorted(keys) if k.startswith(p)}
            cell = fp.digest([fp.arrays_hash(*arrays.values()), sorted(arrays),
                              [meta.get(k) for k in ('dataset', 'model',
                                                     'train_ratio', 'n_features')]])
            yield meta, seed, mode, p, arrays, cell


def _method_keys(cell, methods, specs, timeout, max_n, code):
    return {name: fp.digest([code[name], cell, name, specs[name], timeout,
                             max_n.get(name)])
            for name in methods}


def run_file(path, methods, seeds, timeout=0, max_n=None, code=None, cache=None):
    '''
    rows for one exported file; with `cache`, method results are reused per
    (seed, mode, method) when the data, the method's settings and `code[method]` match
    '''
    max_n = max_n or {}
    specs = {name: fp.digest(LOOKUP[name].describe()) for name in methods}
    store_path = cache and _store_path(path, cache)
    store = _read_store(store_path) if cache else {}
    rows, n_new = [], 0
    with np.load(path, allow_pickle=False) as f:
        for meta, seed, mode, p, arrays, cell in _cells(f, seeds):
            base, clf = _cell(meta, seed, mode, arrays, p)
            rows += _baselines(base, arrays, p)
            new = {}
            for name, key in _method_keys(cell, methods, specs, timeout, max_n,
                                          code).items():
                if key not in store:
                    new[key] = _method_row(name, base, clf, arrays, p,
                                           timeout, max_n)
                rows.append(dict(store.get(key) or new[key]))
            if new and cache:
                _add_to_store(store_path, new)
                store.update(new)
            n_new += len(new)
    return rows, n_new


def prune_file(path, methods, seeds, timeout, max_n, code, cache):
    '''drop the cached results of `path` this run would not reuse; returns how many'''
    store_path = _store_path(path, cache)
    store = _read_store(store_path)
    specs = {name: fp.digest(LOOKUP[name].describe()) for name in methods}
    with np.load(path, allow_pickle=False) as f:
        keep = {key for *_, cell in _cells(f, seeds)
                for key in _method_keys(cell, methods, specs, timeout, max_n,
                                        code).values()}
    kept = {k: v for k, v in store.items() if k in keep}
    if len(kept) < len(store):
        pd.to_pickle(kept, store_path + '.tmp')
        os.replace(store_path + '.tmp', store_path)
    return len(store) - len(kept)


def _cell(meta, seed, mode, a, p):
    '''the fields every row of this (seed, mode) shares, and the frozen classifier'''
    # how optimistic is this classifier on the points it was fitted to? This
    # is the quantity the calibration split exists to neutralise, so record it
    # per row: the coverage shortfall should track it.
    def _bal_err(z, yy, t):
        if not (yy == 0).any() or not (yy == 1).any():
            return np.nan
        return 0.5 * ((z[yy == 0] > t).mean() + (z[yy == 1] <= t).mean())

    y_cert, thr = a[p + 'y_cert'].astype(int), float(a[p + 'threshold'][0])
    fit_err = _bal_err(a[p + 'z_fit'], a[p + 'y_fit'].astype(int), thr)
    test_err = _bal_err(a[p + 'z_test'], a[p + 'y_test'].astype(int), thr)
    base = {'dataset': meta['dataset'], 'model': meta['model'],
            'seed': seed, 'mode': mode,
            'n_cert': int(len(y_cert)),
            'n_cert_minority': int((y_cert == 1).sum()),
            'train_ratio': meta.get('train_ratio', np.nan),
            'n_features': meta.get('n_features', np.nan),
            'clf_fit_err': fit_err,
            'clf_test_err': test_err,
            'optimism': test_err - fit_err}
    return base, FrozenProjection(thr, name=meta['model'])


def _row(base, y_test, method, preds=None, fitted=None, ok=True, reason=''):
    rec = {**base, 'method': method, 'fit': ok, 'reason': reason}
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
    return rec


def _baselines(base, a, p):
    z_test, y_test = a[p + 'z_test'], a[p + 'y_test'].astype(int)
    thr = float(a[p + 'threshold'][0])
    rows = [_row(base, y_test, 'Baseline', preds=(z_test > thr).astype(int))]
    t_star = _best_balanced_threshold(a[p + 'z_fit'], a[p + 'y_fit'].astype(int))
    rows.append(_row(base, y_test, 'Threshold', preds=(z_test > t_star).astype(int)))
    for label, name in (('smote', 'SMOTE'), ('bw', 'Balanced Weights')):
        if p + f'pred_{label}' in a:
            rows.append(_row(base, y_test, name, preds=a[p + f'pred_{label}'].astype(int)))
        else:
            rows.append(_row(base, y_test, name, ok=False))
    for label, name in (('focal', 'Focal'), ('ldam', 'LDAM')):   # deep nets only
        if p + f'pred_{label}' in a:
            rows.append(_row(base, y_test, name, preds=a[p + f'pred_{label}'].astype(int)))
    return rows


def _method_row(name, base, clf, a, p, timeout, max_n):
    y_cert, y_test = a[p + 'y_cert'].astype(int), a[p + 'y_test'].astype(int)
    Xc, Xt = a[p + 'z_cert'][:, None], a[p + 'z_test'][:, None]
    t0 = time.time()
    if len(y_cert) > max_n.get(name, np.inf):
        return {**_row(base, y_test, name, ok=False, reason='too large'),
                'seconds': np.nan}
    try:
        with time_budget(timeout):
            fitted = LOOKUP[name](clf, Xc, y_cert)
        if bool(getattr(fitted, 'is_fit', False)):
            preds = np.asarray(fitted.predict(Xt)).squeeze().astype(int)
            rec = _row(base, y_test, name, preds=preds, fitted=fitted)
        else:
            rec = _row(base, y_test, name, ok=False)
    except _Timeout:
        rec = _row(base, y_test, name, ok=False, reason='timeout')
    except Exception:
        rec = _row(base, y_test, name, ok=False, reason='error')
    return {**rec, 'seconds': round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--methods', nargs='*', default=list(DELTAS_METHODS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=0,
                    help='per-fit seconds budget; 0 disables. A fit that\n                          overruns is recorded as unsolved (reason=timeout).')
    ap.add_argument('--max-n', nargs='*', default=[], metavar='METHOD=N',
                    help='skip a method on certificate sets larger than N '
                         '(reason=too large)')
    ap.add_argument('--out', default='wide.csv',
                    help='results/<out>; the config goes beside it')
    ap.add_argument('--proj', default=PROJ, help='folder of exported .npz files')
    ap.add_argument('--prune', action='store_true',
                    help='drop the cached method results this command would not '
                         'reuse, then stop')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.proj, '*.npz')))
    if args.datasets:
        # the file-name form written by export_projections._slug
        want = {d.replace(':', '').replace(' ', '-').replace('/', '-')
                for d in args.datasets}
        files = [f for f in files
                 if os.path.basename(f).split('__')[0] in want]
    if args.models:
        files = [f for f in files
                 if os.path.basename(f).split('__')[1][:-4] in args.models]
    if not files:
        raise SystemExit(f'no exported projections in {args.proj} - run '
                         f'export_projections.py first (with the '
                         f'projection_models venv)')

    seeds = list(range(args.seeds))
    max_n = {m: int(n) for m, n in (a.rsplit('=', 1) for a in args.max_n)}
    if unknown := set(max_n) - set(args.methods):
        raise SystemExit(f'--max-n names methods not run: {sorted(unknown)}')
    # method results are kept per (file, seed, mode, method) and reused only
    # when the data, the method's settings and its code are unchanged
    code = method_code(args.methods)
    cache = os.path.join(CACHE, os.path.basename(os.path.abspath(args.proj)))
    os.makedirs(cache, exist_ok=True)
    print(f'{len(files)} exported (dataset, model) files, {len(seeds)} seeds, '
          f'{len(args.methods)} deltas methods', flush=True)
    for name, digest in code.items():
        print(f'  method code {digest}  {name}', flush=True)
    t0 = time.time()

    if args.prune:
        from joblib import Parallel, delayed
        dropped = Parallel(n_jobs=args.jobs)(
            delayed(prune_file)(f, args.methods, seeds, args.timeout, max_n,
                                code, cache)
            for f in files if os.path.exists(_store_path(f, cache)))
        print(f'dropped {sum(dropped)} cached method results', flush=True)
        return

    if args.jobs > 1:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_file)(f, args.methods, seeds, args.timeout, max_n,
                              code, cache)
            for f in files)
    else:
        chunks = []
        for i, f in enumerate(files):
            chunks.append(run_file(f, args.methods, seeds, args.timeout, max_n,
                                   code, cache))
            print(f'  [{i + 1}/{len(files)}] {os.path.basename(f)[:-4]}: '
                  f'{chunks[-1][1]} new ({time.time() - t0:.0f}s)', flush=True)
    n_new = sum(n for _, n in chunks)
    chunks = [rows for rows, _ in chunks]
    print(f'{n_new} method results computed, the rest reused', flush=True)

    df = pd.DataFrame([r for c in chunks for r in c])
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, args.out)
    df.to_csv(out, index=False)

    config = {'USE_TWO': use_two_cfg.USE_TWO,
              'USE_GLOBAL_R': use_two_cfg.USE_GLOBAL_R,
              'seeds': seeds, 'methods': args.methods,
              'method_specs': {n: LOOKUP[n].describe() for n in args.methods},
              'n_files': len(files), 'n_rows': len(df),
              'timeout': args.timeout, 'max_n': max_n, 'method_code': code}
    with open(out[:-len('.csv')] + '_config.json', 'w') as fh:
        json.dump(config, fh, indent=2)

    print(f'\nwrote {out}  ({len(df)} rows, {time.time() - t0:.0f}s)')
    solved = df.groupby('method')['fit'].mean().sort_values(ascending=False)
    print('\nsolve rate by method:')
    print(solved.round(3).to_string())


if __name__ == '__main__':
    main()
