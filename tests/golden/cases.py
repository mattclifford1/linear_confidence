'''
Golden (characterisation) cases for the legacy deltas code.

These pin what the existing estimators output *today*, so that the modular
refactor (moving code into deltas/legacy/, rebuilding the overlap methods from
components) can be shown not to change a single number. They are not claims
that the outputs are right - several are known to rest on bugs that are part of
the published results (FINDINGS.md §7.1 B1, B2), and must stay.

Every estimator is imported through its *current public path*
(`deltas.model.downsample`, ...), so after the move these cases also prove the
import shims work.

`USE_TWO` is read at import time all over the legacy package, so the published
setting (`USE_TWO = False`) can only be exercised in a fresh interpreter. That
is what the command-line entry point is for:

    python tests/golden/cases.py --use-two false     # prints JSON to stdout

Record new expectations (only when a change is *meant* to move a number):

    uv run python tests/golden/record.py
'''
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'fixtures')
FIXTURE_NAMES = ('gauss_wide', 'gauss_sep', 'gauss_overlap', 'breast_cancer', 'pima')


def load_fixture(name):
    with np.load(os.path.join(FIXTURES, f'{name}.npz')) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------- the cases ---
# name -> callable(clf, X, y) -> fitted model. Imports are inside the
# callables so that the USE_TWO flag can be set before any deltas module loads.
def _slacks(**kw):
    def run(clf, X, y):
        from deltas.model import downsample
        return downsample.downsample_deltas(clf).fit(X, y, **kw)
    return run


def _non_sep(**kw):
    def run(clf, X, y):
        from deltas.model import non_sep
        return non_sep.deltas(clf).fit(X, y, **kw)
    return run


def _overlap(cls_name, objective):
    def run(clf, X, y):
        from deltas.model import overlap
        return getattr(overlap, cls_name)(clf, objective=objective).fit(X, y)
    return run


def _base(clf, X, y):
    from deltas.model import base
    return base.base_deltas(clf).fit(X, y)


def _ssl(clf, X, y):
    from deltas.model import SSL
    random.seed(0)
    np.random.seed(0)
    return SSL.SSL_deltas(clf).fit(X, y, max_trials=100, parallel=False)


def _reprojection(clf, X, y):
    from deltas.model import reprojection
    return reprojection.reprojection_deltas(clf).fit(X, y)


def _svm_supports(clf, X, y):
    from deltas.model import SVM_supports
    return SVM_supports.SVM_supports_deltas(clf).fit(X, y, parallel=True)


CASES = {
    'base': _base,
    # the published method, configured as the experiment runners use it
    'slacks': _slacks(max_trials=10000, parallel=True),
    # the serial path must agree with the multiprocessing one; 8x slower, so
    # it only runs on one fixture (see ONLY)
    'slacks_serial': _slacks(max_trials=10000, parallel=False),
    'slacks_continuous': _slacks(max_trials=10000, parallel=True,
                                 continuous_slacks=True),
    'nonsep_min': _non_sep(loss_type='min'),
    'nonsep_max': _non_sep(loss_type='max'),
    'nonsep_mean': _non_sep(loss_type='mean'),
    'nonsep_furthest': _non_sep(only_furtherest_k=True),
    'cp_sum': _overlap('binomial_deltas', 'sum'),
    'cp_minimax': _overlap('binomial_deltas', 'minimax'),
    'dkw_sum': _overlap('dkw_deltas', 'sum'),
    'dkw_minimax': _overlap('dkw_deltas', 'minimax'),
    'ssl': _ssl,
    'reprojection': _reprojection,
    'svm_supports': _svm_supports,
}

#: cases restricted to a subset of fixtures (everything else runs on all)
ONLY = {'slacks_serial': ('gauss_overlap',)}


def keys():
    '''every (fixture/case) key, in a stable order'''
    return [f'{fx}/{case}' for fx in FIXTURE_NAMES for case in CASES
            if fx in ONLY.get(case, FIXTURE_NAMES)]


# ------------------------------------------------------------- recording ---
def _num(x):
    '''a plain float (or None) that survives a JSON round trip exactly'''
    if x is None:
        return None
    x = np.asarray(x, dtype=float).ravel()
    return float(x[0]) if x.size == 1 else [float(v) for v in x]


def summarise(model, z_test):
    out = {'is_fit': bool(getattr(model, 'is_fit', False))}
    for attr in ('boundary', 'delta1', 'delta2', 'loss'):
        if hasattr(model, attr):
            out[attr] = _num(getattr(model, attr))
    if hasattr(model, 'class_nums'):
        out['class_nums'] = [int(c) for c in model.class_nums]
    if out['is_fit'] and hasattr(model, 'certified_error'):
        out['certificate'] = {k: _num(v) for k, v in model.certified_error().items()}
    if out['is_fit']:
        preds = np.asarray(model.predict(z_test[:, None])).astype(int).ravel()
        out['preds'] = ''.join(map(str, preds))
    # internals the chosen boundary does not always reveal: the whole candidate
    # sweep and loss curve of the overlap methods (their outermost candidates
    # are rarely chosen, so a change there would otherwise go unseen), and how
    # many points the slack search removed
    for attr in ('candidates', 'losses'):
        if hasattr(model, attr):
            arr = np.ascontiguousarray(np.asarray(getattr(model, attr), dtype=float))
            out[f'{attr}_sha256'] = hashlib.sha256(arr.tobytes()).hexdigest()
    info = getattr(model, 'data_info', None)
    if isinstance(info, dict) and 'num_reduced' in info:
        out['num_reduced'] = _num(info['num_reduced'])
    return out


def run_case(case, fixture):
    from deltas.classifiers.frozen import FrozenProjection
    fx = load_fixture(fixture)
    clf = FrozenProjection(float(fx['threshold']))
    X = fx['z_train'][:, None]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = CASES[case](clf, X, fx['y_train'])
        return summarise(model, fx['z_test'])
    except Exception as e:          # a crash is a behaviour too - pin it
        return {'error': type(e).__name__}


def run_functions():
    '''the pure functions that will move with the legacy code'''
    import deltas.utils.radius as radius
    import deltas.utils.equations as eq
    from deltas.model import base
    fx = load_fixture('breast_cancer')
    info = base.base_deltas.get_data_info(fx['z_train'][:, None], fx['y_train'])
    deltas = [0.01, 0.1, 0.5, 0.9, 0.999]
    out = {
        'error_upper_bound': [_num(radius.error_upper_bound(1.3, n, d))
                              for n in (5, 50, 500) for d in deltas],
        'R_upper_bound': [_num(radius.R_upper_bound(0.7, 2.0, n, d))
                          for n in (5, 50, 500) for d in deltas],
        'loss': [_num(eq.loss(d1, d2, info)) for d1 in deltas for d2 in deltas],
        'delta2_given_delta1': [_num(eq.delta2_given_delta1_matt(d, info))
                                for d in deltas],
        'contraint_eq7': [_num(eq.contraint_eq7(d, d, info)) for d in deltas],
        'data_info': {k: _num(info[k]) for k in (
            'empirical R1', 'empirical R2', 'empirical D', 'R all data',
            'empirical_projected_mean 1', 'empirical_projected_mean 2',
            'N1', 'N2')},
    }
    return out


def run_all():
    results = {}
    for key in keys():
        fx, case = key.split('/')
        results[key] = run_case(case, fx)
    results['functions'] = run_functions()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-two', choices=('true', 'false'), required=True)
    args = ap.parse_args()
    # must happen before any other deltas import: the flag is bound at import
    import deltas.misc.use_two as ut
    ut.USE_TWO = (args.use_two == 'true')
    # legacy code prints freely; keep stdout clean for the JSON
    with contextlib.redirect_stdout(io.StringIO()):
        results = run_all()
    json.dump(results, sys.stdout)


if __name__ == '__main__':
    main()
