'''
The named-method registry (deltas/methods/): the runners get every method
from it, so it must reproduce what their hand-written tables did.
'''
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from deltas.classifiers.frozen import FrozenProjection
from deltas.methods import METHODS, available, describe_all, get, make

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'golden'))
import cases  # noqa: E402


def _old_tables():
    '''the runners' tables before the registry, verbatim'''
    from deltas.model import downsample, non_sep, overlap
    run_experiments = {
        'Slacks Deltas': lambda clf, X, y: downsample.downsample_deltas(clf).fit(
            X, y, max_trials=10000, parallel=True),
        'Min Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(X, y, loss_type='min'),
        'Max Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(X, y, loss_type='max'),
        'Avg Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(X, y, loss_type='mean'),
        'F Deltas': lambda clf, X, y: non_sep.deltas(clf).fit(X, y, only_furtherest_k=True),
        'CP Sum': lambda clf, X, y: overlap.binomial_deltas(clf, objective='sum').fit(X, y),
        'CP Minimax': lambda clf, X, y: overlap.binomial_deltas(clf, objective='minimax').fit(X, y),
        'DKW Sum': lambda clf, X, y: overlap.dkw_deltas(clf, objective='sum').fit(X, y),
        'DKW Minimax': lambda clf, X, y: overlap.dkw_deltas(clf, objective='minimax').fit(X, y),
    }
    run_wide = dict(run_experiments)
    run_wide['Slacks Deltas'] = lambda clf, X, y: downsample.downsample_deltas(clf).fit(
        X, y, max_trials=2000, parallel=False)
    for k in ('Max Deltas', 'Avg Deltas'):
        run_wide.pop(k)
    return run_experiments, run_wide


def _outcome(model, z_test):
    if not getattr(model, 'is_fit', False):
        return (False,)
    return (True, float(np.ravel(model.boundary)[0]),
            np.asarray(model.predict(z_test[:, None])).astype(int).tobytes())


@pytest.mark.golden
@pytest.mark.parametrize('fixture', ['breast_cancer', 'pima', 'gauss_overlap'])
def test_runner_tables_are_reproduced(fixture):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                    'experiments'))
    import run_experiments
    import run_wide
    old_exp, old_wide = _old_tables()
    fx = cases.load_fixture(fixture)
    X, y = fx['z_train'][:, None], fx['y_train']
    clf = FrozenProjection(float(fx['threshold']))
    for old, new in ((old_exp, run_experiments.DELTAS_METHODS),
                     (old_wide, run_wide.DELTAS_METHODS)):
        assert list(old) == list(new)
        for name in old:
            assert _outcome(new[name](clf, X, y), fx['z_test']) == \
                _outcome(old[name](clf, X, y), fx['z_test']), name


@pytest.mark.parametrize('name', sorted(METHODS))
def test_every_method_fits(name):
    fx = cases.load_fixture('gauss_wide')
    X, y = fx['z_train'][:, None], fx['y_train']
    m = METHODS[name](FrozenProjection(float(fx['threshold'])), X, y)
    assert m.is_fit
    preds = np.asarray(m.predict(fx['z_test'][:, None]))
    assert (preds == fx['y_test']).mean() > 0.95


@pytest.mark.parametrize('name', available('envelope'))
def test_envelope_methods_always_report_the_assumption_free_certificate(name):
    fx = cases.load_fixture('breast_cancer')
    m = METHODS[name](None, fx['z_train'][:, None], fx['y_train'])
    assert 'clopper_pearson' in m.certificates_
    # the shape-based ones also report their own model-based certificate
    assert len(m.certificates_) == (1 if name == 'Saw-Yang-Mo Minimax' else 2)
    for cert in m.certificates_.values():
        assert cert['delta1'] == cert['delta2'] == 0.05


def test_configured_and_with_params_copy():
    slacks = get('Slacks Deltas')
    fast = slacks.configured(max_trials=10)
    assert fast.fit_kwargs['max_trials'] == 10
    assert slacks.fit_kwargs['max_trials'] == 10000
    pred = get('Gaussian Predictive')
    lg = pred.with_params(delta_report=0.1)
    assert lg.make().delta_report == 0.1 and pred.make().delta_report == 0.05
    assert make('CP Minimax').objective == 'minimax'
    with pytest.raises(KeyError, match='available'):
        get('No Such Method')


def test_families_and_descriptions():
    assert set(available('overlap')) == {'CP Sum', 'CP Minimax', 'DKW Sum',
                                         'DKW Minimax'}
    assert {'Slacks Deltas', 'F Deltas'} <= set(available())
    json.dumps(describe_all())


def test_importing_the_registry_does_not_import_the_legacy_code():
    '''
    the legacy code binds USE_TWO at import; importing deltas.methods must
    leave the caller free to set the flag afterwards
    '''
    code = ('import sys, deltas.methods; '
            'print(any(m in sys.modules for m in '
            '("deltas.model.downsample", "deltas.model.non_sep", '
            '"deltas.model.base")))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == 'False'
