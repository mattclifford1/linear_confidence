'''The experiment caches reuse results only while their inputs are unchanged.'''
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import fingerprint as fp  # noqa: E402


def test_source_hash_ignores_comments_and_docstrings_only():
    src = "def f(x):\n    return x + 1\n"
    assert fp.source_hash(src) == fp.source_hash(
        "def f(x):\n    '''adds one'''\n    # a comment\n    return x + 1  # here too\n")
    assert fp.source_hash(src) != fp.source_hash("def f(x):\n    return x + 2\n")


def test_stable_is_json_and_repeatable():
    obj = {'a': (1, 2.5), 'f': lambda x: x, 'arr': np.arange(3), 'n': np.int64(4)}
    assert fp.digest(obj) == fp.digest(dict(obj))


@pytest.fixture
def ep():
    pytest.importorskip('projection_models')
    import export_projections
    return export_projections


def test_exports_extend_only_while_settings_and_data_match(ep, tmp_path, monkeypatch):
    out = str(tmp_path)
    assert ep.run_pair('Gaussian', 'Linear', [0], out_dir=out)[1] == 'ok'
    assert ep.run_pair('Gaussian', 'Linear', [0, 1], out_dir=out)[1] == 'ok'
    assert ep.run_pair('Gaussian', 'Linear', [0, 1], out_dir=out)[1] == 'cached'

    path = os.path.join(out, 'Gaussian__Linear.npz')
    old = ep.MODELS['Linear']
    monkeypatch.setitem(ep.MODELS, 'Linear',
                        (lambda **kw: ep.pm.Linear(max_iter=7, **kw), True))
    status, why = ep.run_pair('Gaussian', 'Linear', [0, 1, 2], out_dir=out)[1:]
    assert status == 'stale' and 'model' in why

    monkeypatch.setitem(ep.MODELS, 'Linear', old)
    arrays = ep._read_npz(path)
    arrays['s0_data'] = np.array(['not the data'])
    ep._write_npz(path, arrays)
    status, why = ep.run_pair('Gaussian', 'Linear', [0, 1, 2], out_dir=out)[1:]
    assert status == 'stale' and 'data' in why


def test_files_without_settings_are_adopted(ep, tmp_path):
    out = str(tmp_path)
    ep.run_pair('Gaussian', 'Linear', [0], out_dir=out)
    path = os.path.join(out, 'Gaussian__Linear.npz')
    arrays = ep._read_npz(path)
    del arrays['settings']
    ep._write_npz(path, arrays)
    assert ep.run_pair('Gaussian', 'Linear', [0], out_dir=out)[1] == 'cached'
    assert 'settings' in np.load(path).files


def test_method_results_are_reused_per_seed_and_method(ep, tmp_path):
    import run_wide
    proj, cache = str(tmp_path / 'proj'), str(tmp_path / 'cache')
    os.makedirs(cache)
    ep.run_pair('Gaussian', 'Linear', [0, 1], out_dir=proj)
    path = os.path.join(proj, 'Gaussian__Linear.npz')

    def new(methods, seeds, code='c', timeout=0):
        return run_wide.run_file(path, methods, seeds, timeout, None, code, cache)[1]

    assert new(['CP Minimax'], [0]) == 2               # naive and split
    assert new(['CP Minimax'], [0]) == 0
    assert new(['CP Minimax'], [0, 1]) == 2            # only the new seed
    assert new(['CP Minimax', 'DKW Minimax'], [0, 1]) == 4   # only the new method
    assert new(['CP Minimax'], [0, 1], code='changed') == 4
    assert new(['CP Minimax'], [0, 1], timeout=60) == 4
