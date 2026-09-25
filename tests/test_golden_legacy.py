'''
Golden (characterisation) tests: the legacy deltas estimators must keep
producing exactly the numbers they produce today.

The expectations were recorded by tests/golden/record.py, under both settings
of USE_TWO, and cover the published method (with and without continuous
slacks, serial and parallel), the non-separable variants, the overlap methods,
the exploratory estimators and the pure functions of equations.py/radius.py,
on five fixtures of pre-computed projections (tests/golden/fixtures/).

Comparisons are exact: the refactor moves this code, it does not rewrite it,
so not a single bit may change. Known bugs that feed the published numbers
(FINDINGS.md §7.1) are pinned deliberately - fixing one is a separate, visible
decision, made by re-recording.

Each flag is computed once, in a fresh interpreter (USE_TWO is bound at import
time), both in parallel. Deselect with `-m "not golden"` when iterating on
unrelated code.
'''
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')
sys.path.insert(0, GOLDEN)
import cases  # noqa: E402

FLAGS = ('true', 'false')

pytestmark = pytest.mark.golden


def _load_expected(flag):
    with open(os.path.join(GOLDEN, f'expected_use_two_{flag}.json')) as f:
        return json.load(f)


EXPECTED = {flag: _load_expected(flag) for flag in FLAGS}


def _compute(flag):
    out = subprocess.run([sys.executable, os.path.join(GOLDEN, 'cases.py'),
                          '--use-two', flag],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.fixture(scope='module')
def actual():
    with ThreadPoolExecutor(len(FLAGS)) as pool:
        return dict(zip(FLAGS, pool.map(_compute, FLAGS)))


def _same(a, b):
    '''exact equality, with NaN equal to NaN'''
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _diff(a, b, path=''):
    '''the first place two results differ, for a readable failure'''
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f'{path}/{k}: present in only one side'
            if not _same(a[k], b[k]):
                return _diff(a[k], b[k], f'{path}/{k}')
    return f'{path}: expected {b!r}, got {a!r}'


def test_expectations_cover_every_case():
    for flag in FLAGS:
        recorded = set(EXPECTED[flag]) - {'_meta'}
        assert recorded == set(cases.keys()) | {'functions'}, (
            'golden cases and recorded expectations disagree - re-record with '
            'tests/golden/record.py')


@pytest.mark.parametrize('flag', FLAGS)
@pytest.mark.parametrize('key', cases.keys() + ['functions'])
def test_legacy_output_unchanged(actual, flag, key):
    got, want = actual[flag][key], EXPECTED[flag][key]
    assert _same(got, want), f'USE_TWO={flag} {key}: {_diff(got, want)}'


def test_serial_and_parallel_slacks_agree():
    for flag in FLAGS:
        exp = EXPECTED[flag]
        for fx in cases.ONLY['slacks_serial']:
            assert _same(exp[f'{fx}/slacks_serial'], exp[f'{fx}/slacks'])


def test_use_two_actually_changes_the_published_method():
    # guards the subprocess mechanism itself: if the flag silently failed to
    # take effect, both files would agree and the False run would test nothing
    t, f = EXPECTED['true'], EXPECTED['false']
    assert not _same(t['functions']['error_upper_bound'],
                     f['functions']['error_upper_bound'])
    assert not _same(t['breast_cancer/slacks'], f['breast_cancer/slacks'])
