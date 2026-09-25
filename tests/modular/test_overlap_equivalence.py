'''
The modular rebuild of the overlap methods must be bit-for-bit the original.

Three implementations are compared on many data sets - random draws, the edge
cases of test_overlap.py, ties, integer scores, costs, reversed orientation,
and a size that forces the Clopper-Pearson ternary-search path:

    legacy   deltas.legacy.overlap.overlap  (the frozen original)
    shim     deltas.model.overlap           (what callers import)
    direct   deltas.core.DeltasEstimator    (the composition itself)

Every attribute a caller could read is compared with exact equality.
'''
import numpy as np
import pytest

from deltas.core import DeltasEstimator
from deltas.legacy.overlap import overlap as legacy
from deltas.model import overlap as shim

SCALARS = ['boundary', 'loss', 'delta1', 'delta2', 'error_bound_1',
           'error_bound_2', 'm_low', 'm_high', 'delta_low', 'delta_high',
           'bound_low', 'bound_high', 'N1', 'N2', 'is_fit']
ARRAYS = ['candidates', 'losses', '_L_low', '_L_high', '_low_sorted',
          '_high_sorted', 'class_nums']

BOUNDS = {'clopper_pearson': ('binomial_deltas', 'clopper_pearson'),
          'dkw': ('dkw_deltas', 'dkw')}


def _data():
    rng = np.random.default_rng(2026)
    out = []
    for i in range(14):
        n0, n1 = rng.integers(2, 400), rng.integers(1, 60)
        s0, s1 = rng.uniform(0.3, 3), rng.uniform(0.3, 3)
        z = np.r_[rng.normal(0, s0, n0), rng.normal(rng.uniform(-1, 5), s1, n1)]
        out.append((f'random{i}', z, np.r_[np.zeros(n0), np.ones(n1)]))
    edge = [
        ('tiny', rng.normal(0, 1, 5), rng.normal(0.2, 1, 3)),
        ('one_each', np.array([0.0]), np.array([1.0])),
        ('degenerate', np.zeros(10), np.zeros(10)),
        ('reversed', rng.normal(4, 1, 20), rng.normal(0, 1, 50)),
        ('outlier', np.r_[rng.normal(0, 1, 50), 99.0], rng.normal(3, 1, 20)),
        ('ties', np.round(rng.normal(0, 1, 80), 1),
         np.round(rng.normal(1, 1, 20), 1)),
        ('integers', rng.integers(-5, 5, 60).astype(float),
         rng.integers(-2, 8, 15).astype(float)),
        ('total_overlap', rng.normal(0, 1, 100), rng.normal(0, 1, 100)),
    ]
    for name, z0, z1 in edge:
        out.append((name, np.r_[z0, z1],
                    np.r_[np.zeros(len(z0)), np.ones(len(z1))]))
    return out


DATA = _data()
COSTS = [(1, 1), (3, 1), (1, 7.5)]


def _fit_all(bound, rule, z, y, costs):
    cls_name, reg_name = BOUNDS[bound]
    X = z[:, None]
    return {
        'legacy': getattr(legacy, cls_name)(objective=rule).fit(X, y, costs=costs),
        'shim': getattr(shim, cls_name)(objective=rule).fit(X, y, costs=costs),
        'direct': DeltasEstimator(bound=reg_name, rule=rule).fit(
            X, y, costs=costs),
    }


def _assert_identical(ref, other, label):
    for a in SCALARS:
        assert getattr(other, a) == getattr(ref, a), f'{label}: {a}'
    for a in ARRAYS:
        assert np.array_equal(np.asarray(getattr(other, a)),
                              np.asarray(getattr(ref, a))), f'{label}: {a}'
    assert other.certified_error() == ref.certified_error(), label
    assert other.get_bias() == ref.get_bias(), label


@pytest.mark.parametrize('bound', sorted(BOUNDS))
@pytest.mark.parametrize('rule', ['sum', 'minimax'])
@pytest.mark.parametrize('case', range(len(DATA)), ids=[d[0] for d in DATA])
def test_identical_to_the_original(bound, rule, case):
    name, z, y = DATA[case]
    costs = COSTS[case % len(COSTS)]
    fits = _fit_all(bound, rule, z, y, costs)
    grid = np.linspace(z.min() - 1, z.max() + 1, 301)[:, None]
    for impl in ('shim', 'direct'):
        _assert_identical(fits['legacy'], fits[impl], f'{name}/{impl}')
        assert np.array_equal(fits[impl].predict(grid),
                              fits['legacy'].predict(grid))


@pytest.mark.parametrize('rule', ['sum', 'minimax'])
def test_identical_on_the_ternary_search_path(rule):
    '''N = 3000 puts (N+1) x 2000 over the dense limit: the search path'''
    rng = np.random.default_rng(7)
    z = np.r_[rng.normal(0, 1, 3000), rng.normal(2.5, 1, 40)]
    y = np.r_[np.zeros(3000), np.ones(40)]
    fits = _fit_all('clopper_pearson', rule, z, y, (1, 1))
    assert (3001 * 2000) > legacy.binomial_deltas.dense_table_limit
    for impl in ('shim', 'direct'):
        _assert_identical(fits['legacy'], fits[impl], impl)


def test_loss_tables_identical():
    for N in (1, 17, 250):
        for L, S in ((legacy.binomial_deltas(delta_resolution=300),
                      shim.binomial_deltas(delta_resolution=300)),
                     (legacy.dkw_deltas(delta_resolution=300),
                      shim.dkw_deltas(delta_resolution=300))):
            a, b = L._per_class_loss_table(N), S._per_class_loss_table(N)
            for k in ('loss', 'delta', 'bound'):
                assert np.array_equal(a[k], b[k])


def test_bound_functions_are_the_same_objects_re_exported():
    from deltas.bounds import counts
    assert shim.clopper_pearson_upper is counts.clopper_pearson_upper
    assert shim.dkw_upper is counts.dkw_upper
    m = np.arange(0, 31)
    assert np.array_equal(counts.clopper_pearson_upper(m, 30, 0.05),
                          legacy.clopper_pearson_upper(m, 30, 0.05))
    assert np.array_equal(counts.dkw_upper(m, 30, 0.05),
                          legacy.dkw_upper(m, 30, 0.05))
