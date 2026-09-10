'''
Tests for deltas.model.overlap — run with:  pytest tests/

These cover the properties the write-up actually claims, so if one breaks the
claim in the paper is wrong, not just the code.
'''
import numpy as np
import pytest

from deltas.model import overlap
from deltas.model.overlap import clopper_pearson_upper, dkw_upper


def _toy(z1, z2):
    X = np.concatenate([z1, z2])[:, None]
    y = np.r_[np.zeros(len(z1)), np.ones(len(z2))].astype(int)
    return X, y


MODELS = [overlap.binomial_deltas, overlap.dkw_deltas]
RULES = ['sum', 'minimax']


# ------------------------------------------------------------- the bounds ---
def test_clopper_pearson_covers():
    '''simulated coverage must reach the nominal level'''
    rng = np.random.default_rng(0)
    for N, e, delta in [(50, 0.1, 0.05), (100, 0.02, 0.1), (20, 0.3, 0.05)]:
        m = rng.binomial(N, e, size=20000)
        assert np.mean(clopper_pearson_upper(m, N, delta) >= e) >= 1 - delta


def test_dkw_covers():
    rng = np.random.default_rng(1)
    for N, e, delta in [(50, 0.1, 0.05), (100, 0.02, 0.1), (20, 0.3, 0.05)]:
        m = rng.binomial(N, e, size=20000)
        assert np.mean(dkw_upper(m, N, delta) >= e) >= 1 - delta


def test_bounds_are_probabilities_and_monotone_in_m():
    for fn in (clopper_pearson_upper, dkw_upper):
        ub = fn(np.arange(0, 21), 20, 0.05)
        assert np.all((ub >= 0) & (ub <= 1))
        assert np.all(np.diff(ub) >= -1e-12)      # more errors -> looser bound
        assert ub[-1] == pytest.approx(1.0)       # m == N tells us nothing


def test_separable_corner_matches_published_order():
    '''at m=0 Clopper-Pearson is 1 - delta**(1/N) ~ ln(1/delta)/N'''
    for N in (10, 100, 1000):
        got = float(clopper_pearson_upper(0, N, 0.05))
        assert got == pytest.approx(1 - 0.05 ** (1 / N), rel=1e-9)
        assert got == pytest.approx(np.log(1 / 0.05) / N, rel=0.15)


# ------------------------------------------------------------ the models ---
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('rule', RULES)
def test_always_solvable(model, rule):
    '''the headline claim: there is no infeasible case'''
    rng = np.random.default_rng(2)
    cases = [
        (rng.normal(0, 1, 50), rng.normal(4, 1, 20)),      # separated
        (rng.normal(0, 1, 50), rng.normal(0, 1, 20)),      # total overlap
        (rng.normal(0, 1, 5), rng.normal(0.2, 1, 3)),      # tiny N
        (np.array([0.0]), np.array([1.0])),                # N = 1 each
        (np.zeros(10), np.zeros(10)),                      # degenerate
        (rng.normal(4, 1, 20), rng.normal(0, 1, 50)),      # reversed order
        (np.r_[rng.normal(0, 1, 50), 99.0], rng.normal(3, 1, 20)),  # outlier
    ]
    for z1, z2 in cases:
        X, y = _toy(z1, z2)
        m = model(objective=rule).fit(X, y)
        assert m.is_fit is True
        assert np.isfinite(m.boundary)
        assert set(np.unique(m.predict(X))) <= {0, 1}


@pytest.mark.parametrize('model', MODELS)
def test_separates_well_separated_classes(model):
    rng = np.random.default_rng(3)
    X, y = _toy(rng.normal(0, 1, 200), rng.normal(8, 1, 200))
    m = model(objective='minimax').fit(X, y)
    assert (m.predict(X) == y).mean() > 0.98


@pytest.mark.parametrize('model', MODELS)
def test_certificate_is_reported_and_in_range(model):
    rng = np.random.default_rng(4)
    X, y = _toy(rng.normal(0, 1, 100), rng.normal(3, 1, 40))
    c = model().fit(X, y).certified_error()
    for k in ('class 1', 'class 2', 'delta1', 'delta2'):
        assert 0.0 <= c[k] <= 1.0


@pytest.mark.parametrize('model', MODELS)
def test_minimax_does_not_abandon_the_scarce_class(model):
    '''
    the sum rule pushes the boundary towards a scarce minority (its certificate
    is flat, so the majority term takes over); minimax must not
    '''
    rng = np.random.default_rng(5)
    X, y = _toy(rng.normal(0, 1, 1000), rng.normal(4, 1, 10))
    b_sum = model(objective='sum').fit(X, y).boundary
    b_mmx = model(objective='minimax').fit(X, y).boundary
    assert b_mmx < b_sum      # minimax leaves the minority more room


@pytest.mark.parametrize('model', MODELS)
def test_orientation_is_symmetric(model):
    '''swapping which label is the upper class must mirror the boundary'''
    rng = np.random.default_rng(6)
    z1, z2 = rng.normal(0, 1, 60), rng.normal(3, 1, 30)
    a = model(objective='minimax').fit(*_toy(z1, z2))
    b = model(objective='minimax').fit(*_toy(-z2, -z1))
    assert a.boundary == pytest.approx(-b.boundary, abs=1e-9)


# ------------------------------------------------- the ternary-search path ---
def test_sparse_loss_table_matches_the_dense_one():
    '''
    the ternary search is a speedup, not a different method

    binomial_deltas switches to _sparse_loss_table once (N+1) * resolution
    passes dense_table_limit, which is what makes MIMIC-IV tractable. The
    losses it returns must match the dense grid exactly - only the delta
    reported at an exact tie may differ, and tied deltas give the same loss.
    '''
    for N in (50, 200, 1001):
        m = overlap.binomial_deltas(delta_resolution=200)
        dense = m._per_class_loss_table(N)
        m.dense_table_limit = 1            # force the search path
        sparse = m._per_class_loss_table(N)
        assert np.allclose(sparse['loss'], dense['loss'], atol=1e-12)
        assert np.allclose(sparse['bound'], dense['bound'], atol=1e-12)


def test_dkw_never_takes_the_search_path():
    '''
    DKW's bound clips at 1 for small delta, so its loss carries a flat
    plateau a ternary search can step across. It must stay on the dense path.
    '''
    assert overlap.dkw_deltas().dense_table_limit == np.inf
