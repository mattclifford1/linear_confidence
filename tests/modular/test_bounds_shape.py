'''
Levels 1 and 2 of the assumption ladder: location-scale envelopes, the
Student-t predictive curve, and the moment bounds.

The headline property of a high-probability envelope - it covers the true
class error at every boundary at once with probability >= 1 - delta when its
assumption holds - is checked by simulation.
'''
import numpy as np
import pytest
from scipy import stats

from deltas.bounds import (Cantelli, GaussianConfidence,
                           LocationScaleConfidence, SawYangMo,
                           StudentTPredictive, VysochanskijPetunin)
from deltas.confidence import FixedDelta, OptimisedDelta
from deltas.core import ClassSample, DeltasEstimator

REPS = 1500


def _coverage(make_bound, draw, true_tail, N, delta, side, seed):
    '''fraction of samples whose envelope covers the truth on a whole grid'''
    rng = np.random.default_rng(seed)
    grid = np.linspace(-6, 6, 241)
    ok = 0
    for _ in range(REPS):
        s = ClassSample(draw(rng, N), 0, side)
        U = make_bound().fit(s).curve(grid, delta)
        ok += np.all(U >= true_tail(grid) - 1e-12)
    return ok / REPS


@pytest.mark.parametrize('side', ['low', 'high'])
def test_gaussian_rectangle_covers(side):
    tail = (lambda b: stats.norm.sf(b)) if side == 'low' else \
        (lambda b: stats.norm.cdf(b))
    cov = _coverage(GaussianConfidence, lambda r, n: r.normal(0, 1, n), tail,
                    N=12, delta=0.1, side=side, seed=1)
    assert cov >= 0.9 - 0.02


def test_gaussian_mc_band_covers_and_is_tighter():
    cov = _coverage(lambda: GaussianConfidence(band='mc', draws=20000),
                    lambda r, n: r.normal(0, 1, n), stats.norm.sf,
                    N=12, delta=0.1, side='low', seed=2)
    assert cov >= 0.9 - 0.025
    s = ClassSample(np.random.default_rng(3).normal(0, 1, 12), 0, 'low')
    b = np.array([2.5])
    rect = GaussianConfidence().fit(s).curve(b, 0.1)
    band = GaussianConfidence(band='mc', draws=20000).fit(s).curve(b, 0.1)
    assert band[0] <= rect[0]


def test_mc_band_needs_a_fixed_delta():
    X = np.r_[np.random.default_rng(0).normal(0, 1, 50), np.random.default_rng(1).normal(3, 1, 10)]
    y = np.r_[np.zeros(50), np.ones(10)]
    with pytest.raises(NotImplementedError, match='fixed'):
        DeltasEstimator(bound=GaussianConfidence(band='mc'),
                        confidence=OptimisedDelta(resolution=20)).fit(X[:, None], y)


def test_logistic_family_covers_logistic_data():
    cov = _coverage(lambda: LocationScaleConfidence('logistic', draws=20000),
                    lambda r, n: r.logistic(0, 1, n), stats.logistic.sf,
                    N=15, delta=0.1, side='low', seed=4)
    assert cov >= 0.9 - 0.025


def test_monte_carlo_pivots_reproduce_the_exact_gaussian():
    s = ClassSample(np.random.default_rng(5).normal(0, 1, 20), 0, 'low')
    exact = GaussianConfidence().fit(s).extremes(0.05)
    mc = LocationScaleConfidence('gaussian', draws=200000).fit(s).extremes(0.05)
    assert mc[0] == pytest.approx(exact[0], rel=0.03)
    assert mc[1] == pytest.approx(exact[1], rel=0.03)


def test_fewer_points_mean_more_room():
    '''same mean and s.d., smaller N -> larger envelope beyond the data'''
    base = np.random.default_rng(6).normal(0, 1, 200)
    base = (base - base.mean()) / base.std(ddof=1)
    small = np.r_[-1.2, -0.6, 0.0, 0.6, 1.2]
    small = (small - small.mean()) / small.std(ddof=1)
    b = np.array([3.0])
    for B in (GaussianConfidence(), StudentTPredictive()):
        big_U = type(B)().fit(ClassSample(base, 0, 'low')).curve(b, 0.05)
        small_U = type(B)().fit(ClassSample(small, 0, 'low')).curve(b, 0.05)
        assert small_U[0] > big_U[0]


def test_envelope_shape():
    s = ClassSample(np.random.default_rng(7).normal(0, 1, 30), 0, 'low')
    g = GaussianConfidence().fit(s)
    b = np.linspace(-3, 6, 91)
    U = g.curve(b, 0.05)
    assert np.all((U >= 0) & (U <= 1))
    assert np.all(np.diff(U) <= 1e-15)                 # falls away from the class
    assert np.all(g.curve(b, 0.01) >= U - 1e-15)       # smaller delta, wider
    assert g.curve(np.array([-10.0]), 0.05)[0] == 1.0  # inside the class: 1


def test_degenerate_samples_carry_no_information():
    for z in ([1.0], [2.0, 2.0, 2.0]):
        s = ClassSample(z, 0, 'low')
        for B in (GaussianConfidence(), StudentTPredictive(), SawYangMo()):
            assert np.all(B.fit(s).curve(np.array([5.0, 50.0]), 0.05) == 1.0)


def test_student_t_predictive_is_exact_for_a_fixed_rule():
    N, c = 8, 1.7
    rng = np.random.default_rng(8)
    x = rng.normal(0, 1, (200000, N + 1))
    xb, s = x[:, :N].mean(1), x[:, :N].std(1, ddof=1)
    sim = np.mean(x[:, N] <= xb - c * s * np.sqrt(1 + 1 / N))
    exact = stats.t.cdf(-c, N - 1)
    assert abs(sim - exact) < 4 * np.sqrt(exact * (1 - exact) / 200000)
    sample = ClassSample(x[0, :N], 1, 'high')
    b = sample.mean - c * sample.std * np.sqrt(1 + 1 / N)
    assert StudentTPredictive().fit(sample).curve(np.array([b]))[0] == \
        pytest.approx(exact, rel=1e-12)


# ------------------------------- the closed-form minimax boundary (notes) ---
def _two_class(seed=9, n0=400, n1=12, m1=3.0, s1=1.5):
    rng = np.random.default_rng(seed)
    z = np.r_[rng.normal(0, 1, n0), rng.normal(m1, s1, n1)]
    return z[:, None], np.r_[np.zeros(n0), np.ones(n1)]


def test_minimax_boundary_matches_the_closed_form():
    X, y = _two_class()
    m = DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                        search='grid').fit(X, y)
    b_star, value = m.bounds_['low'].closed_form_minimax(m.bounds_['high'], 0.05)
    assert m.boundary == pytest.approx(b_star, abs=1e-9)
    assert m.loss == pytest.approx(value, rel=1e-9)


class _GaussianExtremesCantelliTail(GaussianConfidence):
    '''same (mu, sigma) bounds as the Gaussian, Cantelli's tail'''

    def tail(self, k):
        return 1.0 / (1.0 + np.asarray(k) ** 2)


def test_minimax_boundary_does_not_depend_on_the_tail_shape():
    '''the "shape-free" result: only the certified value changes'''
    X, y = _two_class()
    kw = dict(confidence=FixedDelta(0.05), search='grid')
    g = DeltasEstimator(bound=GaussianConfidence(), **kw).fit(X, y)
    c = DeltasEstimator(bound=_GaussianExtremesCantelliTail(), **kw).fit(X, y)
    assert g.boundary == pytest.approx(c.boundary, abs=1e-8)
    assert c.loss > g.loss


def test_scarce_or_spread_minority_gets_more_room():
    '''the minimax boundary moves towards the majority as N_2 shrinks'''
    b = []
    for n1 in (6, 30, 300):
        X, y = _two_class(seed=10, n1=n1, s1=1.0)
        b.append(DeltasEstimator(bound='gaussian', confidence=FixedDelta(0.05),
                                 search='grid').fit(X, y).boundary)
    assert b[0] < b[1] < b[2]


# --------------------------------------------------------------- moments ---
def test_saw_yang_mo_formula_and_floor():
    s = ClassSample(np.random.default_rng(11).normal(0, 1, 10), 0, 'low')
    sym = SawYangMo().fit(s)
    lam = np.array([1.5, 3.0, 50.0])
    b = s.mean + lam * s.std
    N = 10
    expected = np.minimum(1, np.floor((N + 1) * (N ** 2 - 1 + N * lam ** 2) /
                                      (N ** 2 * lam ** 2)) / (N + 1))
    assert np.allclose(sym.curve(b), expected)
    assert sym.curve(np.array([1e6]))[0] == pytest.approx(1 / (N + 1))


def test_bounded_moment_bounds_need_a_range():
    s = ClassSample(np.random.default_rng(12).uniform(0, 1, 50), 0, 'low')
    with pytest.raises(ValueError, match='score_range'):
        Cantelli().fit(s)
    with pytest.raises(ValueError, match='outside'):
        Cantelli(score_range=(0.2, 1.0)).fit(s)


def test_cantelli_covers_bounded_data():
    cov = _coverage(lambda: Cantelli(score_range=(-6, 6)),
                    lambda r, n: np.clip(r.normal(0, 1, n), -6, 6),
                    stats.norm.sf, N=40, delta=0.1, side='low', seed=13)
    assert cov >= 0.9


def test_unimodal_tail_is_tighter_than_cantelli_in_the_tail():
    k = np.array([1.5, 2.0, 4.0])
    assert np.all(VysochanskijPetunin(score_range=(0, 1)).tail(k) <
                  Cantelli(score_range=(0, 1)).tail(k))
    # and both tails are continuous where VP switches regime
    kk = np.sqrt(5 / 3)
    vp = VysochanskijPetunin(score_range=(0, 1))
    assert vp.tail(kk - 1e-9) == pytest.approx(vp.tail(kk + 1e-9), abs=1e-7)
