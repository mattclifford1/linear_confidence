'''
Unit tests for the slot components: registry, samples, count bounds, delta
policies, rules, candidate search and transforms.
'''
import numpy as np
import pytest

from deltas.bounds import DKW, ClopperPearson, clopper_pearson_upper
from deltas.confidence import FixedDelta, OptimisedDelta
from deltas.core import Bound, ClassSample, ProjectedData, registry
from deltas.rules import Minimax, Sum
from deltas.search import Auto, DataMidpoints, Grid
from deltas.transforms import Identity, Logit, Standardise, YeoJohnson


# --------------------------------------------------------------- registry ---
def test_registry_lists_the_phase_one_components():
    assert {'clopper_pearson', 'dkw'} <= set(registry.available('bound'))
    assert {'fixed', 'optimised'} <= set(registry.available('confidence'))
    assert {'minimax', 'sum'} <= set(registry.available('rule'))
    assert {'data_midpoints', 'grid', 'auto'} <= set(registry.available('search'))
    assert {'identity', 'logit', 'yeo_johnson'} <= set(
        registry.available('transform'))


def test_resolve_accepts_names_pairs_and_objects():
    assert isinstance(registry.resolve('bound', 'dkw'), DKW)
    cp = registry.resolve('bound', ('clopper_pearson', {'union_bound': False}))
    assert isinstance(cp, ClopperPearson) and cp.union_bound is False
    obj = FixedDelta(0.1)
    assert registry.resolve('confidence', obj) is obj
    with pytest.raises(ValueError, match='available'):
        registry.get('bound', 'no_such_bound')
    with pytest.raises(TypeError):
        registry.resolve('rule', 3.14)


def test_registry_refuses_a_second_class_under_one_name():
    with pytest.raises(ValueError, match='already registered'):
        @registry.register('rule', 'minimax')
        class Impostor(Minimax):
            pass


def test_describe_is_json_friendly():
    import json
    for c in (ClopperPearson(), OptimisedDelta(), Minimax(), Grid(), Logit()):
        json.dumps(c.describe())


# ----------------------------------------------------------------- sample ---
def test_wrong_side_counts_follow_the_prediction_rule():
    low = ClassSample([0.0, 1.0, 2.0], 0, 'low')
    high = ClassSample([1.0, 2.0, 3.0], 1, 'high')
    b = np.array([-1.0, 1.0, 1.5, 5.0])
    # low is wrong when z > b
    assert list(low.wrong_side_count(b)) == [3, 1, 1, 0]
    # high is counted wrong when z < b (a point exactly on b is not counted)
    assert list(high.wrong_side_count(b)) == [0, 0, 1, 3]


def test_orientation_by_mean_and_empty_class():
    d = ProjectedData(np.r_[5.0, 6.0, 0.0, 1.0], np.array([0, 0, 1, 1]))
    assert d.class_nums == [1, 0] and d.low.label == 1
    assert d.by_label(0) is d.high
    with pytest.raises(ValueError, match='no data'):
        ProjectedData(np.r_[1.0, 2.0], np.array([0, 0]))


def test_sample_statistics():
    s = ClassSample([1.0, 2.0, 3.0, 10.0], 0, 'high')
    assert s.mean == 4.0
    assert s.radius == 6.0
    assert s.facing_distance(0.0) == 4.0     # high: distance is mean - b
    assert np.isnan(ClassSample([1.0], 0, 'low').std)


# ------------------------------------------------------- count bounds ---
def test_fixed_delta_applies_the_union_correction():
    s = ClassSample(np.arange(20.0), 0, 'low')
    b = np.array([4.5, 15.5])
    cp = ClopperPearson().fit(s)
    got = FixedDelta(0.05).prepare(cp)(b)
    m = s.wrong_side_count(b)
    assert np.array_equal(got['U'], clopper_pearson_upper(m, 20, 0.05 / 21))
    assert np.all(got['delta'] == 0.05)
    no_union = ClopperPearson(union_bound=False).fit(s)
    assert np.all(FixedDelta(0.05).prepare(no_union)(b)['U'] < got['U'])


def test_optimised_delta_forms():
    s = ClassSample(np.arange(30.0), 0, 'low')
    b = np.array([10.5])
    cp = ClopperPearson().fit(s)
    e = OptimisedDelta(resolution=200, form='expected').prepare(cp)(b)
    u = OptimisedDelta(resolution=200, form='union').prepare(cp)(b)
    assert e['L'][0] == pytest.approx((1 - e['delta'][0]) * e['U'][0] + e['delta'][0])
    assert u['L'][0] == pytest.approx(u['U'][0] + u['delta'][0])
    assert e['L'][0] <= u['L'][0]
    with pytest.raises(ValueError):
        OptimisedDelta(form='nope')


def test_fixed_delta_validates():
    with pytest.raises(ValueError):
        FixedDelta(0.0)


class _Smooth(Bound):
    '''a toy continuous bound: a logistic tail in the facing distance'''
    needs_delta = True

    def curve(self, b, delta):
        d = self.sample.facing_distance(b)
        return 1.0 / (1.0 + np.exp(d)) + 0.0 * np.asarray(delta, dtype=float)


class _NoDelta(_Smooth):
    needs_delta = False
    guarantee = 'average_case'

    def curve(self, b, delta):
        assert delta is None
        return 1.0 / (1.0 + np.exp(self.sample.facing_distance(b)))


def test_policies_on_continuous_and_delta_free_bounds():
    s = ClassSample(np.random.default_rng(0).normal(0, 1, 50), 0, 'low')
    b = np.linspace(-1, 3, 7)
    smooth = _Smooth().fit(s)
    opt = OptimisedDelta(resolution=50).prepare(smooth)(b)
    assert opt['L'].shape == b.shape and np.all(np.isfinite(opt['L']))
    nd = _NoDelta().fit(s)
    for policy in (FixedDelta(), OptimisedDelta(resolution=50)):
        out = policy.prepare(nd)(b)
        assert np.all(np.isnan(out['delta']))
        assert np.array_equal(out['L'], out['U'])


# ------------------------------------------------------------------ rules ---
def test_minimax_takes_the_plateau_midpoint_in_value():
    c = np.array([0.0, 1.0, 3.0, 10.0])
    L_low = np.array([0.9, 0.2, 0.2, 0.2])
    L_high = np.array([0.1, 0.2, 0.2, 0.9])
    b, losses = Minimax().choose(c, L_low, L_high)
    assert b == 2.0                       # middle of [1, 3] in value
    assert np.array_equal(losses, np.maximum(L_low, L_high))
    assert Minimax().combine(0.3, 0.4) == 0.4


def test_sum_takes_the_argmin():
    c = np.array([0.0, 1.0, 2.0])
    b, losses = Sum().choose(c, np.array([0.5, 0.2, 0.1]),
                             np.array([0.1, 0.2, 0.6]))
    assert b == 1.0 and Sum().combine(0.1, 0.2) == pytest.approx(0.3)


def test_minimax_refine_finds_the_crossing():
    low = lambda b: {'L': np.asarray(b, dtype=float)}           # rises
    high = lambda b: {'L': 1.0 - np.asarray(b, dtype=float)}    # falls
    assert Minimax().refine(low, high, (1.0, 1.0), (0.0, 1.0)) == pytest.approx(0.5)
    assert Minimax().refine(low, high, (1.0, 3.0), (0.0, 1.0)) == pytest.approx(0.75)
    assert Minimax().refine(low, high, (1.0, 1.0), (0.6, 1.0)) is None


# ----------------------------------------------------------------- search ---
def test_candidates():
    d = ProjectedData(np.r_[0.0, 1.0, 1.0, 4.0], np.array([0, 0, 1, 1]))
    mids = DataMidpoints().generate(d)
    assert np.allclose(mids[1:-1], [0.5, 2.5])
    assert mids[0] < 0.0 and mids[-1] > 4.0
    single = ProjectedData(np.r_[2.0, 2.0], np.array([0, 1]))
    assert list(DataMidpoints().generate(single)) == [1.0, 3.0]
    g = Grid(n=11, margin=0.5).generate(d)
    assert g[0] == -2.0 and g[-1] == 6.0 and len(g) == 11
    counts = {'low': ClopperPearson(), 'high': DKW()}
    assert np.array_equal(Auto().generate(d, counts), mids)
    smooth = {'low': ClopperPearson(), 'high': _Smooth()}
    union = Auto(n=11).generate(d, smooth)
    assert set(mids) <= set(union) and len(union) > len(mids)


# ------------------------------------------------------------- transforms ---
@pytest.mark.parametrize('t', [Identity(), Logit(), Standardise(), YeoJohnson()])
def test_transforms_are_increasing_and_invertible(t):
    rng = np.random.default_rng(1)
    z = rng.uniform(0.001, 0.999, 400) if isinstance(t, Logit) else \
        rng.gamma(2.0, 1.5, 400) - 1.0
    if t.requires_fit:
        assert not t.is_fitted
        t.fit(z)
        assert t.is_fitted
    s = np.sort(z)
    ts = t(s)
    assert np.all(np.diff(ts) > 0)
    assert np.allclose(t.inverse(ts), s, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('lam_data', [
    np.random.default_rng(2).normal(0, 1, 300),                  # lambda ~ 1
    np.random.default_rng(3).lognormal(0, 1, 300),               # lambda < 1
    -np.random.default_rng(4).lognormal(0, 1, 300) + 5.0,        # lambda > 1
])
def test_yeo_johnson_inverse_across_lambdas(lam_data):
    t = YeoJohnson().fit(lam_data)
    grid = np.linspace(lam_data.min(), lam_data.max(), 50)
    assert np.allclose(t.inverse(t(grid)), grid, atol=1e-8)
