'''
Name -> component lookup: names, (name, kwargs) pairs, objects, and the refusal of a second class under one name.
'''
import pytest

from deltas.bounds import DKW, ClopperPearson
from deltas.confidence import FixedDelta
from deltas.core import registry
from deltas.rules import Minimax


def test_registry_lists_the_components():
    assert {'clopper_pearson', 'dkw', 'gaussian', 'location_scale_mc',
            'predictive_t', 'saw_yang_mo', 'cantelli', 'vysochanskij_petunin',
            'published_fence', 'kth_point_fence'} <= set(registry.available('bound'))
    assert {'risk', 'neyman_pearson'} <= set(registry.available('rule'))
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
