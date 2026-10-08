'''
Named deltas methods - the single place the experiment runners get them from.

    from deltas.methods import METHODS, get, available
    fitted = METHODS['CP Minimax'](clf, X, y)
    available('envelope')

Families: published (ECAI 2024), non_separable (Dec 2024 draft), overlap
(Clopper-Pearson / DKW), envelope (the shape-aware methods), ablation (the
published fences as modular compositions).
'''
from deltas.methods import ablations, envelope, legacy, overlap
from deltas.methods.base import Method

METHODS = {}
for _module in (legacy, overlap, envelope, ablations):
    for _m in _module.METHODS:
        if _m.name in METHODS:
            raise ValueError(f'duplicate method name {_m.name!r}')
        METHODS[_m.name] = _m


def get(name):
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(f'no method {name!r}; available: {sorted(METHODS)}') \
            from None


def available(family=None):
    return [n for n, m in METHODS.items() if family is None or m.family == family]


def make(name, clf=None):
    return get(name).make(clf)


def describe_all(names=None):
    return {n: get(n).describe() for n in (names or METHODS)}


__all__ = ['METHODS', 'Method', 'available', 'describe_all', 'get', 'make']
