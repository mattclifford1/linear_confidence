'''
Name -> component lookup, so that methods can be specified by string
(`DeltasEstimator(bound='clopper_pearson', rule='minimax')`) and recorded in
results files.

Components register themselves with the `register` decorator when their module
is imported; `resolve` imports the component packages on first use, so there
is no import-order trap.
'''
import importlib

KINDS = ('bound', 'confidence', 'rule', 'search', 'transform')

#: kind -> the package whose import registers every component of that kind
_PACKAGES = {'bound': 'deltas.bounds',
             'confidence': 'deltas.confidence',
             'rule': 'deltas.rules',
             'search': 'deltas.search',
             'transform': 'deltas.transforms'}

_REGISTRY = {kind: {} for kind in KINDS}


def register(kind, name):
    '''class decorator: make a component available under `name`'''
    if kind not in KINDS:
        raise ValueError(f'unknown component kind {kind!r}; one of {KINDS}')

    def deco(cls):
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f'{kind} {name!r} is already registered to '
                             f'{existing.__name__}')
        _REGISTRY[kind][name] = cls
        cls.name = name
        return cls
    return deco


def available(kind):
    '''the registered names of one kind of component'''
    importlib.import_module(_PACKAGES[kind])
    return sorted(_REGISTRY[kind])


def get(kind, name, **kwargs):
    '''construct a registered component by name'''
    importlib.import_module(_PACKAGES[kind])
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise ValueError(f'no {kind} called {name!r}; '
                         f'available: {available(kind)}') from None
    return cls(**kwargs)


def resolve(kind, spec):
    '''
    turn a component specification into a component:
        a string         -> the registered component, default parameters
        (name, kwargs)   -> the registered component with those parameters
        a component      -> itself
    '''
    from deltas.core.components import Component
    if isinstance(spec, Component):
        return spec
    if isinstance(spec, str):
        return get(kind, spec)
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return get(kind, spec[0], **dict(spec[1]))
    raise TypeError(f'cannot make a {kind} from {spec!r}')
