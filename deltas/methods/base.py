'''
A named deltas method: how to build it, and how to fit it.

Methods are what the experiment runners select by name. A method is either a
modular composition (a DeltasEstimator, or the overlap shims) or a legacy
estimator run with fixed fit options. Either way:

    method = METHODS['CP Minimax']
    fitted = method(clf, X, y)             # the runners' interface
    model  = method.make(clf)              # unfitted, to inspect or tweak
    method.configured(max_trials=2000)     # a copy with other fit options
    method.describe()                      # JSON-friendly, for results files
'''
import importlib


class Method:
    '''
    name         the name used in results tables
    build        callable(clf) -> an unfitted estimator
    fit_kwargs   extra keyword arguments for .fit (legacy options)
    family       'published', 'non_separable', 'overlap', 'envelope',
                 'ablation'
    reference    where the method is described
    description  one line
    target       'module:Class' of the estimator (for describe(), without
                 constructing it - some legacy estimators need a classifier)
    '''

    def __init__(self, name, build, fit_kwargs=None, family=None,
                 reference=None, description='', target=None):
        self.name = name
        self._build = build
        self.fit_kwargs = dict(fit_kwargs or {})
        self.family = family
        self.reference = reference
        self.description = description
        self.target = target

    def make(self, clf=None):
        return self._build(clf)

    def fit(self, clf, X, y, **fit_kwargs):
        return self.make(clf).fit(X, y, **{**self.fit_kwargs, **fit_kwargs})

    def __call__(self, clf, X, y):
        return self.fit(clf, X, y)

    def configured(self, **fit_kwargs):
        '''a copy with some fit options replaced (the original is unchanged)'''
        return Method(self.name, self._build,
                      {**self.fit_kwargs, **fit_kwargs}, self.family,
                      self.reference, self.description, self.target)

    def with_params(self, **params):
        '''a copy whose estimator gets these constructor parameters'''
        build = self._build

        def _build(clf):
            return build(clf).set_params(**params)
        return Method(self.name, _build, self.fit_kwargs, self.family,
                      self.reference, self.description, self.target)

    def describe(self):
        out = {'name': self.name, 'family': self.family,
               'reference': self.reference, 'fit_kwargs': self.fit_kwargs}
        if self.target is not None:
            out['estimator'] = self.target
        else:
            model = self.make(None)
            out['estimator'] = model.describe() if hasattr(model, 'describe') \
                else type(model).__name__
        return out

    def __repr__(self):
        return f'Method({self.name!r}, family={self.family!r})'


def lazy(module, cls):
    '''
    build(clf) that imports the legacy module only when called: the legacy
    code reads USE_TWO at import time, so importing it early would fix the
    flag before a caller has had the chance to set it
    '''
    def build(clf):
        return getattr(importlib.import_module(module), cls)(clf)
    build.__qualname__ = f'lazy({module}.{cls})'
    return build
