'''
The one thing every slot component shares: a registry name and a JSON-friendly
description of its parameters. The slot base classes themselves live next to
their implementations, in the base.py of each slot package:

    deltas/bounds/base.py       Bound, CountBound
    deltas/confidence/base.py   DeltaPolicy, PreparedCurve, NoDeltaCurve
    deltas/rules/base.py        DecisionRule
    deltas/search/base.py       CandidateSet
    deltas/transforms/base.py   Transform

Components are small, stateless-until-fitted objects; the estimator deep-copies
them before fitting, so the objects passed as parameters are never mutated
(which keeps sklearn's clone/get_params honest).
'''
import inspect


class Component:
    '''shared behaviour: a name, and a description of the parameters'''
    #: registry name (set by the registry decorator)
    name = None

    def params(self):
        '''the constructor parameters and their current values'''
        sig = inspect.signature(type(self).__init__)
        return {p: getattr(self, p) for p in sig.parameters
                if p != 'self' and hasattr(self, p)}

    def describe(self):
        '''a JSON-friendly description, for stamping into results'''
        out = {'component': type(self).__name__, 'name': self.name}
        for k, v in self.params().items():
            if isinstance(v, (int, float, str, bool, type(None))):
                out[k] = v
            elif isinstance(v, (tuple, list)):
                out[k] = list(v)
            else:
                out[k] = repr(v)
        return out

    def __repr__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.params().items())
        return f'{type(self).__name__}({args})'
