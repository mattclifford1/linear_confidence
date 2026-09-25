'''
DeltasEstimator: the one sklearn-shaped class every modular deltas method is
an instance of.

    model = DeltasEstimator(clf,
                            bound='clopper_pearson',      # or a component
                            confidence='optimised',
                            rule='minimax',
                            search='auto',
                            transform=None,
                            certify='decision',
                            delta_report=0.05)
    model.fit(X, y).predict(X_test)
    model.certified_error()

Each slot takes a registered name, a (name, kwargs) pair, or a component
object (see deltas/core/components.py and deltas/core/registry.py). The default
composition is Clopper-Pearson minimax, bit-for-bit the same as
deltas.model.overlap.binomial_deltas(objective='minimax').

Per-class bounds: `bound` may also be a dict {label: spec}, e.g. counts for a
large majority and a location-scale envelope for a scarce minority.

Certificates: certify='decision' reports the (U, delta) of the deciding
curves at the chosen boundary, as the overlap methods always have. Pass a
list of bounds instead to report each of them at the fixed `delta_report` -
the statement that is actually quotable when the decision used an optimised
delta or an average-case curve (working notes, "Deciding and certifying").
'''
import copy

import numpy as np
from sklearn.base import BaseEstimator

import deltas.plotting.plots as plots
from deltas.core import registry
from deltas.core.components import Transform
from deltas.core.sample import ProjectedData


class DeltasEstimator(BaseEstimator):
    #: printed by print_deltas; subclasses may set a fixed one
    bound_name = None

    def __init__(self, clf=None, bound='clopper_pearson',
                 confidence='optimised', rule='minimax', search='auto',
                 transform=None, certify='decision', delta_report=0.05,
                 refine=True, dim_reducer=None, dev=False):
        self.clf = clf
        self.bound = bound
        self.confidence = confidence
        self.rule = rule
        self.search = search
        self.transform = transform
        self.certify = certify
        self.delta_report = delta_report
        self.refine = refine
        self.dim_reducer = dim_reducer
        self.dev = dev

    # --------------------------------------------------------- components ---
    def _components(self):
        '''
        resolve the slots into component objects. Subclasses that fix a
        composition (the overlap shims) override this.
        '''
        return {'bound': self.bound,
                'confidence': registry.resolve('confidence', self.confidence),
                'rule': registry.resolve('rule', self.rule),
                'search': registry.resolve('search', self.search),
                'transform': self._resolve_transform(self.transform),
                'certify': self.certify,
                'refine': self.refine}

    @staticmethod
    def _resolve_transform(spec):
        if spec is None or spec == 'identity':
            return None
        t = registry.resolve('transform', spec)
        if t.requires_fit and not t.is_fitted:
            raise ValueError(
                f'{type(t).__name__} learns its parameters from data and must '
                'be fitted before the deltas fit, on data the certificate '
                'never sees (the classifier\'s fit split) - otherwise the '
                'certificate is no longer valid. Fit it and pass the fitted '
                'object.')
        return t

    @staticmethod
    def _bound_for(spec, label):
        '''a fresh copy of the bound for one class (spec may be per-label)'''
        if isinstance(spec, dict):
            if label not in spec:
                raise KeyError(f'no bound given for class label {label}')
            spec = spec[label]
        return copy.deepcopy(registry.resolve('bound', spec))

    # ---------------------------------------------------------------- fit ---
    def fit(self, X, y, costs=(1, 1), clf=None, _plot=False, _print=False,
            **kwargs):
        if clf is not None:
            if not hasattr(clf, 'get_projection'):
                raise AttributeError(
                    f"Classifier {clf} needs 'get_projection' method")
            self.clf = clf
        comp = self._components()
        self._transform = comp['transform']

        z = self._project(X)
        y = np.asarray(y)
        t = z if self._transform is None else self._transform(z)
        data = ProjectedData(t, y)
        self.data_ = data
        self.N1, self.N2 = data.N[0], data.N[1]
        self.costs = costs
        self.class_nums = data.class_nums
        self._low_sorted, self._high_sorted = data.low.z, data.high.z

        # fit one copy of the bound per class, and resolve its delta handling
        bounds = {side: self._bound_for(comp['bound'], s.label).fit(s)
                  for side, s in data.sides()}
        self.bounds_ = bounds
        if self.bound_name is None:
            self.bound_name = ' / '.join(sorted({type(b).__name__
                                                 for b in bounds.values()}))
        curves = {side: comp['confidence'].prepare(b)
                  for side, b in bounds.items()}

        # a private copy of the rule, shown the data; it sets the per-class
        # weights (costs, given in label order, and for some rules priors)
        rule = copy.deepcopy(comp['rule']).bind(data)
        c_low, c_high = rule.class_weights(data, costs)

        candidates = comp['search'].generate(data, bounds)
        low, high = curves['low'](candidates), curves['high'](candidates)
        L_low = c_low * low['L']
        L_high = c_high * high['L']
        boundary, losses = rule.choose(candidates, L_low, L_high)

        # smooth curves: polish the grid choice between its neighbours
        if comp['refine'] and all(b.smooth for b in bounds.values()):
            boundary = self._refine(rule, curves, (c_low, c_high),
                                    candidates, boundary)

        self._L_low, self._L_high = L_low, L_high
        self.boundary_t_ = boundary          # in transformed units
        self.boundary = boundary if self._transform is None else \
            float(self._transform.inverse(np.array([boundary]))[0])
        self.candidates = candidates
        self.losses = losses

        # recount at the boundary actually chosen, so the reported numbers
        # always describe the boundary that predict() will use
        self.m_low = int(data.low.wrong_side_count(boundary))
        self.m_high = int(data.high.wrong_side_count(boundary))
        at_low = curves['low'](np.array([boundary]))
        at_high = curves['high'](np.array([boundary]))
        ll = c_low * at_low['L'][0]
        lh = c_high * at_high['L'][0]
        self.loss = float(rule.combine(ll, lh))
        self.delta_low = float(at_low['delta'][0])
        self.delta_high = float(at_high['delta'][0])
        self.bound_low = float(at_low['U'][0])
        self.bound_high = float(at_high['U'][0])
        # expose in class-label order (class 0 first) for reporting
        if self.class_nums[0] == 0:
            self.delta1, self.delta2 = self.delta_low, self.delta_high
            self.error_bound_1 = self.bound_low
            self.error_bound_2 = self.bound_high
        else:
            self.delta1, self.delta2 = self.delta_high, self.delta_low
            self.error_bound_1 = self.bound_high
            self.error_bound_2 = self.bound_low

        self.rule_info_ = dict(getattr(rule, 'info_', {}))
        self.certificates_ = self._certify(comp['certify'], comp['bound'],
                                           data, boundary)

        # a curve that is infinite everywhere (e.g. the published fence on
        # data it cannot separate) means no boundary was admissible
        self.solution_possible = bool(np.isfinite(self.loss))
        self.solution_found = self.solution_possible
        self.is_fit = self.solution_possible

        if _print == True:
            self.print_deltas()
        if _plot == True:
            self.plot_loss()
        return self

    def _refine(self, rule, curves, weights, candidates, boundary):
        # the neighbouring candidates on either side; symmetric under
        # mirroring the data, so the refined answer is too
        i = int(np.searchsorted(candidates, boundary))
        on_candidate = i < len(candidates) and candidates[i] == boundary
        lo = candidates[max(i - 1, 0)]
        hi = candidates[min(i + 1 if on_candidate else i, len(candidates) - 1)]
        if not hi > lo:
            return boundary
        new = rule.refine(curves['low'], curves['high'], weights, (lo, hi))
        if new is None or not np.isfinite(new):
            return boundary

        def objective(b):
            b = np.array([b])
            return rule.combine(weights[0] * curves['low'](b)['L'][0],
                                weights[1] * curves['high'](b)['L'][0])
        return float(new) if objective(new) <= objective(boundary) else boundary

    # ------------------------------------------------------- certificates ---
    def _certify(self, certify, bound_spec, data, boundary):
        '''{name: {'guarantee', 'class 1', 'class 2', 'delta1', 'delta2'}}'''
        def by_label(values):
            '''low/high -> class-label order'''
            return (values['low'], values['high']) if self.class_nums[0] == 0 \
                else (values['high'], values['low'])

        if certify == 'decision':
            guarantees = {self.bounds_[s].guarantee for s in ('low', 'high')}
            return {'decision': {
                'guarantee': guarantees.pop() if len(guarantees) == 1
                else 'mixed',
                'class 1': self.error_bound_1, 'class 2': self.error_bound_2,
                'delta1': self.delta1, 'delta2': self.delta2}}

        specs = [certify] if not isinstance(certify, (list, tuple)) else certify
        policy = registry.get('confidence', 'fixed', delta=self.delta_report)
        out = {}
        for spec in specs:
            bs = {side: self._bound_for(spec, s.label).fit(s)
                  for side, s in data.sides()}
            at = {side: policy.prepare(b)(np.array([boundary]))
                  for side, b in bs.items()}
            U1, U2 = by_label({k: float(v['U'][0]) for k, v in at.items()})
            d1, d2 = by_label({k: float(v['delta'][0]) for k, v in at.items()})
            name = bs['low'].name or type(bs['low']).__name__
            while name in out:
                name += "'"
            out[name] = {'guarantee': bs['low'].guarantee,
                         'class 1': U1, 'class 2': U2,
                         'delta1': d1, 'delta2': d2}
        return out

    def certified_error(self):
        '''
        the primary certificate at the chosen boundary, in class-label order
        (class 1 = label 0). All certificates are in `certificates_`.
        '''
        first = next(iter(self.certificates_.values()))
        return {k: first[k] for k in ('class 1', 'class 2', 'delta1', 'delta2')}

    # --------------------------------------------------------- prediction ---
    def _project(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != 1:
            if self.clf is None:
                raise AttributeError(
                    'Deltas classifier needs original classifier to project '
                    'feature space onto 1D classification space')
            X = self.clf.get_projection(X)
        return np.asarray(X).squeeze(axis=-1) if np.asarray(X).ndim > 1 \
            else np.asarray(X)

    def predict(self, X):
        if getattr(self, 'is_fit', False) == False:
            raise AttributeError("Call .fit(X, y) first")
        z = self._project(X)
        if self._transform is None:
            preds = np.where(z <= self.boundary,
                             self.class_nums[0], self.class_nums[1])
        else:
            preds = np.where(self._transform(z) <= self.boundary_t_,
                             self.class_nums[0], self.class_nums[1])
        return preds.squeeze()

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_projection(self, X):
        return self.clf.get_projection(X)

    def get_bias(self):
        '''the new bias, in the classifier's own score units (-boundary)'''
        if getattr(self, 'is_fit', False) == True:
            return -self.boundary
        if hasattr(self.clf, 'get_bias'):
            return self.clf.get_bias()
        return None

    def score(self, *args, **kwargs):
        '''the objective value at the chosen boundary (lower is better)'''
        return self.loss

    # ---------------------------------------------------------- reporting ---
    def describe(self):
        '''the composition, JSON-friendly, for stamping into results'''
        comp = self._components()
        bound = comp['bound']
        if isinstance(bound, dict):
            bounds = {str(k): registry.resolve('bound', v).describe()
                      for k, v in bound.items()}
        else:
            bounds = registry.resolve('bound', bound).describe()
        t = comp['transform']
        return {'estimator': type(self).__name__,
                'bound': bounds,
                'confidence': comp['confidence'].describe(),
                'rule': comp['rule'].describe(),
                'search': comp['search'].describe(),
                'transform': None if t is None else t.describe(),
                'certify': repr(comp['certify']),
                'delta_report': self.delta_report}

    def print_deltas(self):
        print(f'''{self.__class__.__name__} ({self.bound_name})
    boundary : {self.boundary:.5f}
    N        : {self.N1}, {self.N2}
    delta    : {self.delta1:.4g}, {self.delta2:.4g}
    error UB : {self.error_bound_1:.4g}, {self.error_bound_2:.4g}
    loss     : {self.loss:.5f}''')

    def plot_loss(self, ax=None):
        ax, _ = plots._get_axes(ax)
        ax.plot(self.candidates, self.losses)
        ax.axvline(self.boundary_t_, color='k', linestyle='dashed',
                   label='chosen bias')
        ax.set_xlabel('boundary')
        ax.set_ylabel('certified loss')
        ax.legend()
        plots.plt.show()
