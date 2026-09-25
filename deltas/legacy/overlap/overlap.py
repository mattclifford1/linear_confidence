'''
Overlap-native deltas: bias selection from class-conditional tail bounds that
are defined under arbitrary class overlap, with no separability assumption and
no slack variables.

Motivation
----------
The published (ECAI 2024) formulation asks the two inflated class supports to
meet exactly,  R1_hat + R2_hat = D_hat.  When the projected classes overlap no
(delta1, delta2) satisfies that, so the optimiser has nothing to find and the
slack machinery exists purely to delete points until the assumption is true
again.  That answers the question "which points do I throw away so that my
separability assumption holds", when the question we actually care about is
"what is the error at this boundary, given points on both sides".

Here we answer the second question directly.  For a candidate boundary b in the
projected space, count the training points of each class that fall on the wrong
side,

    m_1(b) = #{z in S_1 : z >  b}        (class 1 sits below b)
    m_2(b) = #{z in S_2 : z <= b}

and upper bound the true class-conditional error e_i(b) from m_i(b).  Two
estimators of that bound are provided:

`binomial_deltas`  (Clopper-Pearson)
    m_i(b) ~ Binomial(N_i, e_i(b)), so the exact one-sided Clopper-Pearson
    upper confidence limit gives, with probability at least 1 - delta_i,
        e_i(b) <= BetaInv(1 - delta_i ; m_i + 1, N_i - m_i).
    Pointwise in b, so a union bound is applied over the N_i + 1 distinct
    values m_i can take (cost: log(N_i + 1), not N_i).

`dkw_deltas`  (Dvoretzky-Kiefer-Wolfowitz)
    The projected space is one dimensional, so the empirical CDF is available
    and the one-sided DKW inequality gives, with probability at least
    1 - delta_i, uniformly over every b at once,
        e_i(b) <= m_i(b)/N_i + sqrt(ln(1/delta_i) / (2 N_i)).
    Looser than Clopper-Pearson pointwise but uniform for free.

Both reduce to the published behaviour in the separable limit: at m_i = 0 the
Clopper-Pearson bound is 1 - delta_i**(1/N_i) ~= ln(1/delta_i)/N_i, the same
order as the 1/(N_i + 1) of the original paper, so the separable result is the
m = 0 corner of this one.

Loss
----
Same bookkeeping as Eq. 6 of the published paper: with probability 1 - delta_i
the bound holds and the error is at most UB_i, otherwise the only thing we can
say is that it is at most 1.

    L(b) = sum_i c_i * [ (1 - delta_i) * UB_i(m_i(b), N_i, delta_i) + delta_i ]

Crucially the two classes are *not* coupled by any constraint - b partitions the
line automatically - so for a fixed b each delta_i can be minimised
independently, and the problem is never infeasible.  There is no `is_fit =
False` outcome.

Neither estimator uses the class radius R or the concentration inequality on
the empirical mean: ordering the raw projections needs no mean estimate.

Validity caveat
---------------
The binomial model above needs the projected training points to be i.i.d.
draws from the projected class-conditional law. They are NOT when the
classifier that defines the projection was fitted to those same points - it has
already pushed them towards the correct side, so m_i under-counts and the
certificate comes out optimistic. Measured over 4 datasets x 10 seeds, coverage
on training data is 0.84 against a nominal 0.96, and the shortfall tracks the
classifier's train/test optimism exactly (Breast Cancer optimism .013 ->
coverage 1.00; Hepatitis .208 -> 0.65).

Fit on a held-out calibration split that the classifier never saw and coverage
returns to 1.00, at the cost of a looser bound. So: **treat the reported bound
as a guarantee only when it is computed on held-out data**; on training data it
is an optimistic estimate. See experiments/validate_bounds.py, and note that
the published separable method computes its empirical support the same way and
inherits the same problem.
'''
import numpy as np
from scipy.stats import beta

import deltas.plotting.plots as plots
import deltas.utils.projection as projection


# ----------------------------------------------------------------- bounds ---
def clopper_pearson_upper(m, N, delta):
    '''
    exact one-sided upper confidence limit on a binomial proportion
        P(e <= BetaInv(1 - delta; m + 1, N - m)) >= 1 - delta
    m may be an array; delta a scalar or matching array
    '''
    m = np.asarray(m, dtype=float)
    N = float(N)
    delta = np.asarray(delta, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ub = beta.ppf(1.0 - delta, m + 1.0, N - m)
    # m == N -> no information, error could be 1; also guard nan from ppf
    ub = np.where(m >= N, 1.0, ub)
    ub = np.where(np.isnan(ub), 1.0, ub)
    return np.clip(ub, 0.0, 1.0)


def dkw_upper(m, N, delta, two_sided=False):
    '''
    one-sided DKW upper bound, uniform over all thresholds
        e <= m/N + sqrt(ln(1/delta) / (2N))
    two_sided=True uses ln(2/delta) (only needed if both tails are used)
    '''
    m = np.asarray(m, dtype=float)
    delta = np.asarray(delta, dtype=float)
    inside = np.log((2.0 if two_sided else 1.0) / delta) / (2.0 * N)
    return np.clip(m / N + np.sqrt(inside), 0.0, 1.0)


# ------------------------------------------------------------------ model ---
class base_overlap_deltas:
    '''
    shared machinery: sweep the boundary over every distinct split of the
    projected training data and pick the minimiser of the certified loss
    '''
    #: set by subclasses
    bound_name = None

    def __init__(self, clf=None, dim_reducer=None, objective='sum',
                 union_bound=True, delta_resolution=2000, dev=False):
        '''
            objective:        'sum'     minimise c_1 L_1 + c_2 L_2, i.e. an
                                        upper bound on the (cost weighted)
                                        balanced error
                              'minimax' minimise max_i c_i L_i, the analogue of
                                        the published method's "the two
                                        certified regions meet" constraint.
                                        Ties (common when one class's bound is
                                        flat) are broken towards the larger
                                        margin from the uncertain class.
            union_bound:      correct delta_i for the number of candidate
                              thresholds so the guarantee holds simultaneously
                              for the b that gets selected (Clopper-Pearson
                              only - DKW is already uniform)
            delta_resolution: grid size for the per-class delta_i search
        '''
        if clf is not None and not hasattr(clf, 'get_projection'):
            raise AttributeError(
                f"Classifier {clf} needs 'get_projection' method")
        if objective not in ('sum', 'minimax'):
            raise ValueError("objective must be 'sum' or 'minimax'")
        self.clf = clf
        self.dim_reducer = dim_reducer
        self.objective = objective
        self.union_bound = union_bound
        self.delta_resolution = delta_resolution
        self.dev = dev
        self.is_fit = False

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # -- the bound, supplied by subclasses ---------------------------------
    def _upper_bound(self, m, N, delta):
        raise NotImplementedError

    def _delta_correction(self, N):
        '''divisor applied to delta_i to pay for the choice of threshold'''
        return 1.0

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y, costs=(1, 1), clf=None, _plot=False, _print=False,
            **kwargs):
        if clf is not None:
            if not hasattr(clf, 'get_projection'):
                raise AttributeError(
                    f"Classifier {clf} needs 'get_projection' method")
            self.clf = clf

        z = self._project(X)
        z1 = z[y == 0]
        z2 = z[y == 1]
        if len(z1) == 0 or len(z2) == 0:
            raise ValueError('One class has no data points')

        self.N1, self.N2 = len(z1), len(z2)
        self.costs = costs

        # orientation: class with the smaller mean sits below the boundary
        if np.mean(z1) <= np.mean(z2):
            self.class_nums = [0, 1]
            low, high = z1, z2
        else:
            self.class_nums = [1, 0]
            low, high = z2, z1
        self._low_sorted = np.sort(low)      # predicted as class_nums[0]
        self._high_sorted = np.sort(high)    # predicted as class_nums[1]
        N_low, N_high = len(low), len(high)

        # candidate boundaries: midpoints between consecutive distinct
        # projected values, plus one outside each end. m_i(b) is piecewise
        # constant so this set contains every distinct (m_low, m_high) pair.
        allz = np.unique(np.concatenate([self._low_sorted, self._high_sorted]))
        if len(allz) == 1:
            span = 1.0
            candidates = np.array([allz[0] - span, allz[0] + span])
        else:
            mids = (allz[:-1] + allz[1:]) / 2.0
            pad = (allz[-1] - allz[0]) * 0.01 + 1e-12
            candidates = np.concatenate(
                [[allz[0] - pad], mids, [allz[-1] + pad]])

        # m as a function of b, vectorised
        m_low = N_low - np.searchsorted(self._low_sorted, candidates, 'right')
        m_high = np.searchsorted(self._high_sorted, candidates, 'left')

        # per-class loss lookup tables over the possible m values
        loss_low = self._per_class_loss_table(N_low)
        loss_high = self._per_class_loss_table(N_high)

        c_low = costs[0] if self.class_nums[0] == 0 else costs[1]
        c_high = costs[0] if self.class_nums[1] == 0 else costs[1]
        L_low = c_low * loss_low['loss'][m_low]
        L_high = c_high * loss_high['loss'][m_high]

        if self.objective == 'sum':
            losses = L_low + L_high
            boundary = float(candidates[int(np.argmin(losses))])
        else:
            # minimax. The bound of the scarce class is typically flat over a
            # long run of candidates, so the arg-min is a plateau rather than a
            # point. Every boundary on the plateau carries the same
            # certificate, so spend the remaining freedom on the one furthest
            # from the training points at either end - the midpoint of the
            # plateau *in value*. (Taking the middle index instead makes the
            # answer depend on how the points happen to be spaced, and breaks
            # mirror symmetry.)
            losses = np.maximum(L_low, L_high)
            plateau = np.flatnonzero(losses <= losses.min() + 1e-12)
            boundary = 0.5 * float(candidates[plateau[0]] +
                                   candidates[plateau[-1]])

        self._L_low, self._L_high = L_low, L_high
        self.boundary = boundary
        self.candidates = candidates
        self.losses = losses
        # recount at the boundary actually chosen, so the reported certificate
        # always describes the boundary that predict() will use
        self.m_low = int(N_low - np.searchsorted(
            self._low_sorted, boundary, 'right'))
        self.m_high = int(np.searchsorted(
            self._high_sorted, boundary, 'left'))
        ll = c_low * loss_low['loss'][self.m_low]
        lh = c_high * loss_high['loss'][self.m_high]
        self.loss = float(ll + lh if self.objective == 'sum' else max(ll, lh))
        self.delta_low = float(loss_low['delta'][self.m_low])
        self.delta_high = float(loss_high['delta'][self.m_high])
        self.bound_low = float(loss_low['bound'][self.m_low])
        self.bound_high = float(loss_high['bound'][self.m_high])
        # expose in class-label order (class 0 first) for reporting
        if self.class_nums[0] == 0:
            self.delta1, self.delta2 = self.delta_low, self.delta_high
            self.error_bound_1 = self.bound_low
            self.error_bound_2 = self.bound_high
        else:
            self.delta1, self.delta2 = self.delta_high, self.delta_low
            self.error_bound_1 = self.bound_high
            self.error_bound_2 = self.bound_low

        # always solvable - kept for API compatibility with the other models
        self.solution_possible = True
        self.solution_found = True
        self.is_fit = True

        if _print == True:
            self.print_deltas()
        if _plot == True:
            self.plot_loss()
        return self

    #: above this many (m, delta) pairs, build the table by searching rather
    #: than by evaluating the whole grid - see _per_class_loss_table. Only
    #: worth it when the bound itself is expensive, so subclasses with a cheap
    #: closed form set this to infinity and always take the dense path.
    dense_table_limit = np.inf

    def _per_class_loss_table(self, N):
        '''
        for every possible m in 0..N, minimise
            (1 - delta) * UB(m, N, delta) + delta
        over delta in (0, 1). Returns the minimised loss, the argmin delta and
        the bound value at that delta.
        '''
        deltas = np.linspace(1.0 / self.delta_resolution,
                             1.0 - 1.0 / self.delta_resolution,
                             self.delta_resolution)
        eff = deltas / self._delta_correction(N)      # union bound correction
        ms = np.arange(N + 1)

        if (N + 1) * self.delta_resolution > self.dense_table_limit:
            return self._sparse_loss_table(ms, deltas, eff, N)

        # (N+1, resolution) grids
        bounds = self._upper_bound(ms[:, None], N, eff[None, :])
        # the delta paid in the loss is the *nominal* per-class confidence
        L = (1.0 - deltas[None, :]) * bounds + deltas[None, :]
        idx = np.argmin(L, axis=1)
        return {'loss': L[ms, idx],
                'delta': deltas[idx],
                'bound': bounds[ms, idx]}

    def _sparse_loss_table(self, ms, deltas, eff, N):
        '''
        same table, found by ternary search instead of evaluating every cell

        The dense path costs (N+1) x delta_resolution evaluations of the bound,
        which is 10^8 Beta quantiles at N = 50k and takes minutes. The loss is
        unimodal in delta at fixed m - it tends to 1 at both ends, since a
        vanishing delta buys a vacuous bound and a delta near 1 pays for itself
        - so a ternary search over grid *indices* finds the same grid minimum
        in ~40 evaluations per m rather than 2000.

        This returns the same losses as the dense path (checked in
        tests/test_overlap.py); only the delta reported at an exact tie may
        differ, and tied deltas give the same loss by definition.

        Only safe where the loss is *strictly* unimodal. It is not used for
        DKW, whose bound clips at 1 for small delta and so carries a flat
        plateau there that a ternary search can step across - and DKW needs no
        speedup anyway, having no Beta quantile to evaluate.
        '''
        def loss_at(idx):
            idx = np.clip(idx, 0, len(deltas) - 1)
            b = self._upper_bound(ms, N, eff[idx])
            return (1.0 - deltas[idx]) * b + deltas[idx], b

        lo = np.zeros(len(ms), dtype=int)
        hi = np.full(len(ms), len(deltas) - 1, dtype=int)
        while np.any(hi - lo > 2):
            third = np.maximum((hi - lo) // 3, 1)
            m1, m2 = lo + third, hi - third
            L1, _ = loss_at(m1)
            L2, _ = loss_at(m2)
            take_left = L1 <= L2
            hi = np.where(take_left, np.maximum(m2 - 1, lo), hi)
            lo = np.where(take_left, lo, np.minimum(m1 + 1, hi))

        # settle the last few indices exactly, keeping the lowest index on ties
        # so the tie-breaking matches np.argmin on the dense grid
        best_L = np.full(len(ms), np.inf)
        best_d = np.zeros(len(ms))
        best_b = np.zeros(len(ms))
        for offset in range(0, 3):
            idx = np.clip(lo + offset, 0, hi)
            L, b = loss_at(idx)
            better = L < best_L
            best_L = np.where(better, L, best_L)
            best_d = np.where(better, deltas[idx], best_d)
            best_b = np.where(better, b, best_b)
        return {'loss': best_L, 'delta': best_d, 'bound': best_b}

    # -- prediction ---------------------------------------------------------
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
        if self.is_fit == False:
            raise AttributeError("Call .fit(X, y) first")
        z = self._project(X)
        preds = np.where(z <= self.boundary,
                         self.class_nums[0], self.class_nums[1])
        return preds.squeeze()

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_projection(self, X):
        return self.clf.get_projection(X)

    def get_bias(self):
        if self.is_fit == True:
            return -self.boundary
        if hasattr(self.clf, 'get_bias'):
            return self.clf.get_bias()
        return None

    def score(self, *args, **kwargs):
        return self.loss

    # -- reporting ----------------------------------------------------------
    def certified_error(self):
        '''the class-conditional error bounds certified at the chosen bias'''
        return {'class 1': self.error_bound_1,
                'class 2': self.error_bound_2,
                'delta1': self.delta1,
                'delta2': self.delta2}

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
        ax.axvline(self.boundary, color='k', linestyle='dashed',
                   label='chosen bias')
        ax.set_xlabel('boundary')
        ax.set_ylabel('certified loss')
        ax.legend()
        plots.plt.show()


class binomial_deltas(base_overlap_deltas):
    '''Clopper-Pearson (exact binomial) upper bound on the class error'''
    bound_name = 'Clopper-Pearson'
    #: each cell is a Beta quantile, so the dense (N+1) x resolution table is
    #: 10^8 evaluations at N = 50k. Search instead above this size.
    dense_table_limit = 5_000_000

    def _upper_bound(self, m, N, delta):
        return clopper_pearson_upper(m, N, delta)

    def _delta_correction(self, N):
        # m takes N+1 distinct values as the threshold sweeps, so a union over
        # those hypotheses makes the guarantee hold at the selected boundary
        return (N + 1.0) if self.union_bound else 1.0


class dkw_deltas(base_overlap_deltas):
    '''DKW empirical-CDF bound - already uniform over every threshold'''
    bound_name = 'DKW'

    def _upper_bound(self, m, N, delta):
        return dkw_upper(m, N, delta)

    def _delta_correction(self, N):
        return 1.0      # uniform in b by construction, nothing to pay
