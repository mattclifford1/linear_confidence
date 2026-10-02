'''
The contract every DeltasEstimator composition must honour: sklearn shape,
predictions consistent with the boundary, symmetry, per-class bounds,
certificates at a fixed delta, transforms, and a description for results files.
'''
import json

import numpy as np
import pytest
from sklearn.base import clone

from deltas.bounds import DKW, ClopperPearson
from deltas.classifiers.frozen import FrozenProjection
from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator
from deltas.transforms import Logit, Standardise


def _toy(seed=0, n0=200, n1=25, m1=2.5):
    rng = np.random.default_rng(seed)
    z = np.r_[rng.normal(0, 1, n0), rng.normal(m1, 1, n1)]
    return z[:, None], np.r_[np.zeros(n0), np.ones(n1)].astype(int)


COMPOSITIONS = {
    'cp_minimax': dict(bound='clopper_pearson', rule='minimax'),
    'dkw_sum': dict(bound='dkw', rule='sum'),
    'cp_fixed': dict(bound='clopper_pearson', confidence=('fixed', {'delta': 0.1})),
    'per_class': dict(bound={0: 'dkw', 1: 'clopper_pearson'}),
    'gaussian_fixed': dict(bound='gaussian', confidence=('fixed', {'delta': 0.05})),
    'gaussian_mc': dict(bound=('gaussian', {'band': 'mc', 'draws': 10000}),
                        confidence=('fixed', {'delta': 0.05})),
    'predictive_certified': dict(bound='predictive_t',
                                 certify=['gaussian', 'clopper_pearson']),
    'saw_yang_mo_sum': dict(bound='saw_yang_mo', rule='sum'),
    'neyman_pearson': dict(bound='gaussian', confidence=('fixed', {'delta': 0.05}),
                           rule=('neyman_pearson', {'alpha': 0.2})),
    'mixed_counts_minority_gaussian': dict(
        bound={0: 'clopper_pearson', 1: 'gaussian'},
        confidence=('fixed', {'delta': 0.05})),
}


@pytest.mark.parametrize('name', sorted(COMPOSITIONS))
def test_fit_predict_and_bias(name):
    X, y = _toy()
    m = DeltasEstimator(**COMPOSITIONS[name]).fit(X, y)
    assert m.is_fit is True
    assert np.isfinite(m.boundary)
    assert m.get_bias() == -m.boundary
    pred = m.predict(X)
    assert np.array_equal(pred, np.where(X[:, 0] <= m.boundary,
                                         m.class_nums[0], m.class_nums[1]))
    ce = m.certified_error()
    assert set(ce) == {'class 1', 'class 2', 'delta1', 'delta2'}
    assert 0 <= ce['class 1'] <= 1 and 0 <= ce['class 2'] <= 1


@pytest.mark.parametrize('name', sorted(COMPOSITIONS))
def test_mirror_symmetry(name):
    X, y = _toy(seed=3)
    a = DeltasEstimator(**COMPOSITIONS[name]).fit(X, y)
    b = DeltasEstimator(**COMPOSITIONS[name]).fit(-X, y)
    assert a.boundary == pytest.approx(-b.boundary, abs=1e-9)


def test_sklearn_clone_and_params():
    m = DeltasEstimator(bound='dkw', rule='sum', delta_report=0.1)
    c = clone(m)
    assert c.get_params()['bound'] == 'dkw' and c.get_params()['rule'] == 'sum'
    c.set_params(rule='minimax')
    assert c.rule == 'minimax' and m.rule == 'sum'
    # the component objects passed in are never mutated by fitting
    bound = ClopperPearson()
    DeltasEstimator(bound=bound).fit(*_toy())
    assert not hasattr(bound, 'sample')


def test_projects_through_the_classifier():
    X, y = _toy()
    clf = FrozenProjection(1.0)
    a = DeltasEstimator(clf).fit(X, y)
    assert np.array_equal(a.get_projection(X), X)
    with pytest.raises(AttributeError, match='get_projection'):
        DeltasEstimator().fit(X, y, clf=object())
    with pytest.raises(AttributeError, match='fit'):
        DeltasEstimator().predict(X)


def test_certificates_at_a_fixed_delta():
    X, y = _toy()
    m = DeltasEstimator(certify=['clopper_pearson', 'dkw'],
                        delta_report=0.05).fit(X, y)
    assert list(m.certificates_) == ['clopper_pearson', 'dkw']
    for cert in m.certificates_.values():
        assert cert['delta1'] == cert['delta2'] == 0.05
        assert cert['guarantee'] == 'high_probability'
    # the primary certificate is the first one listed
    assert m.certified_error()['class 1'] == m.certificates_['clopper_pearson']['class 1']
    # and matches a direct evaluation of the bound at the boundary
    ref = DeltasEstimator(confidence=FixedDelta(0.05)).fit(X, y)
    direct = DeltasEstimator(confidence=FixedDelta(0.05),
                             certify=['clopper_pearson']).fit(X, y)
    assert direct.certified_error() == ref.certified_error()


def test_decision_certificate_reports_the_deciding_curves():
    X, y = _toy()
    m = DeltasEstimator().fit(X, y)
    assert list(m.certificates_) == ['decision']
    assert m.certified_error()['class 1'] == m.error_bound_1


def test_count_bounds_do_not_care_about_the_transform():
    '''
    counts are rank-based, so any increasing transform gives the same split of
    the training data (the boundary moves in raw units, the predictions on the
    training points do not)
    '''
    rng = np.random.default_rng(5)
    p = np.r_[rng.beta(1.3, 9, 300), rng.beta(4, 2.5, 30)]
    y = np.r_[np.zeros(300), np.ones(30)].astype(int)
    raw = DeltasEstimator().fit(p[:, None], y)
    lg = DeltasEstimator(transform=Logit()).fit(p[:, None], y)
    assert np.array_equal(raw.predict(p[:, None]), lg.predict(p[:, None]))
    assert raw.loss == lg.loss
    # boundary is reported in the classifier's own (probability) units
    assert 0.0 < lg.boundary < 1.0


def test_fitted_transform_is_required():
    X, y = _toy()
    with pytest.raises(ValueError, match='fit split'):
        DeltasEstimator(transform=Standardise()).fit(X, y)
    t = Standardise().fit(X[:, 0])
    m = DeltasEstimator(transform=t).fit(X, y)
    assert m.is_fit


def test_per_class_bounds_are_used_per_class():
    X, y = _toy()
    m = DeltasEstimator(bound={0: DKW(), 1: ClopperPearson()}).fit(X, y)
    assert isinstance(m.bounds_['low'], DKW)          # label 0 sits low here
    assert isinstance(m.bounds_['high'], ClopperPearson)
    with pytest.raises(KeyError):
        DeltasEstimator(bound={0: 'dkw'}).fit(X, y)


def test_describe_is_serialisable():
    d = DeltasEstimator(bound={0: 'dkw', 1: 'clopper_pearson'},
                        transform=Logit()).describe()
    json.dumps(d)
    assert d['rule']['name'] == 'minimax'
