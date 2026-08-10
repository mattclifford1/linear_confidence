'''
Tests for deltas.pipeline.calibration — run with:  pytest tests/

The headline test is `test_calibration_restores_coverage`: it reproduces the
under-coverage in simulation and shows the split fixing it. If that one breaks,
the claim in CALIBRATION.md is wrong.
'''
import numpy as np
import pytest

from deltas.pipeline import calibration
from deltas.pipeline.calibration import (split_calibration, fit_calibrated,
                                         CalibrationSplitError)
from deltas.model import overlap


class _Projector:
    '''
    minimal classifier with the interface deltas needs. Fits a 1-D projection
    by nearest-class-mean on whatever data it is given, so it overfits small
    samples in exactly the way a real classifier does.
    '''

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).squeeze()
        self.c0 = X[y == 0].mean(axis=0)
        self.c1 = X[y == 1].mean(axis=0)
        w = self.c1 - self.c0
        self.w = w / (np.linalg.norm(w) + 1e-12)
        return self

    def get_projection(self, X):
        return (np.asarray(X, dtype=float) @ self.w)[:, None]


def _blobs(n0, n1, d=20, sep=1.0, seed=0):
    '''two Gaussians in d dims; with d large and n small a projector overfits'''
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 1.0, size=(n0, d))
    X1 = rng.normal(0.0, 1.0, size=(n1, d))
    X1[:, 0] += sep
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    return X, y


# ------------------------------------------------------------- the split ---
def test_split_shapes_and_disjointness():
    X, y = _blobs(80, 40, seed=0)
    X_fit, y_fit, X_cal, y_cal = split_calibration(X, y, cal_frac=0.25, seed=0)
    assert len(y_fit) + len(y_cal) == len(y)
    assert len(y_cal) == pytest.approx(0.25 * len(y), abs=1)
    # no point appears in both parts
    fit_rows = {tuple(r) for r in X_fit}
    assert not any(tuple(r) in fit_rows for r in X_cal)


def test_split_is_stratified():
    X, y = _blobs(200, 20, seed=1)
    _, y_fit, _, y_cal = split_calibration(X, y, cal_frac=0.3, seed=1)
    for part in (y_fit, y_cal):
        assert set(np.unique(part)) == {0, 1}
    # minority proportion preserved to within one point
    assert y_cal.mean() == pytest.approx(y.mean(), abs=0.05)


def test_split_is_deterministic_given_seed():
    X, y = _blobs(60, 30, seed=2)
    a = split_calibration(X, y, cal_frac=0.35, seed=7)
    b = split_calibration(X, y, cal_frac=0.35, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])


def test_split_refuses_when_minority_too_scarce():
    '''a certificate from one minority point is worthless - fail, don't fake it'''
    X, y = _blobs(200, 4, seed=3)
    with pytest.raises(CalibrationSplitError):
        split_calibration(X, y, cal_frac=0.35, seed=0)


def test_split_rejects_bad_fraction():
    X, y = _blobs(50, 50, seed=4)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_calibration(X, y, cal_frac=bad, seed=0)


def test_split_needs_two_classes():
    X = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(CalibrationSplitError):
        split_calibration(X, np.zeros(20, dtype=int), cal_frac=0.3)


# --------------------------------------------------------- the whole recipe ---
def test_fit_calibrated_classifier_never_sees_calibration_data():
    X, y = _blobs(100, 60, seed=5)
    model, info = fit_calibrated(
        clf_factory=_Projector,
        deltas_factory=lambda clf: overlap.binomial_deltas(
            clf, objective='minimax'),
        X=X, y=y, cal_frac=0.35, seed=0)
    assert model.is_fit == True
    assert info['n_fit'] + info['n_cal'] == len(y)
    # the counts the certificate is based on come from the calibration part
    assert model.N1 + model.N2 == info['n_cal']


def test_fit_calibrated_bound_is_looser_than_naive():
    '''the price of validity: fewer points -> a wider certificate'''
    X, y = _blobs(300, 150, seed=6)
    naive_clf = _Projector().fit(X, y)
    naive = overlap.binomial_deltas(naive_clf, objective='minimax').fit(X, y)
    cal, _ = fit_calibrated(
        clf_factory=_Projector,
        deltas_factory=lambda clf: overlap.binomial_deltas(
            clf, objective='minimax'),
        X=X, y=y, cal_frac=0.35, seed=0)
    n = naive.certified_error()
    c = cal.certified_error()
    assert c['class 1'] > n['class 1']
    assert c['class 2'] > n['class 2']


@pytest.mark.parametrize('model_cls', [overlap.binomial_deltas,
                                       overlap.dkw_deltas])
def test_calibration_restores_coverage(model_cls):
    '''
    The claim of CALIBRATION.md, in simulation.

    High dimension and few samples make the nearest-class-mean projector
    overfit, so the naive certificate under-covers. The same certificate
    computed on held-out data must cover at the nominal rate.
    '''
    n_trials = 60
    naive_hits, cal_hits = 0, 0
    naive_total, cal_total = 0, 0

    for seed in range(n_trials):
        X, y = _blobs(120, 60, d=60, sep=1.5, seed=100 + seed)
        Xt, yt = _blobs(2000, 2000, d=60, sep=1.5, seed=9000 + seed)

        # (a) naive - projector fitted to the points the certificate counts
        clf = _Projector().fit(X, y)
        m = model_cls(clf, objective='minimax').fit(X, y)
        naive_hits += _covered(m, Xt, yt)
        naive_total += 2

        # (b) split
        m2, _ = fit_calibrated(
            clf_factory=_Projector,
            deltas_factory=lambda c: model_cls(c, objective='minimax'),
            X=X, y=y, cal_frac=0.35, seed=seed)
        cal_hits += _covered(m2, Xt, yt)
        cal_total += 2

    naive_cov = naive_hits / naive_total
    cal_cov = cal_hits / cal_total
    # the split must cover essentially always, and must beat the naive route
    assert cal_cov >= 0.95, f'calibrated coverage {cal_cov:.2f} too low'
    assert cal_cov > naive_cov, (
        f'calibration did not improve coverage: {naive_cov:.2f} -> '
        f'{cal_cov:.2f}')


def _covered(model, Xt, yt):
    '''how many of the two class certificates hold on the test set'''
    p = np.asarray(model.predict(Xt)).squeeze()
    ce = model.certified_error()
    e0 = (p[yt == 0] != 0).mean()
    e1 = (p[yt == 1] != 1).mean()
    return int(e0 <= ce['class 1']) + int(e1 <= ce['class 2'])
