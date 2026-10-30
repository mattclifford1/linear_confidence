'''
The two synthetic set-ups behind the ECAI 2024 explanation figures, recomputed with the legacy code.

`separated()` is notebooks-ECAI/projection_plots.py; `overlapping()` is notebooks-ECAI/Guassian_plots.py.
'''
import functools

import deltas.misc.use_two as ut
ut.USE_TWO = False  # the published setting; must precede every other deltas import

import numpy as np
from deltas.classifiers import models
from deltas.legacy.ecai2024 import radius
from deltas.model import base, downsample
from deltas.pipeline import classifier, data

N1, N2 = 1000, 10
DELTA_DOT_NINE = 0.9999999999999999999


def R_hat(R_emp, info, N, delta):
    '''Upper bound on R_i (Eq. 5) from the empirical radius.'''
    return radius.R_upper_bound(R_emp, info['R all data'], N, delta)


def class_scores(info):
    '''Projected scores and means of S_1 (majority) and S_2 (minority).'''
    p = info['projected_data']
    x = p['X'].ravel()
    return (x[p['y'] == 0], x[p['y'] == 1],
            info['empirical_projected_mean 1'], info['empirical_projected_mean 2'])


@functools.cache
def separated(m=5, seed=10):
    '''
    Well separated classes at means -/+[m, m] with a linear SVM (Figs. 1-2 of the paper).

    The paper's draw is not recoverable (get_data(seed=None) reseeds from the OS), so this is a new one.
    '''
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-m, 1, (N1, 2)), rng.normal(m, 1, (N2, 2))])
    y = np.r_[np.zeros(N1), np.ones(N2)]
    clf = models.SVM(kernel='linear').fit(X, y)
    return {'X': X, 'y': y, 'clf': clf, 'info': base.base_deltas.get_data_info(X, y, clf)}


@functools.cache
def overlapping(m=1, seed=0):
    '''Overlapping classes at means -/+[m, m] with logistic regression (Figs. 3-6 of the paper).'''
    d = data.get_data(m1=[-m, -m], m2=[m, m], N1=N1, N2=N2, scale=False, seed=seed)
    X, y = d['data']['X'], d['data']['y']
    # SMOTE draws from the global RNG, so reseeding gives get_classifier the same oversample
    np.random.seed(seed)
    smote = data.get_SMOTE_data(d['data'])
    np.random.seed(seed)
    clfs = classifier.get_classifier(data_clf=d, model='Linear', _plot=False)
    clf = clfs['Baseline']
    deltas_model = downsample.downsample_deltas(clf).fit(
        X, y, alpha=1, method='supports-prop-update_mean', _print=False, _plot=False)
    clfs['Our Method'] = deltas_model
    return {'X': X, 'y': y, 'test': d['data_test'], 'smote': smote,
            'clfs': clfs, 'info': base.base_deltas.get_data_info(X, y, clf),
            'deltas': deltas_model}
