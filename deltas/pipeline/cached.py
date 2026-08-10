'''
cached versions of the expensive pipeline steps

Classifier training dominates the wall clock of every experiment script
(MIMIC-III is ~270s per seed to train the three baselines, vs ~35s for all the
deltas fits put together) and the *same* models are retrained by every script.
These wrappers make that a one-off cost.

Note the classifier cache key deliberately does NOT include USE_TWO /
USE_GLOBAL_R: the baselines are trained from data alone and know nothing about
deltas, so one training run serves every deltas configuration.
'''
import deltas.pipeline.data as pipe_data
import deltas.pipeline.classifier as pipe_clf
import deltas.pipeline.calibration as pipe_cal
import deltas.utils.cache as cache
from deltas.misc.use_two import RANDOM_STATE


def get_dataset(dataset, seed=0, scale=True, use_cache=True, _print=False,
                **kwargs):
    '''cached deltas.pipeline.data.get_real_dataset'''
    config = {'dataset': dataset,
              'seed': seed,
              'scale': scale,
              'random_state': RANDOM_STATE,
              'kwargs': kwargs}

    def compute():
        return cache.sanitise(pipe_data.get_real_dataset(
            dataset, _print=False, seed=seed, scale=scale, **kwargs))

    return cache.cached('dataset', config, compute,
                        use_cache=use_cache, _print=_print)


def get_classifiers(data_clf, dataset, model='Linear', seed=0, scale=True,
                    smote=True, balanced_weights=True, costcla_methods=True,
                    calibration=None, use_cache=True, _print=False, **kwargs):
    '''
    cached deltas.pipeline.classifier.get_classifier

    `dataset`, `seed` and `scale` are passed separately only so they can go
    into the cache key - the data itself comes from `data_clf`.

    `calibration` likewise only enters the cache key: when it is set, the
    caller has already replaced data_clf['data'] with the fit part, so these
    classifiers are trained on less data and must not share a cache entry with
    the uncalibrated ones.
    '''
    config = {'dataset': dataset,
              'seed': seed,
              'scale': scale,
              'model': model,
              'smote': smote,
              'balanced_weights': balanced_weights,
              'costcla_methods': costcla_methods,
              'random_state': RANDOM_STATE,
              'kwargs': kwargs}
    # only add the key when it is in use, so the existing (expensive) cache
    # entries for the uncalibrated runs keep their hashes
    if calibration is not None:
        config['calibration'] = calibration

    def compute():
        return pipe_clf.get_classifier(
            data_clf=data_clf,
            model=model,
            smote=smote,
            balanced_weights=balanced_weights,
            costcla_methods=costcla_methods,
            _plot=False,
            _print=False,
            **kwargs)

    return cache.cached('classifiers', config, compute,
                        use_cache=use_cache, _print=_print)


def get_data_and_classifiers(dataset, model, seed=0, scale=True,
                             calibration=None, use_cache=True, _print=False,
                             **clf_kwargs):
    '''
    convenience: the two calls every experiment script starts with

    calibration: None (default) trains on all of the training data, so any
        certificate a deltas model then computes is read off points the
        classifier was fitted to and is optimistic - see
        deltas.pipeline.calibration for the measurement and the reasoning.
        A float in (0, 1) instead holds out that fraction as a calibration set:
        classifiers are trained on the fit part only and the held-out part is
        returned as data_clf['data_cal'], to be passed to the deltas .fit().
        The untouched training set stays available as data_clf['data_full'].
    '''
    data_clf = get_dataset(dataset, seed=seed, scale=scale,
                           use_cache=use_cache, _print=_print)

    if calibration is not None:
        data = data_clf['data']
        X_fit, y_fit, X_cal, y_cal = pipe_cal.split_calibration(
            data['X'], data['y'], cal_frac=calibration, seed=seed)
        data_clf = dict(data_clf)
        data_clf['data_full'] = data
        data_clf['data'] = {**data, 'X': X_fit, 'y': y_fit}
        data_clf['data_cal'] = {**data, 'X': X_cal, 'y': y_cal}

    clfs = get_classifiers(data_clf, dataset, model=model, seed=seed,
                           scale=scale, calibration=calibration,
                           use_cache=use_cache, _print=_print, **clf_kwargs)
    data_clf['clf'] = clfs['Baseline']
    return data_clf, clfs


def get_deltas_fit_data(data_clf):
    '''
    the (X, y) a deltas model should be fitted on: the calibration split when
    one was requested, otherwise the training data itself
    '''
    part = data_clf.get('data_cal') or data_clf['data']
    return part['X'], part['y']
