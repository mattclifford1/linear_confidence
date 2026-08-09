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
                    use_cache=True, _print=False, **kwargs):
    '''
    cached deltas.pipeline.classifier.get_classifier

    `dataset`, `seed` and `scale` are passed separately only so they can go
    into the cache key - the data itself comes from `data_clf`.
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
                             use_cache=True, _print=False, **clf_kwargs):
    '''convenience: the two calls every experiment script starts with'''
    data_clf = get_dataset(dataset, seed=seed, scale=scale,
                           use_cache=use_cache, _print=_print)
    clfs = get_classifiers(data_clf, dataset, model=model, seed=seed,
                           scale=scale, use_cache=use_cache, _print=_print,
                           **clf_kwargs)
    data_clf['clf'] = clfs['Baseline']
    return data_clf, clfs
