'''
Build the input fixtures for the golden (characterisation) tests.

Run once, commit the output. Rerunning is only needed if a fixture is added:

    uv run python tests/golden/make_fixtures.py

Every deltas estimator only ever sees 1-D projections, so the fixtures store
exactly that - projected training and test scores, labels and the untouched
classifier's threshold - rather than a trained classifier. That keeps the
golden tests about the deltas methods alone: a scikit-learn upgrade that moves
a fitted SVM cannot make them fail.

Fixtures
--------
gauss_wide     synthetic, very widely separated: the only fixture on which
               the published method is feasible without slacks
gauss_sep      synthetic, separated, but infeasible without slacks (the
               minimum margin does not fit; see the notes, Fig. 1b)
gauss_overlap  synthetic, overlapping (slacks needed)
breast_cancer  real, SVM-rbf, seed 0 - the published pipeline's projection
pima           real, SVM-rbf, seed 0 - heavy overlap, the published method's
               hardest dataset
'''
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'fixtures')


def _synthetic(name, n0, n1, m1, seed):
    rng = np.random.default_rng(seed)
    def draw(n0_, n1_):
        z = np.concatenate([rng.normal(0.0, 1.0, n0_), rng.normal(m1, 1.0, n1_)])
        y = np.concatenate([np.zeros(n0_, int), np.ones(n1_, int)])
        return z, y
    z_train, y_train = draw(n0, n1)
    z_test, y_test = draw(200, 200)
    return {'z_train': z_train, 'y_train': y_train,
            'z_test': z_test, 'y_test': y_test,
            'threshold': np.array(m1 / 2.0)}, \
        {'source': 'synthetic', 'n0': n0, 'n1': n1, 'mean1': m1, 'seed': seed}


def _real(dataset, model='SVM-rbf', seed=0):
    from deltas.pipeline import cached
    data_clf, clfs = cached.get_data_and_classifiers(dataset, model, seed=seed)
    X, y = cached.get_deltas_fit_data(data_clf)
    clf = clfs['Baseline']
    test = data_clf['data_test']
    arrays = {'z_train': np.asarray(clf.get_projection(X)).squeeze(),
              'y_train': np.asarray(y).astype(int),
              'z_test': np.asarray(clf.get_projection(test['X'])).squeeze(),
              'y_test': np.asarray(test['y']).astype(int),
              'threshold': np.array(-float(np.ravel(clf.get_bias())[0]))}
    import sklearn
    return arrays, {'source': 'deltas.pipeline.cached', 'dataset': dataset,
                    'model': model, 'seed': seed,
                    'sklearn': sklearn.__version__, 'numpy': np.__version__}


def main():
    os.makedirs(FIXTURES, exist_ok=True)
    builders = {
        'gauss_wide': lambda: _synthetic('gauss_wide', 200, 20, 12.0, seed=2),
        'gauss_sep': lambda: _synthetic('gauss_sep', 200, 20, 6.0, seed=0),
        'gauss_overlap': lambda: _synthetic('gauss_overlap', 300, 15, 2.0, seed=1),
        'breast_cancer': lambda: _real('Breast Cancer'),
        'pima': lambda: _real('Pima Indian Diabetes'),
    }
    provenance = {}
    for name, build in builders.items():
        arrays, meta = build()
        np.savez(os.path.join(FIXTURES, f'{name}.npz'), **arrays)
        meta.update({'n_train': [int((arrays['y_train'] == c).sum()) for c in (0, 1)],
                     'n_test': [int((arrays['y_test'] == c).sum()) for c in (0, 1)]})
        provenance[name] = meta
        print(name, meta)
    with open(os.path.join(FIXTURES, 'provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)


if __name__ == '__main__':
    main()
