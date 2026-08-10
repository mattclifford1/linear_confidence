'''
Export 1-D projections from toy_datasets + projection_models for the deltas
experiments.

WHY THIS IS A SEPARATE STEP
---------------------------
It used to be a separate *environment*. The sibling packages need sklearn >= 1.6
and python >= 3.11, and this repo was pinned to sklearn 1.3.2 by ~350 lines of
vendored MLPClassifier internals in deltas/classifiers/models.py, so models had
to be fitted under the sibling venv and their projections shipped across.
scikit-learn#25646 landed sample_weight in MLPClassifier upstream, the vendored
copy is gone, and everything now runs in one uv environment.

The step is kept because it is still worth having:

  * it is a cache. Fitting 224 (dataset, model) pairs x 10 seeds x 2 calibration
    modes takes ~30 min; the deltas methods are then re-runnable in minutes
    without refitting anything.
  * every deltas method needs exactly one thing from a classifier,
    get_projection(X) -> (n, 1). Exporting that and nothing else keeps the
    classifier-agnostic claim structural rather than incidental: the code in
    run_wide.py never sees a model.

    uv run python export_projections.py
    ... --datasets Hepatitis "Heart Disease"     # subset
    ... --models Linear LDA                      # subset
    ... --seeds 10 --jobs 8
    ... --force                                  # ignore existing outputs

Output: one .npz per (dataset, model) in experiments/projections/, holding
every seed and both calibration modes. Resumable - an existing, loadable file
is skipped unless --force.
'''
import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from data_loaders import get_dataset                       # noqa: E402
from data_loaders.utils import proportional_split          # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402
from imblearn.over_sampling import SMOTE                   # noqa: E402

import projection_models as pm                             # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'projections')

#: fraction of the training set held out to compute the certificate on
CAL_FRAC = 0.35
#: aim for roughly this training imbalance where the natural one is milder
TARGET_RATIO = 10.0
#: refuse a dataset/seed that cannot leave this many minority training points
MIN_MINORITY_TRAIN = 8
#: ... nor this many minority points for the (balanced) test set
MIN_MINORITY_TEST = 10

SYNTHETIC = ['Gaussian', 'Moons', 'Circles', 'Blobs']
TABULAR = [
    'Abalone Gender', 'Arrhythmia', 'Banknote Authentication',
    'Breast Cancer', 'Breast Cancer Coimbra', 'Breast Cancer Prognostic',
    'Breast Cancer Wisconsin', 'Cervical Cancer', 'Chronic Kidney Disease',
    'Diabetes Pima Indian', 'Framingham CHD', 'Habermans Breast Cancer',
    'HCC Survival', 'Heart Disease', 'Heart Failure', 'Hepatitis',
    'Indian Liver Patient', 'Ionosphere', 'Mammographic Mass', 'Parkinsons',
    'Sonar Rocks vs Mines', 'SPECTF Heart', 'Stroke Prediction',
    'Thoracic Surgery', 'Thyroid Sick', 'Z-Alizadeh Sani CAD',
]
IMAGE = ['BreastMNIST', 'PneumoniaMNIST']
ALL_DATASETS = SYNTHETIC + TABULAR + IMAGE

#: model name -> (factory, supports_class_weight)
MODELS = {
    'Linear':           (lambda **kw: pm.Linear(max_iter=2000, **kw), True),
    'LDA':              (lambda **kw: pm.LDA(**kw), False),
    'SVM-rbf':          (lambda **kw: pm.SVM(kernel='rbf', **kw), True),
    'MLP':              (lambda **kw: pm.MLP(hidden_layer_sizes=(64, 32),
                                             max_iter=600, **kw), False),
    'RandomForest':     (lambda **kw: pm.RandomForest(n_estimators=200, **kw),
                         True),
    'GradientBoosting': (lambda **kw: pm.GradientBoosting(**kw), False),
    'NearestClassMean': (lambda **kw: pm.NearestClassMean(**kw), False),
}


# ------------------------------------------------------------------ data ---
def _as_minority_class_1(X, y):
    '''deltas convention: class 1 is always the minority / positive class'''
    y = np.asarray(y).squeeze().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f'not binary: {dict(zip(classes.tolist(), counts.tolist()))}')
    if counts[1] > counts[0]:
        y = 1 - y
    return np.asarray(X, dtype=float), y


def load_split(name, seed):
    '''
    train/test split following the deltas convention: half the data to train,
    the training minority thinned towards TARGET_RATIO:1, balanced test set.

    Note on the scaler: toy_datasets' minority_reduce_scaler IS the ratio -
    internally the minority training count is set to
    len(majority_train) / scaler - so it must be sized against the majority
    train count, and capped so the minority still has points left for the test
    split. Asking for more minority training points than exist leaves that
    class with an empty test index list, which makes proportional_split's
    np.concatenate return a float array and blow up downstream.
    '''
    loader = get_dataset(name)
    X, y = _as_minority_class_1(loader.get_X(), loader.get_y())
    if X.ndim > 2:                                  # images -> flat vectors
        X = X.reshape(len(X), -1)

    n_maj, n_min = np.bincount(y, minlength=2)
    natural = n_maj / max(n_min, 1)
    n_maj_train = int(n_maj * 0.5)

    # Never give more than half the minority to training. The test set is
    # what the certificate is *measured* against, and on the datasets with the
    # largest minority pool the 1/TARGET_RATIO rule would otherwise take
    # almost all of it - Stroke Prediction ended up with 10 test points per
    # class, making the measured error a multiple of 0.1 and any coverage
    # number meaningless. Test-set size is not something to trade away.
    max_min_train = min(n_min // 2, n_min - MIN_MINORITY_TEST)
    if max_min_train < MIN_MINORITY_TRAIN:
        raise ValueError(
            f'{n_min} minority points total: cannot leave '
            f'{MIN_MINORITY_TRAIN} for training and keep half for test')
    target = int(n_maj_train / TARGET_RATIO)
    n_min_train = int(np.clip(target, MIN_MINORITY_TRAIN, max_min_train))
    scaler = n_maj_train / n_min_train

    train, test = proportional_split(
        {'X': X, 'y': y}, train_size=0.5, seed=seed,
        minority_reduce_scaler=scaler, equal_test=True)

    if len(np.unique(test['y'])) < 2:
        raise ValueError('test split lost a class')
    if int((train['y'] == 1).sum()) < MIN_MINORITY_TRAIN:
        raise ValueError(f'only {int((train["y"] == 1).sum())} minority train '
                         f'points (need {MIN_MINORITY_TRAIN})')
    return train, test, {'natural_ratio': float(natural),
                         'scaler': float(scaler),
                         'train_ratio': float(
                             (train['y'] == 0).sum() /
                             max((train['y'] == 1).sum(), 1))}


def scale_to_train(train, *others):
    '''min-max to [-1, 1] fitted on the training features only'''
    lo = train['X'].min(axis=0)
    hi = train['X'].max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)

    def apply(d):
        return {**d, 'X': 2.0 * (d['X'] - lo) / span - 1.0}
    return apply(train), [apply(o) for o in others]


# ---------------------------------------------------------------- models ---
def fit_model(model_name, X, y, seed):
    factory, _ = MODELS[model_name]
    kw = {}
    try:
        return factory(random_state=seed, **kw).fit(X, y)
    except TypeError:                       # model has no random_state
        return factory(**kw).fit(X, y)


def fit_weighted(model_name, X, y, seed):
    '''balanced class weights where the estimator supports them, else None'''
    factory, supports = MODELS[model_name]
    if not supports:
        return None
    try:
        try:
            return factory(class_weight='balanced',
                           random_state=seed).fit(X, y)
        except TypeError:
            return factory(class_weight='balanced').fit(X, y)
    except Exception:
        return None


def fit_smote(model_name, X, y, seed):
    n_min = int(np.bincount(y, minlength=2)[1])
    if n_min < 6:
        return None
    try:
        Xs, ys = SMOTE(random_state=seed,
                       k_neighbors=min(5, n_min - 1)).fit_resample(X, y)
        return fit_model(model_name, Xs, ys, seed)
    except Exception:
        return None


def _proj(clf, X):
    return np.asarray(clf.get_projection(X), dtype=float).reshape(-1)


def _thresh(clf):
    return float(np.asarray(clf.get_threshold()).reshape(-1)[0])


# ------------------------------------------------------------------ core ---
def one_case(dataset, model_name, seed):
    '''
    returns a dict of arrays for this (dataset, model, seed), covering both
    calibration modes. Keys are prefixed s{seed}_{mode}_.
    '''
    train, test, meta = load_split(dataset, seed)
    train, (test,) = scale_to_train(train, test)
    Xtr, ytr = train['X'], train['y']
    Xte, yte = test['X'], test['y']

    out = {}
    for mode in ('naive', 'split'):
        if mode == 'naive':
            X_fit, y_fit = Xtr, ytr
            X_cert, y_cert = Xtr, ytr           # certificate on training data
        else:
            n_min = int(np.bincount(ytr, minlength=2)[1])
            if int(np.floor(n_min * CAL_FRAC)) < 2 or \
                    n_min - int(np.floor(n_min * CAL_FRAC)) < 2:
                continue                        # too scarce to split honestly
            X_fit, X_cert, y_fit, y_cert = train_test_split(
                Xtr, ytr, test_size=CAL_FRAC, stratify=ytr, random_state=seed)

        clf = fit_model(model_name, X_fit, y_fit, seed)
        p = f's{seed}_{mode}_'
        out[p + 'z_cert'] = _proj(clf, X_cert)
        out[p + 'y_cert'] = y_cert.astype(np.int8)
        out[p + 'z_test'] = _proj(clf, Xte)
        out[p + 'y_test'] = yte.astype(np.int8)
        out[p + 'z_fit'] = _proj(clf, X_fit)
        out[p + 'y_fit'] = y_fit.astype(np.int8)
        out[p + 'threshold'] = np.array([_thresh(clf)])

        # literature baselines that need the feature space, exported as their
        # test-set predictions (that is all the metrics need)
        for label, built in (('smote', fit_smote(model_name, X_fit, y_fit, seed)),
                             ('bw', fit_weighted(model_name, X_fit, y_fit, seed))):
            if built is not None:
                out[p + f'pred_{label}'] = np.asarray(
                    built.predict(Xte)).reshape(-1).astype(np.int8)

    if not out:
        raise ValueError('no usable calibration mode')
    out['meta'] = np.array([json.dumps(
        {**meta, 'dataset': dataset, 'model': model_name,
         'n_train': int(len(ytr)), 'n_test': int(len(yte)),
         'train_counts': np.bincount(ytr, minlength=2).tolist(),
         'test_counts': np.bincount(yte, minlength=2).tolist(),
         'n_features': int(Xtr.shape[1])})])
    return out


def run_pair(dataset, model_name, seeds, force=False):
    path = os.path.join(OUT, f'{_slug(dataset)}__{model_name}.npz')
    if os.path.exists(path) and not force:
        try:
            with np.load(path, allow_pickle=False) as f:
                if all(f's{s}_naive_z_test' in f for s in seeds):
                    return path, 'cached', None
        except Exception:
            pass                                     # corrupt -> recompute

    merged, errors = {}, []
    for seed in seeds:
        try:
            merged.update(one_case(dataset, model_name, seed))
        except Exception as exc:
            errors.append(f'seed {seed}: {type(exc).__name__}: {exc}')
    if not merged:
        return path, 'failed', '; '.join(errors[:2])

    os.makedirs(OUT, exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **merged)
    os.replace(tmp, path)                            # atomic
    return path, 'ok', ('; '.join(errors[:2]) if errors else None)


def _slug(name):
    return name.replace(' ', '-').replace('/', '-')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=ALL_DATASETS)
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    os.makedirs(OUT, exist_ok=True)
    pairs = [(d, m) for d in args.datasets for m in args.models]
    print(f'{len(pairs)} (dataset, model) pairs x {len(seeds)} seeds '
          f'-> {OUT}', flush=True)

    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_pair)(d, m, seeds, args.force) for d, m in pairs)
    else:
        results = [run_pair(d, m, seeds, args.force) for d, m in pairs]

    n_ok = n_cached = 0
    for (d, m), (_, status, err) in zip(pairs, results):
        if status == 'ok':
            n_ok += 1
        elif status == 'cached':
            n_cached += 1
        if status == 'failed':
            print(f'  FAILED  {d} / {m}: {err}', flush=True)
        elif err:
            print(f'  partial {d} / {m}: {err}', flush=True)
    print(f'\n{n_ok} written, {n_cached} already present, '
          f'{len(pairs) - n_ok - n_cached} failed  '
          f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
