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
    ... --datasets "MNIST: 4 vs 9" "CIFAR-100: fish vs rest" "MNIST: 3+5 vs rest"
    ... --datasets "MNIST: 0 vs rest" --minority-train 5 25 100   # N-sweep
    ... --datasets "Parkinsons (disease-positive)"   # see VARIANTS
    ... --models Linear LDA                      # subset
    ... --seeds 10 --jobs 8
    ... --force                                  # ignore existing outputs

Output: one .npz per (dataset, model) in experiments/projections/, holding
every seed and both calibration modes. Resumable - an existing, loadable file
is skipped unless --force.
'''
import argparse
import functools
import hashlib
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
from projection_models.datasets import binary_pair         # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'projections')
FEATURES = os.path.join(HERE, 'features')

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
#: clinical, and the reason they are listed apart: MIMIC-III Mortality is the
#: dataset both papers lead with and was oddly missing from this grid, and
#: MIMIC-IV is the only set here with a minority large enough (7634) for the
#: certificate to be tight rather than vacuous - see MAJORITY_MAX.
CLINICAL = ['MIMIC-III Mortality', 'MIMIC-IV Ready for Discharge']
ALL_DATASETS = SYNTHETIC + TABULAR + IMAGE + CLINICAL

#: per-dataset cap on majority TRAINING rows (toy_datasets `majority_max`).
#: MIMIC-IV has 1.6M majority rows: fine for linear/tree models, hopeless for
#: a kernel SVM (O(n^2)) and for the slack method, whose downsample loop is
#: O(N) iterations. Capping keeps every model in the grid comparable on the
#: same data. The test split is untouched.
MAJORITY_MAX = {'MIMIC-IV Ready for Discharge': 50_000}
#: the cap for any other dataset whose majority training set would exceed it
#: (the large image tasks); every dataset in the original grid is below it
MAJORITY_MAX_DEFAULT = 10_000

#: named datasets needing loader options a class-pair spec cannot express:
#: name -> (dataset or class-pair spec, extra loader kwargs, make the minority
#: class 1). The disease-positive sets keep the disease as class 1 where it is
#: the larger class. The Gaussians are E2's grid (sep is the distance between
#: the means in class-0 s.d.s): 10^5 points per class, N_1 set by an @n suffix.
VARIANTS = {
    **{f'{d} (disease-positive)': (d, {}, False) for d in (
        'Breast Cancer Coimbra', 'Indian Liver Patient', 'Parkinsons',
        'SPECTF Heart', 'Z-Alizadeh Sani CAD')},
    **{f'CIFAR-10N {noise}: {pair}': (f'CIFAR-10N: {pair}',
                                      {'label_noise_type': f'{noise}_label'},
                                      False)
       for noise in ('aggre', 'worse')
       for pair in ('cat vs dog', 'automobile vs truck')},
    **{f'Gaussian sep{sep:g} spread{spread:g}': (
        'Gaussian', {'num_samples': [100_000, 100_000],
                     'means': [[0.0, 0.0], [float(sep), 0.0]],
                     'cov1_scaler': float(spread) ** 2}, True)
       for sep in (1, 2, 3, 5) for spread in (0.5, 1, 2)},
}

#: model name -> (factory, supports_class_weight)
MODELS = {
    'Linear':           (lambda **kw: pm.Linear(max_iter=2000, **kw), True),
    'LDA':              (lambda **kw: pm.LDA(**kw), False),
    'SVM-rbf':          (lambda **kw: pm.SVM(kernel='rbf', **kw), True),
    'MLP':              (lambda **kw: pm.MLP(hidden_layer_sizes=(64, 32),
                                             max_iter=600, **kw), True),
    'RandomForest':     (lambda **kw: pm.RandomForest(n_estimators=200, **kw),
                         True),
    'GradientBoosting': (lambda **kw: pm.GradientBoosting(**kw), False),
    'NearestClassMean': (lambda **kw: pm.NearestClassMean(**kw), False),
    'SVM-linear':       (lambda **kw: pm.SVM(kernel='linear', **kw), True),
    'LinearSVM':        (lambda **kw: pm.LinearSVM(**kw), True),
    'DecisionTree':     (lambda **kw: pm.DecisionTree(**kw), True),
    'HistGradientBoosting': (lambda **kw: pm.HistGradientBoosting(**kw), True),
    'MLP-small':        (lambda **kw: pm.MLP(hidden_layer_sizes=(20,),
                                             max_iter=600, **kw), True),
    'MLP-deep':         (lambda **kw: pm.MLP(hidden_layer_sizes=(64, 128, 64),
                                             max_iter=600, **kw), True),
}
#: the original 34-dataset grid's models, the --models default
GRID_MODELS = ['Linear', 'LDA', 'SVM-rbf', 'MLP', 'RandomForest',
               'GradientBoosting', 'NearestClassMean']


# ------------------------------------------------------------------ data ---
def resolve_dataset(spec):
    '''
    (toy_datasets name, loader kwargs) for a plain name or a class-pair spec
    "<name>: <positive> vs <negative>", where `+` joins classes, a missing
    or `rest` negative means positive-vs-rest and "except <classes>" drops classes.
    '''
    if ':' not in spec:
        return spec, {}
    name, task = (part.strip() for part in spec.split(':', 1))
    task, _, excluded = (part.strip() for part in task.partition(' except '))
    positive, _, negative = (part.strip() for part in task.partition(' vs '))

    def classes(text):
        names = tuple(c.strip() for c in text.split('+'))
        return names[0] if len(names) == 1 else names

    negative = None if negative in ('', 'rest') else classes(negative)
    kwargs = binary_pair(name, classes(positive), negative)
    if excluded:
        drop = binary_pair(name, classes(excluded))['minority_id']
        kwargs['classes_remove'] = sorted(set(kwargs['classes_remove']) | set(drop))
    return name, kwargs


def _as_minority_class_1(X, y, flip=True):
    '''deltas convention: class 1 is always the minority / positive class'''
    y = np.asarray(y).squeeze().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f'not binary: {dict(zip(classes.tolist(), counts.tolist()))}')
    if flip == True and counts[1] > counts[0]:
        y = 1 - y
    return np.asarray(X, dtype=float), y


def split_options(spec):
    '''
    (dataset spec, minority training count or None, target ratio, test class
    or None) from a name with optional suffixes " @n25" (exactly 25 minority
    training points), " @r100" (thin towards 100:1 instead of TARGET_RATIO)
    and " @test <classes>" (the test minority is these classes instead)
    '''
    n_train, ratio, shift = None, TARGET_RATIO, None
    while ' @' in spec:
        spec, opt = spec.rsplit(' @', 1)
        if opt.startswith('test '):
            shift = opt[len('test '):].strip()
        elif opt[:1] == 'n':
            n_train = int(opt[1:])
        elif opt[:1] == 'r':
            ratio = float(opt[1:])
        else:
            raise ValueError(f'unknown split option @{opt}: use @n<count>, '
                             f'@r<ratio> or @test <classes>')
    return spec.strip(), n_train, ratio, shift


def shift_specs(spec, shift):
    '''
    (training spec, spec whose class 1 is the test minority) of an @test task;
    a "vs rest" training task leaves the test classes out of its rest
    '''
    backbone = ''
    if spec.endswith(']') and ' [' in spec:
        spec, backbone = spec[:-1].rsplit(' [', 1)
        backbone = f' [{backbone}]'
    name, task = (part.strip() for part in spec.split(':', 1))
    positive, _, negative = (part.strip() for part in task.partition(' vs '))
    train = spec
    if negative in ('', 'rest'):
        train = f'{name}: {positive} vs rest except {shift}'
    return train + backbone, f'{name}: {shift} vs {positive}{backbone}'


def source_specs(name):
    '''the _load_xy names a task reads'''
    spec, _, _, shift = split_options(name)
    return list(shift_specs(spec, shift)) if shift else [spec]


def features_path(spec, backbone):
    '''frozen embeddings of a dataset, written by export_torch.py features'''
    return os.path.join(FEATURES, f'{_slug(spec)}__{backbone}.npz')


def x_hash(X):
    return hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()


@functools.lru_cache(maxsize=2)
def _load_xy(name):
    '''
    the whole dataset, binary and flat, cached across seeds (read-only); a
    " [resnet18]" suffix swaps the pixels for that backbone's embeddings
    '''
    backbone = None
    if name.endswith(']') and ' [' in name:
        name, backbone = name[:-1].rsplit(' [', 1)
    spec, extra, flip = VARIANTS.get(name, (name, {}, None))
    base, loader_kwargs = resolve_dataset(spec)
    loader_kwargs = {**loader_kwargs, **extra}
    loader = get_dataset(base, **loader_kwargs)
    if flip is None:                                # a named pair keeps its labels
        flip = 'minority_id' not in loader_kwargs
    X, y = _as_minority_class_1(loader.get_X(), loader.get_y(), flip=flip)
    if X.ndim > 2:                                  # images -> flat vectors
        X = X.reshape(len(X), -1)
    if backbone is not None:
        path = features_path(name, backbone)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path}: run export_torch.py features first')
        with np.load(path) as f:
            if str(f['x_hash']) != x_hash(X):
                raise ValueError(f'{path} was extracted from different data')
            X = f['Z'].astype(float)
    X.setflags(write=False)
    y.setflags(write=False)
    return X, y, loader_kwargs


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

    `name` may be a class-pair spec (see resolve_dataset); its positive class
    stays class 1 even when it is the larger one. Its split_options suffixes
    set the minority size; for one seed, smaller @n samples are subsets of
    larger ones and the majority is the same.
    '''
    spec, n_train_req, ratio, shift = split_options(name)
    if shift:
        spec, test_spec = shift_specs(spec, shift)
    X, y, loader_kwargs = _load_xy(spec)

    n_maj, n_min = np.bincount(y, minlength=2)
    natural = n_maj / max(n_min, 1)
    n_maj_train = int(n_maj * 0.5)

    # cap the majority TRAIN rows where the dataset is too large to fit every
    # model family on. The ratio below is then taken against the capped count,
    # which is what toy_datasets' majority_max guarantees.
    majority_max = MAJORITY_MAX.get(spec, MAJORITY_MAX_DEFAULT)
    if n_maj_train > majority_max:
        n_maj_train = majority_max
    else:
        majority_max = None

    # Never give more than half the minority to training. The test set is
    # what the certificate is *measured* against, and on the datasets with the
    # largest minority pool the 1/TARGET_RATIO rule would otherwise take
    # almost all of it - Stroke Prediction ended up with 10 test points per
    # class, making the measured error a multiple of 0.1 and any coverage
    # number meaningless. Test-set size is not something to trade away.
    max_min_train = min(n_min // 2, n_min - MIN_MINORITY_TEST)
    if n_train_req is not None:
        if not 2 <= n_train_req <= max_min_train:
            raise ValueError(f'@n{n_train_req}: {name} allows 2 to '
                             f'{max_min_train} minority training points')
        n_min_train = n_train_req
        # half a point of slack so proportional_split's int() lands on N
        scaler = n_maj_train / (n_min_train + 0.5)
    else:
        if max_min_train < MIN_MINORITY_TRAIN:
            raise ValueError(
                f'{n_min} minority points total: cannot leave '
                f'{MIN_MINORITY_TRAIN} for training and keep half for test')
        target = int(n_maj_train / ratio)
        n_min_train = int(np.clip(target, MIN_MINORITY_TRAIN, max_min_train))
        scaler = n_maj_train / n_min_train

    train, test = proportional_split(
        {'X': X, 'y': y}, train_size=0.5, seed=seed,
        minority_reduce_scaler=scaler, equal_test=True,
        majority_max=majority_max)

    if shift:
        # a balanced test set of the training task's majority and the new class
        Xs, ys, _ = _load_xy(test_spec)
        Xs = Xs[ys == 1]
        rng = np.random.default_rng(seed)
        maj = np.flatnonzero(test['y'] == 0)
        k = min(len(maj), len(Xs))
        test = {'X': np.concatenate([test['X'][rng.choice(maj, k, replace=False)],
                                     Xs[rng.choice(len(Xs), k, replace=False)]]),
                'y': np.repeat([0, 1], k)}
    if len(np.unique(test['y'])) < 2:
        raise ValueError('test split lost a class')
    if int((train['y'] == 1).sum()) < min(MIN_MINORITY_TRAIN, n_min_train):
        raise ValueError(f'only {int((train["y"] == 1).sum())} minority train '
                         f'points (need {MIN_MINORITY_TRAIN})')
    return train, test, {'natural_ratio': float(natural),
                         'loader_kwargs': loader_kwargs,
                         'test_minority': shift,
                         'majority_max': majority_max,
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
    return factory(seed=seed).fit(X, y)


def fit_weighted(model_name, X, y, seed):
    '''balanced class weights where the estimator supports them, else None'''
    factory, supports = MODELS[model_name]
    if not supports:
        return None
    try:
        return factory(class_weight='balanced', seed=seed).fit(X, y)
    except Exception:
        return None


#: SMOTE balances the minority up to the majority count, so the fit set is
#: roughly 2x the majority. Past this, an O(n^2) kernel SVM stops being a
#: baseline anyone would run - MIMIC-IV would mean fitting an RBF SVM to 100k
#: rows, which took over an hour per seed. Skipped and recorded as unavailable
#: rather than silently distorting the timings.
SMOTE_MAX_RESAMPLED = 40_000


def fit_smote(model_name, X, y, seed):
    n_maj, n_min = np.bincount(y, minlength=2)
    if n_min < 6:
        return None
    if 2 * int(n_maj) > SMOTE_MAX_RESAMPLED:
        return None
    try:
        Xs, ys = SMOTE(random_state=seed,
                       k_neighbors=min(5, int(n_min) - 1)).fit_resample(X, y)
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
    t0 = time.time()
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
         'n_features': int(Xtr.shape[1]),
         'seconds': round(time.time() - t0, 2)})])
    return out


def _read_npz(path):
    '''every array of an existing export, or {} if there is none (or it is corrupt)'''
    try:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    except Exception:
        return {}


def _write_npz(path, arrays):
    '''atomic write; the temporary name does not match *.npz'''
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path + '.part', 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(path + '.part', path)


def run_pair(dataset, model_name, seeds, force=False, out_dir=OUT):
    '''export the seeds not already in the pair's file, keeping those that are'''
    path = os.path.join(out_dir, f'{_slug(dataset)}__{model_name}.npz')
    merged = {} if force else _read_npz(path)
    todo = [s for s in seeds if f's{s}_naive_z_test' not in merged]
    if not todo:
        return path, 'cached', None

    errors, done = [], False
    for seed in todo:
        try:
            merged.update(one_case(dataset, model_name, seed))
        except Exception as exc:
            errors.append(f'seed {seed}: {type(exc).__name__}: {exc}')
            continue
        _write_npz(path, merged)                     # each seed, so a stop loses one
        done = True
    if not done:
        return path, 'failed', '; '.join(errors[:2])
    return path, 'ok', ('; '.join(errors[:2]) if errors else None)


def _slug(name):
    '''file-name form of a dataset name'''
    return name.replace(':', '').replace(' ', '-').replace('/', '-')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=ALL_DATASETS)
    ap.add_argument('--models', nargs='*', default=GRID_MODELS)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', default=OUT, help='folder for the .npz files')
    ap.add_argument('--minority-train', nargs='*', type=int, default=None,
                    help='run each dataset at each of these minority '
                         'training sizes (adds " @n<count>")')
    ap.add_argument('--ratio', nargs='*', type=float, default=None,
                    help='run each dataset at each of these training '
                         'ratios (adds " @r<ratio>")')
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    datasets = args.datasets
    if args.minority_train:
        datasets = [f'{d} @n{n}' for d in datasets for n in args.minority_train]
    if args.ratio:
        datasets = [f'{d} @r{r:g}' for d in datasets for r in args.ratio]
    os.makedirs(args.out, exist_ok=True)
    pairs = [(d, m) for d in datasets for m in args.models]
    print(f'{len(pairs)} (dataset, model) pairs x {len(seeds)} seeds '
          f'-> {args.out}', flush=True)

    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_pair)(d, m, seeds, args.force, args.out)
            for d, m in pairs)
    else:
        results = [run_pair(d, m, seeds, args.force, args.out)
                   for d, m in pairs]

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
