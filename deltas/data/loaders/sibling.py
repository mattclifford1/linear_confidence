'''
Datasets delegated to the sibling `toy_datasets` package.

Every dataset the two papers use exists there, plus about thirty more, and it
is maintained against current numpy/sklearn — three of the loaders in this
directory had silently rotted under numpy 2 before they were patched. New work
should use this path; the local loaders in `deltas/data/loaders/` are kept
because the published results were produced with them and their exact
shuffling defines those splits.

The contract matches the local loaders: return `(train_dict, test_dict)`, each
`{'X', 'y', ...}`, with **class 1 as the minority**.

    from deltas.pipeline import data
    d = data.get_real_dataset('Thyroid Sick', seed=0)      # falls through here

`toy_datasets.proportional_split` is the same routine this repo uses, so
passing the same `ratio` and `seed` reproduces the same style of split. Note
its `minority_reduce_scaler` *is* the imbalance ratio: the minority training
count is set to `len(majority_train) / scaler`, so it is sized against the
majority, and asking for more minority training points than exist leaves that
class an empty test index list (which then fails as a float index).
'''
import numpy as np


def _import_sibling():
    try:
        from data_loaders import get_dataset, proportional_split
    except ImportError as exc:          # pragma: no cover - environment issue
        raise ImportError(
            'toy_datasets is not installed. It is a path dependency of this '
            'project: `uv sync` from the repo root, or see [tool.uv.sources] '
            'in pyproject.toml.') from exc
    return get_dataset, proportional_split


def available():
    '''names that can be passed to get_sibling_dataset'''
    try:
        from data_loaders import AVAILABLE_DATASETS
    except ImportError:
        return []
    return list(AVAILABLE_DATASETS)


def _minority_is_class_1(X, y):
    '''repo-wide convention: class 1 is the minority / positive class'''
    y = np.asarray(y).squeeze().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(
            f'{len(classes)}-class dataset; deltas is binary only '
            f'(counts {dict(zip(classes.tolist(), counts.tolist()))})')
    if counts[1] > counts[0]:
        y = 1 - y
    return np.asarray(X, dtype=float), y


def get_sibling_dataset(name, seed=True, train_size=0.5, ratio=None,
                        equal_test=False, flatten=True, **kwargs):
    '''
    load `name` from toy_datasets and split it the way this repo expects

    ratio:       target training imbalance (majority_train / minority_train).
                 None keeps the dataset's natural balance.
    equal_test:  balance the test set by subsampling the majority.
    flatten:     reshape image datasets to one row per sample.
    '''
    get_dataset, proportional_split = _import_sibling()

    loader = get_dataset(name)
    X, y = _minority_is_class_1(loader.get_X(), loader.get_y())
    if flatten == True and X.ndim > 2:
        X = X.reshape(len(X), -1)

    if ratio is not None:
        n_maj_train = int(np.bincount(y, minlength=2)[0] * train_size)
        n_min = int(np.bincount(y, minlength=2)[1])
        # never hand more than half the minority to training - the test set is
        # what any error estimate is measured against
        wanted = max(1, int(n_maj_train / ratio))
        n_min_train = int(np.clip(wanted, 1, max(1, n_min // 2)))
        ratio = n_maj_train / n_min_train

    train, test = proportional_split(
        {'X': X, 'y': y}, train_size=train_size, seed=seed,
        minority_reduce_scaler=ratio, equal_test=equal_test)

    for part in (train, test):
        part['feature_names'] = _feature_names(loader, X.shape[1])
        part['description'] = _description(loader, name)
    return train, test


def _feature_names(loader, n_features):
    try:
        names = loader.get_feature_names()
    except Exception:
        names = None
    if names is None or len(names) != n_features:
        return [f'f{i}' for i in range(n_features)]
    return list(names)


def _description(loader, name):
    try:
        return loader.get_description() or name
    except Exception:
        return name
