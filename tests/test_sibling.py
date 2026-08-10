'''
Tests for the delegation layer — run with:  pytest tests/

These pin the contract between this repo and the sibling packages
(`toy_datasets`, `projection_models`). If one of them changes shape, these are
what should tell you.
'''
import numpy as np
import pytest

from deltas.classifiers import sibling as clf_sibling
from deltas.data.loaders import sibling as data_sibling
from deltas.model import overlap

pytestmark = pytest.mark.skipif(
    not data_sibling.available() or not clf_sibling.available(),
    reason='sibling packages not installed')


# ------------------------------------------------------------------ data ---
def test_sibling_datasets_cover_the_papers():
    '''every dataset the two papers use must exist in toy_datasets'''
    have = set(data_sibling.available())
    needed = ['Breast Cancer', 'Diabetes Pima Indian', 'Hepatitis',
              'Heart Disease', 'Breast Cancer Wisconsin',
              'Habermans Breast Cancer', 'Ionosphere', 'Abalone Gender',
              'Banknote Authentication', 'Sonar Rocks vs Mines',
              'Wheat Seeds', 'MIMIC-III Mortality']
    assert not [n for n in needed if n not in have]


def test_sibling_split_puts_minority_in_class_1():
    train, test = data_sibling.get_sibling_dataset(
        'Diabetes Pima Indian', seed=0, ratio=10, equal_test=True)
    counts = np.bincount(train['y'].astype(int), minlength=2)
    assert counts[1] < counts[0], 'class 1 must be the minority'
    assert set(np.unique(test['y'])) == {0, 1}


def test_sibling_split_honours_the_ratio():
    train, _ = data_sibling.get_sibling_dataset(
        'Diabetes Pima Indian', seed=0, ratio=10)
    n_maj, n_min = np.bincount(train['y'].astype(int), minlength=2)
    assert 5 <= n_maj / n_min <= 20, f'ratio {n_maj / n_min:.1f} off target'


def test_sibling_split_never_gives_away_most_of_the_minority():
    '''
    the test set is what any error estimate is measured against; a rule that
    trades it away for training imbalance produced a 10-point test set once
    (FINDINGS section 11.6)
    '''
    for name in ('Diabetes Pima Indian', 'Ionosphere'):
        train, test = data_sibling.get_sibling_dataset(
            name, seed=0, ratio=10, equal_test=True)
        n_min_train = int((train['y'] == 1).sum())
        n_min_test = int((test['y'] == 1).sum())
        assert n_min_test >= n_min_train, (
            f'{name}: {n_min_train} minority train vs {n_min_test} test')


def test_sibling_split_is_deterministic():
    a, _ = data_sibling.get_sibling_dataset('Ionosphere', seed=3, ratio=5)
    b, _ = data_sibling.get_sibling_dataset('Ionosphere', seed=3, ratio=5)
    assert np.array_equal(a['X'], b['X']) and np.array_equal(a['y'], b['y'])


def test_multiclass_is_rejected():
    '''
    deltas is binary only. Tested on the guard directly rather than on a named
    dataset, because toy_datasets binarises several of the classically
    multiclass ones and which those are is not our contract.
    '''
    X = np.zeros((9, 2))
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match='binary'):
        data_sibling._minority_is_class_1(X, y)


def test_minority_is_flipped_to_class_1():
    X = np.zeros((10, 2))
    y = np.array([1] * 8 + [0] * 2)          # class 1 is the majority here
    _, out = data_sibling._minority_is_class_1(X, y)
    assert int((out == 1).sum()) == 2


def test_unknown_dataset_names_are_reported_usefully():
    from deltas.pipeline import data as pipe_data
    with pytest.raises(ValueError) as exc:
        pipe_data.get_real_dataset('Definitely Not A Dataset', _print=False)
    assert 'toy_datasets' in str(exc.value)


# ---------------------------------------------------------------- models ---
MODELS = ['Linear', 'LDA', 'SVM-rbf', 'MLP', 'RandomForest',
          'GradientBoosting', 'NearestClassMean']


def _blobs(n0=160, n1=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n0, d)), rng.normal(1.6, 1, (n1, d))])
    y = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    return X, y


@pytest.mark.parametrize('name', MODELS)
def test_built_models_expose_the_deltas_interface(name):
    X, y = _blobs()
    clf = clf_sibling.build(name).fit(X, y)
    z = clf.get_projection(X)
    assert z.shape == (len(X), 1)
    assert np.isfinite(z).all()
    assert np.isfinite(np.ravel(clf.get_bias())).all()


@pytest.mark.parametrize('name', MODELS)
def test_deltas_can_consume_a_sibling_model(name):
    X, y = _blobs()
    clf = clf_sibling.build(name).fit(X, y)
    model = overlap.binomial_deltas(clf, objective='minimax').fit(X, y)
    assert model.is_fit == True
    preds = np.asarray(model.predict(X)).squeeze()
    assert set(np.unique(preds)) <= {0, 1}


def test_get_bias_is_the_negated_threshold():
    '''the one real mismatch between the two APIs'''
    X, y = _blobs()
    clf = clf_sibling.build('Linear').fit(X, y)
    assert np.allclose(np.ravel(clf.get_bias()),
                       -np.ravel(clf.get_threshold()))


def test_threshold_reproduces_predict():
    '''projection > threshold must equal predict, or get_bias is meaningless'''
    X, y = _blobs()
    for name in ('Linear', 'LDA', 'SVM-rbf'):
        clf = clf_sibling.build(name).fit(X, y)
        z = clf.get_projection(X).squeeze(axis=-1)
        t = np.ravel(clf.get_threshold())[0]
        assert np.array_equal((z > t).astype(int),
                              np.asarray(clf.predict(X)).astype(int)), name


def test_mlp_supports_balanced_class_weight():
    '''
    the capability the local NN needed ~350 lines of vendored sklearn for;
    if this breaks, that removal needs revisiting
    '''
    X, y = _blobs(n0=300, n1=30)
    plain = clf_sibling.build('MLP', max_iter=300).fit(X, y)
    weighted = clf_sibling.build(
        'MLP', max_iter=300, class_weight='balanced').fit(X, y)
    rec_plain = (np.asarray(plain.predict(X))[y == 1] == 1).mean()
    rec_weighted = (np.asarray(weighted.predict(X))[y == 1] == 1).mean()
    assert rec_weighted >= rec_plain


def test_unknown_model_names_are_reported_usefully():
    with pytest.raises(ValueError) as exc:
        clf_sibling.build('NotAModel')
    assert 'Available' in str(exc.value)


# ------------------------------------------------------- the sklearn pin ---
def test_no_private_sklearn_imports():
    '''
    The repo was pinned to sklearn 1.3.2 for two years by ~350 lines of
    vendored MLPClassifier internals. Nothing in the package may import a
    private sklearn symbol again, or the pin comes back.
    '''
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / 'deltas'
    pattern = re.compile(r'^\s*(?:from|import)\s+sklearn[\w.]*\._', re.M)
    offenders = [str(p.relative_to(root.parent))
                 for p in root.rglob('*.py')
                 if pattern.search(p.read_text(encoding='utf-8', errors='ignore'))]
    assert offenders == [], f'private sklearn imports in {offenders}'


def test_local_activations_match_sklearn():
    '''
    deltas.classifiers.models carries its own copy of sklearn's activation
    functions to avoid importing the private module. They must stay identical.
    '''
    sk_base = pytest.importorskip('sklearn.neural_network._base')
    import deltas.classifiers.models as models
    rng = np.random.default_rng(0)
    for name, fn in models.ACTIVATIONS.items():
        a = rng.normal(size=(40, 6)) * 3.0
        b = a.copy()
        sk_base.ACTIVATIONS[name](a)
        fn(b)
        assert np.allclose(a, b, rtol=0, atol=0), name
