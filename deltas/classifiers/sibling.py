'''
Classifiers delegated to the sibling `projection_models` package.

`projection_models` exists to give models a `get_projection(X) -> (n, 1)`
method, which is the one thing every deltas estimator needs from a classifier.
It covers eleven families (linear, LDA, kernel SVM, MLP, four tree ensembles,
nearest-class-mean, plus torch trainers) against the three in
`deltas/classifiers/models.py`, and its MLP supports `sample_weight` and
`class_weight='balanced'` directly, which is the only reason the local `NN`
ever needed vendored sklearn internals.

The single mismatch is naming and sign:

    projection_models:  get_threshold()  ->  the decision threshold t,
                                             predict = projection > t
    deltas:             get_bias()       ->  b, with t = -b

`as_deltas_classifier` bridges that. Everything else passes straight through.

    from deltas.classifiers import sibling
    clf = sibling.build('RandomForest').fit(X, y)
    model = overlap.binomial_deltas(clf).fit(X, y)

`build` accepts the model names used by `deltas.pipeline.classifier`
('SVM-rbf', 'MLP', 'Linear', ...) as well as the projection_models class names.
'''
import numpy as np


#: deltas pipeline name -> (projection_models class name, default kwargs)
MODEL_ALIASES = {
    'Linear':            ('Linear', {'max_iter': 2000}),
    'LDA':               ('LDA', {}),
    'SVM':               ('SVM', {'kernel': 'linear'}),
    'SVM-linear':        ('SVM', {'kernel': 'linear'}),
    'SVM-rbf':           ('SVM', {'kernel': 'rbf'}),
    'SVM-rbf-fixed':     ('SVM', {'kernel': 'rbf'}),
    'LinearSVM':         ('LinearSVM', {}),
    'MLP':               ('MLP', {'hidden_layer_sizes': (64, 32),
                                  'max_iter': 600}),
    'MLP-small':         ('MLP', {'hidden_layer_sizes': (20,),
                                  'max_iter': 600}),
    'MLP-deep':          ('MLP', {'hidden_layer_sizes': (64, 128, 64),
                                  'max_iter': 600}),
    'DecisionTree':      ('DecisionTree', {}),
    'RandomForest':      ('RandomForest', {'n_estimators': 200}),
    'GradientBoosting':  ('GradientBoosting', {}),
    'HistGradientBoosting': ('HistGradientBoosting', {}),
    'NearestClassMean':  ('NearestClassMean', {}),
}


def _import_sibling():
    try:
        import projection_models as pm
    except ImportError as exc:          # pragma: no cover - environment issue
        raise ImportError(
            'projection_models is not installed. It is a path dependency of '
            'this project: `uv sync` from the repo root, or see '
            '[tool.uv.sources] in pyproject.toml.') from exc
    return pm


def available():
    '''model names accepted by build()'''
    pm = _import_sibling()
    return sorted(set(MODEL_ALIASES) | set(getattr(pm, '__all__', [])))


class as_deltas_classifier:
    '''
    wrap a projection_models estimator so a deltas model can consume it

    Adds `get_bias()` (the negated threshold) and forwards everything else to
    the wrapped estimator, including attribute access, so the wrapper is
    transparent to code that pokes at `.coef_` and friends.
    '''

    def __init__(self, model):
        if not hasattr(model, 'get_projection'):
            raise AttributeError(
                f'{model} has no get_projection; deltas needs it to reach the '
                f'1-D decision space')
        self.model = model

    # -- the deltas interface ----------------------------------------------
    def get_projection(self, X):
        return np.asarray(self.model.get_projection(X), dtype=float)

    def get_bias(self):
        '''
        deltas stores the boundary as a bias b with threshold t = -b, while
        projection_models reports t directly
        '''
        t = self.model.get_threshold()
        t = np.asarray(t, dtype=float)
        return -t if t.size > 1 else -float(t.reshape(-1)[0])

    def get_threshold(self):
        return self.model.get_threshold()

    # -- sklearn passthrough -----------------------------------------------
    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __getattr__(self, item):
        # only reached for attributes this class does not define
        return getattr(self.__dict__['model'], item)

    def __repr__(self):
        return f'as_deltas_classifier({self.model!r})'


def build(name, wrap=True, **kwargs):
    '''
    construct a projection_models estimator by deltas-pipeline name

    Unknown keyword arguments are passed to the underlying class, overriding
    the defaults in MODEL_ALIASES.
    '''
    pm = _import_sibling()
    cls_name, defaults = MODEL_ALIASES.get(name, (name, {}))
    if not hasattr(pm, cls_name):
        raise ValueError(
            f'unknown model {name!r}. Available: {available()}')
    params = {**defaults, **kwargs}
    model = getattr(pm, cls_name)(**params)
    return as_deltas_classifier(model) if wrap == True else model
