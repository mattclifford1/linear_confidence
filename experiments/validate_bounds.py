'''
Does the certificate actually hold?

The bound of Sec. "Overlap from the Ground Up" assumes the projected training
points are i.i.d. draws from the projected class-conditional law. They are not,
when the classifier that DEFINES the projection was fitted to those same
points: it has pushed them towards the correct side, so the miscount m_i
under-estimates the true error and the bound comes out optimistic.

This script measures the effect and checks the standard fix - hold out a
calibration split that the classifier never sees, and count on that.

(The same criticism applies to the published separable method, whose empirical
support R_i is likewise computed on the classifier's own training data.)
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deltas.pipeline import cached
from deltas.model import overlap
import deltas.classifiers.models as models

DATASETS = ['Breast Cancer', 'Pima Indian Diabetes', 'Hepatitis',
            'Heart Disease']
SEEDS = range(10)
CAL_FRAC = 0.35


def run(cal_frac=CAL_FRAC):
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            dc, clfs = cached.get_data_and_classifiers(ds, 'SVM-rbf', seed=seed)
            X, y = dc['data']['X'], dc['data']['y']
            Xt, yt = dc['data_test']['X'], dc['data_test']['y']

            # (a) naive: classifier and certificate share all the training data
            m = overlap.binomial_deltas(clfs['Baseline'],
                                        objective='minimax').fit(X, y)
            rows += _score(ds, seed, 'naive', m, Xt, yt)

            # (b) split: classifier on the fit part, certificate on the held-out part
            try:
                Xf, Xc, yf, yc = train_test_split(
                    X, y, test_size=cal_frac, stratify=y, random_state=seed)
                clf2 = models.SVM(kernel='rbf',
                                  C=clfs['Baseline'].C,
                                  gamma=clfs['Baseline'].gamma).fit(Xf, yf)
                m2 = overlap.binomial_deltas(clf2,
                                             objective='minimax').fit(Xc, yc)
                rows += _score(ds, seed, 'split', m2, Xt, yt)
            except ValueError:      # too few minority points to stratify
                pass
    return pd.DataFrame(rows)


def _score(ds, seed, mode, m, Xt, yt):
    p = np.asarray(m.predict(Xt)).squeeze()
    c = m.certified_error()
    out = []
    for cls, ub, d in [(0, c['class 1'], c['delta1']),
                       (1, c['class 2'], c['delta2'])]:
        e = (p[yt == cls] != cls).mean()
        out.append(dict(dataset=ds, seed=seed, mode=mode, cls=cls,
                        err=e, ub=ub, delta=d, covered=e <= ub))
    return out


if __name__ == '__main__':
    df = run()
    df.to_csv('results/bound_validation.csv', index=False)
    piv = df.groupby(['dataset', 'mode']).agg(
        coverage=('covered', 'mean'),
        nominal=('delta', lambda s: 1 - s.mean()),
        mean_ub=('ub', 'mean'), mean_err=('err', 'mean'))
    print(piv.round(3).to_string())
    print()
    print(df.groupby('mode').agg(coverage=('covered', 'mean'),
                                 nominal=('delta', lambda s: 1 - s.mean())
                                 ).round(3).to_string())
