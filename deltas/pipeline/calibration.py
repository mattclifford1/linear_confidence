'''
Calibration splitting: make the reported certificate honest.

What it solves
--------------
Every deltas variant reads a statistic off the *projected training data* and
turns it into a confidence statement:

  * the published separable method reads the empirical class support R_i_hat,
  * the overlap-native methods (deltas.model.overlap) read the miscount m_i(b).

Both then claim "with probability at least 1 - delta_i, the class-i error is at
most U_i".  That claim needs the projected points to be i.i.d. draws from the
projected class-conditional law.  They are not, when the classifier that
*defines* the projection was fitted to those same points: it has already pushed
them towards their own side of the boundary.  R_i_hat comes out too small and
m_i comes out too low, so U_i is optimistic and the certificate is violated
more often than delta_i allows.

Measured over 4 datasets x 10 seeds x 2 classes (experiments/validate_bounds.py)
the observed coverage on training data is 0.81 against a nominal 0.96, and the
shortfall tracks the classifier's train/test optimism almost monotonically:

    dataset          optimism   coverage
    Breast Cancer      .013       0.90
    Heart Disease      .027       0.95
    Pima Diabetes      .121       0.75
    Hepatitis          .208       0.65

This is not a bug in the bound - Clopper-Pearson and DKW are exact/valid - it
is a violated assumption upstream of it.

The method
----------
Sample splitting, as in conformal prediction / split-conformal calibration:

    1. partition the training data into a *fit* part and a *calibration* part,
       stratified by class so both keep the minority,
    2. train the classifier on the fit part ONLY,
    3. compute the deltas certificate on the calibration part.

Step 3's points were never seen by the classifier, so conditional on the fitted
projection they are genuine i.i.d. draws from the projected law and the
binomial model holds exactly.  The guarantee becomes conditional on the fitted
classifier, which is the statement one actually wants: "for *this* deployed
model, the class-i error is at most U_i with probability 1 - delta_i".

Cost: the bound gets looser (mean U_i roughly doubles at cal_frac=0.35, since
the count comes from a third of the data and U_i shrinks like 1/N_cal), and the
classifier is trained on less data.  Both are real, and both are the price of
the certificate being true rather than approximately true.

Why this cannot live inside the estimator
-----------------------------------------
By the time `deltas_model.fit(X, y, clf=clf)` is called the classifier has
already been trained.  Splitting X inside that call would compute the count on
points the classifier had already fitted to, which changes nothing.  The split
has to happen upstream of classifier training, which is why it lives in the
pipeline.  `deltas.pipeline.cached.get_data_and_classifiers(...,
calibration=0.35)` does exactly this, and `fit_calibrated` below is the
self-contained version.

Usage
-----
    from deltas.pipeline import calibration

    # self-contained: give it a way to build a fresh classifier
    fitted, info = calibration.fit_calibrated(
        clf_factory=lambda: models.SVM(kernel='rbf'),
        deltas_factory=lambda clf: overlap.binomial_deltas(
            clf, objective='minimax'),
        X=X, y=y, cal_frac=0.35, seed=0)

    # or through the cached pipeline
    data_clf, clfs = cached.get_data_and_classifiers(
        'Hepatitis', 'SVM-rbf', seed=0, calibration=0.35)
    Xc, yc = data_clf['data_cal']['X'], data_clf['data_cal']['y']
    m = overlap.binomial_deltas(clfs['Baseline']).fit(Xc, yc)
'''
import numpy as np
from sklearn.model_selection import train_test_split


#: below this many points of a class in either part, the split is refused
MIN_PER_CLASS = 2


class CalibrationSplitError(ValueError):
    '''raised when the data cannot support a stratified calibration split'''


def split_calibration(X, y, cal_frac=0.35, seed=0, min_per_class=MIN_PER_CLASS):
    '''
    stratified fit/calibration partition of a training set

    Returns (X_fit, y_fit, X_cal, y_cal). Raises CalibrationSplitError when
    either part would be left with fewer than `min_per_class` points of some
    class - a certificate computed from one minority point is worthless, and
    silently returning it would be worse than failing.
    '''
    X = np.asarray(X)
    y = np.asarray(y).squeeze()
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f'cal_frac must be in (0, 1), got {cal_frac}')

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise CalibrationSplitError('need both classes to split')
    smallest = int(counts.min())
    n_cal = int(np.floor(smallest * cal_frac))
    if n_cal < min_per_class or smallest - n_cal < min_per_class:
        raise CalibrationSplitError(
            f'smallest class has {smallest} points; a {cal_frac:.0%} split '
            f'leaves {n_cal} / {smallest - n_cal}, below the minimum of '
            f'{min_per_class} per part')

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=cal_frac, stratify=y, random_state=seed)
    return X_fit, y_fit, X_cal, y_cal


def fit_calibrated(clf_factory, deltas_factory, X, y, cal_frac=0.35, seed=0,
                   fit_kwargs=None, _print=False):
    '''
    the whole recipe: split -> train classifier on fit part -> certify on the
    calibration part

    clf_factory:    callable() -> unfitted classifier exposing get_projection
    deltas_factory: callable(clf) -> unfitted deltas estimator
    returns (fitted_deltas_model, info_dict)
    '''
    X_fit, y_fit, X_cal, y_cal = split_calibration(
        X, y, cal_frac=cal_frac, seed=seed)

    clf = clf_factory().fit(X_fit, y_fit)
    model = deltas_factory(clf).fit(X_cal, y_cal, **(fit_kwargs or {}))

    info = {'cal_frac': cal_frac,
            'seed': seed,
            'n_fit': int(len(y_fit)),
            'n_cal': int(len(y_cal)),
            'fit_counts': np.unique(y_fit, return_counts=True)[1].tolist(),
            'cal_counts': np.unique(y_cal, return_counts=True)[1].tolist(),
            'clf': clf}
    if _print == True:
        print(f"calibrated fit: {info['n_fit']} fit / {info['n_cal']} cal "
              f"(counts {info['fit_counts']} / {info['cal_counts']})")
    return model, info
