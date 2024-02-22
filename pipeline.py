import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import data_utils
import plots
import normal
import madelon
import projection
import models


def get_data(m1 = [1, 1],
             m2 = [10, 10],
             cov1 = [[1, 0], [0, 1]],
             cov2 = [[1, 0], [0, 1]],
             N1 = 10000,
             N2 = 10000,
             scale = True,
             test_nums=[10000, 10000]):
    data = normal.get_two_classes(means=[m1, m2],
                                  covs=[cov1, cov2],
                                  num_samples=[N1, N2])
    data_test = normal.get_two_classes(means=[m1, m2],
                                       covs=[cov1, cov2],
                                       num_samples=[test_nums[0], test_nums[1]])

    scaler = data_utils.normaliser(data)
    if scale == True:
        data = scaler(data)
        data_test = scaler(data_test)
        m1 = scaler.transform_instance(m1)
        m2 = scaler.transform_instance(m2)

    return {'data': data, 'mean1': m1, 'mean2': m2, 'data_test': data_test}

def get_non_sep_data(N1=10000,
                     N2=10000,
                     scale=True,
                     test_nums=[10000, 10000]):
    data, data_test = madelon.get_non_separable(N1=N1, N2=N2, scale=scale, test_nums=test_nums)

    return {'data': data, 'data_test': data_test}

def get_SMOTE_data(data):
    oversample = SMOTE()
    X, y = oversample.fit_resample(data['X'], data['y'])
    return {'X': X, 'y': y}

def get_classifier(data_clf, model='Linear', balance_clf=False, _plot=True):
    data = data_clf['data']
    SMOTE_data = get_SMOTE_data(data)
    if balance_clf == True:
        weights = 'balanced'
    else:
        weights = None
    if model in ['SVM', 'SVM-linear']:
        clf = models.SVM(kernel='linear', class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='linear', class_weight=weights).fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'SVM-rbf':
        clf = models.SVM(kernel='rbf', class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='rbf', class_weight=weights).fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'Linear':
        clf = models.linear(class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.linear(class_weight=weights).fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'MLP':
        clf = models.NN(class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.NN(class_weight=weights).fit(SMOTE_data['X'], SMOTE_data['y'])
    else:
        raise ValueError(f"model: {model} not in list of available models")

    if _plot == True:
        for name, classif, t_data in zip(['clf', 'SMOTE'], [clf, clf_SMOTE], [data, SMOTE_data]):
            print(name)
            ax, _ = plots._get_axes(None)
            plots.plot_classes(t_data, ax=ax)
            plots.plot_decision_boundary(classif, t_data, ax=ax, probs=False)
            ax.set_title(name)
            plots.plt.show()

    return clf, clf_SMOTE


# make precision for each class
def precision0(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=0)

def precision1(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=1)


def eval_test(clfs, test_data, _print=True, _plot=True):
    # using new class for deltas format

    # predict on both classifiers (original and delta adjusted)
    preds = {}
    for name, clf in clfs.items():
        preds[name] = clf.predict(test_data['X'])

    if _print == True:
        metrics = {'accuracy': accuracy_score,
                   'F1': f1_score,
                   'precision0': precision0,
                   'precision1': precision1,
                   }
        for metric, func in metrics.items():
            for name, y_preds in preds.items():
                print(f"{name} {metric}: {func(test_data['y'], y_preds)}")
            print('')

    if _plot == True:
        def _plot_projection_test_and_grid(X, clf, clf_projecter, y_plot, name, grid=False, ax=None):
            proj_data = {'X': clf_projecter.get_projection(X),
                         'y': clf.predict(X)}
            xp1, xp2 = projection.get_classes(proj_data)
            y_plot -= 0.1

            if grid == True:
                names = [f'{name} clf 1', None]
                m = 'x'
            else:
                names = [f'{name} pred 1', f'{name} pred 2']
                m = 'o'
            ax.scatter(xp1, np.ones_like(xp1)*y_plot, c='b', s=10,
                       label=names[0], marker=m)
            ax.scatter(xp2, np.ones_like(xp2)*y_plot, c='r', s=10,
                       label=names[1], marker=m)
            return y_plot

        _, ax = plt.subplots(1, 1)
        y_plot = 0
        for name, clf in clfs.items():
            # plot test data
            X = test_data['X']
            y_plot = _plot_projection_test_and_grid(
                X, clf, clfs['original'], y_plot, name, False, ax)

            # plot linspace/grid
            X, _ = plots.get_grid_pred(
                clf, test_data, probs=False, flat=True, res=25)
            y_plot = _plot_projection_test_and_grid(
                X, clf, clfs['original'], y_plot, name, True, ax)

            y_plot -= 0.2

        ax.legend()
        ax.plot([0], [-1.5], c='w')
        ax.set_title('original (top) vs deltas (bottom) on test dataset in projected space')
        plots.plt.show()

        # plot in original space
        for name, clf in clfs.items():
            print(name)
            ax, _ = plots._get_axes(None)
            data = {'X': test_data['X'], 'y': preds[name]}
            # data = test_data
            plots.plot_classes(data, ax=ax)
            plots.plot_decision_boundary(
                clf, test_data, ax=ax, probs=False)
            ax.set_title(name)
            plots.plt.show()


