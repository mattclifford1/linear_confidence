import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
import umap

import deltas.plotting.plots as plots
import deltas.utils.projection as projection


# make precision for each class
def precision0(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=0)


def precision1(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=1)


def eval_test(clfs, test_data, _print=True, _plot=True, dim_reducer=None):
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
        # plot in original space
        for name, clf in clfs.items():
            print(name)
            ax, _ = plots._get_axes(None)
            data = {'X': test_data['X'], 'y': preds[name]}
            # data = test_data
            plots.plot_classes(data, ax=ax, dim_reducer=dim_reducer)
            plots.plot_decision_boundary(
                clf, test_data, ax=ax, probs=False, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()

        # plot in projected space
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
        ax.set_title(
            'original (top) vs deltas (bottom) on test dataset in projected space')
        plots.plt.show()
